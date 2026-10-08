"""Read-only model audit after validation selection; never trains or deploys."""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_whu_current_20260911 import metrics
from training_safety import sha256,verify_backup

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def values(cm):
    tp=np.diag(cm);return np.array([tp.sum()/cm.sum(),np.mean(2*tp/np.maximum(cm.sum(0)+cm.sum(1),1))])

def main(out,online):
    if read(out/'status.json')['stage']!='complete':raise RuntimeError('Training/evaluation not complete')
    target=out/'audit';target.mkdir(exist_ok=False)
    result=read(out/'results.json');names=read(out/'classes.json');rows=read(out/'test_manifest.json')
    lock=read(out/'selection_lock.json');protocol=read(out/'protocol.json')
    base_dir=Path(protocol['baseline_results']).parent
    old_rows=read(base_dir/'test_manifest.json')
    assert [r['sample_key'] for r in rows]==[r['sample_key'] for r in old_rows]
    old=np.load(base_dir/'all_probabilities.npz');current=np.load(out/'test_probabilities.npz')
    pred={k:current[k].argmax(1) for k in ('baseline','candidate')}
    pred['yolo']=old['whu_yolo11s_cls'].argmax(1)
    assert np.array_equal(pred['baseline'],old['bioclip_whu18_zeroshot'].argmax(1))
    truth=np.array([int(r['class_index']) for r in rows]);computed={}
    for k,p in pred.items():
        computed[k]=metrics(truth,[names[i] for i in p],names)
        assert computed[k]['accuracy']==result[k]['accuracy']
        assert computed[k]['macro_f1']==result[k]['macro_f1']
    segment=np.array(['_'.join(r['sample_key'].split('_')[:2]) for r in rows]);groups=sorted(set(segment))
    cm={k:np.zeros((len(groups),18,18),dtype=int) for k in pred}
    for g,key in enumerate(groups):
        mask=segment==key
        for k,p in pred.items():np.add.at(cm[k][g],(truth[mask],p[mask]),1)
    rng=np.random.default_rng(20260911);draws={k:[] for k in ('baseline','yolo')}
    for _ in range(2000):
        idx=rng.integers(0,len(groups),len(groups));v={k:values(c[idx].sum(0)) for k,c in cm.items()}
        for k in draws:draws[k].append(v['candidate']-v[k])
    comparisons={}
    for k,v in draws.items():
        bounds=np.quantile(v,[.025,.975],axis=0)*100
        observed=(values(cm['candidate'].sum(0))-values(cm[k].sum(0)))*100
        comparisons['candidate_minus_'+k]={name:dict(delta_pp=float(observed[i]),ci95_pp=bounds[:,i].tolist()) for i,name in enumerate(('OA','macro_F1'))}
    changes=[]
    for i,r in enumerate(rows):
        if pred['candidate'][i]!=pred['baseline'][i]:
            changes.append(dict(sample_key=r['sample_key'],source_path=r['source_path'],truth=names[truth[i]],
                baseline=names[pred['baseline'][i]],candidate=names[pred['candidate'][i]],
                baseline_correct=bool(pred['baseline'][i]==truth[i]),candidate_correct=bool(pred['candidate'][i]==truth[i])))
    write(target/'changed_predictions.json',changes)
    integrity=[]
    for run in lock['runs']:
        manifest=verify_backup(run['backup'])
        assert sha256(Path(run['run'])/'best.pt')==run['weights_sha256']
        integrity.append(dict(run=run['recipe'],backup=run['backup'],verified_files=len(manifest['files'])))
    for p,h in protocol['protected_sources'].items():assert sha256(p)==h
    summary=dict(metrics_recomputed_equal=True,baseline_all_predictions_match_previous=True,
        bootstrap=dict(unit='Official test road/sequence cluster (not independent trees)',clusters=groups,
            draws=2000,seed=20260911,comparisons=comparisons,
            caveat='Conditional regression uncertainty, not evidence of unseen-test or seed stability; clusters with few rare classes widen macro-F1 uncertainty.'),
        corrected=sum(not r['baseline_correct'] and r['candidate_correct'] for r in changes),
        regressed=sum(r['baseline_correct'] and not r['candidate_correct'] for r in changes),
        backup_verification=integrity,production_unchanged=True)
    if online:summary['online_reload_checks']=online_check(out,rows,lock,current['candidate'])
    write(target/'verification.json',summary)
    lines=['# WHU18 训练后独立核验','',
        '逐株重算指标与训练报告一致；零样本基线全部 Top-1 预测与前轮实测一致。',
        f"相对零样本：纠正 {summary['corrected']} 株，同时有 {summary['regressed']} 株由对变错；逐株列表见 changed_predictions.json。",'',
        '## 按采集序列配对 bootstrap（2,000 次）','',
        '| 比较 | OA 差值（百分点） | 95% 区间 | Macro-F1 差值（百分点） | 95% 区间 |',
        '|---|---:|---|---:|---|']
    for k,v in comparisons.items():
        a=v['OA'];f=v['macro_F1']
        lines.append(f"| {k} | {a['delta_pp']:+.2f} | [{a['ci95_pp'][0]:+.2f}, {a['ci95_pp'][1]:+.2f}] | {f['delta_pp']:+.2f} | [{f['ci95_pp'][0]:+.2f}, {f['ci95_pp'][1]:+.2f}] |")
    lines+=['','单位为 26 个官方测试采集序列，不把同段树木视为完全独立样本。区间仅反映这批已知回归数据的抽样变动，不代表盲测或多 seed 稳定性。',
        '', '备份、候选权重及原部署文件哈希均核验通过；无自动部署。']
    if online:lines+=['', '另外独立重载教师及所选 head，对预先固定的测试索引 0、1000、2000 原图进行单张推理，与缓存测试预测核对；细节见 verification.json。']
    (target/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def online_check(out,rows,lock,expected):
    import torch
    from PIL import Image,ImageOps
    from tree_species_teacher_demo.engine import Engine,decode
    from bioclip_probe import ResidualHead
    engine=Engine();engine.load()
    bank=torch.tensor(np.load(out/'text_bank.npy'),device='cuda');head=None
    if lock['selected']!='zero_shot':
        ck=torch.load(Path(lock['selected_run']['run'])/'best.pt',map_location='cpu',weights_only=False)
        head=ResidualHead(ck['config']['architecture'],1024,18).cuda().eval()
        head.load_state_dict(ck['model_state'])
    checks=[]
    with torch.inference_mode():
        for idx in (0,1000,2000):
            r=rows[idx];assert sha256(r['source_path'])==r['sha256']
            with Image.open(r['source_path']) as source:
                im=ImageOps.exif_transpose(source).convert('RGB');edge=min(1024,max(im.size))
                im=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128))
                buf=io.BytesIO();im.save(buf,format='JPEG',quality=95)
            pixels=decode(buf.getvalue())[0]
            x=engine.model.encode_image(((pixels[None].cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            z=x@bank.T*engine.model.logit_scale.exp().float()/engine.choice['calibration_temperature']
            p=(head(x,z) if head is not None else z).softmax(-1)[0].cpu().numpy()
            delta=float(np.abs(p-expected[idx]).max());same=bool(p.argmax()==expected[idx].argmax())
            assert same and delta<1e-4,(idx,delta)
            checks.append(dict(index=idx,sample_key=r['sample_key'],same_top1=same,max_probability_delta=delta))
    return checks

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--online-check',action='store_true')
    a=p.parse_args();main(a.out.resolve(),a.online_check)
