"""WHU18 frozen BioCLIP adaptation, explicit review quarantine and per-fit backups.

Never updates an encoder, previous run, human decision or deployed model.
The already inspected official-test-derived set is regression evidence only.
"""
import argparse
import csv
import gc
import io
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training_safety import backup_before_training,verify_backup,deployed_artifacts,sha256
from bioclip_probe import ResidualHead
from evaluate_whu_current_20260911 import metrics,WHU,EXP
from distillation_core import read_json as read,save_json as write

BASELINE=ROOT/'outputs/whu_current_comparison_20260911_v2_single'
REVIEW_PATHS=[ROOT/'lidar data/whu/derived/tscmdl'/p for p in [
    'c1_full_shared_dataset/user_visual_review.json',
    'd1_four_class_image_quality/user_visual_review.json']]
RECIPES=[dict(name='linear',architecture='linear',lr=.03),
         dict(name='residual_mlp',architecture='mlp',lr=.003)]
SEED=20260911

def status(out,stage,**extra):
    data=dict(stage=stage,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),deployed=False,**extra)
    write(out/'status.json',data);print(json.dumps(data,ensure_ascii=False),flush=True)

def review_quarantine(review_docs):
    quarantine={}
    for doc in review_docs:
        for key,row in doc['reviews'].items():
            # A later replacement's acceptance does not accept the old source crop.
            if row.get('action')!='keep' or row.get('rating') not in {'pass','borderline'}:
                quarantine.setdefault(key,[]).append(row)
    return quarantine

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    source_manifest=WHU/'yolo_dataset/inventory.csv'
    inventory=list(csv.DictReader(source_manifest.open(encoding='utf-8-sig')))
    names=read(WHU/'yolo_dataset/dataset_summary.json')['class_names']
    reviews=[read(p) for p in REVIEW_PATHS]; quarantine=review_quarantine(reviews)
    rows={s:[r for r in inventory if r['split']==s and (s=='test' or r['sample_key'] not in quarantine)]
          for s in ('train','val','test')}
    assert len(names)==18
    assert {s:len(r) for s,r in rows.items()}==dict(train=12512,val=481,test=3399)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        for field in ('sample_key','sha256'):
            assert not ({r[field] for r in rows[a]} & {r[field] for r in rows[b]})
    official_path=ROOT/'lidar data/whu/test_split.txt'
    official={(r['road_id'],r['ply_id']) for r in csv.DictReader(official_path.open(encoding='utf-8-sig'))}
    for s,records in rows.items():
        for r in records:
            assert names[int(r['class_index'])]==r['scientific_name']
            segment=tuple(r['sample_key'].split('_')[:2])
            assert (segment in official)==(s=='test')
        write(out/f'{s}_manifest.json',records)
    write(out/'accepted_training_manifest.json',dict(
        acceptance_policy='Official benchmark labels accepted for this task; unreviewed benchmark images are not pseudo-labels. Explicit rejected/rework/pending original crops quarantined using all two relevant human review sources.',
        rows=rows['train']))
    write(out/'review_snapshot.json',dict(sources=[str(p) for p in REVIEW_PATHS],documents=reviews))
    write(out/'quarantine.json',dict(decisions=quarantine,
        excluded_development=[r for r in inventory if r['split']!='test' and r['sample_key'] in quarantine],
        retained_fixed_test=[r for r in rows['test'] if r['sample_key'] in quarantine]))
    write(out/'classes.json',names)
    teacher=Path(read(EXP/'protocol.json')['teacher_weight'])
    selection=read(EXP/'selection_lock.json')
    protected=[teacher,*deployed_artifacts(),
               *[Path(r['run'])/'best.pt' for r in selection['selected_runs']],
               EXP/'classes.json',EXP/'protocol.json',EXP/'teacher_choice.json',EXP/'selection_lock.json',
               EXP/'cache_metadata.json', ROOT/'outputs/distillation/20260908_0045/aliases.json',
               ROOT/'spacescan_trees_demo_v1/species_names.json',ROOT/'spacescan_trees_demo_v1/server.py',
               ROOT/'tree_species_teacher_demo/engine.py']
    protected=list(dict.fromkeys(p.resolve() for p in protected))
    hashes={str(p):sha256(p) for p in protected}
    write(out/'original_evaluation_snapshot.json',read(BASELINE/'results.json'))
    write(out/'protocol.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        architecture='Frozen BioCLIP 2.5 ViT-H-14 + 18-output residual linear / 256-hidden-unit MLP head',
        recipes=RECIPES,seed=SEED,epochs=200,optimizer='AdamW',weight_decay=.01,
        loss='Inverse-class-frequency weighted GT cross entropy; no pseudo-labels, no image augmentation',
        train_count=len(rows['train']),val_count=len(rows['val']),test_count=len(rows['test']),
        label_policy='Official 18 named species; no Ficus merge or other relabel; Others not in this experiment',
        prompt='a photo of {scientific_name}',temperature=read(EXP/'teacher_choice.json')['calibration_temperature'],
        input='Same SpaceScan pad1024+JPEG95 then teacher resize/center224; frozen features float32; train/val encoding batch8, regression encoding batch1',
        selection='Epoch and recipe by validation macro-F1, then OA, then earlier epoch/order; initial zero-shot state eligible. One seed per recipe, no test tuning.',
        baseline_results=str(BASELINE/'results.json'),baseline_sha256=sha256(BASELINE/'results.json'),
        source_inventory=str(source_manifest),source_inventory_sha256=sha256(source_manifest),
        official_split_sha256=sha256(official_path),review_sha256={str(p):sha256(p) for p in REVIEW_PATHS},
        protected_sources=hashes,teacher_weight=str(teacher),teacher_sha256=hashes[str(teacher)],
        test_access='Test images encoded only after selection_lock.json. Test is previously inspected regression, not new unbiased evidence.',
        deployed=False,backbone_trainable=False))
    status(out,'prepared',counts={s:len(r) for s,r in rows.items()},quarantined_development=sum(r['split']!='test' and r['sample_key'] in quarantine for r in inventory))
    return rows,names,protected

def assert_sources(out):
    for p,h in read(out/'protocol.json')['protected_sources'].items():
        if sha256(p)!=h:raise RuntimeError('Protected source changed: '+p)
    for p,h in read(out/'protocol.json')['review_sha256'].items():
        if sha256(p)!=h:raise RuntimeError('Human review changed during experiment: '+p)

def encode(out,engine,bank,rows,phase,batch):
    if phase=='test' and not (out/'selection_lock.json').is_file():
        raise RuntimeError('Test encoding before validation selection lock')
    from tree_species_teacher_demo.engine import decode
    xs=[];zs=[];started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            pixels=[]
            for r in rows[start:start+batch]:
                if sha256(r['source_path'])!=r['sha256']:raise RuntimeError('Image hash changed: '+r['sample_key'])
                with Image.open(r['source_path']) as source:
                    im=ImageOps.exif_transpose(source).convert('RGB'); edge=min(1024,max(im.size))
                    im=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128))
                    buf=io.BytesIO();im.save(buf,format='JPEG',quality=95)
                pixels.append(decode(buf.getvalue())[0])
            x=engine.model.encode_image(((torch.stack(pixels).cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            z=(x@bank.T*engine.model.logit_scale.exp().float())/engine.choice['calibration_temperature']
            xs.append(x.cpu().numpy());zs.append(z.cpu().numpy())
            if start%400==0:
                status(out,'encoding_'+phase,done=min(start+batch,len(rows)),total=len(rows),seconds=round(time.perf_counter()-started,1))
    x=np.concatenate(xs);z=np.concatenate(zs)
    if not np.isfinite(x).all() or not np.isfinite(z).all():raise RuntimeError('Nonfinite cache')
    path=out/f'{phase}_features.npz'
    np.savez_compressed(path,embeddings=x,baseline_logits=z,
                        sample_keys=np.array([r['sample_key'] for r in rows]))
    write(out/f'{phase}_cache.json',dict(sha256=sha256(path),rows_sha256=sha256(out/f'{phase}_manifest.json'),
        teacher_sha256=engine.encoder_sha256,embedding_dtype=str(x.dtype),batch_size=batch,count=len(rows)))
    return torch.tensor(x,device='cuda'),torch.tensor(z,device='cuda')

def scored(head,x,z):
    with torch.inference_mode():
        if head is not None:head.eval()
        return (head(x,z) if head is not None else z).softmax(-1).cpu().numpy()

def score(rows,probs,names):
    return metrics([int(r['class_index']) for r in rows],[names[i] for i in probs.argmax(1)],names)

def fit(out,recipe,names,rows,cache,protected):
    run=out/'runs'/recipe['name'];run.mkdir(parents=True,exist_ok=False)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    config=dict(**recipe,seed=SEED,epochs=200,weight_decay=.01,features=1024,classes=18,baseline_logits_are_temperature_scaled=True)
    head=ResidualHead(recipe['architecture'],features=1024,classes=18).cuda()
    torch.save(dict(model_state=head.state_dict(),config=config),run/'initial_head.pt')
    write(run/'config.json',config)
    snapshots=[out/p for p in ['protocol.json','classes.json','review_snapshot.json','quarantine.json',
        'accepted_training_manifest.json','train_manifest.json','val_manifest.json','test_manifest.json',
        'original_evaluation_snapshot.json','baseline_validation.json','train_cache.json','val_cache.json','text_bank.npy']]
    inputs=[run/'initial_head.pt',run/'config.json',*protected,*snapshots,Path(__file__),ROOT/'tools/bioclip_probe.py',ROOT/'tools/training_safety.py']
    status(out,'backing_up',recipe=recipe['name'])
    backup=backup_before_training(run,inputs,dict(config=config,fit_must_follow_successful_verification=True))
    manifest=verify_backup(backup)
    inventory=[]
    for r in manifest['files']:
        dest=(backup/r['backup']).resolve()
        if os.path.samefile(dest,r['source']):raise RuntimeError('Backup cannot be a hardlink')
        inventory.append(dict(source=r['source'],backup=str(dest),bytes=r['bytes'],sha256=r['sha256']))
    write(run/'backup_verified.json',dict(verified_at=datetime.now(timezone.utc).isoformat(),physical_copies=True,files=inventory))
    # No optimizer construction or fitting before the verified physical backup above.
    status(out,'backup_verified_starting_fit',recipe=recipe['name'],backup=str(backup))
    x,z=cache['train'];xv,zv=cache['val']
    y=torch.tensor([int(r['class_index']) for r in rows['train']],device='cuda')
    counts=torch.bincount(y,minlength=18).float();weights=(1/counts)[y];weights/=weights.mean()
    optimizer=torch.optim.AdamW(head.parameters(),lr=recipe['lr'],weight_decay=.01)
    initial=score(rows['val'],scored(head,xv,zv),names)
    best=(initial['macro_f1'],initial['accuracy'],0);best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};history=[]
    for epoch in range(1,201):
        head.train();optimizer.zero_grad(set_to_none=True)
        loss=(F.cross_entropy(head(x,z),y,reduction='none')*weights).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),10,error_if_nonfinite=True);optimizer.step()
        m=score(rows['val'],scored(head,xv,zv),names);key=(m['macro_f1'],m['accuracy'],-epoch)
        if key>best:
            best=key;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append(dict(epoch=epoch,loss=float(loss),val_macro_f1=m['macro_f1'],val_accuracy=m['accuracy']))
        if epoch%25==0:status(out,'fitting',recipe=recipe['name'],epoch=epoch,best_epoch=-best[2],best_val_macro_f1=best[0])
    assert max(int(v['step'].item()) for v in optimizer.state.values())==200
    torch.save(dict(model_state=head.state_dict(),optimizer_state=optimizer.state_dict(),config=config,epoch=200),run/'last.pt')
    head.load_state_dict(best_state);val_probs=scored(head,xv,zv);val=score(rows['val'],val_probs,names)
    torch.save(dict(model_state=best_state,config=config,class_names=names,epoch=-best[2],
                    teacher_sha256=read(out/'protocol.json')['teacher_sha256'],
                    prompt='a photo of {scientific_name}',temperature=read(out/'protocol.json')['temperature'],
                    deployed=False),run/'best.pt')
    np.save(run/'val_probabilities.npy',val_probs);write(run/'history.json',history);write(run/'val_metrics.json',val)
    verify_backup(backup);assert_sources(out)
    return dict(recipe=recipe['name'],architecture=recipe['architecture'],run=str(run),best_epoch=-best[2],
                val_macro_f1=val['macro_f1'],val_accuracy=val['accuracy'],weights_sha256=sha256(run/'best.pt'),backup=str(backup))

def evaluate(out,rows,names,xt,zt):
    lock=read(out/'selection_lock.json'); probs=scored(None,xt,zt);baseline=score(rows['test'],probs,names)
    if lock['selected']=='zero_shot':candidate_probs=probs
    else:
        selected=lock['selected_run'];path=Path(selected['run'])/'best.pt'
        if sha256(path)!=selected['weights_sha256']:raise RuntimeError('Selected head changed')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['class_names']!=names or ck['teacher_sha256']!=read(out/'protocol.json')['teacher_sha256']:
            raise RuntimeError('Selected checkpoint incompatible')
        head=ResidualHead(ck['config']['architecture'],1024,18).cuda();head.load_state_dict(ck['model_state']);candidate_probs=scored(head,xt,zt)
    candidate=score(rows['test'],candidate_probs,names)
    assert baseline['accuracy']==read(BASELINE/'results.json')['fresh']['bioclip_whu18_zeroshot']['accuracy']
    np.savez_compressed(out/'test_probabilities.npz',baseline=probs,candidate=candidate_probs)
    write(out/'test_predictions.json',[dict(sample_key=r['sample_key'],species=r['scientific_name'],
        baseline=names[probs[i].argmax()],candidate=names[candidate_probs[i].argmax()]) for i,r in enumerate(rows['test'])])
    comparison=read(BASELINE/'results.json')
    results=dict(baseline=baseline,candidate=candidate,yolo=comparison['fresh']['whu_yolo11s_cls'],
        historical_fusion=comparison['historical']['fusion'],selection=lock,
        test_scope='Previously inspected 3399-tree regression; full fixed test including previously flagged images, no cherry-picking',deployed=False)
    write(out/'results.json',results);assert_sources(out)
    write(out/'production_verification.json',dict(unchanged=True,protected_sources=read(out/'protocol.json')['protected_sources']))
    report(out,results,names)
    status(out,'complete',baseline_accuracy=baseline['accuracy'],candidate_accuracy=candidate['accuracy'],
           candidate_macro_f1=candidate['macro_f1'],selected=lock['selected'],report=str(out/'REPORT.md'))

def report(out,result,names):
    b=result['baseline'];c=result['candidate'];y=result['yolo'];lock=result['selection']
    pct=lambda n:f'{100*n:.2f}%'
    lines=['# WHU18 BioCLIP 专项训练结果','',
        '冻结 BioCLIP 2.5 编码器，只训练 18 类轻量残差分类头；没有更新正式 Demo。',
        '', '## 数据与审核','',
        '- 官方 WHU-STree-NJ 18 个具名树种（不含 Others）；训练 12,512 株、验证 481 株、历史测试 3,399 株。',
        '- 合并 C1 和 D1 两轮人工审核，87 个训练样本、59 个验证样本被隔离。待重裁/待换视角不等于可训练。未拿模型预测替代真值。',
        '- 测试集完整保留，包括其中 68 张此前被标为质量问题的原图；不因难样本导致低分而删测试图。',
        '- 每张实际使用图像 SHA 校验；训练/验证/测试 ID 和哈希不重叠，训练和验证不进入官方测试序列。未证明不同序列绝无同株重复。',
        '', '## 预先固定的训练设置','',
        '- 线性残差头（lr=0.03）与 256 隐层 MLP 残差头（lr=0.003），各一个 seed=20260911、200 次全批次更新、AdamW、weight decay=0.01。',
        '- 使用真实类别的逆频率加权交叉熵；无伪标签、无增强、无主干微调。',
        '- 只依据同一验证集 Macro-F1→OA→更早 epoch 选择；保留 epoch 0 的零样本基线获选机会。单次 seed 结果不代表多次训练稳定性。',
        '- 测试图像仅在 selection_lock.json 写入后编码；测试集已在前轮查看，因此本轮属于开发后的回归证据，而非新的无偏测试。',
        '', '## 验证集选模','', '| 配置 | 选中 epoch | 验证 OA | 验证 Macro-F1 |','|---|---:|---:|---:|']
    for fit_result in lock['runs']:
        lines.append(f"| {fit_result['recipe']} | {fit_result['best_epoch']} | {pct(fit_result['val_accuracy'])} | {pct(fit_result['val_macro_f1'])} |")
    lines+=['',f"最终选择：{lock['selected']}；没有按测试成绩换配置。",'', '## 相同 3,399 株测试树的分类结果','',
        '| 模型 | OA | Macro-F1 | 类别 mIoU |','|---|---:|---:|---:|']
    for title,v in [('训练前 BioCLIP18 零样本（本轮重算）',b),('本轮适配候选',c),('原 WHU YOLO11s-cls（前轮实测）',y),('原 YOLO+PTv2 融合（历史记录）',result['historical_fusion'])]:
        lines.append(f"| {title} | {pct(v['accuracy'])} | {pct(v['macro_f1'])} | {pct(v['mean_class_iou'])} |")
    lines+=['',f"相较零样本：OA {(c['accuracy']-b['accuracy'])*100:+.2f} 个百分点，Macro-F1 {(c['macro_f1']-b['macro_f1'])*100:+.2f} 个百分点。相较 YOLO：OA {(c['accuracy']-y['accuracy'])*100:+.2f} 个百分点。",'',
        '这是本项目的 18 类协议，不可据此声称超过论文包含 Others 的 19 类结果；使用既有单木裁片，没有评估检测/分割精度。',
        '', '## 每类变化：召回率','', '| 树种 | 测试株数 | 零样本 | 本轮适配 | YOLO | 适配－零样本 |','|---|---:|---:|---:|---:|---:|']
    for i,n in enumerate(names):
        bc=b['per_class'][i];cc=c['per_class'][i];yc=y['per_class'][i]
        lines.append(f"| {n} | {cc['support']} | {pct(bc['recall'])} | {pct(cc['recall'])} | {pct(yc['recall'])} | {(cc['recall']-bc['recall'])*100:+.2f} pp |")
    lines+=['','## 可恢复性与部署','',
        '- 每个 fit 都有独立的 pre_training_backup：实拷初始 head、冻结教师、已有教师头、YOLO/PTv2/融合及相关部署文件、标签、审核、划分、配置、基线。',
        '- backup_manifest.json 保存 SHA/大小；backup_verified.json 额外保存每份备份的绝对路径并验证不是硬链接。备份验证成功后才构造优化器并拟合。',
        '- training_safety.verify_backup 与 protected_sources 核验通过，原部署文件保持不变。不得直接把 18 类 head 替换 148 类服务；切换需要独立确认并保证标签/编码器兼容。',
        '- 可用候选权重位于所选 runs 子目录的 best.pt；原编码器仍是依赖，轻量 head 并不意味着推理主干变小。',
        '', '## 产物','',
        '- protocol.json / accepted_training_manifest.json / review_snapshot.json / quarantine.json：固定协议、可训练样本及排除依据。',
        '- runs/*/history.json / val_metrics.json / initial_head.pt / best.pt / last.pt：训练轨迹、验证选模和权重。',
        '- selection_lock.json / test_predictions.json / test_probabilities.npz / results.json：选择锁、逐株预测、分数与逐类混淆矩阵。',
        '- production_verification.json：部署文件哈希核验。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main(out):
    rows,names,protected=prepare(out)
    from tree_species_teacher_demo.engine import Engine
    import open_clip
    status(out,'loading_frozen_encoder')
    engine=Engine();engine.load()
    assert not any(p.requires_grad for p in engine.model.parameters())
    with torch.inference_mode():
        tokens=open_clip.get_tokenizer('ViT-H-14')(['a photo of '+s for s in names]).cuda()
        bank=engine.model.encode_text(tokens,normalize=True).float()
    np.save(out/'text_bank.npy',bank.cpu().numpy())
    cache={s:encode(out,engine,bank,rows[s],s,8) for s in ('train','val')}
    base=score(rows['val'],scored(None,*cache['val']),names);write(out/'baseline_validation.json',base)
    assert_sources(out)
    runs=[fit(out,r,names,rows,cache,protected) for r in RECIPES]
    winner=max(runs,key=lambda r:(r['val_macro_f1'],r['val_accuracy']))
    selected=winner['recipe'] if (winner['val_macro_f1'],winner['val_accuracy'])>(base['macro_f1'],base['accuracy']) else 'zero_shot'
    write(out/'selection_lock.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),
        selected=selected,selected_run=winner if selected!='zero_shot' else None,runs=runs,
        baseline_val_macro_f1=base['macro_f1'],baseline_val_accuracy=base['accuracy'],test_images_encoded=False))
    status(out,'selection_locked',selected=selected)
    del cache;gc.collect();torch.cuda.empty_cache()
    xt,zt=encode(out,engine,bank,rows['test'],'test',1)
    del engine,bank;gc.collect();torch.cuda.empty_cache()
    evaluate(out,rows,names,xt,zt)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:main(args.out.resolve())
    except Exception as exc:
        if args.out.exists():status(args.out,'failed',error=repr(exc))
        raise
