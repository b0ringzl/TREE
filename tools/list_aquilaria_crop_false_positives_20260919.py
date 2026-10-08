"""Read-only model replay and error listing. No fitting or model selection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
import numpy as np
from distillation_core import read_json,save_json,digest
from training_safety import sha256
from train_artocarpus_user_20260918 import heads_from_deployment,predict
from train_aq_ar_diagnostic_20260919 import unchanged,verify_images
from aq_ar_dualview_next_20260919 import load_pair,tensor

RUN=ROOT/'outputs/SSTv1_Aquilaria_UserCrops_20260919_v2'
OUT=RUN/'false_positive_audit_v1'


def main():
    OUT.mkdir(exist_ok=False)
    protocol=read_json(RUN/'protocol.json');unchanged(protocol);torch.set_num_threads(4)
    dev=load_pair(RUN,'dev');rows=dev[0];labels=read_json(RUN/'labels.json')
    assert labels[11]=='Aquilaria sinensis'
    with np.load(RUN/'train_features.npz') as f:
        params=dict(groups=read_json(RUN/'classes.json')['groups'],temp=protocol['temperature'],extra=tensor(f['extra_bank']),scale=float(f['scale']))
    with np.load(RUN/'baseline_predictions.npz') as f:before=f['dev_single']
    model=heads_from_deployment(protocol)
    checkpoint=RUN/'H_crops_targeted_002/step_120.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'])
    after=predict(model,dev[:3],params)
    y=np.array([r['target'] for r in rows]);bp=before.argmax(1);cp=after.argmax(1)
    old=(y!=11)&(bp==11);new=(y!=11)&(cp==11)
    assert int(old.sum())==7 and int(new.sum())==10
    union=np.flatnonzero(old|new);verify_images([rows[i] for i in union])
    details=[]
    for i in union:
        r=rows[int(i)];t=int(y[i]);status='persistent' if old[i] and new[i] else 'new' if new[i] else 'corrected'
        def top(p):return [dict(label=labels[int(j)],score=float(p[j])) for j in np.argsort(-p)[:3]]
        details.append(dict(index=int(i),sample_id=r['sample_id'],status=status,
            dataset_label=r.get('species') or r.get('label') or (labels[t] if t<len(labels) else str(t)),
            source_image=r['source_image'],filename=Path(r['source_image']).name,
            sha256=r.get('audit_sha256') or r.get('source_sha256'),group_id=r['group_id'],
            cohort=r.get('audit_cohort'),baseline_top1=labels[int(bp[i])],candidate_top1=labels[int(cp[i])],
            baseline_aquilaria_score=float(before[i,11]),candidate_aquilaria_score=float(after[i,11]),
            baseline_true_score=float(before[i,t]) if t<len(labels) else None,
            candidate_true_score=float(after[i,t]) if t<len(labels) else None,
            baseline_top3=top(before[i]),candidate_top3=top(after[i]),
            label_authority='Existing development source label; this error listing does NOT independently certify the botanical label'))
    counts=dict(before=7,after=10,new=sum(r['status']=='new' for r in details),
        persistent=sum(r['status']=='persistent' for r in details),corrected=sum(r['status']=='corrected' for r in details))
    assert counts['after']-counts['before']==counts['new']-counts['corrected']
    save_json(OUT/'details.json',dict(counts=counts,records=details,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        definition='Top-1 Aquilaria on a non-Aquilaria-labelled development image; no minimum confidence threshold',
        unchanged_production=True,training=False,rows_hash=digest(rows)))
    np.savez_compressed(OUT/'predictions.npz',baseline=before,candidate=after,targets=y)
    lines=['# 土沉香误报：7 → 10 的逐图明细','',
        '对应 H_crops_targeted_002 / step_120 实验权重、单路整图补边输入。不是双取景，也不是正式 Demo 已替换后的结果。','',
        f"原有 7 张误报中，{counts['persistent']} 张仍误报、{counts['corrected']} 张不再误报；另外新增 {counts['new']} 张，最终为 10 张。",'',
        '“误报”按开发集已有树种标签计算：标签不是土沉香，而 Top-1 是土沉香。不要求分数超过 50%。这些来源标签没有在本次被重新做独立植物学鉴定；如发现标签有问题，应先复核再重算，不可用预测替换标签。','']
    titles={'new':'新增误报','persistent':'持续误报','corrected':'已不再误报土沉香'}
    for status in ('new','persistent','corrected'):
        lines += ['## '+titles[status],'','| 图片 | 开发集标签 | 训练前 Top-1 | 土沉香分数：前 → 后 |','|---|---|---|---:|']
        part=[r for r in details if r['status']==status]
        for r in part:
            path=Path(r['source_image']).as_posix()
            lines.append(f"| [{r['filename']}](<{path}>) | {r['dataset_label']} | {r['baseline_top1']} | {r['baseline_aquilaria_score']:.1%} → {r['candidate_aquilaria_score']:.1%} |")
        for r in part:
            path=Path(r['source_image']).as_posix()
            lines += ['',f"### {r['sample_id']} · {r['dataset_label']}",'',
                f"原文件：[{r['filename']}](<{path}>)；补训后 Top-1：{r['candidate_top1']}。",'',f'![{r["dataset_label"]}](<{path}>)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    unchanged(protocol);print(counts,flush=True)
    for r in details:print(r['status'],r['sample_id'],r['dataset_label'],r['filename'],round(r['baseline_aquilaria_score'],4),round(r['candidate_aquilaria_score'],4),flush=True)


if __name__=='__main__':main()
