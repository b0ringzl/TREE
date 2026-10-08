"""Inference-only WHU18 regression. No fit, checkpoint selection or deployment."""
import argparse
import csv
import gc
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_safety import sha256

WHU = ROOT/'experiments/whu18_yolo11_ptv2_fusion_20260826'
EXP = ROOT/'outputs/distillation/bioclip_probe_20260908_0227'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def metrics(truth, predicted, names):
    # Extra prediction-only columns penalize out-of-catalog errors without inventing GT classes.
    n = len(names)
    cm = np.zeros((n, n+1), dtype=int)
    for y, pred in zip(truth, predicted):
        cm[y, names.index(pred) if pred in names else n] += 1
    tp = np.diag(cm[:,:n]); support = cm.sum(1); guessed = cm[:,:n].sum(0)
    recall = tp / np.maximum(support, 1)
    precision = tp / np.maximum(guessed, 1)
    f1 = 2*tp / np.maximum(support+guessed, 1)
    iou = tp / np.maximum(support+guessed-tp, 1)
    return dict(n=int(len(truth)), accuracy=float(tp.sum()/len(truth)), macro_f1=float(f1.mean()),
                mean_class_iou=float(iou.mean()), balanced_accuracy=float(recall.mean()),
                outside_predictions=int(cm[:,-1].sum()), confusion_matrix=cm.tolist(),
                per_class=[dict(species=s,support=int(support[i]),precision=float(precision[i]),
                                recall=float(recall[i]),f1=float(f1[i]),iou=float(iou[i])) for i,s in enumerate(names)])

def main(out, batch):
    out.mkdir(parents=True, exist_ok=False)
    inventory = list(csv.DictReader((WHU/'yolo_dataset/inventory.csv').open(encoding='utf-8-sig')))
    names = read(WHU/'yolo_dataset/dataset_summary.json')['class_names']
    rows = [r for r in inventory if r['split']=='test']
    assert len(rows)==3399 and len(names)==18
    assert len({r['sample_key'] for r in rows})==len(rows)
    splits={s:[r for r in inventory if r['split']==s] for s in ('train','val','test')}
    for a,b in [('train','test'),('val','test'),('train','val')]:
        assert not ({r['sample_key'] for r in splits[a]} & {r['sample_key'] for r in splits[b]})
        assert not ({r['sha256'] for r in splits[a]} & {r['sha256'] for r in splits[b]})
    for r in rows:
        assert names[int(r['class_index'])]==r['scientific_name']
        assert sha256(r['source_path'])==r['sha256'], r['source_path']
    original = read(EXP/'classes.json')
    labels=original['labels']+['Wodyetia bifurcata']
    weights=[Path(read(EXP/'protocol.json')['teacher_weight']),
             WHU/'yolo11s_cls/run/weights/selected_by_val_macro_f1.pt']
    weights += [Path(r['run'])/'best.pt' for r in read(EXP/'selection_lock.json')['selected_runs']]
    before={str(p):sha256(p) for p in weights}
    write(out/'protocol.json', dict(task='WHU18 fixed historical official-test-derived regression',
        training=False,deployed=False,seed=20260911,test_used_for_selection=False,
        sample_count=len(rows),class_names=names,original_demo_labels=labels,
        supported_whu_classes=[s for s in names if s in labels],
        inventory_sha256=sha256(WHU/'yolo_dataset/inventory.csv'),test_images_sha_verified=True,
        split_sample_and_image_hash_disjoint=True,weight_sha256_before=before,
        candidate_policy='WHU18 zero-shot uses the exact 18 GT class names, fixed a photo of {} prompt; no test tuning.',
        preprocessing='Current SpaceScan prepare(): aspect-preserving pad to min(1024,max(H,W)), JPEG95; teacher resize224/center224.',
        caveats=['Paper uses 19 classes including Others; local subset excludes Others.',
                 'Tree classification only, using provided crops; not detection or segmentation accuracy.',
                 'Historical test already inspected; not fresh unbiased evidence.',
                 'BioCLIP pretraining overlap cannot be ruled out.']))
    write(out/'test_manifest.json',rows)
    print('Protocol frozen; all 3399 test hashes verified; loading teacher',flush=True)
    from tree_species_teacher_demo.engine import Engine, decode
    from distillation_core import collapse_logits
    import open_clip
    from torch.nn import functional as F
    engine=Engine(); engine.load()
    tokenizer=open_clip.get_tokenizer('ViT-H-14')
    with torch.inference_mode():
        bank=engine.model.encode_text(tokenizer(['a photo of '+s for s in names]).cuda(),normalize=True).float()
    results={k:[] for k in ['demo148_original','demo148_adapted','bioclip_whu18_zeroshot']}
    started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            pixels=[]
            for r in rows[start:start+batch]:
                with Image.open(r['source_path']) as source:
                    im=ImageOps.exif_transpose(source).convert('RGB')
                    edge=min(1024,max(im.size))
                    im=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128))
                    buf=io.BytesIO(); im.save(buf,format='JPEG',quality=95)
                pixels.append(decode(buf.getvalue())[0])
            x=engine.model.encode_image(((torch.stack(pixels).cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            scale=engine.model.logit_scale.exp().float(); raw=x@engine.bank.T*scale
            extra=x@engine.extra_bank.T*scale; t=engine.choice['calibration_temperature']
            results['demo148_original'].append((torch.cat([collapse_logits(raw,original['groups']),extra],-1)/t).softmax(-1).cpu().numpy())
            xx=x.half().float(); extra_adapted=xx@engine.extra_bank.T*scale
            adapted=torch.stack([(torch.cat([collapse_logits(h(xx,raw),original['groups']),extra_adapted],-1)/t).softmax(-1) for h in engine.heads]).mean(0)
            results['demo148_adapted'].append(adapted.cpu().numpy())
            results['bioclip_whu18_zeroshot'].append((x@bank.T*scale/t).softmax(-1).cpu().numpy())
            if start%(batch*20)==0:
                print(f'Teacher {min(start+batch,len(rows))}/{len(rows)} elapsed {time.perf_counter()-started:.1f}s',flush=True)
    results={k:np.concatenate(v) for k,v in results.items()}
    np.savez_compressed(out/'teacher_probabilities.npz',**results)
    del engine,bank,x,xx,raw,adapted; gc.collect(); torch.cuda.empty_cache()
    print('Teacher complete; loading WHU YOLO',flush=True)
    from ultralytics import YOLO
    from ultralytics.data.augment import classify_transforms
    net=YOLO(str(weights[1])).model.cuda().eval().requires_grad_(False)
    summary=read(WHU/'yolo_dataset/dataset_summary.json')
    assert [net.names[i] for i in range(18)]==[summary['class_folders'][str(i)] for i in range(18)]
    transform=classify_transforms(size=224)
    yolo=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            images=[]
            for r in rows[start:start+32]:
                with Image.open(r['source_path']) as im: images.append(transform(im.convert('RGB')))
            prediction=net(torch.stack(images).cuda())
            if isinstance(prediction,tuple):prediction=prediction[0]
            yolo.append(prediction.cpu().numpy())
    results['whu_yolo11s_cls']=np.concatenate(yolo)
    np.savez_compressed(out/'all_probabilities.npz',**results)
    truth=[int(r['class_index']) for r in rows]
    report={}; all_predictions={}
    for key,prob in results.items():
        candidates=labels if key.startswith('demo148') else names
        predicted=[candidates[i] for i in prob.argmax(1)]
        all_predictions[key]=predicted
        report[key]=metrics(truth,predicted,names)
        report[key]['top3']=float(np.mean([names[y] in [candidates[i] for i in row.argsort()[-3:]] for y,row in zip(truth,prob)]))
    write(out/'predictions.json',[dict(sample_key=r['sample_key'],truth=r['scientific_name'],
        predictions={key:pred[i] for key,pred in all_predictions.items()}) for i,r in enumerate(rows)])
    # Imported earlier measurements are explicitly separated, not claimed as reruns.
    historical={}
    for key,path in [('resnet50',ROOT/'experiments/whu18_resnet50_20260825/run'),
                     ('yolo11s_cls',WHU/'yolo11s_cls/run'),('ptv2',WHU/'ptv2/run'),('fusion',WHU/'fusion/run')]:
        data=read(path/'final_metrics.json'); m=data['metrics']['test']; cm=np.array(m['confusion_matrix'])
        assert cm.shape==(18,18) and cm.sum()==len(rows)
        assert (cm.sum(1)==np.bincount(truth,minlength=18)).all()
        tp=np.diag(cm); denom=cm.sum(0)+cm.sum(1)-tp
        historical[key]=dict(source=str(path/'final_metrics.json'),source_sha256=sha256(path/'final_metrics.json'),
            n=int(cm.sum()),accuracy=float(tp.sum()/cm.sum()),macro_f1=m['macro_f1'],
            mean_class_iou=float(np.mean(tp/np.maximum(denom,1))),source_type='historical_metrics_not_rerun')
    after={str(p):sha256(p) for p in weights}
    assert before==after
    write(out/'results.json',dict(fresh=report,historical=historical,weights_unchanged=True,
                                 elapsed_seconds=time.perf_counter()-started))
    write(out/'weight_integrity.json',dict(before=before,after=after,unchanged=True,training=False))
    print(json.dumps({k:{f:v[f] for f in ['accuracy','macro_f1','mean_class_iou']} for k,v in report.items()},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=8)
    a=p.parse_args(); main(a.out.resolve(),a.batch)
