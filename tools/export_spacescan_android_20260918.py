"""Export an already validation-locked student. No fitting, relabeling or desktop deployment."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'outputs/android_toolchain_20260918_v1/python_deps'))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO
from training_safety import backup_before_training, sha256, verify_backup
from distillation_core import read_json, save_json, digest, metrics

RUN = ROOT/'outputs/distillation/20260908_0045'
OUT = ROOT/'outputs/SpaceScan_Android_Offline_20260918_v1'
ASSETS = ROOT/'spacescan_android/app/src/main/assets'

def tensor(image):
    # Same half-pixel bilinear sampler as Preprocessor.java, no PIL resampling.
    pixels = np.asarray(image.convert('RGB'), dtype=np.float64)
    height, width = pixels.shape[:2]
    scale = min(224/width, 224/height)
    rw, rh = max(1, int(np.floor(width*scale+.5))), max(1, int(np.floor(height*scale+.5)))
    sx = np.clip((np.arange(rw)+.5)*width/rw-.5, 0, width-1)
    sy = np.clip((np.arange(rh)+.5)*height/rh-.5, 0, height-1)
    x0, y0 = sx.astype(int), sy.astype(int)
    x1, y1 = np.minimum(x0+1,width-1), np.minimum(y0+1,height-1)
    fx, fy = (sx-x0)[None,:,None], (sy-y0)[:,None,None]
    a = pixels[y0[:,None],x0[None,:]]*(1-fx)+pixels[y0[:,None],x1[None,:]]*fx
    b = pixels[y1[:,None],x0[None,:]]*(1-fx)+pixels[y1[:,None],x1[None,:]]*fx
    result = np.full((224,224,3),128/255,dtype=np.float32)
    top,left=(224-rh)//2,(224-rw)//2
    result[top:top+rh,left:left+rw] = ((a*(1-fy)+b*fy)/255).astype(np.float32)
    return np.ascontiguousarray(result.transpose(2,0,1))

class MobileStudent(torch.nn.Module):
    def __init__(self, student, groups):
        super().__init__(); self.student = student; self.groups = groups
    def forward(self, images):
        p = self.student(images)
        return torch.stack([p[:,ids].sum(-1) for ids in self.groups], -1)

def load_source():
    protocol, lock = read_json(RUN/'protocol.json'), read_json(RUN/'head_final_lock.json')
    source = Path(protocol['initial_weight'])
    head = Path(lock['kd']['run'])/'best.pt'
    assert sha256(source) == protocol['initial_sha256']
    assert sha256(head) == lock['weights']['kd']
    classes = read_json(RUN/'classes.json')
    state = torch.load(head,map_location='cpu',weights_only=False)
    assert state['class_hash'] == digest(classes)
    student = YOLO(str(source)).model.float().eval()
    summary = read_json(ROOT/'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1/datasets/full/dataset_summary.json')
    assert [student.names[i] for i in range(148)] == [summary['class_folders'][str(i)] for i in range(148)]
    assert summary['class_names'] == classes['original_names']
    student.model[-1].linear.load_state_dict(state['model_state'])
    student.model[-1].export = True
    return MobileStudent(student,classes['groups']).eval().requires_grad_(False), classes, source, head

def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False); ASSETS.mkdir(exist_ok=True)
    model, classes, source, head = load_source()
    current = ROOT/'tree_species_teacher_demo/models/SSTv1_Artocarpus_User_20260918_v1'
    protected = [source,head,RUN/'classes.json',RUN/'protocol.json',RUN/'head_final_lock.json',
        ROOT/'tree_species_teacher_demo/engine.py',ROOT/'spacescan_trees_demo_v1/server.py',
        *current.glob('*.pt'), current/'deployment.json',
        ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib',
        ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json']
    backup=backup_before_training(OUT, protected, {'purpose':'Pre-export physical copies only; no training', 'desktop_unchanged':True})
    graph=OUT/'tree_model.onnx'
    with torch.inference_mode():
        torch.onnx.export(model,torch.zeros(1,3,224,224),str(graph),input_names=['image'],output_names=['probabilities'],opset_version=17,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(str(graph))
    session_options=ort.SessionOptions();session_options.intra_op_num_threads=4
    session=ort.InferenceSession(str(graph),sess_options=session_options,providers=['CPUExecutionProvider'])
    rows=read_json(RUN/'planning_rows.json')
    names=read_json(ROOT/'spacescan_trees_demo_v1/species_names.json')
    species=[dict(id=i,label=s,scientific_name=names[s]['scientific_name'],chinese_name=names[s]['chinese_name']) for i,s in enumerate(classes['labels'])]
    parity=[]
    for row in rows[:12]:
        with Image.open(row['source_image']) as im: x=tensor(ImageOps.exif_transpose(im))[None]
        with torch.inference_mode(): expected=model(torch.from_numpy(x)).numpy()
        start=time.perf_counter(); actual=session.run(None,{'image':x})[0];duration=time.perf_counter()-start
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-4)
        assert actual.argmax()==expected.argmax()
        parity.append(dict(sample_id=row['sample_id'],max_abs=float(abs(actual-expected).max()),cpu_seconds=duration))
    # Java/Android self-test uses a generated image, never a bundled third-party photo.
    w,h=173,291
    yy,xx=np.mgrid[:h,:w]
    gradient=np.stack(((xx*7+yy*3)%256,(xx*2+yy*5)%256,(xx+yy*11)%256),axis=-1).astype(np.uint8)
    x=tensor(Image.fromarray(gradient))[None]
    selftest=dict(width=w,height=h,recipe='r=(x*7+y*3)%256,g=(x*2+y*5)%256,b=(x+y*11)%256',
                  probabilities=session.run(None,{'image':x})[0][0].tolist(),tolerance=2e-5)
    save_json(OUT/'parity.json',dict(samples=parity,training_performed=False))
    save_json(OUT/'selftest.json',selftest)
    # Evaluate the actual mobile framing, not the historical center-crop scores.
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    probabilities=[]
    started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            batch=[]
            for row in rows[start:start+32]:
                with Image.open(row['source_image']) as im: batch.append(tensor(ImageOps.exif_transpose(im)))
            probabilities.append(model(torch.from_numpy(np.stack(batch)).to(device)).cpu().numpy())
            if start%256==0:print('VALIDATION',start,'/',len(rows),flush=True)
    p=np.concatenate(probabilities)
    result=metrics([r['target'] for r in rows],p)
    result.update(cohort='historical planning regression; not fresh mobile or street-tree accuracy',
                  preprocessing='224 RGB whole-image gray128 letterbox, float half-pixel bilinear; no stochastic augmentation',
                  elapsed_seconds=time.perf_counter()-started,
                  focus={s:dict(support=result['support'][classes['labels'].index(s)],recall=result['per_class_recall'][classes['labels'].index(s)],f1=result['per_class_f1'][classes['labels'].index(s)]) for s in ('Aquilaria sinensis','Artocarpus hypargyreus')})
    save_json(OUT/'mobile_validation.json',result)
    np.savez_compressed(OUT/'mobile_validation_predictions.npz',probabilities=p)
    metadata=dict(version='sst-mobile-0.1-20260918',architecture='YOLO11s-cls + validation-locked frozen KD head',
                  class_count=147,model_sha256=sha256(graph),model_bytes=graph.stat().st_size,
                  source_backbone_sha256=sha256(source),source_head_sha256=sha256(head),
                  input_shape=[1,3,224,224],preprocessing=result['preprocessing'],output='147 normalized probabilities; historical Ficus pair summed',
                  scope='Offline image-only experimental mobile model, NOT desktop SSTv1/BioCLIP',
                  not_supported=['Wodyetia bifurcata','Rhodoleia championii','Keteleeria fortunei','point-cloud species inference','automatic panorama tree detection'],
                  candidate_scores_are_not_accuracy=True,validation_top1=result['top1'],validation_n=result['n'],
                  rights='Local evaluation APK only; Ultralytics AGPL-3.0 and ONNX Runtime MIT notices included; reference/training images not bundled',
                  trained_this_run=False)
    save_json(OUT/'model_manifest.json',metadata);save_json(OUT/'species.json',species)
    for name in ('tree_model.onnx','model_manifest.json','species.json','selftest.json'):
        dest=ASSETS/name
        if dest.exists():raise FileExistsError(dest)
        shutil.copy2(OUT/name,dest)
        assert sha256(dest)==sha256(OUT/name)
    verify_backup(backup)
    save_json(OUT/'EXPORT_COMPLETE.json',dict(model=str(graph),metadata=metadata,backup=str(backup),desktop_unchanged=True,training_performed=False))
    print(json.dumps(metadata,ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
