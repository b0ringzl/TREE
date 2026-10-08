"""Revalidate exported graph with original preprocessing; no fitting or weight edits."""
import json, subprocess, sys, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/android_toolchain_20260918_v1/python_deps'))
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
from distillation_core import read_json,save_json,canonical_image,metrics
from export_spacescan_android_20260918 import load_source
OUT=ROOT/'outputs/SpaceScan_Android_Offline_20260918_v2'
ASSETS=ROOT/'spacescan_android/app/src/main/assets'
def canonical(im):
    return TF.to_tensor(TF.center_crop(TF.resize(im,224,interpolation=InterpolationMode.BICUBIC),[224,224])).numpy()
def main():
    OUT.mkdir(exist_ok=False)
    paths=read_json(ROOT/'outputs/android_toolchain_20260918_v1/paths.json');jdk=Path(paths['jdk'])/'bin'
    src=ROOT/'spacescan_android/app/src/main/java/org/spacescan/trees/offline/Preprocessor.java'
    subprocess.run([str(jdk/'javac.exe'),'-d',str(OUT/'java'),str(src),str(ROOT/'spacescan_android/tests/PreprocessorCheck.java')],check=True)
    checks=[]
    for w,h in [(173,291),(300,201),(224,224),(1024,2048),(11,19),(865,869)]:
        yy,xx=np.mgrid[:h,:w];gradient=np.stack(((xx*7+yy*3)%256,(xx*2+yy*5)%256,(xx+yy*11)%256),-1).astype(np.uint8)
        target=canonical(Image.fromarray(gradient));file=OUT/f'java_{w}_{h}.bin'
        subprocess.run([str(jdk/'java.exe'),'-cp',str(OUT/'java'),'PreprocessorCheck',str(w),str(h),str(file)],check=True)
        actual=np.fromfile(file,dtype='>f4').reshape(3,224,224)
        delta=float(np.abs(target-actual).max());checks.append(dict(width=w,height=h,max_abs=delta));assert delta<1e-6,checks
    save_json(OUT/'java_preprocessing_parity.json',checks)
    sess=ort.InferenceSession(str(ASSETS/'tree_model.onnx'),providers=['CPUExecutionProvider'])
    w,h=173,291;yy,xx=np.mgrid[:h,:w]
    gradient=np.stack(((xx*7+yy*3)%256,(xx*2+yy*5)%256,(xx+yy*11)%256),-1).astype(np.uint8)
    selftest=dict(width=w,height=h,recipe='r=(x*7+y*3)%256,g=(x*2+y*5)%256,b=(x+y*11)%256',probabilities=sess.run(None,{'image':canonical(Image.fromarray(gradient))[None]})[0][0].tolist(),tolerance=2e-5)
    model,classes,_,_=load_source();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';model=model.to(device)
    rows=read_json(ROOT/'outputs/distillation/20260908_0045/planning_rows.json');prob=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            batch=torch.stack([canonical_image(r['source_image']) for r in rows[start:start+32]])
            prob.append(model(batch.to(device)).cpu().numpy())
            if start%256==0:print('VALIDATION',start,flush=True)
    p=np.concatenate(prob);report=metrics([r['target'] for r in rows],p)
    report.update(cohort='Historical planning regression; not fresh mobile or street-tree accuracy',preprocessing='EXIF RGB; antialiased bicubic shortest-side resize224; center224; /255',trained_this_run=False)
    save_json(OUT/'mobile_validation.json',report);np.savez_compressed(OUT/'mobile_validation_predictions.npz',probabilities=p)
    metadata=read_json(ASSETS/'model_manifest.json');metadata.update(version='sst-mobile-0.1-20260918-preproc2',preprocessing=report['preprocessing'],validation_top1=report['top1'],validation_n=report['n'])
    for name,data in [('selftest.json',selftest),('model_manifest.json',metadata)]:save_json(OUT/name,data);shutil.copy2(OUT/name,ASSETS/name)
    print(json.dumps({k:report[k] for k in ['n','top1','macro_f1','top5']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
