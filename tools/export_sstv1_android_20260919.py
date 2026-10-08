"""Port deployed SSTv1 to bounded-memory FP32 ONNX stages; never fit weights."""
import gc
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools'), str(ROOT/'outputs/android_toolchain_20260918_v1/python_deps')]
import numpy as np
import onnx
import onnxruntime as ort
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from training_safety import backup_before_training, verify_backup, sha256
from tree_species_teacher_demo.engine import Engine, SSTV1_MANIFEST
from distillation_core import collapse_logits

OUT = ROOT/'outputs/SpaceScan_Android_SSTv1_20260919_v1'
PACK = OUT/'model_pack'


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


class Start(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.conv=visual.conv1; self.cls=visual.class_embedding
        self.pos=visual.positional_embedding; self.norm=visual.ln_pre
        self.register_buffer('mean',torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None])
    def forward(self,x):
        x=self.conv((x-self.mean)/self.std).flatten(2).transpose(1,2)
        return self.norm(torch.cat([self.cls[None,None,:].expand(1,1,-1),x],1)+self.pos)


class Blocks(torch.nn.Module):
    def __init__(self, layers, batch_first):
        super().__init__();self.layers=torch.nn.ModuleList(layers);self.batch_first=batch_first
    def forward(self,x):
        if not self.batch_first:x=x.transpose(0,1)
        for layer in self.layers:x=layer(x)
        if not self.batch_first:x=x.transpose(0,1)
        return x


class Finish(torch.nn.Module):
    def __init__(self, visual, engine):
        super().__init__();self.norm=visual.ln_post;self.proj=visual.proj
        self.heads=torch.nn.ModuleList([h.cpu().float() for h in engine.sstv1_heads])
        self.register_buffer('bank',engine.bank.cpu().float())
        self.register_buffer('extra',engine.extra_bank.cpu().float())
        self.register_buffer('scale',engine.model.logit_scale.exp().float().detach().cpu())
        self.groups=engine.classes['groups'];self.temp=engine.choice['calibration_temperature']
    def forward(self,x):
        embedding=F.normalize(self.norm(x[:,0])@self.proj,dim=-1)
        raw=embedding@self.bank.T*self.scale
        feature=embedding.half().float()
        extra=feature@self.extra.T*self.scale
        return torch.stack([(torch.cat([collapse_logits(h(feature,raw),self.groups),extra],-1)/self.temp).softmax(-1) for h in self.heads]).mean(0)


def ort_session(path):
    opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
    opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opt.enable_cpu_mem_arena=False;opt.enable_mem_pattern=False
    opt.add_session_config_entry('session.disable_prepacking','1')
    return ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])


def demo_pixels(path):
    with Image.open(path) as source: im=ImageOps.exif_transpose(source).convert('RGB')
    edge=min(1024,max(im.size))
    im=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128))
    buf=io.BytesIO();im.save(buf,format='JPEG',quality=95)
    im=Image.open(io.BytesIO(buf.getvalue())).convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
    return im


def tensor(im):return np.ascontiguousarray(np.asarray(im,dtype=np.float32).transpose(2,0,1)[None]/255)


def run(sessions,x):
    for s in sessions:x=s.run(None,{'input':x})[0]
    return x[0]


def main():
    torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
    OUT.mkdir(exist_ok=False);PACK.mkdir()
    source_manifest=json.loads(SSTV1_MANIFEST.read_text(encoding='utf-8'))
    assets=ROOT/'spacescan_android/app/src/main/assets'
    protected=[SSTV1_MANIFEST,*[SSTV1_MANIFEST.parent/m['file'] for m in source_manifest['members']],
        ROOT/'external_datasets/distillation_models/bioclip25/open_clip_model.safetensors',
        ROOT/'tree_species_teacher_demo/engine.py',ROOT/'spacescan_trees_demo_v1/server.py',
        ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json',
        ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib',
        *[assets/n for n in ['tree_model.onnx','species.json','model_manifest.json']]]
    backup=backup_before_training(OUT,protected,{'purpose':'Inference-only model format conversion; no fitting','keep_desktop_unchanged':True})
    print('BACKUP_VERIFIED',flush=True)
    engine=Engine();engine.load();assert engine.sstv1_version==source_manifest['version']
    visual=engine.model.visual.cpu().float().eval().requires_grad_(False)
    assert visual.pool_type=='tok' and visual.attn_pool is None
    modules=[Start(visual)]+[Blocks(list(visual.transformer.resblocks[i:i+2]),visual.transformer.batch_first) for i in range(0,32,2)]+[Finish(visual,engine)]
    labels=engine.labels
    names=json.loads((ROOT/'spacescan_trees_demo_v1/species_names.json').read_text(encoding='utf-8'))
    species=[dict(id=i,label=s,scientific_name=names[s]['scientific_name'],chinese_name=names[s]['chinese_name']) for i,s in enumerate(labels)]
    save(PACK/'species.json',species)
    del engine
    torch.cuda.empty_cache();gc.collect()
    yy,xx=np.mgrid[:224,:224]
    gradient=Image.fromarray(np.stack(((xx*7+yy*3)%256,(xx*2+yy*5)%256,(xx+yy*11)%256),-1).astype(np.uint8))
    x=torch.from_numpy(tensor(gradient));files=[];parity=[]
    with torch.inference_mode():
        for i,module in enumerate(modules):
            module=module.eval().requires_grad_(False);path=PACK/f'stage_{i:02d}.onnx';input_shape=list(x.shape)
            expected=module(x)
            torch.onnx.export(module,x,str(path),input_names=['input'],output_names=['output'],
                              opset_version=17,dynamo=False,do_constant_folding=True)
            onnx.checker.check_model(str(path))
            session=ort_session(path);actual=session.run(None,{'input':x.numpy()})[0]
            error=float(np.abs(actual-expected.numpy()).max())
            assert np.allclose(actual,expected.numpy(),rtol=2e-3,atol=3e-4),(i,error)
            files.append(dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size,input_shape=input_shape,output_shape=list(expected.shape)))
            parity.append(dict(stage=i,max_abs=error));print('EXPORTED',i,files[-1]['bytes'],'parity',error,flush=True)
            x=expected;del session;gc.collect()
    save(OUT/'stage_export_parity.json',parity)
    del modules,visual;gc.collect()
    sessions=[ort_session(PACK/f['file']) for f in files]
    selftest=dict(width=224,height=224,recipe='r=(x*7+y*3)%256,g=(x*2+y*5)%256,b=(x+y*11)%256',
                  probabilities=run(sessions,tensor(gradient)).tolist(),tolerance=.0002)
    save(PACK/'selftest.json',selftest)
    manifest=dict(version='sstv1-artocarpus-user-20260918-android-fp32-v1',desktop_version=source_manifest['version'],
        architecture='BioCLIP 2.5 ViT-H/14 + same three SSTv1 MLP heads',class_count=148,
        pack_id='sstv1-artocarpus-20260918-fp32-v1',graph_files=files,
        model_sha256=hashlib.sha256(''.join(f['sha256'] for f in files).encode()).hexdigest(),
        model_bytes=sum(f['bytes'] for f in files),source_deployment=source_manifest,
        preprocessing='Aspect-preserving gray128 pad to max edge <=1024, JPEG95, bicubic resize224, RGB /255; normalization inside stage00',
        numerical_policy='CPU FP32 computation from deployed effective encoder weights; same MLPs, prompts, temperature and class collapse; not bit-identical to CUDA BF16',
        scope='Same latest deployed image classifier; no YOLO substitute, no server',
        not_supported=['point-cloud inference','automatic tree detection','panorama reprojection','Rhodoleia championii','Keteleeria fortunei'],
        candidate_scores_are_not_accuracy=True,training_performed=False,
        rights='Private local evaluation only. No training/reference photos included; no public distribution clearance for adapted weights.',
        memory_policy='Load, run and close one stage at a time; no concurrent sessions on Android')
    save(PACK/'model_manifest.json',manifest)
    rows=json.loads((ROOT/'outputs/distillation/20260908_0045/planning_rows.json').read_text(encoding='utf-8'))
    selected=[rows[i]['source_image'] for i in np.linspace(0,len(rows)-1,20,dtype=int)]
    selected.insert(0,'C:/Users/25041725G/Desktop/验证/微信图片_20260916084639_130_2.jpg')
    results=[]
    for index,path in enumerate(selected):
        pixels=demo_pixels(path);t=time.perf_counter();p=run(sessions,tensor(pixels));elapsed=time.perf_counter()-t
        b=io.BytesIO();pixels.save(b,format='PNG')
        r=requests.post('http://127.0.0.1:8039/api/predict',files={'image':('input.png',b.getvalue(),'image/png')},data={'mode':'sstv1'},timeout=180)
        r.raise_for_status();data=r.json();q=np.array([data['sstv1_distribution'][s] for s in labels])
        row=dict(path=path,sha256=sha256(path),top1_mobile=labels[int(p.argmax())],top1_desktop=labels[int(q.argmax())],
            same_top1=bool(p.argmax()==q.argmax()),max_abs=float(abs(p-q).max()),l1=float(abs(p-q).sum()),seconds=elapsed,
            mobile_top3=[dict(species=labels[i],score=float(p[i])) for i in p.argsort()[::-1][:3]],
            desktop_top3=[dict(species=labels[i],score=float(q[i])) for i in q.argsort()[::-1][:3]])
        results.append(row);save(OUT/'paired_regression.json',results);print('PAIRED',index,row['same_top1'],row['max_abs'],row['top1_mobile'],elapsed,flush=True)
    matched=sum(r['same_top1'] for r in results);max_abs=max(r['max_abs'] for r in results)
    # Equivalence gate declared before data: >=95% top1 agreement, max score drift <= .04.
    gate=dict(n=len(results),same_top1=matched,max_abs=max_abs,pass_gate=matched/len(results)>=.95 and max_abs<=.04 and results[0]['top1_mobile']=='Terminalia mantaly',
              interpretation='Conversion regression agreement, not accuracy or independent species validation')
    save(OUT/'equivalence_gate.json',gate)
    assert gate['pass_gate'],gate
    with zipfile.ZipFile(OUT/'SSTv1-20260918-Offline-Model.zip','w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for path in sorted(PACK.iterdir()):archive.write(path,path.name)
    verify_backup(backup)
    save(OUT/'COMPLETE.json',dict(backup=str(backup),source_unchanged=True,training_performed=False,gate=gate,model_bytes=manifest['model_bytes']))
    print('EXPORT_COMPLETE',gate,flush=True)


if __name__=='__main__':main()
