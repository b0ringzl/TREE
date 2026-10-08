"""Inference-only Real-ESRGAN x2 comparison; never invent verified botanical detail.

RRDB architecture compatible with BasicSR; the retrieved upstream license and
official checkpoint provenance downloaded alongside the isolated experiment.
"""
import argparse
import base64
import html
import io
from pathlib import Path
import requests
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from run_vmms20m_teacher import ROOT, Survey, Engine, quality, lift_shadows, write
from training_safety import sha256


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(5):setattr(self,f'conv{i+1}',nn.Conv2d(64+i*32,32 if i<4 else 64,3,1,1))
    def forward(self,x):
        parts=[x]
        for i in range(4):parts.append(F.leaky_relu(getattr(self,f'conv{i+1}')(torch.cat(parts,1)),.2))
        return x+.2*self.conv5(torch.cat(parts,1))


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.rdb1=Dense();self.rdb2=Dense();self.rdb3=Dense()
    def forward(self,x):return x+.2*self.rdb3(self.rdb2(self.rdb1(x)))


class X2(nn.Module):
    def __init__(self):
        super().__init__();self.conv_first=nn.Conv2d(12,64,3,1,1)
        self.body=nn.Sequential(*[Block() for _ in range(23)])
        for name in ('conv_body','conv_up1','conv_up2','conv_hr'):setattr(self,name,nn.Conv2d(64,64,3,1,1))
        self.conv_last=nn.Conv2d(64,3,3,1,1)
    def forward(self,x):
        x=self.conv_first(F.pixel_unshuffle(x,2));x=x+self.conv_body(self.body(x))
        x=F.leaky_relu(self.conv_up1(F.interpolate(x,scale_factor=2,mode='nearest')),.2)
        x=F.leaky_relu(self.conv_up2(F.interpolate(x,scale_factor=2,mode='nearest')),.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(x),.2))


def upscale(model,im):
    x=torch.from_numpy(np.array(im).copy()).permute(2,0,1)[None].float().cuda()/255
    h,w=x.shape[-2:];x=F.pad(x,(0,w%2,0,h%2),mode='reflect')
    output=torch.zeros((1,3,x.shape[2]*2,x.shape[3]*2),device='cuda')
    with torch.inference_mode():
        for y in range(0,x.shape[2],192):
            for col in range(0,x.shape[3],192):
                bottom=min(y+192,x.shape[2]);right=min(col+192,x.shape[3])
                y0=max(0,y-16);x0=max(0,col-16);y1=min(x.shape[2],bottom+16);x1=min(x.shape[3],right+16)
                tile=model(x[:,:,y0:y1,x0:x1])
                output[:,:,y*2:bottom*2,col*2:right*2]=tile[:,:,(y-y0)*2:(bottom-y0)*2,(col-x0)*2:(right-x0)*2]
    return Image.fromarray((output[0,:,:h*2,:w*2].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    response=requests.get(url,timeout=(15,120));response.raise_for_status()
    checkpoint=out/'RealESRGAN_x2plus.pth';checkpoint.write_bytes(response.content)
    license_url='https://raw.githubusercontent.com/XPixelGroup/BasicSR/master/LICENSE.txt'
    response=requests.get(license_url,timeout=30);response.raise_for_status();(out/'BasicSR_LICENSE.txt').write_bytes(response.content)
    write(out/'model_provenance.json',{'source':url,'sha256':sha256(checkpoint),'license_url':license_url,
        'purpose':'super-resolution inference comparison only','training':False,'deployed':False,
        'limitation':'Generated fine texture is not recovered ground-truth botanical evidence; no HR reference available to measure reconstruction fidelity.'})
    net=X2().cuda().eval().requires_grad_(False)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state.get('params_ema',state.get('params')),strict=True)
    images=[];rows=[]
    source=ROOT/'outputs/focus8_20260909_v1/vmms_positive_audit/crops'
    for name in ('0016','0003'):
        with Image.open(source/(name+'.jpg')) as image:im=image.convert('RGB')
        variants=[('original',im),('bounded_shadow_lift',lift_shadows(im)),
            ('gamma_0.55_comparison',Image.fromarray((np.power(np.asarray(im)/255.,.55)*255).round().astype('uint8'))),
            ('bicubic_x2',im.resize((im.width*2,im.height*2),Image.Resampling.BICUBIC)),('RealESRGAN_x2',upscale(net,im))]
        for variant,img in variants:
            path=out/(name+'_'+variant+'.png');img.save(path)
            rows.append({'crop_id':name,'variant':variant,'file':str(path),'quality':quality(img),
                'synthetic_detail':variant=='RealESRGAN_x2'})
            images.append(img)
        print('Restoration compared: '+name,flush=True)
    del net;torch.cuda.empty_cache()
    survey=Survey.__new__(Survey);survey.teacher=Engine();survey.teacher.load()
    survey.labels=survey.teacher.classes['original_names']+['Wodyetia bifurcata']
    survey.bank=torch.cat([survey.teacher.bank,survey.teacher.extra_bank],0)
    _,p=survey.encode(images)
    for row,prob in zip(rows,p):row['top3']=survey.top(prob)
    write(out/'results.json',rows)
    cards=[]
    for row in rows:
        data=base64.b64encode(Path(row['file']).read_bytes()).decode('ascii')
        cards.append('<article><h2>'+html.escape(row['crop_id']+' / '+row['variant'])+'</h2><img src="data:image/png;base64,'+data+'"><p>'+html.escape(str(row['top3']))+'</p><p>Dark pixels / 暗像素: '+f"{row['quality']['dark_fraction_lt35']:.1%}"+'</p></article>')
    (out/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Restoration trial / 修復對照</title><style>body{font:16px system-ui;margin:25px;background:#edf3ef}main{display:grid;grid-template-columns:repeat(5,minmax(240px,1fr));gap:12px}article{background:white;padding:12px}img{width:100%;height:350px;object-fit:contain}</style><h1>Restoration comparison / 修復對照</h1><p>Super-resolution generates texture; it is not verified leaf detail. This is a two-case diagnostic, not an accuracy benchmark. / 超分辨率紋理不是已驗證的葉片細節；這是兩個案例的對照，不是準確率測試。</p><main>'+''.join(cards)+'</main>',encoding='utf8')
    print('Quality trial finished',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out.resolve())
