"""Scene-wide overview and deterministic candidate contact sheets for inspection."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from hk_fullscene_extract_20260912 import OUT,read,write
def make(map_id):
    d=OUT/'maps'/map_id;rows=read(d/'candidates.json');x=np.load(d/'coarse.npy');dest=d/'qa';dest.mkdir(exist_ok=True)
    im=Image.new('RGB',(1500,1100),'#f5f6f0');draw=ImageDraw.Draw(im);lo=x[:,:2].min(0);extent=np.ptp(x[:,:2],axis=0);scale=min(1380/max(extent[0],1),950/max(extent[1],1));ix=np.linspace(0,len(x)-1,min(180000,len(x)),dtype=int);zlo,zhi=np.quantile(x[:,2],[.02,.98])
    for p in x[ix]:
        xx=60+(p[0]-lo[0])*scale;yy=1020-(p[1]-lo[1])*scale;v=np.clip((p[2]-zlo)/max(zhi-zlo,1),0,1);draw.point((xx,yy),fill=(int(150-80*v),int(175-60*v),int(170-95*v)))
    for i,r in enumerate(rows):
        xx=60+(r['axis_xy'][0]-lo[0])*scale;yy=1020-(r['axis_xy'][1]-lo[1])*scale;color='#b65336' if r['seed_kind']=='crown_only_proposal' else '#2863a0';draw.ellipse((xx-3,yy-3,xx+3,yy+3),outline=color,width=2)
    draw.text((25,20),map_id+f' | {len(x):,} real voxels | {len(rows)} UNVERIFIED proposals',fill='black');draw.text((25,44),'Blue: vertical trace + crown. Red: crown-only peak. NOT confirmed trees. XY local metres.',fill='black');im.save(dest/'overview.jpg')
    selected=sorted(rows,key=lambda r:r['id']);ids=np.unique(np.linspace(0,max(len(rows)-1,0),min(16,len(rows)),dtype=int));chosen=[selected[i] for i in ids]
    for page in range((len(chosen)+7)//8):
        sheet=Image.new('RGB',(1800,1240),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(chosen[page*8:(page+1)*8]):
            xx=i%2*900;yy=i//2*310
            with Image.open(r['views']) as pic:sheet.paste(ImageOps.contain(pic,(895,270)),(xx,yy))
            draw.text((xx+5,yy+265),r['id']+f" | {r['coarse_points']} pts | {r['seed_kind']}\nH={r['observed_z_span_m']:.1f} m | base support={r['base_support_points']}",fill='black')
        sheet.save(dest/f'contact_{page+1}.jpg')
    write(dest/'contact_selection.json',[r['id'] for r in chosen])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('ids',nargs='+');a=p.parse_args()
    for name in a.ids:make(name)
