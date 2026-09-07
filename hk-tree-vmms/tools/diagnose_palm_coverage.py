"""Reproducible coverage diagnosis for the user's JTS palm example (read-only sources)."""
from pathlib import Path
import csv
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
from build_hk_lidar_pairs import PROJECT, VMMS, read, write, project, render_preview, voxel

KEY='jianshazui_pano_2__001474'
ID=KEY+'__79dca14b84'
PILOT=PROJECT/'derived/hk_lidar_pairs_pilot_20260907'
OUT=PROJECT/'derived/palm_coverage_diagnosis_20260907'

def main():
    OUT.mkdir(exist_ok=True)
    row=next(r for r in csv.DictReader((PROJECT/'outputs/coordinates/frame_coordinates.csv').open(encoding='utf-8-sig')) if r['stream_id']+'__'+r['frame_id']==KEY)
    anno=next(a for a in read(PILOT/'annotation_snapshot.json') if a['frame_key']==KEY)
    poly=np.array(anno['label']['points'])
    xyz=np.load(PILOT/'map_cache/20250210-jianshazui.npy')
    pos=np.array([float(row['local_'+k]) for k in 'xyz'])
    xyz=xyz[np.linalg.norm(xyz[:,:2]-pos[:2],axis=1)<65]
    mask=np.zeros((2048,4096),np.uint8);cv2.fillPoly(mask,[np.rint(poly*[4095,2047]).astype(np.int32)],1)
    crown_y=float(poly[:,1].min()+.27*np.ptp(poly[:,1]))
    trials=[]
    with Image.open(VMMS/row['source_image_relpath']) as src: im=src.convert('RGB');im.thumbnail((4096,2048))
    for yaw in (174,177,180,183,186):
        uv,dist=project(xyz,row,yaw)
        inside=mask[np.clip((uv[:,1]*2047).astype(int),0,2047),np.clip((uv[:,0]*4095).astype(int),0,4095)]>0
        take=inside&(dist>2)&(dist<60); selected=xyz[take];v=uv[take];d=dist[take]
        labs=DBSCAN(eps=.65,min_samples=5,n_jobs=2).fit_predict(selected)
        clusters=[]
        for lab in sorted(set(labs)):
            if lab<0:continue
            p=selected[labs==lab]; eig=np.linalg.eigvalsh(np.cov(p.T));eig=eig/max(eig.sum(),1e-9)
            clusters.append({'cluster':int(lab),'n':len(p),'range_median':float(np.median(d[labs==lab])),'crown_points':int((v[labs==lab,1]<crown_y).sum()),'eigen_fraction':eig.tolist(),'extent':np.ptp(p,axis=0).tolist(),'center':p.mean(0).tolist()})
        trials.append({'yaw':yaw,'all_mask_points':len(selected),'crown_points':int((v[:,1]<crown_y).sum()),'within_45m':int((d<45).sum()),'clusters':clusters})
        folder=OUT/('yaw_'+str(yaw));folder.mkdir(exist_ok=True)
        render_preview(im,poly,v,selected,folder,'unfiltered')
        # Enlarged vicinity: hue encodes range, so foreground/background are visible.
        canvas=im.copy();dr=ImageDraw.Draw(canvas)
        region=(uv[:,0]<.18)&(uv[:,1]>.1)&(uv[:,1]<.55)&(dist<60)&(dist>2)
        for (u,w),r in zip(uv[region],dist[region]):
            color='#00e8ff' if r<20 else '#ffff00' if r<30 else '#ff657f'
            x,y=int(u*im.width),int(w*im.height);dr.ellipse((x-1,y-1,x+1,y+1),fill=color)
        dr.line([(int(x*im.width),int(y*im.height)) for x,y in np.vstack((poly,poly[:1]))],fill='white',width=3)
        canvas.crop((0,int(.1*im.height),int(.18*im.width),int(.56*im.height))).save(folder/'vicinity_depth.jpg')
    write(OUT/'diagnosis.json',{'frame':KEY,'crown_region':'upper 27% of annotation height (diagnostic proxy only)','crown_y':crown_y,'trials':trials})
    print(json.dumps(trials,indent=2))

if __name__=='__main__': main()
