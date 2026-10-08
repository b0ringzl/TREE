"""Read-only file/segmentation audit. No species relabeling or model training."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,laspy
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'lidar data/hk/data/data';OUT=ROOT/'outputs/single_tree_delivery_audit_v1';sys.path.insert(0,str(ROOT/'tools'))
from rebuild_banyan_coverage_preview import sample_ids,orthographic
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
 OUT.mkdir(exist_ok=False);prior={r['id']:r for r in read(ROOT/'outputs/hk_point_training_20260912_v1/new_las_associations.json')};allrows=[];scenes=[];stats=[]
 for sid in ['01','03','04']:
  folder=SRC/sid;d=OUT/sid;d.mkdir();summary=read(folder/(sid+'_summary.json'));files=sorted(folder.glob('**/TreeID_*.las'),key=lambda p:int(p.stem.split('_')[-1]));rows=[];renderpoints=[]
  for p in files:
   before=p.stat();f=laspy.read(p);x=np.column_stack([f.x,f.y,f.z]);key=sid+'_'+p.stem;tid=int(p.stem.split('_')[-1]);q=x[sample_ids(x,6000)];hag=np.asarray(f.height_above_ground);dims=list(f.point_format.dimension_names);hcounts=np.histogram(x[:,2],bins=np.linspace(x[:,2].min(),x[:,2].max()+1e-6,21))[0];hgap=max((len(v) for v in ''.join('1' if n==0 else '0' for n in hcounts).split('0')),default=0);sha=hashlib.sha256(p.read_bytes()).hexdigest();view=d/(p.stem+'.jpg')
   orthographic(x,sample_ids(x,6000),view,float(np.quantile(x[:,2],.25)));renderpoints.append((tid,q));rec=dict(id=key,path=str(p),points=len(x),sha256=sha,same_as_previous=sha==prior.get(key,{}).get('sha256'),bounds=[x.min(0).tolist(),x.max(0).tolist()],extent_m=np.ptp(x,axis=0).tolist(),hag_quantiles=np.quantile(hag,[0,.01,.5,.99,1]).tolist(),fraction_hag_below_1m=float((hag<1).mean()),tree_ids=np.unique(f.tree_id).tolist(),wood_leaf_labels=dict(zip(*[a.tolist() for a in np.unique(f.wood_leaf_label,return_counts=True)])),largest_empty_vertical_bin_run=hgap,views=str(view),geometry_verified=False,species_verified=False)
   rows.append(rec);assert before.st_size==p.stat().st_size and before.st_mtime_ns==p.stat().st_mtime_ns;stats.append(dict(path=str(p),size=before.st_size,mtime_ns=before.st_mtime_ns));print(key,len(x),'H',round(rec['extent_m'][2],1),flush=True)
  segpath=folder/(sid+'_single_tree_output')/(sid+'_segmented_trees.las');seg=laspy.read(segpath);labels,counts=np.unique(seg.tree_id,return_counts=True);idsdict={int(k):int(v) for k,v in zip(labels,counts)};treecounts={int(r['id'].split('_')[-1]):r['points'] for r in rows};stages={}
  for name in ['vegetation','wood_points','wood_points_filtered','leaf_points','leaf_points_filtered','understory_points']:
   p=folder/(sid+'_single_tree_output')/(sid+'_'+name+'.las')
   with laspy.open(p) as f:stages[name]=dict(points=f.header.point_count,min_z=float(f.header.mins[2]),max_z=float(f.header.maxs[2]))
  s=dict(scene=sid,provided_summary=summary,instances=len(rows),individual_points=sum(r['points'] for r in rows),segmented_total=len(seg.points),segmented_ids=idsdict,individual_export_matches_id_counts=all(idsdict.get(k)==v for k,v in treecounts.items()),missing_export_positive_ids=[k for k in idsdict if k>0 and k not in treecounts],stages=stages,segmented_max_z=float(np.max(seg.z)),all_same_as_previous=all(r['same_as_previous'] for r in rows));save(d/'summary.json',s);save(d/'instances.json',rows);scenes.append(s);allrows+=rows
  # Context panorama is not assumed: show full semantic scene as grey XY support.
  with laspy.open(folder/(sid+'_full3cls.las')) as reader:
   background=[]
   for chunk in reader.chunk_iterator(1000000):background.append(np.column_stack([chunk.x[::200],chunk.y[::200],chunk.z[::200]]))
  bg=np.concatenate(background);lo=bg[:,:2].min(0);ext=np.ptp(bg[:,:2],axis=0);scale=min(1300/ext[0],1100/ext[1]);im=Image.new('RGB',(1500,1320),'#f7f8f2');dr=ImageDraw.Draw(im)
  for x,y,z in bg:dr.point((80+(x-lo[0])*scale,1220-(y-lo[1])*scale),fill='#c8cdc8')
  colors=['#d55032','#167958','#526bbe','#be8a1f','#a6468d','#328eac']
  for tid,q in renderpoints:
   for x,y,z in q:dr.point((80+(x-lo[0])*scale,1220-(y-lo[1])*scale),fill=colors[tid%6])
   center=np.median(q,axis=0);xx=80+(center[0]-lo[0])*scale;yy=1220-(center[1]-lo[1])*scale;dr.rectangle((xx-2,yy-2,xx+25,yy+14),fill='white');dr.text((xx,yy),str(tid),fill='black')
  dr.text((30,20),sid+' | individual IDs on full scene XY | colours are IDs, NOT species',fill='black');im.save(d/'overview.jpg')
  for start in range(0,len(rows),12):
   im=Image.new('RGB',(2100,1480),'white');dr=ImageDraw.Draw(im)
   for i,r in enumerate(rows[start:start+12]):
    xx=i%2*1050;yy=i//2*245
    with Image.open(r['views']) as pic:im.paste(ImageOps.contain(pic,(1040,210)),(xx,yy))
    dr.text((xx+10,yy+212),f"{r['id']} | {r['points']:,} points | XYZ span "+','.join(f'{v:.1f}' for v in r['extent_m'])+' m',fill='black')
   im.save(d/f'contact_{start//12+1}.jpg')
 save(OUT/'summary.json',scenes);save(OUT/'instances.json',allrows);save(OUT/'source_stat_snapshot.json',stats)
 print('AUDIT COMPLETE',len(allrows),flush=True)
if __name__=='__main__':main()
