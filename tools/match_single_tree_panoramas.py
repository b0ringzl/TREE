"""Read-only, all delivered instances -> registered VMMS multi-view evidence."""
import sys,csv,json,math
from pathlib import Path
import numpy as np,laspy,cv2
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'hk-tree-vmms/tools'))
from build_hk_lidar_pairs import project
from rebuild_banyan_coverage_preview import sample_ids,orthographic
from spacescan_trees_demo_v1 import panorama
OUT=ROOT/'outputs/single_tree_panorama_selection_v1';VMMS=ROOT.parent/'vmms'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
 OUT.mkdir(exist_ok=True);rows=read(ROOT/'outputs/single_tree_delivery_audit_v1/instances.json');coords=list(csv.DictReader((ROOT/'hk-tree-vmms/outputs/coordinates/frame_coordinates.csv').open(encoding='utf-8-sig')));old={r['id']:r for r in read(ROOT/'outputs/hk_point_training_20260912_v1/new_las_evidence/manifest.json')};cache={};results=[]
 if (OUT/'total_only/manifest.json').exists():rows+=read(OUT/'total_only/manifest.json')
 for rec in rows:
  d=OUT/rec['id'];d.mkdir(exist_ok=True)
  if (d/'record.json').exists():results.append(read(d/'record.json'));continue
  f=laspy.read(rec['path']);x=np.column_stack([f.x,f.y,f.z]);q=x[sample_ids(x,7000)];low=x[np.asarray(f.height_above_ground)<1.5];axis=np.median(low[:,:2],axis=0) if len(low)>30 else np.median(x[:,:2],axis=0);sid=rec['id'][:2];survey='Stubbs Road_2024-07-25_HyD_collect' if sid=='04' else '20250210-jianshazui';options=[]
  for c in coords:
   if Path(c['source_image_relpath']).parts[0]!=survey:continue
   pos=np.array([float(c['local_x']),float(c['local_y'])]);dist=np.linalg.norm(pos-axis)
   if not 8<dist<40:continue
   key=c['stream_id']+'__'+c['frame_id'];angle=math.atan2(*(axis-pos)[::-1]);options.append((0 if old.get(rec['id'],{}).get('frame_key')==key else dist,c,angle,dist))
  selected=[]
  for _,c,angle,dist in sorted(options,key=lambda v:v[0]):
   if selected and min(abs(math.atan2(math.sin(angle-a),math.cos(angle-a))) for _,a,_ in selected)<math.radians(12):continue
   if not (VMMS/c['source_image_relpath']).is_file():continue
   selected.append((c,angle,dist))
   if len(selected)==4:break
  views=[]
  for i,(c,angle,dist) in enumerate(selected):
   path=VMMS/c['source_image_relpath'];key=c['stream_id']+'__'+c['frame_id']
   try:
    if str(path) not in cache:
     if len(cache)>=5:cache.pop(next(iter(cache)))
     with Image.open(path) as im:cache[str(path)]=im.convert('RGB')
    im=cache[str(path)];uv,_=project(q,c);lon=(uv[:,0]-.5)*2*np.pi;lat=(.5-uv[:,1])*np.pi;rays=np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],axis=1);yaw=math.degrees(math.atan2(np.sin(lon).mean(),np.cos(lon).mean()));pitch=float(np.median(lat)*180/np.pi);forward,right,up=panorama.basis(yaw,pitch);depth=rays@forward;good=depth>.05
    tx=(rays@right)/np.maximum(depth,.05);ty=(rays@up)/np.maximum(depth,.05)
    hfov=float(np.clip(np.degrees(2*np.arctan(np.quantile(abs(tx[good]),.998)*1.13)),10,115))
    vfov=float(np.clip(np.degrees(2*np.arctan(np.quantile(abs(ty[good]),.998)*1.13)),10,120))
    spec=panorama.view_spec(im.size,yaw,pitch,hfov,vfov,1400);crop=panorama.render(im,spec);over=crop.copy();dr=ImageDraw.Draw(over);px=(tx/np.tan(np.radians(hfov)/2)+1)*.5*crop.width;py=(1-ty/np.tan(np.radians(vfov)/2))*.5*crop.height;inside=good&(px>=0)&(px<crop.width)&(py>=0)&(py<crop.height)
    for a,b in zip(px[inside],py[inside]):dr.ellipse((a-1,b-1,a+1,b+1),fill='#f542d7')
    crop.save(d/f'view_{i}.jpg',quality=94);over.save(d/f'overlay_{i}.jpg',quality=94);gray=np.asarray(crop.convert('L'));score=float(np.std(gray))*(1-float((gray<30).mean()));views.append(dict(index=i,frame_key=key,panorama=str(path),image=str(d/f'view_{i}.jpg'),overlay=str(d/f'overlay_{i}.jpg'),camera_wgs84=[float(c['wgs84_latitude']),float(c['wgs84_longitude'])],capture_date=c['capture_date'],camera_local_xyz=[float(c['local_'+k]) for k in 'xyz'],view_angle_rad=angle,camera_distance_m=float(dist),perspective=spec,projected_sample_inside_fraction=float(inside.mean()),dark_pixel_fraction=float((gray<30).mean()),contrast_score=score))
   except Exception as e:views.append(dict(index=i,frame_key=key,error=str(e)))
  valid=sorted([v for v in views if 'image' in v],key=lambda v:-v['contrast_score']);best=valid[:2];r=dict(rec,axis_xy=axis.tolist(),views=views,display_views=[v['index'] for v in best],prior_species_evidence=old.get(rec['id']),pair_decision='pending_visual_review',training_eligible=False);save(d/'record.json',r);results.append(r);print(rec['id'],'views',len(valid),flush=True)
 save(OUT/'manifest.json',results)
 for sid in ['01','03','04']:
  rs=[r for r in results if r['id'].startswith(sid)]
  for start in range(0,len(rs),6):
   sheet=Image.new('RGB',(1800,1740),'white');dr=ImageDraw.Draw(sheet)
   for i,r in enumerate(rs[start:start+6]):
    xx=i%2*900;yy=i//2*580;dr.text((xx+10,yy+8),r['id'],fill='black')
    with Image.open(r.get('geometry_preview') or ROOT/'outputs/single_tree_delivery_audit_v1'/sid/(r['id'][3:]+'.jpg')) as pic:sheet.paste(ImageOps.contain(pic,(880,230)),(xx,yy+28))
    vs=[v for v in r['views'] if v.get('index') in r['display_views'] and 'image' in v]
    for j,v in enumerate(vs):
     for k,kind in enumerate(['image','overlay']):
      with Image.open(v[kind]) as pic:pic=ImageOps.contain(pic,(215,305));sheet.paste(pic,(xx+j*440+k*218,yy+264))
   sheet.save(OUT/f'{sid}_contact_{start//6+1}.jpg')
 print('ALL MATCHED',len(results),flush=True)
if __name__=='__main__':main()
