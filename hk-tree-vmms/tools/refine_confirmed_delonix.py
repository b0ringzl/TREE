"""Versioned re-extraction of explicitly confirmed Delonix scene targets.

Keep final crown growth inside the reviewed image mask, select crown components
by projected support rather than nearest range alone, and attempt branched stem
tracking. Crown-only results remain explicitly partial and never auto-train.
"""
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from build_hk_lidar_pairs import read,write,project,voxel,VMMS
from build_hk_dense_dataset import coordinates,polygon_inside,coverage_sample,CACHE
from expand_requested_five_species import DenserFrameMaps
from pilot_banyan_tracking import track
from audit_hk_metric_height import ground_candidate
from repair_hk_banyan_stems import views,balanced_ids
from build_hk_five_species_pairs import OUT as PAIRS

SCENES=PAIRS.parent/'delonix_scene_review_20260907'
OUT=PAIRS.parent/'delonix_refined_v1_20260907'

def choose_crown(local,row,poly):
    uv,dist=project(local,row);mask=polygon_inside(uv,poly,dilation=0)
    top=poly[:,1].min()+.75*np.ptp(poly[:,1])
    upper=local[mask&(uv[:,1]<top)&(dist>2)&(dist<60)]
    upper=upper[voxel(upper,.12)]
    if len(upper)<30:raise ValueError('insufficient_measured_upper_support')
    labels=DBSCAN(eps=.65,min_samples=4).fit_predict(upper);options=[]
    for label in set(labels)-{-1}:
        p=upper[labels==label]
        if len(p)<30:continue
        eig=np.linalg.eigvalsh(np.cov(p.T));span=np.ptp(p,axis=0)
        if eig[0]/max(eig.sum(),1e-9)<.015 or span[2]<.8 or max(span[:2])>22:continue
        puv,_=project(p,row)
        bins=np.floor(puv*[512,256]).astype(int)
        support=len(np.unique(bins,axis=0))
        options.append((support,p))
    if not options:raise ValueError('no_supported_nonplanar_crown')
    _,seed=max(options,key=lambda x:x[0])
    crown=local[mask&(cKDTree(seed).query(local)[0]<.45)]
    crown=crown[voxel(crown,.08)]
    if len(crown)<100:raise ValueError('fewer_than_100_real_crown_points')
    return crown,mask,{'candidate_crown_components':len(options),'selected_projected_support_cells':max(x[0] for x in options)}

def main():
    if OUT.exists():raise SystemExit('Preserve existing revision; choose a new output version.')
    OUT.mkdir();(OUT/'instances').mkdir()
    protected=[SCENES/'scene_reviews.json',PAIRS/'reviews.json',PAIRS/'manifest.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    reviews=read(SCENES/'scene_reviews.json');frames=read(SCENES/'manifest.json')
    chosen=[{**t,'frame_key':f['id'],'review':reviews[t['id']]} for f in frames for t in f['targets']
            if reviews.get(t['id'],{}).get('identification')=='delonix' and reviews[t['id']]['alignment']=='aligned']
    chosen.sort(key=lambda t:('重点训练' not in t['review']['note'],t['frame_key'],t['id']))
    write(OUT/'selection.json',chosen);write(OUT/'protected_hashes.json',hashes)
    rows=coordinates();maps=DenserFrameMaps(CACHE);results=[];cached_key=None;cloud=None
    for item in chosen:
        ident=item['id'];folder=OUT/'instances'/ident;folder.mkdir();row=rows[item['frame_key']];poly=np.array(item['polygon'])
        r={**item,'status':'needs_rework','species':'Delonix regia','training_eligible':False,
           'total_height_verified':False,'priority_by_user':'重点训练' in item['review']['note']}
        with Image.open(VMMS/row['source_image_relpath']) as src:im=src.convert('RGB');im.thumbnail((3072,1536))
        im.save(folder/'panorama.jpg',quality=94)
        lo=poly.min(0);hi=poly.max(0);im.crop((*np.rint(lo*[im.width,im.height]).astype(int),*np.rint(hi*[im.width,im.height]).astype(int))).save(folder/'image.jpg',quality=94)
        oldpath=PAIRS/'instances'/ident/'cleaned_local.npz';r['has_previous']=oldpath.exists()
        if oldpath.exists():views(np.load(oldpath)['xyz'],folder/'before_views.jpg')
        try:
            if cached_key!=item['frame_key']:cloud,sources=maps.nearby(row);cached_key=item['frame_key']
            r.update(source_lidar=sources,source_sampling_m=maps.sampling_m)
            crown,inside,stats=choose_crown(cloud,row,poly);axis=np.median(crown[:,:2],axis=0)
            floor=float(np.quantile(crown[:,2],.02)-.25)
            local=cloud[np.linalg.norm(cloud[:,:2]-axis,axis=1)<8]
            ground=ground_candidate(local,axis);z0=ground['ground_z_m'] if ground['status']=='local_plane_candidate' else float(np.quantile(local[:,2],.01))
            lower=cloud[inside&(cloud[:,2]>z0+.25)&(cloud[:,2]<floor+.8)&(np.linalg.norm(cloud[:,:2]-axis,axis=1)<6)]
            stem,tracking=track(lower,axis,z0,floor)
            p=crown if stem is None else np.r_[crown,stem]
            p=p[voxel(p,.08)];ti=np.flatnonzero(p[:,2]<floor);ci=np.flatnonzero(p[:,2]>=floor)
            count=min(2048,len(p));nt=min(len(ti),768);nc=min(len(ci),count-nt);nt=min(len(ti),count-nc)
            ids=np.r_[ti[coverage_sample(p[ti],nt)] if nt else np.array([],int),ci[coverage_sample(p[ci],nc)]]
            center=p.mean(0);scale=float(np.linalg.norm(p-center,axis=1).max())
            normalized=((p[ids]-center)/scale).astype(np.float32)
            assert len(np.unique(normalized,axis=0))==len(ids)<=2048
            assert cKDTree(cloud).query(p)[0].max()<1e-7
            np.savez_compressed(folder/'point.npz',points_xyz=normalized,centroid_xyz=center,scale=scale,metric_units='m')
            np.savez_compressed(folder/'cleaned_local.npz',xyz=p)
            views(p,folder/'point_views.jpg',floor)
            canvas=im.copy();draw=ImageDraw.Draw(canvas);puv,_=project(p,row)
            pp=poly*[canvas.width,canvas.height];draw.line([tuple(x) for x in np.r_[pp,pp[:1]]],fill='#ffd027',width=3)
            for i in balanced_ids(p,7000):
                x,y=puv[i]*[canvas.width,canvas.height];draw.ellipse((x-1,y-1,x+1,y+1),fill='#e58e3b' if p[i,2]<floor else '#13d9bb')
            canvas.save(folder/'projection.jpg',quality=94)
            r.update(status='candidate_pending_review',crown_component_stats=stats,ground=ground,tracking=tracking,
                 quality={'clean_points':len(p),'crown_points':len(ci),'lower_points':len(ti),'sample_unique_points':len(ids),
                          'observed_extent_m':np.ptp(p,axis=0).tolist(),'axis_local_xy':axis.tolist(),
                          'projection_inside_fraction':float(polygon_inside(puv,poly,dilation=0).mean()),
                          'stem_path_found':stem is not None,'crown_floor_z':floor},
                 limitation='Supported lower path is not proof of a complete root or trunk.' if stem is not None else 'Crown-only: no supported low stem-to-crown path; do not claim a complete tree.')
        except (ValueError,IndexError) as exc:r['reason']=str(exc)
        write(folder/'record.json',r);results.append(r);write(OUT/'manifest.progress.json',results)
        print(ident,r['status'],r.get('quality',{}).get('stem_path_found'),flush=True)
    # Suggest same-tree links from real 3D overlap; never merge labels/decisions automatically.
    links=[];valid=[r for r in results if r['status']=='candidate_pending_review']
    for i,a in enumerate(valid):
        pa=np.load(OUT/'instances'/a['id']/'cleaned_local.npz')['xyz']
        for b in valid[i+1:]:
            if np.linalg.norm(np.array(a['quality']['axis_local_xy'])-b['quality']['axis_local_xy'])>5:continue
            pb=np.load(OUT/'instances'/b['id']/'cleaned_local.npz')['xyz']
            ab=float((cKDTree(pb).query(pa)[0]<.2).mean());ba=float((cKDTree(pa).query(pb)[0]<.2).mean())
            if min(ab,ba)>.5:links.append({'a':a['id'],'b':b['id'],'overlap_a_to_b':ab,'overlap_b_to_a':ba,'identity_verified':False})
    write(OUT/'possible_same_tree_links.json',links);write(OUT/'manifest.json',results)
    write(OUT/'summary.json',{'attempted':len(results),'statuses':dict(Counter(r['status'] for r in results)),
          'with_stem_path':sum(r.get('quality',{}).get('stem_path_found',False) for r in results),
          'possible_same_tree_links':len(links),'auto_accepted':0,'model_changed':False})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())

if __name__=='__main__':main()
