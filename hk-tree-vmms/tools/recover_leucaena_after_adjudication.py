"""Materialize resolved labels, then recover remaining silver-leadtree candidates.
All outputs are a new revision. Source assets and human decisions are protected.
"""
import sys,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from refine_confirmed_delonix import choose_crown
from build_hk_lidar_pairs import read,write,project,voxel,VMMS
from build_hk_dense_dataset import coordinates,coverage_sample,CACHE
from expand_requested_five_species import DenserFrameMaps
from pilot_banyan_tracking import track
from audit_hk_metric_height import ground_candidate
from repair_hk_banyan_stems import views,balanced_ids

WORK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WORK/'tree_species_demo'))
import albizia_conflict_api as conflict
DATA=WORK/'hk-tree-vmms/derived/hk_dualsource_five_species_20260907'
OUT=DATA.parent/'leucaena_recovery_v1_20260907'
SILVER='Leucaena leucocephala'

def main():
    if OUT.exists():raise SystemExit('Preserve existing revision; do not overwrite.')
    exported=conflict.candidates();adjudication=conflict.latest()
    if not adjudication or adjudication['status']!='identity_and_species_confirmed':raise ValueError('Resolve source conflict first')
    OUT.mkdir();(OUT/'instances').mkdir()
    source=read(DATA/'manifest.json');reviews=read(DATA/'reviews.json')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'manifest.json',DATA/'reviews.json',conflict.ROOT/'adjudications.json']}
    write(OUT/'protected_hashes.json',hashes)
    snapshot=[];byid={r['id']:r for r in source}
    for entry in exported['items']:
        r=byid[entry['id']]
        snapshot.append({**entry,'image':r['image'],'point':r['point'],
            'split_group':entry.get('split_group',r['tree_group']),
            'group_identity_verified':entry.get('reason','').startswith('explicit_species_adjudication'),
            'label_provenance':{'original_review':reviews[entry['id']],'adjudication':adjudication if entry['id'] in {x['id'] for x in conflict.manifest()['items'] if x['role']=='conflict'} else None}})
    assert len(snapshot)==len({r['id'] for r in snapshot})
    for group in {r['split_group'] for r in snapshot}:
        assert len({r['species'] for r in snapshot if r['split_group']==group})==1
    write(OUT/'accepted_training_candidates.json',snapshot)
    write(OUT/'preserved_species_corrections.json',read(conflict.ROOT/'preserved_species_corrections.json'))
    accepted_ids={r['id'] for r in snapshot}
    selected=[r for r in source if r['species']==SILVER and r['id'] not in accepted_ids]
    selected.sort(key=lambda r:r['id']);write(OUT/'selection.json',selected)
    reference=[(r,np.load(DATA/'instances'/r['id']/'cleaned_local.npz',allow_pickle=False)['xyz']) for r in snapshot if r['species']=='Albizia lebbeck' and r['group_identity_verified']]
    rows=coordinates();maps=DenserFrameMaps(CACHE);records=[];cached=None
    for item in selected:
        ident=item['id'];row=rows[item['frame_key']];poly=np.array(item['polygon']);folder=OUT/'instances'/ident;folder.mkdir()
        r={'id':ident,'frame_key':item['frame_key'],'species':SILVER,'original_species':item['original_species'],
           'polygon':item['polygon'],'original_status':item['status'],'original_reason':item.get('reason'),
           'original_review':reviews.get(ident),'annotation_provenance':item['annotation_provenance'],
           'status':'needs_rework','training_eligible':False,'total_height_verified':False}
        with Image.open(VMMS/row['source_image_relpath']) as src:
            im=src.convert('RGB');w,h=im.size;lo=poly.min(0);hi=poly.max(0)
            im.crop((*np.rint(lo*[w,h]).astype(int),*np.rint(hi*[w,h]).astype(int))).save(folder/'image.jpg',quality=95)
            im.thumbnail((3072,1536));im.save(folder/'panorama.jpg',quality=94)
        label=im.copy();d=ImageDraw.Draw(label);pp=poly*np.array(im.size);d.line([tuple(v) for v in np.r_[pp,pp[:1]]],fill='#ffd027',width=3);label.save(folder/'label.jpg',quality=94)
        old=DATA/'instances'/ident/'cleaned_local.npz';r['has_previous']=old.exists()
        if old.exists():views(np.load(old,allow_pickle=False)['xyz'],folder/'before_views.jpg')
        try:
            if cached!=item['frame_key']:cloud,paths=maps.nearby(row);cached=item['frame_key']
            r.update(source_lidar=paths,source_sampling_m=maps.sampling_m)
            crown,inside,stats=choose_crown(cloud,row,poly);axis=np.median(crown[:,:2],0);floor=float(np.quantile(crown[:,2],.02)-.25)
            local=cloud[np.linalg.norm(cloud[:,:2]-axis,axis=1)<8];ground=ground_candidate(local,axis)
            base=ground['ground_z_m'] if ground['status']=='local_plane_candidate' else float(np.quantile(local[:,2],.01))
            lower=cloud[inside&(cloud[:,2]>base+.25)&(cloud[:,2]<floor+.8)&(np.linalg.norm(cloud[:,:2]-axis,axis=1)<6)]
            stem,tracking=track(lower,axis,base,floor);p=crown if stem is None else np.r_[crown,stem];p=p[voxel(p,.08)]
            ti=np.flatnonzero(p[:,2]<floor);ci=np.flatnonzero(p[:,2]>=floor);count=min(2048,len(p));nt=min(len(ti),768);nc=min(len(ci),count-nt);nt=min(len(ti),count-nc)
            idx=np.r_[ti[coverage_sample(p[ti],nt)] if nt else np.array([],int),ci[coverage_sample(p[ci],nc)] if nc else np.array([],int)]
            center=p.mean(0);scale=float(np.linalg.norm(p-center,axis=1).max());normalized=((p[idx]-center)/scale).astype(np.float32)
            assert np.isfinite(normalized).all() and len(np.unique(normalized,axis=0))==len(idx)<=2048
            assert cKDTree(cloud).query(p)[0].max()<1e-7
            assert cKDTree(p).query(normalized*scale+center)[0].max()<1e-4
            np.savez_compressed(folder/'point.npz',points_xyz=normalized,centroid_xyz=center,scale=scale,metric_units='m');np.savez_compressed(folder/'cleaned_local.npz',xyz=p)
            views(p,folder/'point_views.jpg',floor);canvas=label.copy();d=ImageDraw.Draw(canvas);uv,_=project(p,row)
            for i in balanced_ids(p,7000):
                x,y=uv[i]*np.array(im.size);d.ellipse((x-1,y-1,x+1,y+1),fill='#e58e3b' if p[i,2]<floor else '#13d9bb')
            canvas.save(folder/'projection.jpg',quality=94)
            links=[]
            for ref,other in reference:
                ab=float((cKDTree(other).query(p)[0]<.2).mean());ba=float((cKDTree(p).query(other)[0]<.2).mean())
                if max(ab,ba)>.5:links.append({'id':ref['id'],'species':ref['species'],'overlap_new_to_reference':ab,'overlap_reference_to_new':ba,'identity_verified':False})
            r.update(status='candidate_pending_review',quality={'clean_points':len(p),'sample_unique_points':len(idx),'stem_path_found':stem is not None,'observed_extent_m':np.ptp(p,axis=0).tolist(),'axis_local_xy':axis.tolist(),'crown_floor_z':floor},crown_stats=stats,tracking=tracking,ground=ground,conflicting_reference_links=links,
                species_status='requires_identity_review_against_adjudicated_albizia' if links else 'original_label_pending_new_pair_review')
        except (ValueError,IndexError,OSError) as e:r['reason']=str(e)
        write(folder/'record.json',r);records.append(r);write(OUT/'manifest.progress.json',records)
        print(ident,r['status'],r.get('quality',{}).get('stem_path_found'),flush=True)
    write(OUT/'manifest.json',records)
    write(OUT/'summary.json',{'accepted_candidate_views':len(snapshot),'accepted_by_species':dict(Counter(r['species'] for r in snapshot)),
        'split_groups_by_species':{s:len({r['split_group'] for r in snapshot if r['species']==s}) for s in {r['species'] for r in snapshot}},
        'attempted_recovery':len(records),'recovery_status':dict(Counter(r['status'] for r in records)),
        'training_started':False,'reason_not_training':'Silver leadtree has only one accepted identity group; cannot form species-complete group-disjoint validation.',
        'snapshot_adjudication':adjudication})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print('Source reviews unchanged; training not started.',flush=True)

if __name__=='__main__':main()
