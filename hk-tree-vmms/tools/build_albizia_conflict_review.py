"""Versioned identity/species conflict evidence. Never changes source reviews."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from build_hk_dense_dataset import coordinates,polygon_inside
from build_hk_lidar_pairs import project

PROJECT=Path(__file__).resolve().parents[2]
DATA=PROJECT/'hk-tree-vmms/derived/hk_dualsource_five_species_20260907'
OUT=PROJECT/'hk-tree-vmms/derived/albizia_conflict_review_v1_20260907'
IDS=['hewentian_pano__004276__479647ba27','hewentian_pano__004287__9515e6f1a6']
CORRECTION='hewentian_pano__004496__f9261fbcdc'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    rows={r['id']:r for r in read(DATA/'manifest.json')};reviews=read(DATA/'reviews.json');coords=coordinates()
    protected={str(p):sha(p) for p in [DATA/'manifest.json',DATA/'reviews.json']}
    for ident in IDS+[CORRECTION]:
        for name in ['cleaned_local.npz','point.npz','record.json']:
            p=DATA/'instances'/ident/name;protected[str(p)]=sha(p)
    xyz=[np.load(DATA/'instances'/i/'cleaned_local.npz',allow_pickle=False)['xyz'] for i in IDS]
    dist=[cKDTree(xyz[1-i]).query(xyz[i])[0] for i in range(2)]
    metrics={'point_counts':[len(x) for x in xyz],
        'directed_overlap':{str(t):[float((d<t).mean()) for d in dist] for t in [.05,.10,.15,.30]},
        'axis_distance_m':float(np.linalg.norm(np.array(rows[IDS[0]]['quality']['axis_local_xy'])-rows[IDS[1]]['quality']['axis_local_xy'])),
        'shared_lidar_paths':sorted(set(rows[IDS[0]]['dense_source_paths'])&set(rows[IDS[1]]['dense_source_paths'])),
        'identity_assessment':'likely_same_target_or_overlapping_extraction_not_verified',
        'species_resolution':'pending_human_adjudication','cross_projection':[]}
    items=[]
    for k,ident in enumerate(IDS+[CORRECTION]):
        r=rows[ident];folder=OUT/ident;folder.mkdir(exist_ok=True)
        with Image.open(r['source_panorama']) as source:
            raw=source.convert('RGB');w,h=raw.size;poly=np.array(r['polygon']);lo=np.maximum(poly.min(0)-.015,0);hi=np.minimum(poly.max(0)+.015,1)
            crop=raw.crop((int(lo[0]*w),int(lo[1]*h),int(hi[0]*w),int(hi[1]*h)))
            crop.save(folder/'native_crop.jpg',quality=96)
            raw.thumbnail((3072,1536));raw.save(folder/'panorama.jpg',quality=94)
        draw=ImageDraw.Draw(raw);pp=[tuple(v) for v in poly*np.array(raw.size)];draw.line(pp+[pp[0]],fill='#ffd64a',width=3);raw.save(folder/'label.jpg',quality=94)
        if k<2:
            cross=raw.copy();draw=ImageDraw.Draw(cross)
            # Both measured clouds in each camera; colors are source IDs, not species.
            for j in range(2):
                uv,_=project(xyz[j],coords[r['frame_key']]);inside=polygon_inside(uv,poly,dilation=0)
                metrics['cross_projection'].append({'camera_id':ident,'cloud_id':IDS[j],'inside_original_mask_fraction':float(inside.mean())})
                for u,v in uv[::5]:
                    x,y=u*cross.width,v*cross.height;draw.ellipse((x-1,y-1,x+1,y+1),fill=['#ff4bc3','#04dce5'][j])
            cross.save(folder/'cross_projection.jpg',quality=95)
        items.append({'id':ident,'frame_key':r['frame_key'],'original_species':r['species'],'human_review':reviews[ident],
            'tree_group':r['tree_group'],'image':r.get('image'),'point':r.get('point'),'source_panorama':r['source_panorama'],'native_panorama_size':[w,h],
            'crop_size':list(crop.size),'model_evidence':r.get('image_evidence',{}),'role':'conflict' if k<2 else 'preserved_correction'})
    # All accepted views of any cross-label group are held from this new export.
    groups={}
    for i,r in rows.items():
        rv=reviews.get(i,{})
        if rv.get('decision')=='accept':groups.setdefault(r.get('tree_group',i),set()).add(rv.get('species',r['species']))
    conflicts={g for g,s in groups.items() if len(s)>1};candidates=[];excluded=[]
    for i,rv in reviews.items():
        r=rows[i];record={'id':i,'species':rv['species'],'original_species':r['species'],'tree_group':r['tree_group'],
            'original_pair_decision':rv['decision'],'image':r.get('image'),'point':r.get('point'),'training_eligible':False}
        if rv['decision']!='accept':record['reason']='pair_rejected';excluded.append(record)
        elif r['tree_group'] in conflicts:record['reason']='cross_species_identity_conflict';excluded.append(record)
        else:record['reason']='accepted_candidate_requires_identity_grouped_split';candidates.append(record)
    corrections=[{'id':i,'original_species':rows[i]['species'],'effective_species':rv['species'],'pair_decision':rv['decision'],
                  'note':rv.get('note',''),'image_training_status':'requires_separate_image_quality_review','pair_training_eligible':False}
                 for i,rv in reviews.items() if rv['species']!=rows[i]['species']]
    write(OUT/'manifest.json',{'items':items,'metrics':metrics,'source_review_sha256':protected[str(DATA/'reviews.json')],
        'candidate_export_count':len(candidates),'quarantined_accepted_count':sum(x['reason']=='cross_species_identity_conflict' for x in excluded),
        'policy':'Species remain separate. No automatic adjudication, relabeling, acceptance or training.'})
    write(OUT/'training_candidates.json',candidates);write(OUT/'training_exclusions.json',excluded)
    write(OUT/'preserved_species_corrections.json',corrections)
    assert all(sha(Path(p))==h for p,h in protected.items())
    write(OUT/'protected_hashes.json',protected)
    print(json.dumps({'metrics':metrics,'candidates':len(candidates),'excluded':len(excluded),'protected_files_unchanged':len(protected)},ensure_ascii=False))

if __name__=='__main__':main()
