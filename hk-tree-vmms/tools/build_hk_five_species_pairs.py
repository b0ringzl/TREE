"""Full, resumable VMMS pairing for the five requested expansion classes.

Reuses original-map selection, verified PCD streaming, real-point sampling and
single-tree extraction from the three-class pipeline. Does not update weights
or existing human reviews. Every label is attempted, not just a 40-case pilot.
"""
import argparse
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import build_hk_dense_dataset as dense
from build_hk_lidar_pairs import manual_records, read, write, VMMS, voxel
from expand_requested_five_species import DenserFrameMaps

OUT=dense.PROJECT/'derived/hk_dualsource_five_species_20260907'
NAMES={'Delonix regia':'凤凰木','Aleurites moluccana':'石栗',
       'Leucaena leucocephala':'银合欢','Albizia lebbeck':'大叶合欢',
       'Araucaria columnaris(Araucaria heterophylla)':'南洋杉（原训练合并类）'}

def canonical(raw):
    if raw in ('银合欢','銀合歡'):return 'Leucaena leucocephala'
    if raw.startswith(('Araucaria columnaris','Araucaria heterophylla')):return list(NAMES)[-1]
    return next((n for n in NAMES if raw.startswith(n)),None)

def effective_records():
    """Resolve older numeric frame keys without duplicating newer full-key reviews."""
    original=manual_records();result={k:r for k,r in original.items() if '__' in k}
    reviews=read(dense.PROJECT/'derived/exhaustive_species_frame_labeler/runtime/frame_review_state.json')
    audit=[]
    for key,r in original.items():
        if '__' in key:continue
        full=r.get('stream_id','')+'__'+r.get('frame_id',key)
        if full in result:action='superseded_by_full_key_record'
        elif full in reviews:action='excluded_by_explicit_review'
        elif r.get('stream_id'):
            result[full]={**r,'frame_key':full,'legacy_frame_key':key};action='resolved_from_stream_and_frame'
        else:action='unresolved_missing_stream'
        audit.append({'legacy_key':key,'canonical_key':full,'action':action})
    return result,audit

def discover(out):
    rows=dense.coordinates();maps=DenserFrameMaps(dense.CACHE)
    previous=read(out/'seeds.progress.json') if (out/'seeds.progress.json').exists() else []
    done={r['id']:r for r in previous};items=[]
    records,key_audit=effective_records();write(out/'frame_key_audit.json',key_audit)
    for key,record in sorted(records.items()):
        labels=[l for l in record.get('labels',[]) if canonical(l.get('species',''))]
        if not labels:continue
        row=rows.get(key);local=None;sources=[];error=None
        for label in labels:
            ident=key+'__'+hashlib.sha256(str(label.get('label_id')).encode()).hexdigest()[:10]
            if ident in done:items.append(done[ident]);continue
            name=canonical(label['species'])
            item={'id':ident,'frame_key':key,'route':record['route'],'species':name,
                  'chinese_name':NAMES[name],'original_species':label['species'],
                  'label_id':label.get('label_id'),'polygon':label.get('points',[]),
                  'annotation_provenance':record['annotation_provenance'],
                  'morphology':'conifer' if name.startswith('Araucaria') else 'broadleaf',
                  'status':'rejected_seed','training_eligible':False}
            if row is None:item['reason']='missing_frame_coordinates'
            elif len(item['polygon'])<3:item['reason']='invalid_polygon'
            else:
                item['survey']=Path(row['source_image_relpath']).parts[0]
                if local is None and error is None:
                    try:local,sources=maps.nearby(row)
                    except (ValueError,OSError) as exc:error=str(exc)
                if error:item['reason']=error
                else:
                    seed,reason=dense.crown_seed(local[voxel(local,.2)],row,np.array(item['polygon']),False)
                    item.update(seed_source_lidar=sources,source_sampling_m=maps.sampling_m)
                    if reason:item['reason']=reason
                    else:item.update(status='seed_found',seed_center=np.median(seed,axis=0).tolist(),seed_points=seed.tolist())
            items.append(item)
            write(out/'seeds.progress.json',items)
            print('seed',len(items),ident,item['status'],flush=True)
    write(out/'seeds.json',items)
    write(out/'scope.json',{'classes':NAMES,'input_annotations':len(items),
          'counts':dict(Counter(x['species'] for x in items)),
          'policy':'All effective labels attempted. No auto-acceptance. Keep original taxonomy. 15m selection only at review, not before source extraction.'})
    return items

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','seeds','cache','extract'],default='all')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    # Protect all previously accepted three-class reviews by content hash.
    protected=[p for p in dense.OUT.rglob('*reviews.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    write(OUT/'protected_reviews.json',hashes)
    dense.Maps=DenserFrameMaps
    items=discover(OUT) if args.stage in ('all','seeds') else read(OUT/'seeds.json')
    if args.stage in ('all','seeds'):dense.choose_maps(items,OUT)
    if args.stage in ('all','cache'):dense.build_dense_cache(items,OUT)
    if args.stage in ('all','extract'):dense.extract(items,OUT)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'Protected review changed during run'
    print('Protected review files unchanged:',len(hashes),flush=True)

if __name__=='__main__':main()
