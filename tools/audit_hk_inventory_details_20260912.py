"""Content checks supplement the metadata inventory. Never edits source data."""
import json,hashlib,csv
from pathlib import Path
from collections import Counter
import numpy as np,laspy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/hk_data_inventory_20260912_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    inv=read(OUT/'inventory.json');trees=[]
    for h in inv['point_headers']:
        if 'individual_tree_output' not in Path(h['path']).parts:continue
        p=Path(h['path']);cloud=laspy.read(p);xyz=np.c_[cloud.x,cloud.y,cloud.z];unique=len(np.unique(xyz,axis=0));bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()]
        trees.append(dict(path=str(p),block=p.parents[2].name,id=p.stem,points=len(xyz),unique_points=unique,all_finite=bool(np.isfinite(xyz).all()),actual_bounds=bounds,extent=np.ptp(xyz,axis=0).tolist(),classification_values=np.unique(cloud.classification).tolist(),header_bounds_match=bool(np.allclose(bounds[0],h['bounds_min']) and np.allclose(bounds[1],h['bounds_max'])),sha256=sha(p)))
    large=[]
    for b in inv['processed_blocks']:
        h=b['primary_header']
        if not h:continue
        p=Path(h['path']);chunks=[]
        with laspy.open(p) as r:
            for start in [0,h['points']//2,max(0,h['points']-4096)]:
                r.seek(start);c=r.read_points(4096);chunks.append(np.c_[c.x,c.y,c.z])
        xyz=np.concatenate(chunks);large.append(dict(path=str(p),sampled_points=len(xyz),all_finite=bool(np.isfinite(xyz).all()),sample_bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()],header_zero_bounds=bool(np.allclose(h['bounds_min'],0) and np.allclose(h['bounds_max'],0))))
    maps=[ROOT/'lidar data/hk/tree981'/n for n in ['map.pcd','map_dense.pcd']];hs={str(p):sha(p) for p in maps}
    coord=read(ROOT/'hk-tree-vmms/outputs/coordinates/coordinate_summary.json')
    streams=[]
    for s in coord['surveys']:
        for v in s['streams']:streams.append(dict(survey=s['survey_id'],stream=v['stream_id'],frames=v['frame_count'],distance_km=v['route_distance_km']))
    cat=read(ROOT/'outputs/hk_point_assist_20260911_v1/catalog.json');review=read(ROOT/'outputs/hk_point_assist_20260911_v1/geometry_reviews.json')
    geometry=dict(rows=len(review),flags={k:sum(bool(r.get(k)) for r in review.values()) for k in ['same_tree','isolated_tree','base_visible','crown_complete','species_verified']})
    refs=read(ROOT/'external_datasets/spacescan_whole_tree_references_20260910_v2/mixed_v3/accepted_manifest.json')
    reference=dict(rows=len(refs),species=len({r['species'] for r in refs}),unique_hashes=len({r['sha256'] for r in refs}),training_eligible=sum(bool(r.get('training_eligible')) for r in refs))
    files=[json.loads(line) for line in (OUT/'file_inventory.jsonl').open(encoding='utf-8')];web={}
    for key,rule in [('herbarium',lambda r:r['collection']=='external_herbarium_training_candidates_20260907' and r['relative_path'].startswith('images/')),('inat_max150',lambda r:r['collection']=='external_inat_max150' and 'images' in Path(r['relative_path']).parts),('inaturalist_dated',lambda r:r['collection']=='external_inaturalist_dated'),('legacy',lambda r:r['collection']=='legacy_web_dataset')]:
        a=[r for r in files if rule(r) and r['extension'] in ['.jpg','.jpeg','.png','.webp']];web[key]=dict(image_files=len(a),species_folders=len({Path(r['relative_path']).parts[0] for r in a}),note='Physical image files; may include originals and derived copies, not globally deduplicated')
    result=dict(single_tree_content=trees,large_las_samples=large,tree981_hashes=hs,tree981_maps_identical=len(set(hs.values()))==1,coordinate_streams=streams,current_pair_species=dict(Counter(r['species'] for r in cat)),current_geometry_reviews=geometry,reference_library=reference,web_image_subsets=web,
        single_tree_counts=dict(files=len(trees),points=sum(r['points'] for r in trees),less_than_512=sum(r['unique_points']<512 for r in trees),less_than_2048=sum(r['unique_points']<2048 for r in trees),zero_or_negative_height=sum(r['extent'][2]<=0 for r in trees),less_than_1m_height=sum(r['extent'][2]<1 for r in trees),bad_header_bounds=sum(not r['header_bounds_match'] for r in trees)),source_mutations=False)
    result['supersedes']='content_audit.json (corrected nested iNaturalist image scope and single-tree block ID extraction)'
    p=OUT/'content_audit_v2.json'
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['single_tree_content','tree981_hashes']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
