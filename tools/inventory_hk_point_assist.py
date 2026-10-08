"""Snapshot existing assets/reviews; no training, relabeling or original-file writes."""
import csv
import json
import hashlib
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'hk-tree-vmms/derived'
OUT=ROOT/'outputs/hk_point_assist_20260911_v1'
THREE=D/'hk_dualsource_three_species_20260907'
def read(p,default=None):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def species(s):
    if s.startswith(('榕树','Ficus microcarpa')):return 'Ficus microcarpa'
    for n in ['Livistona chinensis','Wodyetia bifurcata','Delonix regia','Aleurites moluccana','Leucaena leucocephala','Albizia lebbeck']:
        if s.startswith(n):return n
    if s.startswith('Araucaria'):return 'Araucaria columnaris(Araucaria heterophylla)'
    return s

def main():
    OUT.mkdir(exist_ok=False)
    manifests={};reviews={};allrows=[];sources={}
    def load(p,default=None):
        v=read(p,default)
        if p.exists():sources[str(p)]=sha(p)
        return v
    for key in ['hk_dualsource_three_species_20260907','hk_dualsource_five_species_20260907']:
        manifests[key]=load(D/key/'manifest.json',[])
        reviews[key]=load(D/key/('pair_reviews.json' if key.startswith('hk_dualsource_three') else 'reviews.json'),{})
    original={r['id']:r for r in manifests[THREE.name]}
    measured=load(THREE/'metric_height_v1/measurements.json',{})
    candidates={};revisions=[]
    def add(r,version,review,folder=None,original_row=None):
        base=original_row or r;folder=folder or Path(r.get('point','')).parent
        p=folder/'point.npz';clean=folder/'cleaned_local.npz'
        img=folder/'image.jpg'
        if not img.is_file():img=Path(base.get('image',''))
        m=measured.get(r['id'],{}) if version==THREE.name else {}
        row=dict(id=r['id'],version=version,species=species(review.get('species') or base.get('species','榕树')),
            review=review,review_decision=review.get('decision','pending'),survey=base.get('survey',r.get('survey','')),
            provisional_tree_group=base.get('tree_group') or base.get('identity_group') or r['id'],
            leakage_group=base.get('leakage_group'),group_label_conflict=base.get('group_label_conflict',False),
            asset_paths={'point':str(p),'cleaned':str(clean),'image':str(img),'projection':str(folder/'projection.jpg')},
            source_lidar=base.get('source_lidar') or base.get('dense_source_paths') or base.get('seed_source_lidar') or [],
            dense_source=base.get('dense_source'),source_cache_paths=r.get('source_cache_paths',[]),
            quality=r.get('quality',base.get('quality',{})),ground=r.get('ground',m.get('ground',{})),
            historical_measurement=m,tracking=r.get('tracking',{}),total_height_verified=False,
            coverage_scope=base.get('coverage_scope','Pair acceptance is not ground/crown completeness certification'),
            classification_test_eligible=False)
        row['assets_exist']={k:Path(v).is_file() for k,v in row['asset_paths'].items()}
        row['current_accepted']=review.get('decision')=='accept' and all(row['assets_exist'][k] for k in ['point','image','cleaned'])
        row['key']=hashlib.sha256((version+'|'+r['id']).encode()).hexdigest()[:20]
        allrows.append(row)
        if row['current_accepted']:
            if r['id'] in candidates:revisions.append(dict(id=r['id'],old=candidates[r['id']]['version'],new=version))
            candidates[r['id']]=row
    for r in manifests[THREE.name]:
        if r.get('point'):add(r,THREE.name,reviews[THREE.name].get(r['id'],{}))
    for version in ['banyan_tracking_pilot_v1','banyan_tracking_batch_v2']:
        rr=load(THREE/version/'reviews.json',{})
        for r in load(THREE/version/'results.json',[]):
            if r['id'] not in original:continue
            add(r,version,rr.get(r['id'],{}),THREE/version/r['id'],original[r['id']])
    five=load(D/'reviewed_five_species_pilot_v1_20260907/accepted_manifest.json',[])
    for r in five:
        version=r['source_version']
        if version not in reviews:reviews[version]=load(D/version/'reviews.json',{})
        rev=reviews[version].get(r['id'],{})
        # Keep explicit conflict adjudications in the curated manifest; never infer a new label.
        rev={**rev,'species':r['species']} if rev.get('decision')=='accept' else rev
        add(r,version,rev,Path(r['asset_dir']))
    accepted=list(candidates.values());bad=[]
    for r in accepted:
        r['sha256']={k:sha(Path(v)) for k,v in r['asset_paths'].items() if r['assets_exist'][k]}
        try:
            with np.load(r['asset_paths']['cleaned'],allow_pickle=False) as f:
                xyz=np.asarray(f['xyz'],dtype=np.float64)
            if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('invalid XYZ')
            r['observed']=dict(point_count=len(xyz),unique_points=len(np.unique(xyz,axis=0)),
                extent_m=np.ptp(xyz,axis=0).tolist(),z_quantiles=np.quantile(xyz[:,2],[.005,.5,.995]).tolist(),
                centroid_local=xyz.mean(0).tolist(),units='metres',height_verified=False)
        except Exception as e:r['current_accepted']=False;bad.append(dict(key=r['key'],error=str(e)))
    accepted=[r for r in accepted if r['current_accepted']]
    raw_path=ROOT/'hk-tree-vmms/outputs/coordinates/pointcloud_assets.csv';raw=list(csv.DictReader(raw_path.open(encoding='utf-8-sig')))
    sources[str(raw_path)]=sha(raw_path);raw_summary=[]
    vmms=ROOT.parent/'vmms'
    for survey in sorted({r['survey_id'] for r in raw}):
        subset=[r for r in raw if r['survey_id']==survey]
        found=[r for r in subset if (vmms/r['source_relpath']).is_file()]
        raw_summary.append(dict(survey=survey,indexed_files=len(subset),existing_files=len(found),
            actual_bytes=sum((vmms/r['source_relpath']).stat().st_size for r in found),
            warning='Raw map variants overlap; point counts are not counts of independent trees. Presence/size audit only; no fresh full raw-file hash scan.'))
    counts=[]
    for n in sorted({r['species'] for r in accepted}):
        rr=[r for r in accepted if r['species']==n]
        counts.append(dict(species=n,accepted_views=len(rr),provisional_groups=len({(r['survey'],r['provisional_tree_group']) for r in rr}),
            surveys=len({r['survey'] for r in rr}),complete_height_verified=0))
    summary=dict(created_at=datetime.now(timezone.utc).isoformat(),raw_assets=raw_summary,
        canonical_manifest_rows={k:len(v) for k,v in manifests.items()},
        canonical_review_counts={k:dict(Counter(r.get('decision','pending') for r in v.values())) for k,v in reviews.items()},
        accepted_current_views=len(accepted),species=counts,asset_errors=bad,asset_replacements=revisions,
        validated_auxiliary_gain=False,reason='No current BioCLIP + accepted independent-tree point-cloud comparison; existing image-transfer and tiny holdouts do not establish benefit.',
        historical_height_summary=load(THREE/'metric_height_v1/summary.json',{}))
    write(OUT/'catalog.json',accepted);write(OUT/'all_asset_versions.json',allrows);write(OUT/'summary.json',summary)
    write(OUT/'source_hashes.json',sources)
    write(OUT/'review_snapshot.json',reviews)
    for p,h in sources.items():assert sha(Path(p))==h
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
