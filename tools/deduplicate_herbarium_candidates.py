"""Read-only existing-image index and conservative candidate leakage checks."""
import argparse,csv,hashlib,io,json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from collect_herbarium_candidates import OUT,INVENTORY,ROOT,perceptual_hash
from inventory_herbarium_images import write_json

INDEX=OUT/'existing_image_index.jsonl'

def references():
    refs=[]
    base=ROOT/'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1/datasets'
    for view in ['full','crop']:
        p=base/view/'inventory.csv'
        if not p.exists():continue
        with p.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                refs.append({'path':r['source_image'] if view=='full' else r['materialized_path'],'split':r['split'],'species':r['species'],'dataset':'image_baseline_'+view,'sample_id':r['sample_id'],'declared_sha256':r.get('source_sha256') if view=='full' else None})
    for version in ['reviewed_five_species_pilot_v1_20260907','hk_dualsource_three_species_20260907/experiment']:
        p=ROOT/'hk-tree-vmms/derived'/version/'split.json'
        if not p.exists():continue
        data=json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(data,list):raise ValueError('Unrecognized split schema: '+version)
        for r in data:
            if r.get('image'):refs.append({'path':r['image'],'split':r['split'],'species':r['species'],'dataset':version,'sample_id':r['id'],'declared_sha256':r.get('image_sha256')})
    return refs

def scan(item):
    p=Path(item)
    try:
        stat=p.stat();raw=p.read_bytes()
        with Image.open(io.BytesIO(raw)) as original:im=ImageOps.exif_transpose(original).convert('RGB')
        return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'phash':perceptual_hash(im),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'status':'ok'}
    except Exception as e:return {'path':str(p),'status':'error','error':type(e).__name__+': '+str(e)}

def build_index():
    OUT.mkdir(parents=True,exist_ok=True);refs=references();write_json(OUT/'reference_scope.json',{'records':len(refs),'by_dataset':dict(Counter(r['dataset'] for r in refs)),'by_split':dict(Counter(r['split'] for r in refs)),'references':refs})
    paths=sorted({r['path'] for r in refs});cache={}
    if INDEX.exists():
        for line in INDEX.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line);cache[r['path']]=r
            except json.JSONDecodeError:raise ValueError('Incomplete index; inspect before resuming')
    pending=[]
    for p in paths:
        old=cache.get(p)
        try:st=Path(p).stat()
        except OSError:pending.append(p);continue
        if not old or old.get('status')!='ok' or old.get('size')!=st.st_size or old.get('mtime_ns')!=st.st_mtime_ns:pending.append(p)
    print(f'Indexing {len(pending)} uncached paths / {len(paths)} unique references',flush=True)
    with INDEX.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(scan,pending)):
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
            if (i+1)%500==0:f.flush();print(f'Indexed {i+1}/{len(pending)}',flush=True)
    print('Existing image index ready',flush=True)

def crosscheck():
    refs=references();lookup={}
    for r in refs:lookup.setdefault(r['path'],[]).append({k:v for k,v in r.items() if k!='path'})
    indexed={json.loads(l)['path']:json.loads(l) for l in INDEX.read_text(encoding='utf-8').splitlines() if l}
    entries=[v for k,v in indexed.items() if k in lookup and v['status']=='ok']
    missing=[p for p in lookup if p not in indexed or indexed[p]['status']!='ok']
    candidates=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));results=[]
    hashes=np.array([int(e['phash'],16) for e in entries],dtype=np.uint64)
    popcount=np.array([bin(n).count('1') for n in range(256)],dtype=np.uint8)
    for c in candidates:
        if c['status']!='downloaded':continue
        distance=popcount[np.bitwise_xor(hashes,np.uint64(int(c['phash'],16))).view(np.uint8).reshape(-1,8)].sum(axis=1)
        matches=[]
        for j in np.flatnonzero(distance<=6):
            e=entries[int(j)];matches.append({'path':e['path'],'references':lookup[e['path']],'phash_distance':int(distance[j]),'same_sha256':e['sha256']==c['sha256']})
        # Exact match must not rely on a perceptual-hash filter.
        matched_paths={m['path'] for m in matches}
        for e in entries:
            if e['sha256']==c['sha256'] and e['path'] not in matched_paths:matches.append({'path':e['path'],'references':lookup[e['path']],'same_sha256':True,'phash_distance':None})
        results.append({'id':c['id'],'matches':matches,'status':'quarantine_duplicate_candidate' if matches else 'no_match_at_checked_threshold','training_eligible':False})
    write_json(OUT/'cross_dataset_duplicates.json',{'threshold':6,'scope_records':len(refs),'indexed_unique_files':len(entries),'unreadable_or_missing':missing,'candidate_results':results,'note':'pHash similarity requires visual confirmation; no match is not proof of independence. Other project datasets outside the named scope are not checked.'})
    print(json.dumps({'indexed_unique_files':len(entries),'missing':len(missing),'new_candidates_with_matches':sum(bool(r['matches']) for r in results)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['index','cross']);a=p.parse_args()
    build_index() if a.stage=='index' else crosscheck()
