"""Read-only provenance/dedup firewall over a new institutional image batch."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from artocarpus_fresh_split_20260917 import record_identity,lookup_maps,historical_matches,merge_records,HELD
from import_artocarpus_user_20260918 import current_human_review_exclusions,REGISTRY,perceptual_hash
from PIL import Image,ImageOps,ImageDraw,ImageFont

BASE=ROOT/'outputs/conservation_authoritative_collection_20260918_v1'
SPECIES=['Aquilaria sinensis','Artocarpus hypargyreus','Rhodoleia championii','Keteleeria fortunei']

def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2)

def main(paths):
    out=BASE/'audit';out.mkdir(exist_ok=False)
    reg=load(REGISTRY);live=current_human_review_exclusions()
    save(out/'live_review_snapshot.json',live)
    forbidden=set(live['blocked_sha256']); blocked_ids=set(live['blocked_ids'])
    manifests=[Path(p).resolve() for p in paths]
    inputs=[dict(path=str(p),sha256=sha(p)) for p in [REGISTRY,*manifests]]
    review_path=BASE/'hk_official/visual_review_root.json'
    reviews={r['id']:r for r in load(review_path)['records']}
    inputs.append(dict(path=str(review_path),sha256=sha(review_path)))
    extra_review=BASE/'visual_review_additional.json'
    if extra_review.is_file():
        reviews.update({r['id']:r for r in load(extra_review)['records']})
        inputs.append(dict(path=str(extra_review),sha256=sha(extra_review)))
    training_index={}; latest_records=[];prior=ROOT/'outputs/SST_Conservation_Iterative_20260918_v1'
    for phase in ['train','planning','newdev']:
        f=prior/f'{phase}_rows.json';inputs.append(dict(path=str(f),sha256=sha(f)))
        for r in load(f):
            training_index.setdefault(r['sha256'],[]).append(dict(id=r['sample_id'],phase=phase))
            rec=record_identity(r,f);rec['phase']='planning' if phase=='newdev' else phase
            latest_records.append(rec)
    user_manifest=ROOT/'external_datasets/artocarpus_user_20260918_v1/final_training_manifest.json'
    inputs.append(dict(path=str(user_manifest),sha256=sha(user_manifest)))
    for r in load(user_manifest):
        rec=record_identity(r,user_manifest);rec['phase']='train';latest_records.append(rec)
    # Preserve historic registry IDs, append latest metadata-only identities.
    # No historical/test image pixels are decoded for deduplication.
    added=merge_records(latest_records)
    reg['records'].extend(added);maps,visual=lookup_maps(reg)
    save(out/'latest_identity_records.json',added)
    rows=[];seen={};seen_visual=[];metadata_only=0
    for manifest in manifests:
        raw=load(manifest)
        if isinstance(raw,dict):
            raw=next((raw[k] for k in ['records','items','images','rows'] if isinstance(raw.get(k),list)),None)
        if not isinstance(raw,list): raise RuntimeError('Unknown manifest schema '+str(manifest))
        for old in raw:
            path=old.get('source_image') or old.get('local_path') or old.get('image_path')
            if not path or not Path(path).is_file():
                if path or old.get('sha256') or old.get('downloaded') is True or old.get('download_status')=='downloaded':
                    raise RuntimeError('Downloaded image missing: '+str(path))
                metadata_only+=1;continue
            path=Path(path).resolve()
            if not path.is_relative_to(BASE): raise RuntimeError('Incoming artifact outside acquisition directory')
            r=dict(old);r['source_image']=str(path);r['source_manifest']=str(manifest)
            r['source_species']=old.get('source_species') or old.get('scientific_name') or old.get('species')
            if r['source_species'] not in SPECIES: raise RuntimeError('Unexpected species '+str(r['source_species']))
            r['id']=str(old.get('id') or old.get('record_id') or path.stem)
            r['source_host']=urlsplit(old.get('source_page') or old.get('source_url') or '').netloc
            content_domains={'specimen_domain','live_detail_photo','live_crown_photo','live_whole_tree_photo'}
            r['domain']=old.get('domain') if old.get('domain') in content_domains else 'unclassified_image'
            if r['id'] in reviews:
                r['domain']=reviews[r['id']]['domain'];r['agent_visual_review']=reviews[r['id']]
            r['image_url']=old.get('image_url') or old.get('raw_url') or old.get('original_url')
            r['license']=old.get('license') or old.get('media_license')
            r['source_page']=old.get('source_page') or old.get('source_url')
            r['group_id']=old.get('group_id') or old.get('source_group') or old.get('occurrence_identifier') or r['source_page']
            actual=sha(path)
            if old.get('sha256') and actual!=old['sha256']: raise RuntimeError('Incoming image hash changed')
            r['sha256']=actual
            with Image.open(path) as im:
                rgb=ImageOps.exif_transpose(im).convert('RGB');r['phash']=perceptual_hash(rgb);r['dimensions']=list(rgb.size)
            rec=record_identity(r,manifest);hits,ids,cached=historical_matches(rec,reg,maps,visual,6)
            content_keys={'sha256','path','photo','sample','observation','image_url','near_phash'}
            existing_content={i for k,vs in hits.items() if k in content_keys for i in vs}
            existing_cached={k:v for k,v in cached.items() if k in content_keys}
            held_related={i for k,vs in hits.items() if k in content_keys|{'group','gallery','observer'} for i in vs if HELD & set(reg['records'][i]['phases'])}
            same_new=seen.get(actual)
            near_new=[j for h,j in seen_visual if (int(r['phash'],16)^h).bit_count()<=6]
            # Do not count two scans or resized derivatives as new images.
            flags=list(old.get('quality_flags') or [])
            flags.extend(reviews.get(r['id'],{}).get('flags',[]))
            if actual in forbidden or r['id'] in blocked_ids: flags.append('actual_human_reject_pending_or_uncertain')
            if existing_content or existing_cached or actual in training_index: flags.append('existing_project_image_or_near_duplicate')
            if same_new is not None: flags.append('within_batch_exact_duplicate')
            elif near_new: flags.append('within_batch_near_duplicate')
            if min(r['dimensions'])<224: flags.append('low_resolution')
            if not r['license']: flags.append('missing_image_license')
            elif not str(r['license']).upper().startswith(('CC-','CC BY','CC0','ATTRIBUTED NON-COMMERCIAL')):
                flags.append('image_license_requires_review')
            if r['domain']=='unclassified_image': flags.append('image_domain_unreviewed')
            if not r['source_page'] or not r['image_url']: flags.append('incomplete_provenance')
            if held_related: flags.append('historical_heldout_related_source_group')
            if r['domain']=='specimen_domain':
                lane='specimen_domain_reference_only'
            elif r['domain'] in content_domains-{'specimen_domain'}: lane='live_image_candidate_not_training_approved'
            else: lane='quarantine_image_domain_unreviewed'
            if flags: lane='quarantine_'+flags[0]
            r.update(collection_flags=flags,collection_lane=lane,
                historical_identity_matches={k:v for k,v in hits.items()},historical_cache_matches=cached,
                latest_training_study_matches=training_index.get(actual,[]),
                historical_heldout_related_ids=sorted(held_related),within_batch_duplicate_index=same_new,
                within_batch_near_indices=near_new,training_eligible=False,split='quarantine',
                collection_review='source audit and technical duplicate screen, not expert taxonomic certification',
                new_content_after_screen=not bool(existing_content or existing_cached or actual in training_index or same_new is not None or near_new),
                independent_tree_verified=False)
            seen.setdefault(actual,len(rows));seen_visual.append((int(r['phash'],16),len(rows)));rows.append(r)
    save(out/'manifest.json',rows)
    save(out/'source_snapshot.json',inputs)
    summaries={}
    for species in SPECIES:
        rr=[r for r in rows if r['source_species']==species]
        summaries[species]=dict(downloaded=len(rr),unique_bytes=len({r['sha256'] for r in rr}),
            new_content_after_screen=sum(r['new_content_after_screen'] for r in rr),
            lanes=dict(Counter(r['collection_lane'] for r in rr)),
            non_specimen_candidates=sum(r['collection_lane']=='live_image_candidate_not_training_approved' for r in rr))
    summary=dict(created_at=datetime.now(timezone.utc).isoformat(),downloaded=len(rows),
        unique_bytes=len(seen),new_content_after_screen=sum(r['new_content_after_screen'] for r in rows),
        metadata_only=metadata_only,by_species=summaries,training_eligible=0,trained=False,deployed=False,
        limitations=['Historical index plus current train/dev identity metadata; not exhaustive proof of no unrecorded cross-site reuse.',
        'pHash<=6 is a conservative duplicate flag, not botanical verification.',
        'Collector/record/source groups are not verified independent trees.',
        'Specimen sheets never promoted into street-tree image training by this script.'])
    save(out/'SUMMARY.json',summary)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    for species in SPECIES:
        rr=[r for r in rows if r['source_species']==species]
        for start in range(0,len(rr),20):
            subset=rr[start:start+20];sheet=Image.new('RGB',(1100,275*((len(subset)+4)//5)),(240,242,237));draw=ImageDraw.Draw(sheet)
            for i,r in enumerate(subset):
                x=(i%5)*220;y=(i//5)*275
                with Image.open(r['source_image']) as im: pic=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(210,210))
                sheet.paste(pic,(x+(220-pic.width)//2,y));draw.text((x+3,y+211),r['id'][:27],font=font,fill='black')
                draw.text((x+3,y+229),r['domain'][:28],font=font,fill='black')
                draw.text((x+3,y+247),'NEW' if r['new_content_after_screen'] else 'KNOWN / DUPLICATE',font=font,fill='darkgreen')
            sheet.save(out/(species.replace(' ','_')+f'_{start//20+1}.jpg'),quality=90)
    for p in inputs:
        if sha(p['path'])!=p['sha256']: raise RuntimeError('Historical/input manifest changed during audit')
    for d in live['databases']:
        if sha(d['path'])!=d['sha256']: raise RuntimeError('Review database changed during audit')
    save(out/'COMPLETE.json',dict(source_manifests_unchanged=True,human_databases_unchanged=True,
         new_image_hashes_verified=True,training_performed=False,model_files_written=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifests',nargs='+');args=p.parse_args();main(args.manifests)
