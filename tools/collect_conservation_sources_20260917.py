"""New source-labelled, licensed conservation candidates. NO fitting or auto-acceptance.

No old image, pending decision, rejection, model, or deployment is modified.
Weak source labels are explicitly quarantined for user review, never training.
"""
import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps
from collect_sstv1_incremental import ROOT, LICENSES, UA, download, old_index, save

SPECIES = [
    ('Aquilaria sinensis', 80), ('Ficus benjamina', 12), ('Ficus microcarpa', 12),
    ('Ilex rotunda', 12), ('Artocarpus hypargyreus', 12), ('Diospyros morrisiana', 12),
    ('Bridelia tomentosa', 12), ('Lagerstroemia speciosa', 12), ('Cratoxylum cochinchinense', 12),
]
GUIDE = 'https://www.herbarium.gov.hk/tc/publications/books/book2/text/aquilaria-sinensis/index.html'


def query(endpoint, params):
    time.sleep(1.1)
    response = requests.get('https://api.inaturalist.org/v1/' + endpoint, params=params, headers=UA, timeout=40)
    response.raise_for_status()  # Never circumvent an access denial / quota response.
    return response.json()


def augmented_index(out):
    obs, hashes, photos, visual = old_index()
    files = list((ROOT / 'external_datasets').rglob('*manifest*.json'))
    files += [ROOT / 'outputs/SSTv1_Aquilaria_User_20260914_v1/user_import_manifest.json']
    examined = []
    def walk(value):
        if isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif isinstance(value, dict):
            yield value
            for item in value.values():
                if isinstance(item, (dict, list)):
                    yield from walk(item)
    for path in sorted(set(files)):
        if out in path.parents:
            continue
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        examined.append({'path':str(path.resolve()), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for r in walk(data):
            for key in ('inat_observation_id', 'observation_id'):
                if r.get(key):
                    obs.add(str(r[key]))
            for key in ('source_page', 'source_url', 'url', 'group_id', 'source_group'):
                if not isinstance(r.get(key), str):
                    continue
                for pattern in (r'/observations/(\d+)', r'observation[_:](\d+)'):
                    for match in re.findall(pattern, r[key]):
                        obs.add(match)
            for key in ('photo_id', 'inat_photo_id'):
                if str(r.get(key, '')).isdigit():
                    photos.add(int(r[key]))
            for key in ('sha256', 'source_sha256'):
                if r.get(key):
                    hashes.add(r[key])
            if r.get('phash'):
                visual.append((int(r['phash'], 16), str(r.get('source_image') or r.get('file') or path)))
    save(out / 'prior_index_audit.json', {'manifests':examined, 'old_observations':len(obs),
         'old_photos':len(photos), 'old_hashes':len(hashes), 'old_phashes':len(visual),
         'scope':'Existing old_index plus ALL external dataset manifests, including pending/rejected and user 84 images',
         'limitation':'Unrecorded observation identities or visually very different photos of the same tree cannot be excluded with certainty.'})
    return obs, hashes, photos, visual


def identifiers(o, taxon):
    current = [i for i in o.get('identifications', []) if i.get('current') and not i.get('hidden') and not i.get('spam')]
    exact = sorted({i['user']['id'] for i in current if i.get('taxon_id', (i.get('taxon') or {}).get('id')) == taxon})
    conflicts = [i['id'] for i in current if (i.get('taxon') or {}).get('rank_level', 100) <= 10
                 and (i.get('taxon') or {}).get('id') != taxon]
    return {'exact_identifier_ids':exact, 'independent_of_observer':bool(set(exact)-{o['user']['id']}),
            'conflicting_current_identifications':conflicts,
            'tier':'multiple_independent_exact_ids' if len(exact) >= 2 and set(exact)-{o['user']['id']} and not conflicts
                   else 'source_label_only_requires_human_confirmation'}


def contact_sheets(rows, out):
    for prefix in ('positive', 'negative'):
        selected = [r for r in rows if ('positive' if r['source_species']=='Aquilaria sinensis' else 'negative') == prefix]
        for start in range(0, len(selected), 12):
            sheet = Image.new('RGB', (1440, 1140), 'white'); draw = ImageDraw.Draw(sheet)
            for j, r in enumerate(selected[start:start+12]):
                im = ImageOps.exif_transpose(Image.open(r['source_image'])).convert('RGB'); im.thumbnail((350, 290))
                x=(j % 4)*360; y=(j // 4)*380
                sheet.paste(im, (x+(350-im.width)//2, y))
                draw.text((x+4,y+295), r['id'], fill='black')
                draw.text((x+4,y+315), r['source_species'], fill='black')
                draw.text((x+4,y+335), 'PENDING: '+r['source_evidence']['tier'][:40], fill='black')
                draw.text((x+4,y+355), 'obs '+str(r['observation_id']), fill='black')
            sheet.save(out/'contact_sheets'/f'{prefix}_{start//12:03d}.jpg', quality=93)


def main(out):
    out = out.resolve(); out.mkdir(parents=True, exist_ok=False)
    for directory in ('images', 'source_responses', 'contact_sheets'):
        (out/directory).mkdir()
    old_obs, old_hashes, old_photos, old_visual = augmented_index(out)
    save(out/'PLAN.json', {'created_at':datetime.now(timezone.utc).isoformat(), 'species_caps':SPECIES,
        'license_gate':sorted(LICENSES), 'rate_limit':'Serial requests; >=1.1s delay; stop on 401/403/429',
        'negative_selection':'Categories derived from pre-existing training-only hard_negative_manifest, not test errors',
        'manual_authority':'All downloaded images remain pending. Source labels are not model predictions or human acceptance.',
        'positive_evidence':'Multiple exact independent identifiers preferred; sparse weak-source positive candidates separated for human confirmation.',
        'training_eligible':False, 'split':'quarantine', 'taxonomy_guide':GUIDE})
    all_rows=[]; audits=[]; summaries=[]; errors=[]; seen_hashes=set(); seen_photos=set(); seen_visual=[]
    for index, (species, cap) in enumerate(SPECIES):
        print('SOURCE', species, flush=True)
        try:
            taxa=query('taxa/autocomplete', {'q':species,'per_page':10})
            save(out/'source_responses'/f'{index:02d}_taxon.json', taxa)
            exact=[t for t in taxa.get('results',[]) if t.get('is_active') and t.get('name')==species]
            if len(exact)!=1:
                raise ValueError('Exact active taxon not uniquely resolved; no silent synonym mapping')
            taxon=exact[0]['id']; candidates=[]; examined_obs=set(); total=None
            for page in range(1,4):
                params={'taxon_id':taxon,'photos':'true','photo_license':','.join(sorted(LICENSES)),
                        'per_page':200,'page':page,'order_by':'id','order':'desc'}
                if species!='Aquilaria sinensis':
                    params['quality_grade']='research'
                response=query('observations',params)
                save(out/'source_responses'/f'{index:02d}_observations_page{page}.json',response)
                total=response.get('total_results',0)
                for o in response.get('results',[]):
                    if o['id'] in examined_obs:
                        continue
                    examined_obs.add(o['id']); ev=identifiers(o,taxon); reasons=[]
                    if (o.get('taxon') or {}).get('id')!=taxon: reasons.append('not_exact_taxon')
                    if str(o['id']) in old_obs: reasons.append('old_observation')
                    if o.get('flags') or ev['conflicting_current_identifications']: reasons.append('flags_or_taxon_conflict')
                    if species!='Aquilaria sinensis' and ev['tier']!='multiple_independent_exact_ids': reasons.append('insufficient_negative_source_evidence')
                    audits.append({'species':species,'observation_id':o['id'],'source_evidence':ev,'exclusions':reasons})
                    if reasons: continue
                    allowed=[p for p in o.get('photos',[]) if p.get('license_code') in LICENSES
                             and p['id'] not in old_photos and p['id'] not in seen_photos
                             and not p.get('flags') and not p.get('hidden')]
                    # Preserve two views for positives where available; one negative per observation.
                    for p in allowed[:2 if species=='Aquilaria sinensis' else 1]:
                        loc=o.get('location'); grid=None
                        if loc:
                            lat,lng=map(float,loc.split(',')); grid=f'{math.floor(lat*100)}:{math.floor(lng*100)}'
                        candidates.append({'id':'conserve_inat_'+str(p['id']), 'source':'iNaturalist',
                            'source_species':species,'source_label_binary':int(species=='Aquilaria sinensis'),
                            'source_url':o.get('uri'),'source_page':o.get('uri'),'source_evidence':ev,
                            'observation_id':o['id'],'inat_observation_id':o['id'],'observer_id':o['user']['id'],
                            'observed_on':o.get('observed_on'),'location_public':loc,'location_obscured':o.get('obscured'),
                            'place_guess':o.get('place_guess'),'location_grid_0p01deg':grid,
                            'group_id':'inat_observation_'+str(o['id']),'observer_group':'inat_observer_'+str(o['user']['id']),
                            'identity_status':'Source observation group; different observations may still depict the same tree',
                            'photo_id':p['id'],'photo_url':p['url'].replace('/square.','/large.'),
                            'license':p['license_code'],'photo_license_code':p['license_code'],
                            'attribution':p.get('attribution'),'photo_attribution':p.get('attribution'),
                            'review_priority':'high' if species=='Aquilaria sinensis' else 'normal',
                            'human_review_status':'pending','training_eligible':False,'split':'quarantine',
                            'organ_review':'pending; visible content has not been classified by a model'})
                        seen_photos.add(p['id'])
                if len(candidates)>=cap*2 or page*200>=total:
                    break
            candidates.sort(key=lambda r:(r['source_evidence']['tier']!='multiple_independent_exact_ids',r['observation_id']))
            # Spread observers and geographic grids, rather than fill from one photographer.
            bins={}
            for r in candidates:
                bins.setdefault(r['observer_id'],[]).append(r)
            balanced=[]
            while any(bins.values()):
                for values in bins.values():
                    if values: balanced.append(values.pop(0))
            rows=[]; clean_count=0; exclusions=Counter()
            for r in balanced:
                if clean_count>=cap: break
                time.sleep(1.1); r=download((r,out)); flags=[]
                if r['status']!='downloaded':
                    flags.append('download_failed')
                    if any(str(code) in r.get('error','') for code in (401,403,429)):
                        raise RuntimeError('Access denied / rate limited while downloading; stop this source')
                else:
                    r['source_image']=str((out/r['file']).resolve()); h=int(r['phash'],16)
                    if r['sha256'] in old_hashes: flags.append('existing_exact_image')
                    if r['sha256'] in seen_hashes: flags.append('new_exact_duplicate')
                    matches=[p for v,p in old_visual if (h^v).bit_count()<=6]
                    if matches: flags.append('existing_near_image')
                    if any((h^v).bit_count()<=6 for v in seen_visual): flags.append('new_near_duplicate')
                    if min(r['dimensions'])<224: flags.append('low_resolution')
                    if r['quality']['dark_fraction']>.65: flags.append('mostly_dark')
                    r['near_existing_matches']=matches[:20]
                    seen_hashes.add(r['sha256']); seen_visual.append(h)
                r['quality_flags']=flags; r['candidate_lane']='excluded_duplicate_quality' if flags else 'pending_human_review'
                exclusions.update(flags); rows.append(r)
                if not flags: clean_count+=1
                all_rows.append(r)
                save(out/'manifest.json',all_rows)
                save(out/'review_manifest.json',[x for x in all_rows if x['status']=='downloaded' and not x['quality_flags']])
            summary={'species':species,'source_total_observations':total,'queried_observations':len(examined_obs),
                'downloads':sum(r['status']=='downloaded' for r in rows),'pending_clean':clean_count,
                'strong_evidence_pending':sum(not r['quality_flags'] and r['source_evidence']['tier']=='multiple_independent_exact_ids' for r in rows),
                'weak_source_pending':sum(not r['quality_flags'] and r['source_evidence']['tier']!='multiple_independent_exact_ids' for r in rows),
                'quality_exclusions':dict(exclusions), 'distinct_observers':len({r['observer_id'] for r in rows if not r['quality_flags']})}
            summaries.append(summary); print(json.dumps(summary),flush=True)
        except Exception as exc:
            errors.append({'species':species,'error':str(exc)}); print('ERROR',str(exc),flush=True)
            if isinstance(exc,requests.HTTPError) and exc.response.status_code in (401,403,429):
                break
        save(out/'source_audit.json',audits)
        save(out/'collection_summary.json',{'by_species':summaries,'errors':errors,'training_performed':False,'deployed_files_changed':False})
    clean=[r for r in all_rows if r['status']=='downloaded' and not r['quality_flags']]
    contact_sheets(clean,out)
    totals={'downloaded':sum(r['status']=='downloaded' for r in all_rows),'pending_clean':len(clean),
        'positive_pending':sum(r['source_species']=='Aquilaria sinensis' for r in clean),
        'negative_pending':sum(r['source_species']!='Aquilaria sinensis' for r in clean),
        'human_accepted':0,'training_eligible':0,'by_species':summaries,'errors':errors}
    save(out/'totals.json',totals)
    print('COMPLETE',json.dumps(totals),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
