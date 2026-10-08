"""Licensed, unseen-observer external-check candidates; NEVER training data."""
import json
import re
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import requests
from collect_conservation_sources_20260917 import query,identifiers
from collect_sstv1_incremental import ROOT,LICENSES,download,save,old_index

OUT=ROOT/'external_datasets/aq_ar_external_check_20260919_v2'
REG=ROOT/'outputs/artocarpus_fresh_split_20260917_v1/refined_v2/history_registry.json'
# Negatives fixed from previous TRAIN-only top160 focal-confusion counts.
TARGETS=[('Aquilaria sinensis',20),('Artocarpus hypargyreus',20),
         ('Diospyros morrisiana',10),('Syzygium levinei',10),
         ('Sterculia lanceolata',10),('Artocarpus heterophyllus',10)]


def walk(v):
    if isinstance(v,list):
        for x in v:yield from walk(x)
    elif isinstance(v,dict):
        yield v
        for x in v.values():
            if isinstance(x,(list,dict)):yield from walk(x)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for s in ('images','source_responses'):(OUT/s).mkdir()
    obs,hashes,photos,visual=old_index()
    registry=json.loads(REG.read_text('utf-8'));observers=set()
    for r in registry['records']:
        for token in r['tokens']['observer']:
            m=re.fullmatch(r'inat:id:(\d+)',token)
            if m:observers.add(int(m[1]))
    # Refresh later acquisitions/user-photo provenance without accessing predictions.
    for file in (ROOT/'external_datasets').rglob('*manifest*.json'):
        if OUT in file.parents:continue
        try:
            for r in walk(json.loads(file.read_text('utf-8-sig'))):
                if str(r.get('observer_id','')).isdigit():observers.add(int(r['observer_id']))
                for key in ('observation_id','inat_observation_id'):
                    if str(r.get(key,'')).isdigit():obs.add(str(r[key]))
                for key in ('photo_id','inat_photo_id'):
                    if str(r.get(key,'')).isdigit():photos.add(int(r[key]))
                for key in ('sha256','source_sha256'):
                    value=r.get(key)
                    if isinstance(value,str) and re.fullmatch('[a-fA-F0-9]{64}',value):hashes.add(value)
                value=r.get('phash')
                if isinstance(value,str) and re.fullmatch('[a-fA-F0-9]{16}',value):
                    visual.append((int(value,16),str(r.get('source_image') or file)))
        except (ValueError,UnicodeError):pass
    for r in registry['records']:
        for token in r['tokens']['observation']:
            if token.startswith('inat:'):obs.add(token.split(':')[-1])
        for token in r['tokens']['photo']:
            if token.startswith('inat:'):photos.add(int(token.split(':')[-1]))
        hashes.update(r['tokens']['sha256'])
        for h in r.get('phashes',[]):
            try:visual.append((int(h,16),'historical_registry'))
            except ValueError:pass
    save(OUT/'PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),targets=TARGETS,
        licenses=sorted(LICENSES),pages_per_species=3,photos_per_observation=1,
        source_gate='Research grade; exact active taxon; multiple current independent IDs, no species disagreement; new observer/observation/photo/SHA and pHash distance>4',
        label_authority='Community identifications; not guaranteed expert truth. Visual/user review before acceptance.',
        negative_choice='Only training focal-probability ranking, not regression errors.',
        historical_observers_excluded=len(observers),role='External-check quarantine ONLY; no training or validation selection',
        api_policy='Serial >=1.1s; max18 observation pages; stop source on401/403/429/503; no access-control bypass',
        rights='Per-photo CC0/CC-BY/CC-BY-SA; keep original bytes and attribution; not published in Demo',
        sources=['https://www.inaturalist.org/pages/api+recommended+practices',
                 'https://help.inaturalist.org/en/support/solutions/articles/151000169936',
                 'https://help.inaturalist.org/en/support/solutions/articles/151000169918']))
    rows=[];audits=[];errors=[];selected_observers=set();seenhash=set();seenvisual=[];stop=False
    for species,cap in TARGETS:
        selected=0;counts=Counter()
        try:
            taxa=query('taxa/autocomplete',dict(q=species,per_page=5))
            save(OUT/'source_responses'/f'{species}_taxon.json',taxa)
            exact=[t for t in taxa['results'] if t['name']==species and t.get('is_active')]
            assert len(exact)==1
            tid=exact[0]['id']
            for page in range(1,4):
                data=query('observations',dict(taxon_id=tid,quality_grade='research',photos='true',
                    photo_license=','.join(sorted(LICENSES)),order_by='id',order='desc',per_page=200,page=page))
                save(OUT/'source_responses'/f'{species}_{page}.json',data)
                for o in data['results']:
                    ev=identifiers(o,tid);reasons=[]
                    if o.get('quality_grade')!='research' or (o.get('taxon') or {}).get('id')!=tid:reasons.append('not_exact_research_species')
                    if ev['tier']!='multiple_independent_exact_ids' or o.get('flags'):reasons.append('insufficient_source_agreement')
                    if str(o['id']) in obs:reasons.append('historical_observation')
                    if o['user']['id'] in observers:reasons.append('historical_observer')
                    if o['user']['id'] in selected_observers:reasons.append('observer_already_in_external_batch')
                    ps=[p for p in o.get('photos',[]) if p.get('license_code') in LICENSES and p['id'] not in photos
                        and not p.get('flags') and not p.get('hidden')]
                    if not ps:reasons.append('no_new_licensed_photo')
                    audits.append(dict(species=species,observation_id=o['id'],observer_id=o['user']['id'],evidence=ev,exclusions=reasons))
                    counts.update(reasons)
                    if reasons or selected>=cap:continue
                    p=ps[0]
                    row=dict(id='aq_ar_external_'+str(p['id']),source='iNaturalist',source_species=species,
                        source_page=o.get('uri'),source_url=o.get('uri'),source_taxon_id=tid,source_evidence=ev,
                        observation_id=o['id'],inat_observation_id=o['id'],observer_id=o['user']['id'],
                        group_id='inat_observation_'+str(o['id']),observer_group='inat_observer_'+str(o['user']['id']),
                        photo_id=p['id'],photo_url=p['url'].replace('/square.','/large.'),
                        photo_license_code=p['license_code'],photo_attribution=p.get('attribution'),
                        observed_on=o.get('observed_on'),place_guess=o.get('place_guess'),
                        label_authority='Multiple independent source community identifications, not model labels or expert certification',
                        training_eligible=False,split='external_check_quarantine',human_review_status='not_requested',
                        agent_content_review='pending',identity_warning='Unseen observer and observation do not prove a different individual tree')
                    time.sleep(1.1);row=download((row,OUT))
                    flags=[]
                    if row['status']=='downloaded':
                        row['source_image']=str((OUT/row['file']).resolve())
                        h=int(row['phash'],16)
                        if row['sha256'] in hashes or row['sha256'] in seenhash:flags.append('exact_duplicate')
                        near=[str(path) for hv,path in visual if (h^hv).bit_count()<=4]
                        if near:flags.append('near_historical_duplicate')
                        if any((h^hv).bit_count()<=4 for hv in seenvisual):flags.append('near_batch_duplicate')
                        if min(row['dimensions'])<224:flags.append('small_image')
                        if row['quality']['dark_fraction']>.65:flags.append('mostly_dark')
                        row['near_matches']=near[:10]
                        seenhash.add(row['sha256']);seenvisual.append(h)
                        selected_observers.add(o['user']['id']);photos.add(p['id'])
                        if not flags:selected+=1
                    else:
                        flags.append('download_failed')
                        if any(str(c) in row.get('error','') for c in (401,403,429,503)):stop=True
                    row['quality_flags']=flags;rows.append(row)
                    save(OUT/'manifest.json',rows)
                    print('PHOTO',species,row['id'],row['status'],flags,flush=True)
                    if stop:break
                if stop or selected>=cap or page*200>=data.get('total_results',0):break
            print('TAXON',species,'retained_candidates',selected,'exclusions',dict(counts),flush=True)
        except Exception as e:
            errors.append(dict(species=species,error=str(e)))
            print('ERROR',errors[-1],flush=True)
            if isinstance(e,requests.HTTPError) and e.response.status_code in (401,403,429,503):stop=True
        save(OUT/'source_audit.json',audits);save(OUT/'errors.json',errors)
        if stop:break
    save(OUT/'manifest.json',rows)
    save(OUT/'SUMMARY.json',dict(downloaded=sum(r['status']=='downloaded' for r in rows),
        clean_candidates=dict(Counter(r['source_species'] for r in rows if r['status']=='downloaded' and not r['quality_flags'])),
        errors=errors,stopped=stop,training_used=0,predictions_used_for_filtering=False,
        note='All candidates still require visual review; no human review decision fabricated.'))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
