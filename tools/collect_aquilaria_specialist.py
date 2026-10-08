"""Conservative source-evidence gate; agent review is NOT human acceptance."""
import argparse,json,re,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw,ImageOps
from collect_sstv1_incremental import get_json,download,old_index,save,ROOT,LICENSES

TAXON=189373
def main(out):
    out.mkdir(parents=True,exist_ok=False)
    for d in ('images','source_responses','contact_sheets'):(out/d).mkdir()
    obs,hashes,photos,visual=old_index()
    # Include the prior incremental collection, including rejected/pending images.
    prior=ROOT/'external_datasets/sstv1_incremental_20260914_v1/manifest.json'
    for r in json.loads(prior.read_text(encoding='utf8')):
        obs.add(str(r.get('inat_observation_id')));photos.add(r.get('photo_id'));hashes.add(r.get('sha256'))
        if r.get('phash'):visual.append((int(r['phash'],16),r.get('file')))
    query=get_json('observations',{'taxon_id':TAXON,'photos':'true','photo_license':','.join(sorted(LICENSES)),'per_page':200,'order_by':'id','order':'desc'})
    save(out/'source_responses/inat_observations.json',query)
    if query.get('total_results',0)>200:raise RuntimeError('Pagination required; do not silently truncate')
    rows=[];audit=[]
    for o in query['results']:
        ids=[i for i in o.get('identifications',[]) if i.get('current') and not i.get('hidden') and not i.get('spam')]
        exact={i['user']['id'] for i in ids if i.get('taxon_id')==TAXON or i.get('taxon',{}).get('id')==TAXON}
        conflict=[i['id'] for i in ids if i.get('taxon',{}).get('rank_level',100)<=10 and i.get('taxon',{}).get('id')!=TAXON]
        reasons=[]
        if o.get('taxon',{}).get('id')!=TAXON:reasons.append('wrong_exact_taxon')
        if len(exact)<2 or not (exact-{o['user']['id']}):reasons.append('insufficient_independent_exact_ids')
        if conflict or o.get('flags'):reasons.append('current_taxon_conflict_or_flags')
        if str(o['id']) in obs:reasons.append('previous_observation_excluded')
        ev={'observation':o['id'],'exact_identifiers':sorted(exact),'conflicting_current_ids':conflict,'quality_grade':o.get('quality_grade'),'reasons':reasons}
        audit.append(ev)
        if reasons:continue
        for p in o['photos']:
            if p.get('license_code') not in LICENSES or p['id'] in photos or p.get('flags') or p.get('hidden'):continue
            rows.append({'id':'aq_inat_'+str(p['id']),'source_species':'Aquilaria sinensis','source_page':o['uri'],
                'photo_url':p['url'].replace('/square.','/large.'),'photo_id':p['id'],'photo_license_code':p['license_code'],
                'photo_attribution':p.get('attribution'),'inat_observation_id':o['id'],'observer_id':o['user']['id'],
                'observed_on':o.get('observed_on'),'source_evidence':ev,'source_group':'inat_observation_'+str(o['id']),
                'training_eligible':False,'review_authority':'agent_review_pending_not_human','split':'quarantine'})
    print('SOURCE_GATE',len(query['results']),'observations;',len(rows),'new photo candidates',flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(download,[(r,out) for r in rows]))
    seen=[]
    for r in rows:
        flags=[]
        if r['status']!='downloaded':flags=['download_failed']
        else:
            h=int(r['phash'],16)
            if r['sha256'] in hashes:flags.append('known_exact_image')
            if any((h^v).bit_count()<=4 for v,p in visual):flags.append('known_near_image')
            if any((h^v).bit_count()<=4 for v in seen):flags.append('new_near_image')
            if min(r['dimensions'])<224:flags.append('low_resolution')
            if r['quality']['dark_fraction']>.65:flags.append('mostly_dark')
            seen.append(h)
        r['quality_flags']=flags
    save(out/'manifest.json',rows);save(out/'source_audit.json',audit)
    clean=[r for r in rows if not r['quality_flags']]
    for start in range(0,len(clean),12):
        sheet=Image.new('RGB',(1440,1080),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(clean[start:start+12]):
            im=ImageOps.exif_transpose(Image.open(out/r['file'])).convert('RGB');im.thumbnail((350,290))
            x=(j%4)*360;y=(j//4)*360;sheet.paste(im,(x+(350-im.width)//2,y))
            draw.text((x+3,y+297),r['id'],fill='black');draw.text((x+3,y+319),'obs '+str(r['inat_observation_id']),fill='black')
        sheet.save(out/'contact_sheets'/f'{start//12:03d}.jpg',quality=94)
    save(out/'collection_summary.json',{'query_observations':len(query['results']),'new_downloads':sum(r['status']=='downloaded' for r in rows),'ready_for_agent_visual_review':len(clean),'source_exclusions':dict(Counter(x for r in audit for x in r['reasons'])),'quality_exclusions':dict(Counter(x for r in rows for x in r['quality_flags'])),'human_review_overridden':False,'fitting_performed':False})
    print(json.dumps(json.loads((out/'collection_summary.json').read_text()),ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
