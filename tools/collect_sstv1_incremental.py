"""New, licensed source-labelled photos for SSTv1; never fit pending images.

Preserves official taxonomy, original downloads, source observation identity,
review decisions, and existing fixed split boundaries. No model is modified.
"""
import argparse
import hashlib
import io
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
PAGE = 'https://www.greening.gov.hk/tc/resource-centre/plant-species-knowledge/common-tree-in-urban-environment-of-hong-kong/index.html'
SPECIES = [
 ('Araucaria columnaris','柱狀南洋杉'), ('Archontophoenix alexandrae','假檳榔'),
 ('Bischofia javanica','秋楓'), ('Callistemon viminalis','串錢柳'),
 ('Casuarina equisetifolia','木麻黃'), ('Cinnamomum burmannii','陰香'),
 ('Cinnamomum camphora','樟'), ('Elaeocarpus hainanensis','水石榕'),
 ('Ficus benjamina','垂葉榕'), ('Ficus microcarpa','細葉榕'),
 ('Grevillea robusta','銀樺'), ('Jacaranda mimosifolia','藍花楹'),
 ('Lagerstroemia speciosa','大花紫薇'), ('Livistona chinensis','蒲葵'),
 ('Magnolia grandiflora','荷花玉蘭'), ('Mallotus paniculatus','白楸'),
 ('Melaleuca cajuputi subsp. cumingiana','白千層'), ('Michelia chapensis','樂昌含笑'),
 ('Roystonea regia','大王椰子'), ('Schefflera heptaphylla','鵝掌柴'),
 ('Spathodea campanulata','火焰樹'), ('Sterculia lanceolata','假蘋婆'),
 ('Tabebuia chrysantha','黃鐘木'), ('Tabebuia rosea','紅花風鈴木'),
 ('Terminalia catappa','欖仁'), ('Terminalia mantaly','小葉欖仁'),
 ('Yulania x soulangeana','二喬木蘭')]
LICENSES = {'cc0','cc-by','cc-by-sa'}
UA = {'User-Agent':'SpaceScanTrees-SSTv1/1.0 (local botanical research; limited public API collection)'}

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def norm(s):
    return re.sub(r'\s+', ' ', s.replace('×','x').replace(' subsp. ',' ')).strip().lower()

def get_json(endpoint, params):
    time.sleep(.8)
    response = requests.get('https://api.inaturalist.org/v1/'+endpoint, params=params, headers=UA, timeout=30)
    response.raise_for_status()  # Do not circumvent denial or rate limiting.
    return response.json()

def phash(im):
    values = dctn(np.asarray(im.convert('L').resize((32,32)),dtype=float),norm='ortho')[:8,:8].flatten()
    return int(''.join('1' if v else '0' for v in values>np.median(values[1:])),2)

def old_index():
    observations, hashes, photos, visual = set(), set(), set(), []
    source = ROOT/'outputs/distillation/20260908_0045'
    for phase in ('train','planning','selection','test','field','case'):
        for row in json.loads((source/f'{phase}_rows.json').read_text(encoding='utf-8')):
            group = row.get('group_id','')
            if 'observation:' in group: observations.add(group.split('observation:')[-1])
            if row.get('source_sha256'): hashes.add(row['source_sha256'])
    index = ROOT/'external_datasets/herbarium_training_candidates_20260907/existing_image_index.jsonl'
    for line in index.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r.get('sha256'): hashes.add(r['sha256'])
        if r.get('phash'): visual.append((int(r['phash'],16),r['path']))
        match = re.search(r'inat_photo_(\d+)',r.get('path',''))
        if match: photos.add(int(match[1]))
    return observations, hashes, photos, visual

def download(task):
    row, out = task
    try:
        response = requests.get(row['photo_url'],headers=UA,timeout=25)
        response.raise_for_status()
        raw = response.content
        if len(raw)>25_000_000: raise ValueError('Image exceeds 25 MB limit')
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            rgb = ImageOps.exif_transpose(im).convert('RGB')
        gray = np.asarray(rgb.convert('L'))
        row.update(sha256=hashlib.sha256(raw).hexdigest(),dimensions=list(rgb.size),
            bytes=len(raw),phash=format(phash(rgb),'016x'),
            quality={'dark_fraction':float(np.mean(gray<25)), 'bright_fraction':float(np.mean(gray>245))})
        path = out/'images'/f"{row['id']}.jpg"
        if path.exists(): raise FileExistsError(path)
        path.write_bytes(raw)  # Original licensed bytes, no watermark removal.
        row.update(file=path.relative_to(out).as_posix(),status='downloaded')
    except Exception as exc:
        row.update(status='failed',error=str(exc))
    return row

def main(out, cap, only=None):
    out.mkdir(parents=True,exist_ok=False)
    (out/'images').mkdir(); (out/'source_responses').mkdir(); (out/'contact_sheets').mkdir()
    labels=json.loads((ROOT/'outputs/distillation/20260908_0045/classes.json').read_text(encoding='utf-8'))['labels']
    catalog=[]
    for i,(name,zh) in enumerate(SPECIES,1):
        exact = labels.index(name) if name in labels else None
        catalog.append({'index':i,'scientific_name':name,'traditional_chinese':zh,'official_page':PAGE,
            'existing_exact_label_index':exact, 'label_action':'existing_exact_label' if exact is not None else 'taxonomy_review_before_training'})
    save(out/'official_catalog.json',catalog)
    save(out/'SSTv1_plan.json',{'name':'SSTv1','initialization':'Current BioCLIP 2.5 + three adapted MLP heads, NOT random restart',
        'created_at':datetime.now(timezone.utc).isoformat(),'official_page':PAGE,'catalog_entries':27,
        'collection_cap_per_entry':cap,'licenses':sorted(LICENSES),'review_required':True,
        'training':'Continue existing heads with accepted new images plus old-class replay. Preserve label/source identity; resolve unmapped taxa first.',
        'selection':'Fixed existing validation; previous test/VMMS images are regression diagnostics only.',
        'safety':'Physical SHA-256 verified initialization/deployed-weight backups before EVERY fit. New run directories. No automatic deployment.',
        'note':'Research-grade community IDs are evidence, not botanical expert certification. Page English names contain spelling errors corrected against scientific taxa; Tabebuia rosea is NOT Tabebuia impetiginosa.'})
    # Save the source page for provenance; no official photographs are downloaded.
    for lang in ('tc','en'):
        try:
            response=requests.get(PAGE.replace('/tc/','/'+lang+'/'),headers=UA,timeout=25)
            response.raise_for_status()
            (out/'source_responses'/f'official_{lang}.html').write_bytes(response.content)
        except Exception as exc: save(out/'source_responses'/f'official_{lang}_error.json',{'error':str(exc)})
    old_obs, old_hashes, old_photos, old_visual = old_index()
    all_rows, summary, errors, seen_ids, seen_hashes, seen_visual = [], [], [], set(), set(), []
    for item in catalog:
        name, zh = item['scientific_name'], item['traditional_chinese']
        if only and name != only: continue
        print(f"COLLECT {item['index']}/27 {name}",flush=True)
        try:
            query=get_json('taxa/autocomplete',{'q':name.replace(' subsp. ',' '),'per_page':10})
            save(out/'source_responses'/f"{item['index']:02d}_taxa.json",query)
            hits=[t for t in query.get('results',[]) if t.get('is_active') and
                (norm(t['name'])==norm(name) or norm(t.get('matched_term',''))==norm(name))]
            if len(hits)!=1: raise ValueError(f'Exact name/synonym match is not unique: {len(hits)}')
            taxon=hits[0]; item['inat_taxon_id']=taxon['id']; item['inat_accepted_name']=taxon['name']
            selected=[]; groups=set(); skips=Counter()
            for region in ('hong_kong','global'):
                if len(selected)>=cap: break
                params={'taxon_id':taxon['id'],'quality_grade':'research','photos':'true',
                    'photo_license':','.join(sorted(LICENSES)), 'per_page':200,'order_by':'created_at','order':'desc'}
                if region=='hong_kong': params.update(lat=22.32,lng=114.17,radius=45)
                data=get_json('observations',params)
                save(out/'source_responses'/f"{item['index']:02d}_{region}.json",data)
                for obs in data.get('results',[]):
                    if len(selected)>=cap: break
                    oid=str(obs['id'])
                    if oid in old_obs: skips['old_observation']+=1; continue
                    if oid in groups: continue
                    if (obs.get('taxon') or {}).get('id')!=taxon['id']:
                        skips['different_or_descendant_taxon']+=1; continue
                    choices=[p for p in obs.get('photos',[]) if p.get('license_code') in LICENSES
                        and p['id'] not in old_photos and p['id'] not in seen_ids]
                    if not choices: skips['no_new_licensed_photo']+=1; continue
                    p=choices[0]; seen_ids.add(p['id']); groups.add(oid)
                    selected.append({'id':'sstv1_inat_'+str(p['id']), 'source_species':name,'source_names':[zh],
                        'source_group':'inat_observation_'+oid,'inat_observation_id':obs['id'],
                        'inat_taxon_id':taxon['id'],'inat_taxon_name':taxon['name'],
                        'inat_quality_grade':obs.get('quality_grade'),'inat_observed_on':obs.get('observed_on'),
                        'source_page':obs.get('uri'),'photo_id':p['id'],'photo_license_code':p['license_code'],
                        'photo_attribution':p.get('attribution'),'photo_url':p['url'].replace('/square.','/large.'),
                        'place_guess':obs.get('place_guess'),'region_query':region,
                        'training_eligible':False,'split':'quarantine','organ_review':{'status':'pending','organ_tags':[]},
                        'existing_exact_label_index':item['existing_exact_label_index']})
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows=list(pool.map(download,[(r,out) for r in selected]))
            for r in rows:
                if r['status']!='downloaded': continue
                flags=[]; h=int(r['phash'],16)
                if r['sha256'] in old_hashes or r['sha256'] in seen_hashes: flags.append('exact_duplicate')
                near=[p for v,p in old_visual if (h^v).bit_count()<=4]
                if near: flags.append('near_existing_image')
                if any((h^v).bit_count()<=4 for v in seen_visual): flags.append('near_new_image')
                if min(r['dimensions'])<224: flags.append('low_resolution')
                if r['quality']['dark_fraction']>.65: flags.append('mostly_dark')
                r.update(exact_duplicate_matches=['known_image'] if 'exact_duplicate' in flags else [],
                    near_existing_matches=near[:10],quality_flags=flags,
                    candidate_lane='excluded_pending_duplicate_quality_check' if flags else 'Species and tree/crown framing review / 樹種與全株構圖核驗')
                seen_hashes.add(r['sha256']); seen_visual.append(h)
            all_rows.extend(rows)
            summary.append({'species':name,'zh':zh,'resolved_taxon':taxon['name'],'downloaded':sum(r['status']=='downloaded' for r in rows),
                'clean_pending_review':sum(r['status']=='downloaded' and not r.get('quality_flags') for r in rows),'skips':dict(skips)})
        except Exception as exc:
            errors.append({'species':name,'error':str(exc)})
            print('ERROR',name,str(exc),flush=True)
            if isinstance(exc,requests.HTTPError) and exc.response.status_code in (401,403,429):
                save(out/'access_stop.json',errors[-1]); break
        save(out/'official_catalog.json',catalog); save(out/'manifest.json',all_rows)
        save(out/'curated_manifest.json',[r for r in all_rows if r['status']=='downloaded' and not r.get('quality_flags')])
        save(out/'collection_summary.json',{'by_species':summary,'errors':errors,'training_performed':False,'deployed_files_changed':False})
        print('PROGRESS',sum(s['downloaded'] for s in summary),'new downloads',flush=True)
    clean=[r for r in all_rows if r['status']=='downloaded' and not r.get('quality_flags')]
    for start in range(0,len(clean),24):
        subset=clean[start:start+24]; sheet=Image.new('RGB',(1440,1440),'white'); draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(subset):
            im=Image.open(out/r['file']).convert('RGB'); im.thumbnail((230,300))
            x=(j%6)*240; y=(j//6)*360; sheet.paste(im,(x+(230-im.width)//2,y))
            draw.text((x+3,y+305),f"{start+j+1}: {r['source_species'][:27]}",fill='black')
            draw.text((x+3,y+324),r['id'].replace('sstv1_',''),fill='black')
        sheet.save(out/'contact_sheets'/f'batch_{start//24+1:03d}.jpg',quality=90)
    html=(ROOT/'tree_species_demo/static/herbarium-review.html').read_text(encoding='utf-8')
    html=html.replace('Herbarium image review','SSTv1 image review').replace('植物標本室影像驗收','SSTv1 新增影像驗收')
    foot=html.find('<p class="foot">')
    if foot>=0:
        end=html.find('</p>',foot)+4
        html=html[:foot]+'<p class="foot">SSTv1 · Newly collected iNaturalist photographs. Source labels require review. CC0 / CC BY / CC BY-SA; attribution preserved. 新增網絡圖片，來源標籤需核驗；保留作者及授權。</p>'+html[end:]
    (out/'review.html').write_text(html,encoding='utf-8')
    totals={'downloaded':sum(s['downloaded'] for s in summary),'clean_pending_review':len(clean),
        'entries_with_images':sum(s['downloaded']>0 for s in summary),'catalog_entries':27,
        'errors':errors,'training_performed':False,'deployed_files_changed':False}
    save(out/'totals.json',totals)
    lines=['# SSTv1 — incremental data collection','','Base: existing BioCLIP 2.5 + adapted MLP ensemble. No fitting or deployment yet.',
        '',f"New downloaded originals: {totals['downloaded']}; clean candidates awaiting review: {len(clean)}.",
        '','Old observation IDs / photo IDs / exact hashes are excluded; near-image matches are quarantined. New photos are not automatically accepted.',
        '','## Taxonomy gates','',
        '- Keep source Ficus benjamina distinct from Ficus microcarpa; do not inherit the Demo display rename as botanical ground truth.',
        '- Araucaria old mixed label and Melaleuca subspecies need an explicit mapping decision.',
        '- Michelia chapensis, Tabebuia rosea and Yulania x soulangeana are not exact existing output labels. Do not relabel them as related species.',
        '', '## Collection counts','', '| Source taxon | New originals | Clean, pending |','|---|---:|---:|']
    lines.extend(f"| {s['species']} | {s['downloaded']} | {s['clean_pending_review']} |" for s in summary)
    lines.extend(['','After acceptance: verify physical backups; continue existing adapted heads with new accepted data and old-class replay; compare on fixed validation; retain deployed Demo.'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(totals),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True); parser.add_argument('--cap',type=int,default=40)
    parser.add_argument('--only',choices=[s for s,z in SPECIES])
    args=parser.parse_args(); main(args.out,args.cap,args.only)
