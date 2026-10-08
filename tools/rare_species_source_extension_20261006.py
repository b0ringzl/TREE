"""Licensed NC scientific-evaluation extension and Europe PMC official OA API.

Does not change training eligibility, human decisions, or model files.
"""
import argparse
import hashlib
import io
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import fresh_bioclip146_benchmark_20261006 as base
from training_safety import verify_backup, sha256

TARGETS = ('Aquilaria sinensis', 'Artocarpus hypargyreus')
LICENSES = {'cc0', 'cc-by', 'cc-by-sa', 'cc-by-nc', 'cc-by-nc-sa'}
PAPERS = ('PMC7050300', 'PMC10935121', 'PMC8048387', 'PMC11945267')
SEED = 2026100607
XLINK = '{http://www.w3.org/1999/xlink}href'


def nc_collect(out):
    verify_backup(out / 'pre_training_backup')
    if (out / 'nc_manifest.json').exists(): raise FileExistsError('Completed collection is immutable')
    base.save(out / 'NC_EXTENSION_PLAN.json', {
        'reason': 'CC0/BY/BY-SA exact research-grade pools exhausted by historical project collection',
        'license_extension': sorted(LICENSES), 'noncommercial_only': True,
        'excluded_license': 'All-rights-reserved and ND excluded; no adaptation/redistribution ambiguity',
        'purpose': 'Private noncommercial scientific evaluation, NOT training or redistributed demo examples',
        'selection': 'All exact labelled research-grade observations through 2026-10-06, >=2 current independent exact IDs; up to3 photos/observation and100 observations/species',
        'quality': 'Same predeclared native224/dedup/lighting rules; no prediction-based selection',
        'locked_before_predictions': True,
    })
    old_obs, old_photos, old_hashes, old_visual = base.prior_index(out)
    api = base.Api(); rows = []; exclusions = []; summaries = []
    hashes, visuals = set(), {}
    rng = random.Random(SEED)
    for label in TARGETS:
        data = base.load(out / 'source_responses' / (label.replace(' ', '_') + '_taxon.json'))
        taxon, mapping = base.resolve_taxon(data, label)
        pool = []; total = 0
        for page in range(1, 11):
            data = api.get('observations', {'taxon_id': taxon['id'], 'photos':'true', 'quality_grade':'research',
                'photo_license':','.join(sorted(LICENSES)), 'd2':'2026-10-06', 'per_page':100, 'page':page,
                'order_by':'id', 'order':'desc'})
            base.save(out / 'source_responses' / f"{taxon['id']}_nc_page{page}.json", data)
            total = data.get('total_results',0); pool += data.get('results',[])
            if len(pool) >= total or not data.get('results'): break
        if len(pool) < total: raise RuntimeError('Unexamined pagination would silently bias sample')
        candidates = []; reasons = Counter()
        for obs in pool:
            ev = base.identifiers(obs, taxon['id']); why = []
            if obs.get('taxon',{}).get('id') != taxon['id']: why.append('not_exact_taxon')
            if obs.get('flags') or ev['tier'] != 'multiple_independent_exact_ids': why.append('weak_current_identification')
            if str(obs['id']) in old_obs: why.append('historical_observation')
            photos = [p for p in obs.get('photos',[]) if p.get('license_code') in LICENSES
                      and str(p['id']) not in old_photos and not p.get('hidden') and not p.get('flags')]
            if not photos: why.append('no_new_licensed_photo')
            if why:
                exclusions.append({'label':label,'observation_id':obs['id'],'reasons':why}); reasons.update(why)
            else: candidates.append((obs,photos,ev))
        rng.shuffle(candidates); groups = 0
        for obs,photos,ev in candidates:
            if groups >= 100: break
            n = 0
            for photo in photos:
                if n >= 3: break
                url=photo['url'].replace('/square.','/large.')
                response=requests.get(url,headers=base.UA,timeout=35)
                if response.status_code in (401,403,429,503): raise base.Denied('Media access stopped, no bypass')
                response.raise_for_status(); raw=response.content
                with Image.open(io.BytesIO(raw)) as src: im=ImageOps.exif_transpose(src).convert('RGB')
                digest=hashlib.sha256(raw).hexdigest(); visual=base.phash(im); gray=np.asarray(im.convert('L'))
                old=base.nearest_hash(visual,old_visual); new=base.nearest_hash(visual,visuals); why=[]
                if digest in old_hashes or digest in hashes: why.append('exact_duplicate')
                if old: why.append('near_historical_image')
                if new: why.append('near_new_image')
                if min(im.size)<224: why.append('low_resolution')
                if len(raw)>20_000_000 or im.width*im.height>40_000_000: why.append('oversized')
                if (gray<15).mean()>.85 or (gray>245).mean()>.85: why.append('extreme_lighting')
                if why:
                    exclusions.append({'label':label,'observation_id':obs['id'],'photo_id':photo['id'],
                        'reasons':why,'near_old':old,'near_new':new}); reasons.update(why); continue
                path=out/'images'/f"inat_{photo['id']}.jpg"
                if path.exists(): raise FileExistsError(path)
                path.write_bytes(raw)
                rows.append({'id':f"inat_{photo['id']}",'label':label,'source_species':label,
                    'taxon_id':taxon['id'],'taxonomy_mapping':mapping,'observation_id':obs['id'],
                    'photo_id':photo['id'],'observer_id':obs['user']['id'],
                    'source_group':f"inat_observation_{obs['id']}",'source_kind':'community_observation',
                    'observed_on':obs.get('observed_on'),'source_created_at':obs.get('created_at'),
                    'source_page':obs['uri'],'photo_url':url,'source_evidence':ev,
                    'license':photo['license_code'],'attribution':photo.get('attribution'),
                    'noncommercial_only':'nc' in photo['license_code'],'hong_kong':7613 in obs.get('place_ids',[]),
                    'recent_after_base_project':(obs.get('observed_on') or '')>='2026-09-08',
                    'region_query':'global','file':path.relative_to(out).as_posix(),'sha256':digest,
                    'phash':f'{visual:016x}','dimensions':list(im.size),'bytes':len(raw),
                    'training_eligible':False,'human_review_status':'source_labelled_not_expert_confirmed',
                    'split':'prospective_evaluation'})
                hashes.add(digest); visuals[visual]=str(path); n+=1
                base.save(out/'nc_manifest_partial.json',rows)
                print('PHOTO',label,len(rows),obs['id'],photo['id'],flush=True); time.sleep(.3)
            groups+=int(n>0)
        summaries.append({'label':label,'queried_observations':len(pool),'candidate_observations':len(candidates),
            'accepted_observations':groups,'photos':sum(r['label']==label for r in rows),'exclusions':dict(reasons)})
        base.save(out/'nc_collection_summary.json',summaries); base.save(out/'nc_exclusions.json',exclusions)
    base.save(out/'nc_manifest.json',rows); base.sheets(out,rows)
    print('NC COLLECTION COMPLETE',summaries,flush=True)


def papers(out):
    verify_backup(out/'pre_training_backup')
    root=out/'paper_originals'; root.mkdir(exist_ok=True)
    session=requests.Session(); session.headers.update(base.UA); summaries=[]
    for pmcid in PAPERS:
        dest=root/pmcid; dest.mkdir(exist_ok=False)
        url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
        r=session.get(url,timeout=40)
        if r.status_code in (401,403,429,503): raise base.Denied('Europe PMC API denial; no bypass')
        r.raise_for_status(); (dest/'article.xml').write_bytes(r.content)
        xml=ET.fromstring(r.content)
        license_text=' '.join(''.join(x.itertext()) for x in xml.findall('.//license'))
        metadata={'pmcid':pmcid,'source_page':f'https://europepmc.org/articles/{pmcid}',
            'xml_api':url,'title':''.join(xml.find('.//article-title').itertext()),
            'doi':[x.text for x in xml.findall('.//article-id') if x.get('pub-id-type')=='doi'],
            'license':license_text,'figures':[],'training_eligible':False,
            'rights':'Private noncommercial scientific evaluation only; figure attribution retained'}
        for fig in xml.findall('.//fig'):
            metadata['figures'].append({'id':fig.get('id'),'caption':''.join(fig.find('caption').itertext()),
                'graphics':[g.get(XLINK) for g in fig.findall('.//graphic')]})
        base.save(dest/'metadata.json',metadata)
        if 'creativecommons.org/licenses/' not in license_text:
            metadata['download_status']='No explicit CC article licence; excluded'
            summaries.append(metadata); continue
        package_url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles'
        r=session.get(package_url,timeout=60,stream=True)
        if r.status_code in (401,403,429,503): raise base.Denied('Europe PMC API denial; no bypass')
        r.raise_for_status(); blocks=[]; total=0
        for block in r.iter_content(1024*1024):
            total+=len(block)
            if total>100_000_000: raise RuntimeError('Predeclared100MB article package cap exceeded')
            blocks.append(block)
        payload=b''.join(blocks); (dest/'oa_assets.zip').write_bytes(payload)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members=archive.namelist(); metadata['package_members']=members
            for fig in metadata['figures']:
                for graphic in fig['graphics']:
                    matches=[m for m in members if Path(m).name==Path(graphic).name or Path(m).stem==Path(graphic).stem]
                    for index,name in enumerate(matches):
                        target=dest/(Path(name).name if index==0 else f'{index}_'+Path(name).name)
                        if target.exists(): continue
                        target.write_bytes(archive.read(name))
        metadata['package_api']=package_url; metadata['package_sha256']=sha256(dest/'oa_assets.zip')
        base.save(dest/'metadata.json',metadata); summaries.append(metadata)
        print('PAPER',pmcid,metadata['doi'],[(x['id'],x['graphics']) for x in metadata['figures'][:2]],flush=True)
        time.sleep(1.2)
    base.save(out/'paper_sources.json',summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('nc_collect','papers')); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); {'nc_collect':nc_collect,'papers':papers}[a.phase](a.out.resolve())
