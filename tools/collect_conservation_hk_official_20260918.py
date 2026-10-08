"""Small attributed official HK photo acquisition; quarantine only, never fit."""
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/conservation_authoritative_collection_20260918_v1/hk_official'
SPECIES = ['Aquilaria sinensis', 'Artocarpus hypargyreus', 'Rhodoleia championii', 'Keteleeria fortunei']
BASE = 'https://www.herbarium.gov.hk'
NOTICE = BASE+'/en/important-notices/index.html'
UA = 'SpaceScanTreesResearch/1.0 (limited local botanical reference collection)'


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.images=[]; self.text=[]; self.feed(text)
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=='img' and '/filemanager/' in a.get('src',''): self.images.append(a['src'])
    def handle_data(self, value): self.text.append(value)


def save(path, data):
    with path.open('x', encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)


def main():
    OUT.mkdir(parents=True, exist_ok=False); (OUT/'images').mkdir(); (OUT/'evidence').mkdir()
    session=requests.Session(); session.headers['User-Agent']=UA
    events=[]; rows=[]; blocked=False
    def get(url):
        nonlocal blocked
        if blocked: raise RuntimeError('Source stopped after denied/rate-limited response')
        time.sleep(1.2)
        r=session.get(url,timeout=30)
        events.append(dict(url=url,final_url=r.url,status=r.status_code,date=datetime.now(timezone.utc).isoformat()))
        if r.status_code in (401,403,429): blocked=True
        r.raise_for_status(); return r
    try:
        robots=get(BASE+'/robots.txt'); (OUT/'evidence/robots.response').write_bytes(robots.content)
        rp=RobotFileParser(); is_text=not robots.text.lstrip().lower().startswith(('<!doctype','<html'))
        if is_text: rp.parse(robots.text.splitlines())
        else: rp.parse([])
        policy=dict(url=BASE+'/robots.txt',http_status=robots.status_code,
                    interpretation='parsed robots directives' if is_text else 'HTTP200 HTML home redirect, not robots directives; no explicit rules supplied',
                    raw_sha256=hashlib.sha256(robots.content).hexdigest(),rate='serial >=1.2 seconds')
        save(OUT/'robots_audit.json',policy)
        if not rp.can_fetch(UA,NOTICE): raise RuntimeError('Notice disallowed by robots')
        notice=get(NOTICE); (OUT/'evidence/copyright.html').write_bytes(notice.content)
        if not all(t in notice.text for t in ['non-commercial use','source of information','acknowledged']):
            raise RuntimeError('Expected noncommercial attributed copying basis not found')
        for species in SPECIES:
            slug=species.lower().replace(' ','-')
            page=BASE+'/en/publications/books/book2/text/'+slug+'/index.html'
            if not rp.can_fetch(UA,page):
                events.append(dict(url=page,skip='robots_disallow')); continue
            try:
                response=get(page); (OUT/f'evidence/{slug}.html').write_bytes(response.content)
                parsed=Page(response.text)
                if species.casefold() not in ' '.join(parsed.text).casefold():
                    events.append(dict(url=page,skip='exact_species_not_found')); continue
                for source in dict.fromkeys(parsed.images):
                    url=urljoin(page,source)
                    if not rp.can_fetch(UA,url): continue
                    try:
                        response=get(url)
                        with Image.open(io.BytesIO(response.content)) as im:
                            im.verify()
                        with Image.open(io.BytesIO(response.content)) as im: size=list(im.size); fmt=im.format
                        sha=hashlib.sha256(response.content).hexdigest()
                        rid='hkbook_'+sha[:18]; ext={'JPEG':'.jpg','PNG':'.png'}.get(fmt,'.img')
                        dest=OUT/'images'/(rid+ext)
                        if not dest.exists(): dest.write_bytes(response.content)
                        rows.append(dict(id=rid,source_species=species,source='Hong Kong Herbarium (AFCD)',
                            source_page=page,source_url=page,image_url=url,source_image=str(dest.resolve()),
                            sha256=sha,bytes=len(response.content),dimensions=size,
                            attribution='Hong Kong Herbarium, Agriculture, Fisheries and Conservation Department',
                            photographer='not named on species page',license='attributed non-commercial reproduction notice',
                            license_url=NOTICE,license_evidence_sha256=hashlib.sha256(notice.content).hexdigest(),
                            rights_scope='local_noncommercial_research_only_no_distribution; explicit ML-specific grant not established',
                            label_authority='official species page exact binomial; no independent expert certification',
                            source_group='HKHerbarium_book2_'+slug,group_id='HKHerbarium_book2_'+slug,
                            domain='official_species_illustration_pending_visual_type',
                            retrieved_at=datetime.now(timezone.utc).isoformat(),
                            quality_flags=['low_resolution'] if min(size)<224 else [],
                            training_eligible=False,split='quarantine',human_acceptance=False,
                            review_status='not_submitted_to_human_review',original_bytes_preserved=True))
                    except Exception as ex: events.append(dict(url=url,error=str(ex)))
            except Exception as ex: events.append(dict(url=page,error=str(ex)))
    finally:
        save(OUT/'manifest.json',rows); save(OUT/'network_log.json',events)
        save(OUT/'summary.json',dict(downloaded=len(rows),training_eligible=0,trained=False,deployed=False,
            by_species={s:sum(r['source_species']==s for r in rows) for s in SPECIES},
            novelty='Not yet screened; downloads are NOT claimed new to project',
            excluded_source=dict(name='HKBU Medicinal Plant Images Database',
                url='https://sys01.lib.hkbu.edu.hk/cmed/mpid/detail.php?herb_id=D00492&lang=cht',
                reason='robots endpoint returned403 in preliminary check; no image crawl attempted')))
        print(json.dumps(dict(downloaded=len(rows),out=str(OUT)),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
