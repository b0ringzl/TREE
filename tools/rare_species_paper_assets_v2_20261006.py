"""Recover a terminal OA-package timeout using public figure resources.

Uses the official Europe PMC XML API for captions/licences. No cookies, auth,
challenge bypass, mirrors of restricted content, or generated imagery.
"""
import hashlib
import io
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin

import requests
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from fresh_bioclip146_benchmark_20261006 import save,load,UA

PAPERS={'PMC7050300':'fig1','PMC10935121':'molecules-29-01075-f001',
        'PMC8048387':'F1','PMC11945267':'plants-14-00951-f001'}
OUT=ROOT/'outputs/rare_species_fresh_validation_20261006_v1'
XLINK='{http://www.w3.org/1999/xlink}href'


def main():
    summaries=[]
    for pmcid, wanted in PAPERS.items():
        dest=OUT/'paper_originals'/(pmcid+'_assets_v2');dest.mkdir(exist_ok=False)
        api=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
        r=requests.get(api,headers=UA,timeout=40);r.raise_for_status();(dest/'article.xml').write_bytes(r.content)
        root=ET.fromstring(r.content)
        figs=root.findall('.//fig'); fig=next((f for f in figs if f.get('id')==wanted),figs[0])
        md={'pmcid':pmcid,'title':''.join(root.find('.//article-title').itertext()),
            'doi':[a.text for a in root.findall('.//article-id') if a.get('pub-id-type')=='doi'],
            'license':' '.join(''.join(a.itertext()) for a in root.findall('.//license')),
            'source_page':f'https://europepmc.org/articles/{pmcid}',
            'xml_api':api,'figure_id':fig.get('id'),'caption':''.join(fig.find('caption').itertext()),
            'graphics':[a.get(XLINK) for a in fig.findall('.//graphic')],'attempts':[]}
        save(dest/'metadata.json',md)
        for name in md['graphics']:
            if not name.lower().endswith(('.jpg','.jpeg','.png')):continue
            url=f'https://europepmc.org/articles/{pmcid}/bin/{name}'
            try:
                r=requests.get(url,headers=UA,timeout=45)
                attempt={'url':url,'status':r.status_code,'content_type':r.headers.get('content-type'), 'bytes':len(r.content)}
                md['attempts'].append(attempt)
                if r.status_code in (401,403,429,503):
                    md['access_denied_stop']=True;break
                r.raise_for_status()
                with Image.open(io.BytesIO(r.content)) as im:attempt['dimensions']=list(im.size)
                target=dest/Path(name).name;target.write_bytes(r.content)
                md['image_file']=str(target.relative_to(OUT)).replace('\\','/')
                md['image_sha256']=hashlib.sha256(r.content).hexdigest()
                print('FIGURE',pmcid,name,attempt['dimensions'],flush=True)
                break
            except Exception as exc:
                md['error']=str(exc);print('SOURCE_LIMITATION',pmcid,str(exc)[:160],flush=True)
            time.sleep(1.2)
        save(dest/'metadata.json',md);summaries.append(md);save(OUT/'paper_sources_v2.json',summaries)
        time.sleep(1.2)


if __name__=='__main__':main()
