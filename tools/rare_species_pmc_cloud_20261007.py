"""Download licensed figures from PMC's documented public cloud dataset.

No access-control bypass: the independently published, public S3 distribution
is documented at https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/ .
"""
import hashlib
import io
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fresh_bioclip146_benchmark_20261006 import load, save, UA
OUT = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'
S3 = 'https://pmc-oa-opendata.s3.amazonaws.com/'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def main():
    plan = OUT / 'PMC_CLOUD_PLAN_20261007.json'
    if plan.exists():
        raise FileExistsError('Preserve the previous collection version')
    save(plan, {
        'official_distribution_documentation': 'https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/',
        'purpose': 'Private noncommercial source-labelled scientific evaluation, not training',
        'selection': 'Only figure1 of four preselected papers; natural plant-photo panels selected from captions before scores',
        'excluded': 'Microscopy, charts, agarwood products, other Artocarpus species, mixed cultivar montage',
        'minimum_panel_native_side': 224,
        'forbid': 'No upscaling to meet native resolution, generated imagery, or score-based selection',
        'prediction_inspected': False,
    })
    summaries = []
    for original in load(OUT / 'paper_sources_v2.json'):
        pmcid = original['pmcid']
        dest = OUT / 'paper_originals' / (pmcid + '_cloud_v1')
        dest.mkdir(exist_ok=False)
        response = requests.get(S3, params={'list-type': '2', 'prefix': pmcid + '.', 'max-keys': 100}, headers=UA, timeout=30)
        response.raise_for_status()
        (dest / 'bucket_listing.xml').write_bytes(response.content)
        listing = ET.fromstring(response.content)
        if listing.findtext('s:IsTruncated', namespaces=NS) == 'true':
            raise RuntimeError('Do not silently truncate available versions')
        entries = [{name: node.findtext('s:' + name, namespaces=NS) for name in ('Key', 'ETag', 'Size')}
                   for node in listing.findall('s:Contents', NS)]
        desired = original['graphics'][0]
        matches = [e for e in entries if Path(e['Key']).name == desired]
        if len(matches) != 1:
            raise RuntimeError(f'Ambiguous figure version: {pmcid}: {matches}')
        figure = matches[0]
        prefix = figure['Key'].rsplit('/', 1)[0]
        selected = [e for e in entries if e['Key'] == figure['Key'] or
                    (e['Key'].startswith(prefix + '/') and e['Key'].endswith(('.json', '.xml')))]
        md = dict(original)
        md['cloud_documentation'] = 'https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/'
        md['cloud_downloads'] = []
        for entry in selected:
            if int(entry['Size']) > 10_000_000:
                raise RuntimeError('Predeclared 10MB individual asset cap')
            url = S3 + entry['Key']
            response = requests.get(url, headers=UA, timeout=35)
            response.raise_for_status()
            raw = response.content
            if len(raw) != int(entry['Size']):
                raise RuntimeError('Asset size mismatch')
            etag = entry['ETag'].strip('"')
            if '-' not in etag and hashlib.md5(raw).hexdigest() != etag:
                raise RuntimeError('Published S3 object MD5 mismatch')
            path = dest / Path(entry['Key']).name
            path.write_bytes(raw)
            asset = {'url': url, 'file': path.relative_to(OUT).as_posix(),
                     'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 's3_etag': etag}
            if entry['Key'] == figure['Key']:
                with Image.open(io.BytesIO(raw)) as im:
                    asset['dimensions'] = list(im.size)
                md['image_file'] = asset['file']
                md['image_sha256'] = asset['sha256']
                md['dimensions'] = asset['dimensions']
            md['cloud_downloads'].append(asset)
            time.sleep(.5)
        md['training_eligible'] = False
        save(dest / 'metadata.json', md)
        summaries.append(md)
        save(OUT / 'paper_cloud_sources.json', summaries)
        print('CLOUD_FIGURE', pmcid, md['dimensions'], md['doi'], flush=True)


if __name__ == '__main__':
    main()
