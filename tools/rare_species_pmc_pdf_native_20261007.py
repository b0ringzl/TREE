"""Read public OA PDFs, preserving original native embedded image resolution."""
import hashlib
import sys
import time
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fresh_bioclip146_benchmark_20261006 import load, save, UA
OUT = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'

def main():
    records = []
    for md in load(OUT / 'paper_cloud_sources.json'):
        dest = OUT / 'paper_originals' / (md['pmcid'] + '_cloud_v1')
        fig = next(a for a in md['cloud_downloads'] if 'dimensions' in a)
        prefix = fig['url'].rsplit('/', 1)[0]
        version = prefix.rsplit('/', 1)[-1]
        path = dest / (version + '.pdf')
        if path.exists(): raise FileExistsError(path)
        url = prefix + '/' + version + '.pdf'
        response = requests.get(url, headers=UA, timeout=35)
        response.raise_for_status()
        if not response.content.startswith(b'%PDF') or len(response.content) > 20_000_000:
            raise RuntimeError('Unexpected PDF source')
        path.write_bytes(response.content)
        records.append({'pmcid': md['pmcid'], 'url': url, 'file': path.relative_to(OUT).as_posix(),
                        'sha256': hashlib.sha256(response.content).hexdigest(), 'bytes': len(response.content)})
        save(OUT / 'paper_pdf_sources.json', records)
        print('PDF', md['pmcid'], len(response.content), flush=True)
        time.sleep(.5)

if __name__ == '__main__': main()
