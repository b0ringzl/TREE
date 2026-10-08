"""Preserve all browser acquisitions in the current-run evidence directory.

The raw archive is NOT an approved dataset: duplicates and the identified wrong
gallery image remain here for provenance only. No image byte is modified.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/conservation_browser_collection_20260917_v1'

def main():
    records = json.loads((RUN/'source_records.json').read_text(encoding='utf-8-sig'))
    if isinstance(records, dict):
        records = records.get('items', records.get('records'))
    out = RUN/'raw_browser_downloads'
    out.mkdir(exist_ok=False)
    manifest = []
    for i, row in enumerate(records, 1):
        original = Path(row['local_file'])
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        destination = out/f'{i:03d}_{digest[:16]}{original.suffix.lower()}'
        with original.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        manifest.append(dict(row, archived_original=str(destination.resolve()), sha256=digest,
                             training_eligible=False, archive_role='raw_acquisition_evidence_not_approved_dataset'))
    with (out/'acquisition_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)
    shutil.copyfile(RUN/'COPYRIGHT_GREening.txt',out/'COPYRIGHT_GREening.txt')
    print(json.dumps({'raw_source_files_archived':len(manifest),'bytes':sum(Path(r['archived_original']).stat().st_size for r in manifest),'out':str(out)},ensure_ascii=False))

if __name__=='__main__':
    main()
