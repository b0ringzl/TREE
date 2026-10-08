"""Prepare isolated, pending human review; source labels are NOT accepted labels."""
import json
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image
from training_safety import sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external_datasets/palm_confusion_20260908'


def main():
    output = DATA / 'curated_manifest.json'
    if output.exists():
        raise FileExistsError('Preserve existing review manifest; use a new collection directory')
    (DATA / 'images').mkdir(exist_ok=True)
    existing = {}
    for line in (ROOT / 'external_datasets/herbarium_training_candidates_20260907/existing_image_index.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('sha256'): existing.setdefault(row['sha256'], []).append(row['path'])
    records, seen = [], {}
    for file in sorted(DATA.glob('*/metadata.jsonl')):
        for line in file.read_text(encoding='utf-8').splitlines():
            source = json.loads(line)
            if source['inat_taxon_name'] != source['target_species']:
                continue
            path = DATA / source['local_path']
            digest = sha256(path)
            ident = 'inat_' + str(source['photo_id'])
            target = DATA / 'images' / (ident + path.suffix)
            if not target.exists(): shutil.copy2(path, target)
            assert sha256(target) == digest
            with Image.open(target) as image: dimensions = list(image.size)
            duplicates = existing.get(digest, []) + ([seen[digest]] if digest in seen else [])
            seen[digest] = ident
            records.append({**source, 'id': ident, 'file': target.relative_to(DATA).as_posix(), 'sha256': digest,
                'source_species': source['target_species'], 'source_names': ['狐尾椰子' if source['target_species'].startswith('Wodyetia') else '王棕'],
                'source_page': source['inat_uri'], 'source_group': 'inat_observation_' + str(source['inat_observation_id']),
                'organ_review': {'organ_tags': [], 'status': 'pending'}, 'training_eligible': False, 'split': 'quarantine',
                'candidate_lane': 'cross_dataset_match_quarantine' if duplicates else 'Verify species and usable tree detail / 請核對樹種及可辨識細節',
                'exact_duplicate_matches': duplicates, 'dimensions': dimensions})
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    html = (ROOT / 'tree_species_demo/static/herbarium-review.html').read_text(encoding='utf-8')
    start = html.index('<p class="foot">')
    end = html.index('</p>', start) + 4
    html = html[:start] + '<p class="foot">Photos: iNaturalist contributors / 圖片：iNaturalist 貢獻者。CC0 / CC BY / CC BY-SA. Attribution and observation links are preserved in the exported records. / 匯出記錄保留作者及觀察來源。</p>' + html[end:]
    (DATA / 'review.html').write_text(html, encoding='utf-8')
    summary = {'photos_by_species': dict(Counter(r['source_species'] for r in records)),
        'observations_by_species': {s: len({r['source_group'] for r in records if r['source_species'] == s}) for s in {r['source_species'] for r in records}},
        'pending': len(records), 'accepted': 0, 'training_performed': False,
        'exact_duplicate_candidates': sum(bool(r['exact_duplicate_matches']) for r in records),
        'required_next': 'Human review; perceptual/source-group leakage audit; freeze splits; physical backups before any head fit. Existing teacher heads cannot directly score the missing class.'}
    (DATA / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__': main()
