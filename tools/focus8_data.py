"""Auditable eight-species scope; public authoritative photos stay in quarantine."""
import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import urljoin

from inventory_herbarium_images import Client, BASE, ASSETS, NOTICE, norm, quality, write_json
from training_safety import sha256

ROOT = Path(__file__).resolve().parents[1]
TAXA = [
    ('Aleurites moluccana', '石栗', 'Aleurites moluccana'),
    ('Melaleuca cajuputi', '白千層', 'Melaleuca cajuputi subsp. cumingiana'),
    ('Livistona chinensis', '蒲葵', 'Livistona chinensis'),
    ('Delonix regia', '鳳凰木', 'Delonix regia'),
    ('Ficus microcarpa', '細葉榕', 'Ficus microcarpa'),
    ('Hibiscus tiliaceus', '黃槿', 'Hibiscus tiliaceus'),
    ('Acacia confusa', '台灣相思', 'Acacia confusa'),
    ('Bauhinia x blakeana', '洋紫荊', 'Bauhinia x blakeana'),
]
LABELS = [t[0] for t in TAXA]
HERB = ROOT / 'external_datasets/herbarium_training_candidates_20260907'


def review_snapshot():
    with sqlite3.connect((HERB / 'human_review.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        return {r['id']: dict(r) for r in db.execute('SELECT * FROM decisions')}


def collect(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    client = Client(out)
    client.policy()
    client.get('/filemanager/template/common/js/plantSearch.js')
    decisions = review_snapshot()
    write_json(out / 'prior_human_review_snapshot.json', decisions)
    existing = {r['id']: r for r in json.loads((HERB / 'curated_manifest.json').read_text('utf8'))}
    records, inventory = [], []
    for label, chinese, query in TAXA:
        catalog = client.json(ASSETS + 'js/species/species_' + query[0].lower() + '.json')
        matches = [r for group in catalog['records'].values() for r in group if norm(r['scientific_name']) == norm(query)]
        inventory.append({'label': label, 'chinese': chinese, 'official_query': query, 'matches': matches})
        for match in matches:
            detail = client.json('/plantdb/GetSpecies.php?id=' + match['species_id'])
            for photo in detail.get('photo', []):
                ident = match['species_id'] + '_' + photo['photo_multimedia_id']
                row = {'id': ident, 'label': label, 'chinese': chinese,
                    'source_species': detail['scientific_name_without_authority'],
                    'image_url': urljoin(BASE, photo['filepath']),
                    'source_page': BASE + '/tc/hk-plant-database/plant-detail/index.html?pType=species&oID=' + match['species_id'],
                    'copyright_holder': 'Hong Kong Herbarium, AFCD', 'rights_url': NOTICE,
                    'training_eligible': False, 'split': 'quarantine', 'identity_authority': 'official_botanical_catalog',
                    'review_status': 'pending', 'rights_status': 'ML training not explicitly granted; no public redistribution'}
                # Do not reset rejected decisions by downloading under another batch.
                if ident in existing:
                    old = existing[ident]
                    path = HERB / old['file']
                    if sha256(path) != old['sha256']:
                        raise ValueError('Original image changed: ' + ident)
                    row.update(file=str(path), sha256=old['sha256'], reused_original=True,
                        previous_decision=decisions.get(ident, {}), review_status=decisions.get(ident, {}).get('status', 'pending'))
                else:
                    raw, meta = client.get(row['image_url'])
                    q, _ = quality(raw)
                    path = out / 'images' / (ident + '.jpg')
                    path.write_bytes(raw)
                    row.update(file=str(path.resolve()), sha256=meta['sha256'], quality=q, reused_original=False)
                records.append(row)
                write_json(out / 'manifest.json', records)
        write_json(out / 'inventory.json', inventory)
        print(label + ': ' + str(sum(r['label'] == label for r in records)), flush=True)
    write_json(out / 'summary.json', {'photos': len(records), 'new_downloads': sum(not r['reused_original'] for r in records),
        'training_started': False, 'new_photos_training_eligible': 0,
        'counts': {s: sum(r['label'] == s for r in records) for s in LABELS}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    collect(parser.parse_args().out.resolve())
