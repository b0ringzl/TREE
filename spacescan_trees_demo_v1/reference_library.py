"""Read-only, allowlisted reference assets, separate from all training manifests."""
import hashlib
import json
from pathlib import Path

from .species_names import bilingual_names
from . import manual_references

ROOT = Path(__file__).resolve().parents[1] / 'external_datasets/spacescan_whole_tree_references_20260910_v2'
MANIFEST = ROOT / 'mixed_v3/accepted_manifest.json'
ALLOWED_LICENSES = {'cc0', 'cc-by', 'cc-by-sa'}
ROWS = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else []
ASSETS = {r['id']: r for r in ROWS if r.get('visual_review') == 'accepted_for_reference'
          and r.get('framing') in {'whole_tree_or_crown','whole_tree_or_flower_leaf_detail'}
          and r.get('license') in ALLOWED_LICENSES and r.get('attribution') and not r.get('training_eligible')}


def asset_path(key):
    row = ASSETS.get(key)
    if row is None:
        raise KeyError(key)
    path = (ROOT / row['file']).resolve()
    if not path.is_relative_to((ROOT / 'images').resolve()) or not path.is_file():
        raise KeyError(key)
    return path


def references(species, exclude_hash=None):
    result = []
    for key, row in ASSETS.items():
        if row['species'] != species or row['sha256'] == exclude_hash:
            continue
        try:
            path = asset_path(key)
        except KeyError:
            continue
        if exclude_hash and hashlib.sha256(path.read_bytes()).hexdigest() == exclude_hash:
            continue
        result.append({**bilingual_names(species), 'species': species, 'url': '/api/web-reference/' + key,
            'source': 'iNaturalist', 'source_url': row['source_url'], 'photo_page': row['photo_page'],
            'attribution': row['attribution'], 'author': row['author'], 'license': row['license'].upper(),
            'license_url': row['license_url'], 'date': row['date'], 'reference_taxon': row['observed_taxon'],
            'note': 'Species reference, not the same individual. / 樹種參考圖，並非同一株樹。'})
        if len(result) == 2:
            break
    manual = {r['slot']: r for r in manual_references.rows(species) if r['sha256'] != exclude_hash}
    manual_hashes = {r['sha256'] for r in manual.values()}
    def usable(ref):
        key = ref['url'].rsplit('/',1)[-1]
        return ref['url'] not in used and ASSETS.get(key,{}).get('sha256') not in manual_hashes
    combined = []
    used = set()
    for slot in (1, 2):
        if slot in manual:
            ref = manual[slot]
        else:
            preferred = result[slot-1] if slot <= len(result) else None
            ref = preferred if preferred and usable(preferred) else next((r for r in result if usable(r)), None)
        if ref:
            combined.append({**ref, 'display_slot': slot})
            used.add(ref['url'])
    return combined
