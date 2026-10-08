"""Bundle one unchanged, licensed offline reference photo per model class."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from training_safety import sha256, verify_backup

RUN = ROOT / 'outputs/android_h120_ui_sync_20260927_v1'
SOURCE = ROOT / 'external_datasets/spacescan_whole_tree_references_20260910_v2'
ASSETS = ROOT / 'spacescan_android_sstv1/app/src/main/assets'
GALLERY = ASSETS / 'species_examples'


def main():
    verify_backup(RUN / 'pre_training_backup')
    assert not GALLERY.exists(), 'Gallery already exists; use a new run-specific output'
    labels = json.loads((ASSETS / 'species.json').read_text(encoding='utf-8'))
    reviewed = json.loads((SOURCE / 'mixed_v3/accepted_manifest.json').read_text(encoding='utf-8'))
    allowed = {}
    for row in reviewed:
        if (row.get('visual_review') != 'accepted_for_reference'
            or row.get('framing') not in {'whole_tree_or_crown', 'whole_tree_or_flower_leaf_detail'}
            or row.get('license') not in {'cc0', 'cc-by', 'cc-by-sa'}
            or not row.get('attribution') or row.get('training_eligible')):
            continue
        source = (SOURCE / row['file']).resolve()
        assert source.is_relative_to((SOURCE / 'images').resolve())
        if not source.is_file() or sha256(source) != row['sha256']:
            continue
        allowed.setdefault(row['species'], row)
    GALLERY.mkdir()
    output = []
    for label in labels:
        row = allowed.get(label['label'])
        if row is None:
            output.append({'label': label['label'], 'available': False})
            continue
        source = SOURCE / row['file']
        name = f"example_{label['id']:03d}.jpg"
        target = GALLERY / name
        shutil.copy2(source, target)  # Preserve the licensed original bytes/watermark.
        assert sha256(target) == row['sha256']
        output.append({'label': label['label'], 'available': True, 'asset': 'species_examples/' + name,
                       'sha256': row['sha256'], 'bytes': target.stat().st_size,
                       'attribution': row['attribution'], 'license': row['license'].upper(),
                       'license_url': row['license_url'], 'source_url': row['source_url'],
                       'photo_page': row['photo_page'], 'observed_taxon': row['observed_taxon'],
                       'reference_only': True})
    assert len(output) == 148
    assert sum(x['available'] for x in output) == 147
    manifest = ASSETS / 'species_examples.json'
    manifest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (RUN / 'reference_gallery_audit.json').write_text(json.dumps({
        'raw_classes': 148, 'photos': 147,
        'without_licensed_photo': [x['label'] for x in output if not x['available']],
        'source_manifest': str(SOURCE / 'mixed_v3/accepted_manifest.json'),
        'source_images_unchanged': True,
        'policy': 'Reviewed reference-only CC0/CC-BY/CC-BY-SA originals; no training use or rights inference',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    verify_backup(RUN / 'pre_training_backup')
    print('GALLERY_READY', len(output), sum(x['available'] for x in output))


if __name__ == '__main__':
    main()
