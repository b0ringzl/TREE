"""Verify reviewed gallery before enabling it; does not change any model."""
import hashlib
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from .collect_references import write_json, ROOT, APP, LICENSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'external_datasets/spacescan_whole_tree_references_20260910_v2'
    rows = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    review = json.loads((root/'visual_review.json').read_text(encoding='utf-8'))
    names = json.loads((APP/'species_names.json').read_text(encoding='utf-8'))
    classes = json.loads((ROOT/'outputs/distillation/bioclip_probe_20260908_0227/classes.json').read_text(encoding='utf-8'))['labels'] + ['Wodyetia bifurcata']
    accepted = []
    hashes = set()
    for row in rows:
        decision = review.get(row['id'], {})
        if decision.get('status') != 'accepted_for_reference':
            continue
        row = {**row, 'visual_review': decision['status'], 'review_note': decision['note'], 'framing': 'whole_tree_or_crown'}
        path = (root/row['file']).resolve()
        assert path.is_relative_to((root/'images').resolve())
        assert row['license'] in LICENSES and row['attribution'] and row['author']
        assert not row['training_eligible']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert row['sha256'] not in hashes
        hashes.add(row['sha256'])
        with Image.open(path) as im:
            im.verify()
        accepted.append(row)
    counts = Counter(r['species'] for r in accepted)
    missing = {s: counts[s] for s in classes if counts[s] != 2}
    print(json.dumps({'accepted':len(accepted), 'complete_classes':sum(counts[s] == 2 for s in classes), 'missing':missing}, ensure_ascii=True), flush=True)
    if (missing or len(accepted) != 296) and not args.allow_partial:
        raise SystemExit('Not enabled: every class must have two visually reviewed, licensed references.')
    assert all(n <= 2 for n in counts.values())
    write_json(root/'accepted_manifest.json', accepted)
    write_json(root/'coverage.json', {'target_classes':148, 'target_images':296, 'accepted_images':len(accepted),
        'covered_classes':len(counts), 'complete_classes':sum(counts[s]==2 for s in classes), 'missing':missing,
        'criterion':'Whole tree or substantial visible crown; no isolated seed, flower, leaf or trunk close-ups.', 'training_eligible':False})
    lines = ['# Reference gallery collection / 參考圖庫採集', '',
        f'Target: 148 model classes, 296 images. Current accepted: {len(accepted)} images across {len(counts)} classes; {sum(counts[s]==2 for s in classes)} classes have two images. Source: iNaturalist research-grade observations. All included images received an assistant whole-tree/crown visual suitability check, not independent botanical certification. Unfilled slots are left empty, not filled with organ-only photos.', '',
        'For candidate comparison only. No training, evaluation, human acceptance database or deployed weights changed. Downloaded large image variants are retained without content editing. All photographs retain source URLs, author attribution, license links, observed dates and SHA-256. No public image upload was performed.', '',
        'Only CC0, CC BY and CC BY-SA photo licenses were accepted; observation licenses alone were not used. License terms: https://help.inaturalist.org/en/support/solutions/articles/151000169918 . API practice: https://www.inaturalist.org/pages/api+recommended+practices .', '',
        'The model\'s existing grouped classes retain the actual photographed species in each reference record. Related species were not silently substituted. Two images can occasionally share an observation when alternatives are unavailable; this is not independent validation data.', '',
        '| Scientific name / 學名 | Traditional Chinese / 中文名 | Photos / 圖片 |', '|---|---|---|']
    for s in classes:
        refs = [r for r in accepted if r['species'] == s]
        links = ' · '.join('[' + r['id'] + '](' + r['photo_page'] + ')' for r in refs)
        lines.append('| ' + names[s]['scientific_name'] + ' | ' + names[s]['chinese_name'] + ' | ' + (links or 'Pending / 待補整株或樹冠照') + ' |')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
