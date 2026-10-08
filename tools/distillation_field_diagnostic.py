"""Read-only human-reviewed VMMS diagnostic; never a tuning or training dataset."""
import argparse
from pathlib import Path
import numpy as np
from distillation_core import read_json, save_json, fuse, metrics
from run_distillation import ROOT, teacher, teacher_prob, student_features, write_csv
from training_safety import sha256


def prepare(out):
    labels = read_json(out / 'classes.json')['labels']
    base = ROOT / 'hk-tree-vmms/derived'
    split_path = base / 'reviewed_five_species_pilot_v1_20260907/split.json'
    reviews_path = base / 'hk_dualsource_five_species_20260907/reviews.json'
    reviews = read_json(reviews_path)
    conflict = base / 'albizia_conflict_review_v1_20260907'
    manifest = read_json(conflict / 'manifest.json')
    decision = read_json(conflict / 'adjudications.json')[-1]
    current = sha256(reviews_path)
    if decision['source_review_sha256'] != current or manifest['source_review_sha256'] != current:
        raise RuntimeError('Explicit adjudication source has changed; cannot use old identity/label')
    if decision['status'] != 'identity_and_species_confirmed' or decision['identity'] != 'same_tree':
        raise RuntimeError('No current explicit same-tree adjudication')
    confirmed = {r['id'] for r in manifest['items'] if r['role'] == 'conflict'}
    rows = []
    for r in read_json(split_path):
        if r['split'] != 'heldout':
            continue
        acceptance = r.get('accepted_review') or reviews.get(r['id'], {})
        if acceptance.get('decision') != 'accept':
            continue
        species = decision['species'] if r['id'] in confirmed else acceptance.get('species', r['species'])
        if species not in labels:
            # Original free-text species is never guessed into a different class.
            continue
        if sha256(r['image']) != r['image_sha256']:
            raise RuntimeError('Field image changed')
        rows.append({'sample_id': r['id'], 'materialized_path': r['image'],
            'source_sha256': r['image_sha256'], 'label': species, 'target': labels.index(species),
            'phase': 'field_diagnostic', 'group_id': r['leakage_group'],
            'tree_id': 'explicit_4276_4287' if r['id'] in confirmed else None,
            'review_provenance': acceptance})
    three = base / 'hk_dualsource_three_species_20260907'
    three_reviews = read_json(three / 'pair_reviews.json')
    # Only explicit accepted rows from the old holdout. Absent output taxa remain excluded.
    three_map = {'榕树': 'Ficus microcarpa / Ficus benjamina (榕樹)',
                 'Livistona chinensis 蒲葵': 'Livistona chinensis'}
    excluded = []
    for r in read_json(three / 'experiment/split.json'):
        acceptance = three_reviews.get(r['id'], {})
        if r['split'] != 'heldout' or acceptance.get('decision') != 'accept':
            continue
        species = three_map.get(acceptance.get('species'))
        if species not in labels:
            excluded.append({'id': r['id'], 'species': acceptance.get('species'), 'reason': 'not_in_student_catalog'})
            continue
        rows.append({'sample_id': r['id'], 'materialized_path': r['image'],
            'source_sha256': sha256(r['image']), 'label': species, 'target': labels.index(species),
            'phase': 'field_diagnostic', 'group_id': r['tree_group'], 'tree_id': None,
            'review_provenance': acceptance})
    save_json(out / 'field_rows.json', rows)
    case = []
    for r in manifest['items']:
        if r['id'] not in confirmed:
            continue
        case.append({'sample_id': r['id'], 'materialized_path': r['image'],
            'source_sha256': sha256(r['image']), 'label': decision['species'],
            'target': labels.index(decision['species']), 'phase': 'case_diagnostic',
            'group_id': 'explicit_4276_4287', 'tree_id': 'explicit_4276_4287'})
    save_json(out / 'case_rows.json', case)
    save_json(out / 'audit/field_diagnostic_provenance.json', {
        'reviews_sha256': current, 'three_reviews_sha256': sha256(three / 'pair_reviews.json'),
        'adjudication': decision, 'excluded_absent_output': excluded,
        'case_tree_count': 1, 'case_views': len(case), 'field_views': len(rows),
        'usage': 'Descriptive diagnostics only, no hyperparameter or checkpoint selection. No training. One verified tree cannot support a multiview generalization claim.'})


def run(out):
    if not (out / 'final_lock.json').exists():
        raise RuntimeError('Field diagnostics run only after configuration lock')
    prepare(out)
    teacher(out, ['field', 'case'])
    student_features(out, ['field', 'case'])
    labels = read_json(out / 'classes.json')['labels']
    for phase in ['field', 'case']:
        rows = read_json(out / f'{phase}_rows.json')
        if not rows:
            continue
        tp = teacher_prob(out, phase)
        sp = np.load(out / f'student_cache/{phase}.npz')['probabilities']
        results = []
        for i, r in enumerate(rows):
            results.append({'id': r['sample_id'], 'truth': r['label'],
                'YOLO': labels[int(sp[i].argmax())], 'BioCLIP': labels[int(tp[i].argmax())],
                'teacher_GT_probability': float(tp[i, r['target']]),
                'teacher_top1_probability': float(tp[i].max())})
        write_csv(out / f'analysis/{phase}_diagnostic.csv', results)
        if phase == 'field':
            save_json(out / 'metrics/field_diagnostic.json', {
                'baseline': metrics([r['target'] for r in rows], sp),
                'teacher': metrics([r['target'] for r in rows], tp),
                'warning': 'Small historical reviewed VMMS holdout; not independent operational accuracy.'})
        else:
            fusion = []
            for method in ['mean', 'logprob', 'weighted']:
                p = fuse(tp, method)
                fusion.append({'method': method, 'tree_id': rows[0]['tree_id'], 'views': len(rows),
                    'truth': rows[0]['label'], 'prediction': labels[int(p.argmax())],
                    'GT_probability': float(p[rows[0]['target']]), 'top1_probability': float(p.max())})
            write_csv(out / 'analysis/one_verified_tree_fusion_case.csv', fusion)
            save_json(out / 'metrics/one_verified_tree_case.json', {
                'single_views': results, 'fusion': fusion,
                'warning': 'ONE explicitly adjudicated tree. Illustrative case only; no multiview training or aggregate gain claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.out.resolve())
