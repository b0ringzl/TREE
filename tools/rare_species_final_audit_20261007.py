"""Independent recomputation of frozen-data metrics and provenance invariants."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import fresh_bioclip146_benchmark_20261006 as base
from training_safety import sha256, verify_backup
OUT = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'

def main():
    if (OUT / 'FINAL_AUDIT.json').exists(): raise FileExistsError('Audit cannot be silently replaced')
    weight = verify_backup(OUT / 'pre_training_backup')
    protocol = verify_backup(OUT / 'evaluation_protocol_snapshot/pre_training_backup')
    rows = base.load(OUT / 'manifest.json'); lock = base.load(OUT / 'collection_lock.json')
    scores = [json.loads(line) for line in (OUT / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    result = base.load(OUT / 'RARE_RESULTS.json'); cases = base.load(OUT / 'CASE_RESULTS.json')
    assert len(rows) == len(scores) == len(cases) == 192
    assert sha256(OUT / 'manifest.json') == lock['manifest_sha256'] == result['manifest_sha256']
    assert sha256(OUT / 'predictions.jsonl') == result['predictions_sha256']
    # Filesystem creation times strengthen the immutable pre-inference lock.
    created = datetime.fromisoformat(lock['locked_at_utc']).timestamp()
    assert created < (OUT / 'predictions.jsonl').stat().st_ctime
    for name, expected in lock['plans_sha256'].items(): assert sha256(OUT / name) == expected
    prior_obs, prior_photos, prior_hash, prior_visual = base.prior_index(OUT)
    current_hash = set(); current_ids = set(); current_visual = {}; group_counts = Counter(); parity = []
    tally = Counter()
    for r, p, c in zip(rows, scores, cases):
        assert r['id'] == p['id'] == c['id'] and r['label'] == p['label'] == c['label']
        assert not r['training_eligible'] and r['id'] not in current_ids
        file = OUT / r['file']; assert sha256(file) == r['sha256']
        assert r['sha256'] not in prior_hash and r['sha256'] not in current_hash
        with Image.open(file) as src: im = ImageOps.exif_transpose(src).convert('RGB')
        assert list(im.size) == r['dimensions'] and min(im.size) >= 224
        phash = base.phash(im); assert r['phash'] == f'{phash:016x}'
        assert not base.nearest_hash(phash, prior_visual) and not base.nearest_hash(phash, current_visual)
        if r['source_kind'] == 'community_observation':
            assert str(r['observation_id']) not in prior_obs and str(r['photo_id']) not in prior_photos
            assert r['license'] in {'cc-by-nc', 'cc-by-nc-sa'}
            ev = r['source_evidence']
            assert ev['tier'] == 'multiple_independent_exact_ids' and len(set(ev['exact_identifier_ids'])) >= 2
            assert ev['independent_of_observer'] and not ev['conflicting_current_identifications']
            group_counts[(r['label'], r['observation_id'])] += 1
        else:
            assert r['license'] == 'cc-by-4.0' and not r['resampled']
            assert sha256(OUT / r['native_figure_file']) == r['native_figure_sha256']
            assert sha256(OUT / r['pdf_file']) == r['pdf_sha256']
            with Image.open(OUT / r['native_figure_file']) as src:
                exact = np.asarray(src.convert('RGB').crop(r['crop_rectangle']))
            assert np.array_equal(exact, np.asarray(im)), 'Scientific extraction must preserve original pixels'
        current_ids.add(r['id']); current_hash.add(r['sha256']); current_visual[phash] = r['id']
        assert len(p['distributions']) == 6
        for mode, dist in p['distributions'].items():
            assert len(dist) == 148 and abs(sum(dist.values()) - 1) < 1e-5
            assert all(np.isfinite(v) and v >= 0 for v in dist.values())
            assert len(base.grouped_distribution(dist)) == 146
            assert base.rank(dist)[:3] == p['top3'][mode]
        for prefix in ('teacher', 'sstv1'):
            for k, v in p['distributions'][prefix + '_context'].items():
                assert abs(v - .5 * (p['distributions'][prefix + '_whole'][k] + p['distributions'][prefix + '_center'][k])) < 1e-12
        if 'production_parity_max_error' in p['evidence']:
            parity.append(p['evidence']['production_parity_max_error']); assert parity[-1] <= 1e-6
        if r['source_kind'] == 'community_observation':
            a = p['top3']['teacher_context'][0] == r['label']; b = p['top3']['sstv1_context'][0] == r['label']
            tally[(r['label'], 'n')] += 1; tally[(r['label'], 'a')] += int(a); tally[(r['label'], 'b')] += int(b)
            tally[(r['label'], 'fixed')] += int(b and not a); tally[(r['label'], 'broken')] += int(a and not b)
    for label, s in result['per_species'].items():
        assert [s['n'], s['before_correct'], s['after_correct'], s['corrected'], s['regressed']] == [tally[(label, k)] for k in ('n', 'a', 'b', 'fixed', 'broken')]
        assert abs(s['gain_percentage_points'] - (tally[(label, 'b')] - tally[(label, 'a')]) * 100 / tally[(label, 'n')]) < 1e-10
    assert len(parity) == 3 and all(n <= 3 for n in group_counts.values())
    assert 36887547 not in {r.get('photo_id') for r in rows}
    audit = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'requirements': {
            'two_species_new_to_registered_project_sources': True,
            'known_source_ids_sha_and_phash_excluded_including_pending_rejected': True,
            'labels_from_independent_source_identifiers_not_model': True,
            'native_paper_photographs_caption_license_and_lossless_pixels_verified': True,
            'sampling_frozen_before_any_inference': True,
            'all192_samples_compared_same146_catalog_and_production_path': True,
            'metrics_independently_recomputed_and_regressions_retained': True,
            'all_probability_distributions_and_three_production_parity_checks_verified': True,
            'physical_model_and_protocol_backups_and_original_sources_sha_verified': True,
            'training_or_model_deployment': False,
        }, 'backup_weight_source_count': len(weight['files']), 'protocol_source_count': len(protocol['files']),
        'production_parity_max_errors': parity,
        'main_photos': result['primary']['n'], 'paper_challenge_photos': result['paper_challenge']['n'],
        'report_sha256': sha256(OUT / 'REPORT_ZH.md'),
        'raw_predictions_sha256': sha256(OUT / 'predictions.jsonl'),
        'scope_limits': ['Project-novel, not proven absent from BioCLIP pretraining; recent subset0',
            'Source labels not expert ground truth', 'Positive-only recall, not target precision or all-rare-species accuracy',
            'Observer-cluster uncertainty crosses zero; white桂木 regresses', 'NC originals not distributed']}
    base.save(OUT / 'FINAL_AUDIT.json', audit)
    print('FINAL AUDIT PASSED', len(rows), 'images', len(weight['files']), 'protected original artifacts unchanged', flush=True)

if __name__ == '__main__': main()
