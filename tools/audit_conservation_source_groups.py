"""Read-only grouping audit; proposals are not accepted labels or training splits."""
import argparse
import json
from pathlib import Path
from collections import Counter
from training_safety import sha256

ROOT = Path(__file__).resolve().parents[1]


def main(output):
    source = ROOT / 'external_datasets/conservation_sources_20260917_v1/review_manifest.json'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    rows = read(source)
    prior = ROOT / 'outputs/SSTv1_Aquilaria_User_20260914_v1'
    old_audit = read(prior / 'old_source_audit.json')
    old_extra = read(prior / 'accepted_new_manifest.json')
    known_aq_authors = {r['observer'] for r in old_audit if r.get('observer')}
    known_aq_authors |= {r['observer_id'] for r in old_extra if r.get('observer_id')}
    old_obs = {str(r['observation']) for r in old_audit}
    old_obs |= {str(r['inat_observation_id']) for r in old_extra if r.get('inat_observation_id')}
    old_hashes = {r.get('sha256') for r in old_extra if r.get('sha256')}
    for phase in ('train', 'planning', 'selection', 'test', 'field', 'case'):
        for r in read(prior / f'old_{phase}_rows.json'):
            if 'observation:' in r['group_id']:
                old_obs.add(r['group_id'].split('observation:')[-1])
            old_hashes.add(r['source_sha256'])
    for r in read(prior / 'user_import_manifest.json'):
        old_hashes.add(r['sha256'])
    positive = [r for r in rows if r['source_label_binary'] == 1]
    negative = [r for r in rows if r['source_label_binary'] == 0]
    checks = []
    for r in rows:
        assert r['human_review_status'] == 'pending' and r['training_eligible'] is False
        assert r['split'] == 'quarantine'
        assert sha256(r['source_image']) == r['sha256']
        checks.append({'id': r['id'], 'observation_id': r['observation_id'], 'observer_id': r['observer_id'],
                       'source_label_binary': r['source_label_binary'],
                       'existing_observation': str(r['observation_id']) in old_obs,
                       'existing_exact_image': r['sha256'] in old_hashes,
                       'observer_previously_in_aquilaria_sources': r['observer_id'] in known_aq_authors,
                       'location_obscured': r['location_obscured']})
    assert not any(r['existing_observation'] or r['existing_exact_image'] for r in checks)
    unseen = [r for r in positive if r['observer_id'] not in known_aq_authors]
    seen = [r for r in positive if r['observer_id'] in known_aq_authors]
    proposal = {
        'does_not_change_manifest_or_human_labels': True,
        'first_choice': 'Freeze current pilot and thresholds first. After human confirmation, '
                        'evaluate all newly downloaded observations as a prospective challenge. '
                        'Report the unseen-Aquilaria-author positive stratum separately. '
                        'Do not fit any of this challenge before this evaluation.',
        'strict_aquilaria_source_author_positive_ids': [r['id'] for r in unseen],
        'seen_aquilaria_source_author_positive_ids': [r['id'] for r in seen],
        'negative_challenge_ids': [r['id'] for r in negative],
        'future_training': 'If a later training round consumes any challenge images, those images '
                           'become development/regression permanently; acquire and lock another holdout. '
                           'All images of an observation or known same tree, and preferably author, stay together.',
        'not_claimed': ['whole-148-class author independence', 'new physical trees', 'location independence',
                        'no BioCLIP pretraining overlap', 'expert species confirmation before user review'],
    }
    summary = {
        'manifest_path': str(source), 'manifest_sha256': sha256(source),
        'photos': len(rows), 'positive_source_photos': len(positive), 'negative_source_photos': len(negative),
        'positive_observations': len({r['observation_id'] for r in positive}),
        'positive_authors': len({r['observer_id'] for r in positive}),
        'unseen_aquilaria_source_author_positive_photos': len(unseen),
        'unseen_aquilaria_source_authors': len({r['observer_id'] for r in unseen}),
        'seen_aquilaria_source_author_positive_photos': len(seen),
        'seen_aquilaria_source_authors': sorted({r['observer_id'] for r in seen}),
        'negative_observations': len({r['observation_id'] for r in negative}),
        'source_species_counts': dict(Counter(r['source_species'] for r in rows)),
        'all_aquilaria_locations_obscured': all(r['location_obscured'] for r in positive),
        'zero_fp_one_sided_95pct_binomial_upper_bound_if_all_negative_confirmed':
            1 - .05 ** (1 / len(negative)) if negative else None,
        'checks': checks, 'proposal_only': proposal,
        'limitation': 'Public obscured coordinates cannot establish site or single-tree independence; '
                      'old 148-species observer metadata and user-supplied images have incomplete authorship. '
                      'Grouping is source evidence, not independent physical tree identity.',
    }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('checks', 'proposal_only')}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
