"""Re-authorize previously source-verified TRAIN rows with current review firewall.

Read identity metadata only for heldout data; no predictions or heldout rasters.
No model fitting, relabeling, DB mutations, or changes to old manifests.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from artocarpus_fresh_split_20260917 import HELD, record_identity, lookup_maps, historical_matches
from import_artocarpus_user_20260918 import current_human_review_exclusions, REGISTRY
from prepare_artocarpus_fit_20260918 import identities, row_ids
from training_safety import sha256

OLD = ROOT/'outputs/SST_Conserve_Browser_20260917_v3'
BASE = ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1'
OUT = ROOT/'outputs/conservation_fourclass_extra_audit_20260918_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    original = read(OLD/'new_manifest.json')
    live = current_human_review_exclusions()
    blocked_sha, blocked_ids = set(live['blocked_sha256']), set(live['blocked_ids'])
    reg = read(REGISTRY); maps, visual = lookup_maps(reg)
    held_metadata = read(BASE/'heldout_metadata.json')
    held_tokens = {t for r in held_metadata['records'] for t in r['identity_tokens']}
    observed_review = live['current_live_queue_decisions']
    held_keys = {'sha256', 'path', 'photo', 'sample', 'observation', 'image_url', 'group', 'gallery', 'near_phash'}
    accepted, excluded, audits, counts = [], [], [], defaultdict(Counter)
    for row in original:
        phase = row['phase']; species = row['source_species']
        counts[species][phase] += 1
        normalized = record_identity(row, OLD/'new_manifest.json')
        hits, ids, cached = historical_matches(normalized, reg, maps, visual, 6)
        content_ids = {i for k, vs in hits.items() if k in held_keys for i in vs}
        content_hold = sorted(i for i in content_ids if HELD & set(reg['records'][i]['phases']))
        observer_hold = sorted(i for i in hits.get('observer', []) if HELD & set(reg['records'][i]['phases']))
        token_conflicts = sorted(identities(row)&held_tokens)
        human = [r for r in observed_review if r.get('sha256') == row['sha256'] or r.get('id') in row_ids(row)]
        current_blocked = row['sha256'] in blocked_sha or bool(row_ids(row)&blocked_ids)
        integrity = sha256(row['source_image']) == row['sha256']
        reasons = []
        if phase != 'train': reasons.append('preserve_previously_used_development')
        if not integrity: reasons.append('source_content_changed')
        if current_blocked: reasons.append('actual_human_reject_pending_uncertain')
        if phase == 'train' and (content_hold or token_conflicts): reasons.append('historical_heldout_content_observation_or_group_overlap')
        if phase == 'train' and observer_hold: reasons.append('historical_heldout_observer_overlap')
        if row.get('label_authority') != 'institutional_source_plus_agent_visual_review_not_human_certified': reasons.append('unexpected_label_authority')
        if row.get('rights_scope') != 'local_noncommercial_research_only_no_distribution': reasons.append('unexpected_rights_scope')
        audit = dict(sample_id=row['sample_id'], species=species, original_phase=phase,
            integrity_verified=integrity, actual_human_queue_matches=human, actual_human_blocked=current_blocked,
            historical_heldout_content_record_ids=content_hold, historical_heldout_observer_record_ids=observer_hold,
            explicit_heldout_identity_tokens=token_conflicts,
            heldout_record_evidence=[dict(record_id=i, ids=reg['records'][i]['ids'], phases=reg['records'][i]['phases'],
                                       origins=reg['records'][i]['origins']) for i in sorted(set(content_hold)|set(observer_hold))],
            rejection_reasons=reasons)
        audits.append(audit)
        if phase != 'train': continue
        if reasons:
            excluded.append(dict(row, revalidation_reasons=reasons)); continue
        result = dict(row)
        result.update(phase='train', split='train', training_eligible=True,
            final_lane='accepted_existing_source_verified_training',
            human_review_status='no_actual_queue_decision_source_verified_only' if not human else 'existing_actual_decision_preserved',
            imported_human_review_status_placeholder=row.get('human_review_status'),
            review_status='agent_source_consistent_previously_authorized_local_experiment',
            human_acceptance=False, actual_human_review_match=bool(human),
            previously_used_for_training=True, new_image_claim=False,
            data_role='existing_institutional_train_only',
            revalidated_against_current_human_review=True,
            heldout_content_observation_group_conflicts=[], heldout_observer_conflicts=[],
            rights_limitation='Inherited local non-commercial research scope only; explicit ML licence not established; no new publication, Demo image storage or weight distribution authority.',
            source_manifest=str(OLD/'new_manifest.json'))
        accepted.append(result)
    assert len(accepted)+len(excluded) == 38
    assert len({r['sha256'] for r in accepted}) == len(accepted)
    assert not ({r['sha256'] for r in accepted} & {r['sha256'] for r in original if r['phase'] != 'train'})
    summary = dict(original_counts={k:dict(v) for k,v in counts.items()},
        accepted_counts=dict(Counter(r['source_species'] for r in accepted)), excluded_train_count=len(excluded),
        accepted_groups={s:sorted({r['group_id'] for r in accepted if r['source_species']==s}) for s in counts},
        development_groups={s:sorted({r['group_id'] for r in original if r['source_species']==s and r['phase']!='train'}) for s in counts},
        current_human_review_queue_matches=sum(bool(r['actual_human_queue_matches']) for r in audits),
        current_human_review_blocked_count=sum(r['actual_human_blocked'] for r in audits),
        development_history='The same15 source-held-out images have already been used for development/selection in v2/v3 and other optimization experiments; not a fresh independent test. Never move them into training.',
        independent_tree_identity_verified=False,
        independent_never_inspected_red_or_keteleeria_test_count=0,
        missing_positive_class='Aquilaria sinensis has no extra rows in this batch; retain215 existing accepted training rows from BASE separately.',
        rights_scope=read(OLD/'rights_scope.json'),
        label_boundary='Source plus prior agent visual consistency; NOT human taxonomic certification. Legacy pending strings were importer placeholders, not live decisions.',
        history_screen_limit='Recorded identity union + pHash distance<=6 + current live review SHA/ID. Unknown observers or cross-site reuse cannot be proven absent.',
        no_fit=True, no_test_prediction_access=True, old_files_unchanged=True)
    write(OUT/'extra_train_accepted.json', accepted)
    write(OUT/'excluded_train.json', excluded)
    write(OUT/'eligibility_audit.json', audits)
    write(OUT/'current_human_review_snapshot.json', live)
    write(OUT/'SUMMARY.json', summary)
    write(OUT/'input_contract.json', dict(artifacts=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in (
        OLD/'new_manifest.json', OLD/'user_training_authorization.json', OLD/'rights_scope.json', REGISTRY, BASE/'heldout_metadata.json', Path(__file__))]
        + [dict(path=r['path'], sha256=sha256(r['path'])) for r in live['databases']]))
    print(json.dumps({k:v for k,v in summary.items() if k in ('accepted_counts','excluded_train_count','accepted_groups','development_groups','current_human_review_queue_matches','current_human_review_blocked_count')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
