"""Read-only, hash-bound audit of actual conservation human review decisions."""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from training_safety import sha256

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'conservation_review/data/conservation_sources_20260917_v1/human_review.sqlite3'
MANIFEST = ROOT / 'external_datasets/conservation_sources_20260917_v1/review_manifest.json'
CRITICAL = 'conserve_inat_368821903'
CHOICES = {'aquilaria', 'not_aquilaria', 'uncertain', 'unusable'}


def main(output, compare_challenge=None):
    report = output.with_suffix('.md')
    if output.exists() or report.exists():
        raise FileExistsError('Audit output is versioned and must not overwrite a prior audit')
    db_before = sha256(DB)
    manifest_hash = sha256(MANIFEST)
    rows = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert len({r['id'] for r in rows}) == len(rows)
    with sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        decisions = {r['item_id']: dict(r) for r in connection.execute('SELECT * FROM decisions')}
        history = [dict(r) for r in connection.execute('SELECT * FROM history ORDER BY event_id')]
        metadata = {r['key']: r['value'] for r in connection.execute('SELECT * FROM metadata')}
        integrity = connection.execute('PRAGMA integrity_check').fetchone()[0]
        connection.rollback()
    current = []
    by_observation = defaultdict(list)
    for r in rows:
        d = decisions.get(r['id'])
        actual = sha256(r['source_image'])
        reasons = []
        if actual != r['sha256']:
            reasons.append('image_sha256_mismatch')
        if d and d['sha256'] != r['sha256']:
            reasons.append('stale_review_sha256')
        if d and d['decision'] not in CHOICES:
            reasons.append('invalid_decision_value')
        if d and d['reviewer_identity'] != 'human':
            reasons.append('not_recorded_as_human')
        valid = d is not None and not reasons
        status = d['decision'] if valid else 'pending'
        if not valid:
            reasons.append('no_valid_current_human_decision')
        elif status in ('uncertain', 'unusable'):
            reasons.append('human_' + status)
        row = {'id': r['id'], 'observation_id': r.get('observation_id'),
               'observer_id': r.get('observer_id'), 'source_species': r['source_species'],
               'source_label_binary': r['source_label_binary'], 'source_url': r['source_url'],
               'image_sha256': r['sha256'], 'actual_image_sha256': actual,
               'human_review': d, 'valid_current_human_decision': valid, 'status': status,
               'binary_target': 1 if status == 'aquilaria' else 0 if status == 'not_aquilaria' else None,
               'quarantine_reasons': reasons}
        current.append(row)
        if row['observation_id'] is not None:
            by_observation[str(row['observation_id'])].append(row)
    conflicts = []
    mixed = []
    for obs, members in by_observation.items():
        targets = {r['binary_target'] for r in members if r['binary_target'] is not None}
        if len(targets) > 1:
            conflicts.append({'observation_id': obs, 'members': [r['id'] for r in members],
                              'decisions': {r['id']: r['status'] for r in members}})
            for r in members:
                r['quarantine_reasons'].append('same_observation_binary_label_conflict')
        if len(members) > 1 and len({r['status'] for r in members}) > 1:
            mixed.append({'observation_id': obs, 'members': [r['id'] for r in members],
                          'decisions': {r['id']: r['status'] for r in members}})
    counts = Counter(r['status'] for r in current)
    usable = [r for r in current if r['binary_target'] is not None and not r['quarantine_reasons']]
    by_source = {}
    for species in sorted({r['source_species'] for r in current}):
        by_source[species] = dict(Counter(r['status'] for r in current if r['source_species'] == species))
    db_after = sha256(DB)
    assert manifest_hash == sha256(MANIFEST), 'Source manifest changed during read-only audit'
    summary = {
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'database': str(DB), 'database_open_mode': 'SQLite mode=ro; query_only=ON; read transaction',
        'database_sha256_before': db_before, 'database_sha256_after': db_after,
        'database_file_unchanged_during_audit': db_before == db_after,
        'database_integrity_check': integrity, 'database_metadata': metadata,
        'manifest': str(MANIFEST), 'manifest_sha256': manifest_hash,
        'manifest_count': len(rows), 'current_database_decisions': len(decisions),
        'decided_valid_count': sum(r['valid_current_human_decision'] for r in current),
        'pending_count': counts['pending'],
        'counts': {k: counts[k] for k in ('aquilaria', 'not_aquilaria', 'uncertain', 'unusable', 'pending')},
        'history_events': len(history),
        'decision_snapshot_sha256': hashlib.sha256(json.dumps(decisions, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'orphan_database_ids': sorted(set(decisions) - {r['id'] for r in rows}),
        'by_source_species': by_source,
        'same_observation_binary_conflicts': conflicts,
        'same_observation_mixed_status_not_necessarily_conflicts': mixed,
        'critical': next((r for r in current if r['id'] == CRITICAL), None),
        'critical_history': [r for r in history if r['item_id'] == CRITICAL],
        'usable_binary_after_isolation': {'photos': len(usable), 'positive': sum(r['binary_target'] == 1 for r in usable),
                                          'negative': sum(r['binary_target'] == 0 for r in usable),
                                          'observation_groups': len({r['observation_id'] for r in usable})},
        'isolate': [{'id': r['id'], 'reasons': r['quarantine_reasons']} for r in current if r['quarantine_reasons']],
        'all_current_rows': current,
        'scope': 'Actual database decisions only; pending is never accepted. Binary species review does not '
                 'grant rights, confirm physical-tree independence or authorize automatic training/deployment. '
                 'No model was run; source labels and database remain unmodified.'}
    if compare_challenge is not None:
        previous_cfg = json.loads((compare_challenge / 'protocol.json').read_text(encoding='utf-8'))
        previous_rows = json.loads((compare_challenge / 'challenge_rows.json').read_text(encoding='utf-8'))
        previous_db = compare_challenge / 'human_review.sqlite3'
        assert sha256(previous_db) == previous_cfg['database_snapshot_sha256']
        with sqlite3.connect(previous_db.resolve().as_uri() + '?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA query_only=ON')
            old_decisions = {r['item_id']: dict(r) for r in connection.execute('SELECT * FROM decisions')}
        current_map = {r['id']: r for r in current}
        previous_score_ids = {r['id'] for r in previous_rows}
        appended = sorted(set(decisions) - set(old_decisions))
        removed = sorted(set(old_decisions) - set(decisions))
        relabeled = []
        revised_metadata = []
        for item_id in sorted(set(old_decisions) & set(decisions)):
            a, b = old_decisions[item_id], decisions[item_id]
            if a['decision'] != b['decision'] or a['sha256'] != b['sha256']:
                relabeled.append({'id': item_id, 'previous_decision': a['decision'],
                                  'current_decision': b['decision'], 'image_hash_changed': a['sha256'] != b['sha256'],
                                  'previously_scored': item_id in previous_score_ids})
            elif a != b:
                revised_metadata.append({'id': item_id, 'decision_unchanged': b['decision'],
                                         'previous_revision': a['revision'], 'current_revision': b['revision'],
                                         'changed_fields': [k for k in a if a[k] != b[k]]})
        previous_score_changes = []
        for r in previous_rows:
            now = current_map[r['id']]
            if r['binary_target'] != now['binary_target'] or now['quarantine_reasons']:
                previous_score_changes.append({'id': r['id'], 'previous_truth': r['binary_target'],
                                               'current_truth': now['binary_target'],
                                               'quarantine_reasons': now['quarantine_reasons']})
        frozen_lock_path = compare_challenge / 'model_selection_lock.json'
        assert sha256(frozen_lock_path) == previous_cfg['selection_lock_sha256']
        frozen_lock = json.loads(frozen_lock_path.read_text(encoding='utf-8'))
        pilot = Path(previous_cfg['pilot_run'])
        chosen = frozen_lock['chosen']
        weight_hash = sha256(chosen['path'])
        lock_unchanged = sha256(pilot / 'selection_lock.json') == previous_cfg['selection_lock_sha256']
        weights_unchanged = weight_hash == chosen['sha256'] == previous_cfg['candidate_sha256']
        assert lock_unchanged and weights_unchanged, 'Previously locked candidate or selection changed'
        first_history = min((r['created_at'] for r in history), default=None)
        lock_time = datetime.fromtimestamp((pilot / 'selection_lock.json').stat().st_mtime, timezone.utc).isoformat()
        protected = json.loads((pilot / 'contract.json').read_text(encoding='utf-8'))['artifacts']
        for a in protected:
            assert sha256(a['path']) == a['sha256'], 'Protected deployment artifact changed: ' + a['path']
        summary['comparison_to_previous_challenge'] = {
            'previous_challenge': str(compare_challenge), 'previous_eligible': len(previous_rows),
            'previous_database_decisions': len(old_decisions), 'newly_appended_decision_count': len(appended),
            'newly_appended_decision_ids': appended,
            'appended_decision_counts': dict(Counter(decisions[i]['decision'] for i in appended)),
            'removed_decision_ids': removed, 'changed_labels_or_source_hashes': relabeled,
            'same_label_revisions_or_note_changes': revised_metadata,
            'previous_scored_rows_changed_or_newly_isolated': previous_score_changes,
            'previous_scored_rows_all_retained_unchanged': not previous_score_changes,
        }
        summary['frozen_candidate_audit'] = {
            'pilot_run': str(pilot), 'selection_lock_sha256': previous_cfg['selection_lock_sha256'],
            'selection_lock_unchanged': lock_unchanged, 'candidate_sha256': weight_hash,
            'candidate_unchanged': weights_unchanged, 'selection_lock_mtime_utc': lock_time,
            'first_human_history_event_utc': first_history,
            'selection_lock_precedes_first_human_event': bool(first_history and datetime.fromisoformat(lock_time) < datetime.fromisoformat(first_history)),
            'thresholds': previous_cfg['thresholds_fixed_before_new_labels'],
            'protected_deployment_artifacts_verified_unchanged': len(protected),
            'no_model_loaded_or_run': True,
        }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    critical = summary['critical']
    lines = ['# 保育图像真实人工复核结果审计', '',
             f"清单 {len(rows)} 张；有效已决策 {summary['decided_valid_count']} 张，待审 {counts['pending']} 张。",
             f"土沉香确认 {counts['aquilaria']}；非土沉香 {counts['not_aquilaria']}；无法确定 {counts['uncertain']}；质量不可用 {counts['unusable']}。", '',
             f"关键样本 `{CRITICAL}`：{critical['status'] if critical else '不在当前清单'}。",
             f"同一 observation 的相反二分类标签冲突：{len(conflicts)} 组；混合状态（例如确认＋不可用，不一定是标签冲突）：{len(mixed)} 组。", '',
             f"隔离无效/未定/不可用/冲突项后，可用二分类记录 {len(usable)} 张：正例 {summary['usable_binary_after_isolation']['positive']}，负例 {summary['usable_binary_after_isolation']['negative']}。",
             '这是供已冻结候选进行新来源挑战的有效标签数，不自动转为训练许可或独立实测树数。', '', '## 按源网站类别核对', '',
             '| 来源类别 | 土沉香 | 非土沉香 | 不确定 | 不可用 | 待审 |', '|---|---:|---:|---:|---:|---:|']
    for species, cs in by_source.items():
        lines.append('| ' + species + ' | ' + ' | '.join(str(cs.get(k, 0)) for k in ('aquilaria','not_aquilaria','uncertain','unusable','pending')) + ' |')
    lines += ['', '## 必须隔离', '']
    lines += [f"- `{r['id']}`：{', '.join(r['reasons'])}" for r in summary['isolate']] or ['- 无。']
    if compare_challenge is not None:
        comparison = summary['comparison_to_previous_challenge']
        frozen = summary['frozen_candidate_audit']
        lines += ['', '## 与前一评估快照比较', '',
                  f"前版评分 {comparison['previous_eligible']} 张；前版数据库 {comparison['previous_database_decisions']} 条决策。本轮新增 {comparison['newly_appended_decision_count']} 条：{comparison['appended_decision_counts']}。",
                  f"改标或源哈希改变 {len(comparison['changed_labels_or_source_hashes'])} 条；同标签备注/revision变化 {len(comparison['same_label_revisions_or_note_changes'])} 条；删除决策 {len(comparison['removed_decision_ids'])} 条。",
                  f"前版已评分样本全部保留且真值不变：{comparison['previous_scored_rows_all_retained_unchanged']}。扩展集包含旧样本，不能把前42张和最终全量当作两个独立测试累计。", '',
                  '## 固定候选检查', '',
                  f"选择锁和候选权重均未改变：{frozen['selection_lock_unchanged'] and frozen['candidate_unchanged']}；选择锁 {frozen['selection_lock_mtime_utc']} 早于首个人审 {frozen['first_human_history_event_utc']}。",
                  f"阈值仍固定为 {frozen['thresholds']}。候选 SHA-256：`{frozen['candidate_sha256']}`。",
                  f"另有 {frozen['protected_deployment_artifacts_verified_unchanged']} 个受保护部署文件哈希未变。未加载或执行模型。"]
    lines += ['', f"数据库完整性：{integrity}；数据库文件审计前后哈希相同：{db_before == db_after}。",
              '数据库以只读连接和一致性读事务查询，108张图片均核对源哈希；未写回数据库、清单、标签或模型。',
              '详细当前决策、备注、revision、关键样本历史与冲突组保存在同名 JSON；root 执行前应以自己的事务快照再次核验，避免用户后续修改造成时点差异。']
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k in ('manifest_count','decided_valid_count','pending_count','counts','same_observation_binary_conflicts','usable_binary_after_isolation','isolate')}, ensure_ascii=True))
    print('CRITICAL', json.dumps(critical, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-challenge', type=Path)
    args = parser.parse_args()
    main(args.output, args.compare_challenge.resolve() if args.compare_challenge else None)
