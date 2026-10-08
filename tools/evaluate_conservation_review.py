"""Freeze human decisions and evaluate a PREVIOUSLY locked model. Never fit or deploy."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from training_safety import sha256
from distillation_core import read_json, save_json, digest


def prepare(run, manifest, database, out):
    if not (run / 'selection_lock.json').is_file():
        raise ValueError('A previously selected model is required')
    lock = read_json(run / 'selection_lock.json')
    assert sha256(lock['chosen']['path']) == lock['chosen']['sha256']
    out.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as source:
        with sqlite3.connect(out / 'human_review.sqlite3') as target:
            source.backup(target)
    with sqlite3.connect(out / 'human_review.sqlite3') as db:
        db.row_factory = sqlite3.Row
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        decisions = {r['item_id']: dict(r) for r in db.execute('SELECT * FROM decisions')}
    shutil.copy2(manifest, out / 'source_manifest.json')
    shutil.copy2(run / 'selection_lock.json', out / 'model_selection_lock.json')
    rows = read_json(out / 'source_manifest.json')
    shutil.copy2(ROOT / 'outputs/conservation_new_source_group_audit_20260917_v1.json', out / 'source_group_audit.json')
    audit = {r['id']: r for r in read_json(out / 'source_group_audit.json')['checks']}
    counts = Counter(); accepted = []; excluded = []
    for row in rows:
        d = decisions.get(row['id'])
        state = d['decision'] if d else 'pending'
        counts[state] += 1
        if not d or state not in ('aquilaria', 'not_aquilaria'):
            excluded.append({'id': row['id'], 'reason': state})
            continue
        assert d['sha256'] == row['sha256'] == sha256(row['source_image'])
        assert d['reviewer_identity'] == 'human'
        check = audit[row['id']]
        assert not check['existing_observation'] and not check['existing_exact_image']
        accepted.append({**row, 'binary_target': int(state == 'aquilaria'), 'human_review': d,
                         'previously_seen_aquilaria_author': check['observer_previously_in_aquilaria_sources'],
                         'split': 'locked_prospective_challenge', 'training_eligible': False})
    # A contradictory observation is not quietly resolved with a model prediction.
    by_group = defaultdict(set)
    for r in accepted:
        by_group[r['group_id']].add(r['binary_target'])
    conflicts = {g for g, labels in by_group.items() if len(labels) > 1}
    excluded += [{'id': r['id'], 'reason': 'conflicting_human_labels_within_observation'} for r in accepted if r['group_id'] in conflicts]
    accepted = [r for r in accepted if r['group_id'] not in conflicts]
    # Recheck exact image overlap against every fixed pilot split independently.
    old_hashes = {r.get('sha256') or r.get('source_sha256') for p in ('train', 'planning', 'test') for r in read_json(run / f'{p}_rows.json')}
    assert not old_hashes.intersection(r['sha256'] for r in accepted)
    assert len({r['sha256'] for r in accepted}) == len(accepted)
    if not accepted:
        raise ValueError('No unambiguous human-confirmed rows')
    save_json(out / 'challenge_rows.json', accepted)
    save_json(out / 'excluded_rows.json', excluded)
    protocol = {
        'prepared_at': datetime.now(timezone.utc).isoformat(), 'stage': 'before_challenge_inference',
        'pilot_run': str(run), 'database_source': str(database),
        'database_snapshot_sha256': sha256(out / 'human_review.sqlite3'),
        'source_manifest_sha256': sha256(out / 'source_manifest.json'), 'rows_sha256': sha256(out / 'challenge_rows.json'),
        'source_group_audit_sha256': sha256(out / 'source_group_audit.json'),
        'selection_lock_sha256': sha256(out / 'model_selection_lock.json'),
        'candidate_sha256': lock['chosen']['sha256'], 'human_decisions': dict(counts),
        'eligible_photos': len(accepted), 'positive_photos': sum(r['binary_target'] for r in accepted),
        'observation_groups': len({r['group_id'] for r in accepted}), 'conflicting_groups': sorted(conflicts),
        'thresholds_fixed_before_new_labels': {'baseline': lock['baseline_threshold'], 'specialist': lock['chosen']['planning']['threshold']},
        'primary_comparison': 'Both use old-planning-fixed thresholds; same unmodified image preprocessing. No new threshold tuning.',
        'no_training_or_model_selection_on_this_challenge': True, 'automatic_deployment': False,
        'limitations': ['Source observation is not proven unique physical tree.', 'Author independence only checked against historical Aquilaria sources, not whole encoder pretraining.',
                       'Source candidate batch, not random street prevalence; image-label review is not exact localization.',
                       'Pending, uncertain and conflicting labels never used. New batch remains reserved from training.',
                       'All positive-source locations are obscured; no claim of new-site independence.']}
    save_json(out / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), out / 'evaluation_code.py')
    print(json.dumps(protocol, ensure_ascii=True), flush=True)


def metrics(rows):
    from conservation_pilot import binary_metrics
    if not rows:
        return {'n': 0}
    y = np.array([r['truth'] for r in rows])
    result = {}
    for key in ('baseline', 'specialist', 'baseline_argmax'):
        m = binary_metrics(y, np.array([r[key + '_positive'] for r in rows], float), .5)
        m.pop('threshold')
        if m['positive'] == 0:
            m['recall'] = None
        if m['negative'] == 0:
            m['fpr'] = None
        if m['tp'] + m['fp'] == 0:
            m['precision'] = None
        m['binary_accuracy'] = (m['tp'] + m['tn']) / m['n']
        result[key] = m
    return result


def evaluate(out):
    import torch
    from conservation_pilot import AQ, odds
    from train_sstv1_incremental import prepare_pixels, logits
    from tree_species_teacher_demo.engine import Engine
    if (out / 'predictions.json').exists() or (out / 'features.npz').exists():
        raise FileExistsError('Immutable evaluation already started; use a new version for a new snapshot')
    cfg = read_json(out / 'protocol.json'); rows = read_json(out / 'challenge_rows.json')
    assert sha256(out / 'human_review.sqlite3') == cfg['database_snapshot_sha256']
    assert sha256(out / 'source_manifest.json') == cfg['source_manifest_sha256']
    if 'source_group_audit_sha256' in cfg:
        assert sha256(out / 'source_group_audit.json') == cfg['source_group_audit_sha256']
    assert sha256(out / 'challenge_rows.json') == cfg['rows_sha256']
    assert sha256(out / 'model_selection_lock.json') == cfg['selection_lock_sha256']
    lock = read_json(out / 'model_selection_lock.json'); chosen = lock['chosen']
    assert sha256(chosen['path']) == chosen['sha256'] == cfg['candidate_sha256']
    saved = torch.load(chosen['path'], map_location='cpu', weights_only=False)
    engine = Engine(); engine.load()
    assert engine.sstv1_version == saved['baseline_deployment']['version']
    assert engine.encoder_sha256 == saved['encoder_sha256']
    # Match fixed-evaluation CPU head numerics (the expensive image encoder runs on GPU).
    head = torch.nn.Linear(1024, 1).eval(); head.load_state_dict(saved['model_state'])
    predictions = []; xs = []; ps = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            assert sha256(row['source_image']) == row['sha256']
            pixels = prepare_pixels(row['source_image'])[None].cuda()
            e = engine.model.encode_image(((pixels - engine.mean) / engine.std).to(engine.dtype), normalize=True).float()
            z = e @ engine.bank.T * engine.model.logit_scale.exp().float(); x = e.half().float()
            p = torch.stack([logits(h, x, z, engine.classes['groups'], engine.choice['calibration_temperature'], engine.extra_bank, engine.model.logit_scale.exp().float()).softmax(-1) for h in engine.sstv1_heads]).mean(0)[0].cpu().numpy()
            xx = x.cpu(); baseline = float(p[AQ])
            specialist = float(torch.sigmoid(torch.tensor(odds(np.array([baseline], dtype=np.float32))) + head(xx).squeeze(1))[0])
            top = np.argsort(p)[::-1][:3]
            predictions.append({'id': row['id'], 'sha256': row['sha256'], 'source_image': row['source_image'],
                'source_species': row['source_species'], 'group_id': row['group_id'], 'observer_id': row['observer_id'],
                'previously_seen_aquilaria_author': row['previously_seen_aquilaria_author'], 'truth': row['binary_target'],
                'baseline_score': baseline, 'specialist_score': specialist,
                'baseline_positive': baseline >= lock['baseline_threshold'], 'specialist_positive': specialist >= saved['threshold'],
                'baseline_argmax_positive': int(top[0]) == AQ,
                'baseline_top3': [{'species': engine.labels[int(j)], 'score': float(p[j])} for j in top]})
            xs.append(xx.numpy()[0]); ps.append(p)
            if i % 10 == 0:
                print('SCORE', i, len(rows), flush=True)
    np.savez_compressed(out / 'features.npz', x=np.array(xs), p=np.array(ps), rows_hash=digest(rows), encoder_sha256=engine.encoder_sha256)
    save_json(out / 'predictions.json', predictions)
    report = {'all_confirmed': metrics(predictions),
              'previously_unseen_aquilaria_author': metrics([r for r in predictions if not r['previously_seen_aquilaria_author']]),
              'previously_seen_aquilaria_author': metrics([r for r in predictions if r['previously_seen_aquilaria_author']]),
              'retuned': False, 'trained_on_challenge': False, 'deployed': False}
    # Equal observation weight sensitivity, no claim of independent trees.
    for branch in ('baseline', 'specialist'):
        grouped = defaultdict(list)
        for r in predictions:
            grouped[(r['truth'], r['group_id'])].append(float(r[branch + '_positive']))
        report[branch + '_equal_observation_rates'] = {
            str(y): float(np.mean([np.mean(v) for (truth, _), v in grouped.items() if truth == y]))
            for y in (0, 1) if any(truth == y for truth, _ in grouped)}
    for artifact in read_json(Path(cfg['pilot_run']) / 'contract.json')['artifacts']:
        assert sha256(artifact['path']) == artifact['sha256']
    report['production_artifacts_unchanged'] = True
    save_json(out / 'evaluation.json', report)
    lines = ['# 人工复核后的土沉香专项模型检验', '',
        f"本快照包含 {len(rows)} 张明确人工判断；来自 {cfg['observation_groups']} 个来源观察组，不等于已确认独立树。", '',
        '这批图片未用于拟合、选模型或调阈值。按审核快照筛选，不根据模型结果删图。来源标签与您的判断分开保存。', '',
        '| 分支 | 土沉香检出 | 非土沉香误报 | 二分类正确 |', '|---|---:|---:|---:|']
    for key, title in [('baseline', '当前模型（旧验证集阈值）'), ('specialist', '专用复核器（旧验证集阈值）'), ('baseline_argmax', '当前148类首选，仅参考')]:
        m = report['all_confirmed'][key]
        lines.append(f"| {title} | {m['tp']}/{m['positive']} | {m['fp']}/{m['negative']} | {m['tp']+m['tn']}/{m['n']} |")
    lines += ['', '不能将少量负例上的误报率外推成实际街道误报率；不能将源观察组直接当作不同树。两种阈值分支才是主要公平比较，不能只与148类首选比较后声称训练提升。', '',
              '未审核及不确定图片均未评分。该快照后新增决策须另建版本，并保留当前结果。模型及阈值保持锁定；生产Demo、图像模型、点云模型均未修改。', '',
              '逐图证据：predictions.json；分组结果：evaluation.json；可恢复人工决策快照：human_review.sqlite3；评估约定：protocol.json。']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['prepare', 'evaluate'], required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--run', type=Path)
    p.add_argument('--manifest', type=Path); p.add_argument('--database', type=Path)
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a.run.resolve(), a.manifest.resolve(), a.database.resolve(), a.out.resolve())
    else:
        evaluate(a.out.resolve())
