"""Controlled refits of initial Hong Kong head recipe; no production switch.

Four predeclared recipes x three seeds. Accepted training data only. Every fit
physically backs up initial, frozen encoder and deployed weights. Validation
chooses epoch, then all recipes are reported on fixed regression data.
"""
from __future__ import annotations
import argparse
import gc
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'external_datasets/distillation_runtime')]
import numpy as np
import torch
import torch.nn.functional as F
import ablate_sstv1_accuracy_20261007 as audit
from bioclip_probe import ResidualHead, load_phase, scored, TEACHER
from distillation_core import collapse_logits, metrics, digest
from training_safety import backup_before_training, verify_backup, sha256

SEEDS = (20260909, 20260910, 20260911)
RECIPES = (
    {'name': 'control_mlp', 'architecture': 'mlp', 'balanced': True, 'auxiliary': True},
    {'name': 'no_class_balance', 'architecture': 'mlp', 'balanced': False, 'auxiliary': True},
    {'name': 'no_herbarium', 'architecture': 'mlp', 'balanced': True, 'auxiliary': False},
    {'name': 'linear_same_lr', 'architecture': 'linear', 'balanced': True, 'auxiliary': True},
)
CN = {'control_mlp': '完整初期训练配方（重跑对照）', 'no_class_balance': '移除类别均衡',
      'no_herbarium': '移除已审核标本图片', 'linear_same_lr': '改为线性残差头，同学习率'}


def save(path, value): audit.save(path, value)


def prepare(out, inference):
    out.mkdir(parents=True, exist_ok=False)
    assert (inference / 'production_verification.json').is_file(), 'First complete frozen-checkpoint inference audit'
    verify_backup(inference / 'pre_training_backup')
    training, _, _ = load_phase(audit.PROBE, 'train', device='cpu')
    planning, _, _ = load_phase(audit.PROBE, 'planning', device='cpu')
    selection, _, _ = load_phase(audit.PROBE, 'selection', device='cpu')
    dbpath = ROOT / 'external_datasets/herbarium_training_candidates_20260907/human_review.sqlite3'
    with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        decisions = {r['id']: dict(r) for r in db.execute('SELECT * FROM decisions')}
    excluded, accepted = [], []
    for i, row in enumerate(training):
        if row.get('auxiliary'):
            decision = decisions.get(row['sample_id'].removeprefix('herbarium_'), {})
            if decision.get('status') != 'accepted' or decision.get('sha256') != row['source_sha256']:
                excluded.append({'sample_id': row['sample_id'], 'reason': 'not_currently_human_accepted_matching_sha256'})
                continue
        accepted.append(i)
    retained = [training[i] for i in accepted]
    test_rows = audit.load(audit.SOURCE / 'test_rows.json')
    for held in (planning, selection, test_rows):
        assert not {r['group_id'] for r in retained} & {r['group_id'] for r in held}, 'Training/validation group overlap'
    evaluation_rows = audit.load(inference / 'rows.json')
    training_observations = {m for r in retained for m in re.findall(r'observation:(\d+)', r.get('group_id', ''))}
    evaluation_observations = {str(r.get('observation_id', r.get('inat_observation_id', ''))) for r in evaluation_rows}
    assert not training_observations & evaluation_observations, 'Training/evaluation observation overlap'
    assert not {r.get('source_sha256') for r in retained} & {r['sha256'] for r in evaluation_rows}, 'Training/evaluation SHA256 overlap'
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'seed_list': list(SEEDS), 'epochs': 80,
            'optimizer': 'AdamW', 'lr': .003, 'weight_decay': .01, 'gradient_clip': 10,
            'recipes': list(RECIPES), 'encoder_frozen': True, 'backbone_sha256': sha256(TEACHER),
            'temperature': audit.load(audit.PROBE / 'teacher_choice.json')['calibration_temperature'],
            'train_count': len(retained), 'auxiliary_count': sum(bool(r.get('auxiliary')) for r in retained),
            'nonauxiliary_count': sum(not r.get('auxiliary') for r in retained),
            'auxiliary_weight': 'x3 per image retained in all recipes where auxiliary data exists',
            'balanced': 'Each class 1/count weight, mean-normalized; no_class_balance disables only 1/count',
            'selection': 'Epoch0 or strict maximum planning Macro-F1. All three fixed seeds probability-averaged. Selection set recorded; no recipe picked using regression.',
            'inference_audit': str(inference), 'source_training': str(audit.PROBE),
            'scope': 'Attribution to initial head recipe, NOT a counterfactual removal from the final H120 continuation.',
            'training_data_note': 'Only the historical accepted training split, rechecked against current human_review.sqlite3 for auxiliary photos. Common398 and rare192 evaluation images never used for fitting.',
            'production_change': False}
    save(out / 'plan.json', plan); save(out / 'human_review_snapshot.json', decisions)
    save(out / 'accepted_indices.json', accepted); save(out / 'accepted_training_manifest.json', retained)
    save(out / 'excluded_training_rows.json', excluded)
    for phase, rows in (('planning', planning), ('selection', selection)):
        save(out / (phase + '_rows.json'), rows)
    # Heldout split assignments are snapshotted without scoring.
    save(out / 'test_rows.json', test_rows)
    save(out / 'classes.json', audit.load(audit.PROBE / 'classes.json'))
    stages = audit.load(inference / 'stages.json')
    protected = [TEACHER, *[Path(m['path']) for m in stages['h120']['members']],
                 Path(stages['h120']['manifest']), ROOT / 'spacescan_trees_demo_v1/release.json',
                 ROOT / 'spacescan_trees_demo_v1/server.py', ROOT / 'spacescan_trees_demo_v1/presentation_groups.py',
                 ROOT / 'tree_species_teacher_demo/engine.py', ROOT / 'tools/training_safety.py',
                 ROOT / 'tools/bioclip_probe.py', ROOT / 'tools/distillation_core.py', Path(__file__),
                 ROOT / 'tools/test_sstv1_component_ablation_20261007.py',
                 ROOT / 'tools/ablate_sstv1_accuracy_20261007.py', audit.PROBE / 'protocol.json',
                 audit.PROBE / 'classes.json', audit.PROBE / 'teacher_choice.json', audit.PROBE / 'cache_metadata.json',
                 audit.SOURCE / 'aliases.json', inference / 'features.npz', inference / 'inference_complete.json',
                 inference / 'rows.json', inference / 'results.json']
    for name in ('ptv2_seven_20260913_v1/model.joblib', 'ptv2_seven_20260913_v1/deployment.json'):
        protected.append(ROOT / 'spacescan_trees_demo_v1/models' / name)
    protected += [Path(m['path']) for m in stages['adapted_mlp']['members']]
    for phase in ('train', 'planning', 'selection', 'test'):
        protected += [audit.SOURCE / (phase + '_rows.json'), audit.SOURCE / ('teacher_cache/bioclip25/' + phase + '.npz')]
    save(out / 'protected_inputs.json', [str(p) for p in protected])
    backup_before_training(out, protected + list(out.glob('*.json')), {'operation': 'controlled_ablation_plan', **plan})
    print('PLAN BACKUP VERIFIED', out / 'pre_training_backup', flush=True)


def compact(m):
    return {k: m[k] for k in ('n', 'top1', 'macro_f1', 'balanced_accuracy', 'nll', 'mean_confidence', 'support', 'per_class_f1', 'per_class_recall')}


def training_weights(rows, balanced):
    counts = Counter(r['target'] for r in rows)
    w = torch.tensor([(3 if r.get('auxiliary') else 1)/(counts[r['target']] if balanced else 1) for r in rows], dtype=torch.float32)
    return w/w.mean()


def fit(out):
    verify_backup(out / 'pre_training_backup')
    if (out / 'fits.json').exists() or (out / 'runs').exists(): raise FileExistsError('Never resume/overwrite a fitting run')
    (out / 'runs').mkdir()
    plan = audit.load(out / 'plan.json'); classes = audit.load(out / 'classes.json')
    groups, temp = classes['groups'], plan['temperature']
    rows, x, z = load_phase(audit.PROBE, 'train'); ids = audit.load(out / 'accepted_indices.json')
    rows, x, z = [rows[i] for i in ids], x[ids], z[ids]
    planning, xv, zv = load_phase(audit.PROBE, 'planning')
    selection, xs, zs = load_phase(audit.PROBE, 'selection')
    yp = np.array([r['target'] for r in planning]); ys = np.array([r['target'] for r in selection])
    baseline = compact(metrics(yp, (collapse_logits(zv, groups)/temp).softmax(-1).cpu().numpy()))
    save(out / 'baseline_planning.json', baseline)
    fits = []
    for recipe in RECIPES:
        keep = [i for i, r in enumerate(rows) if recipe['auxiliary'] or not r.get('auxiliary')]
        rr, xt, zt = [rows[i] for i in keep], x[keep], z[keep]
        y = torch.tensor([r['target'] for r in rr], dtype=torch.long, device='cuda')
        w = training_weights(rr, recipe['balanced']).cuda()
        for seed in SEEDS:
            run = out / 'runs' / (recipe['name'] + '_seed_' + str(seed)); run.mkdir(exist_ok=False)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            head = ResidualHead(recipe['architecture']).cuda()
            initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            torch.save({'model_state': initial, 'encoder_sha256': plan['backbone_sha256'], 'classes': classes}, run / 'initial.pt')
            cfg = {**plan, **recipe, 'seed': seed, 'training_rows': len(rr), 'training_rows_sha256': digest(rr)}
            save(run / 'config.json', cfg); save(run / 'training_manifest.json', rr)
            save(run / 'baseline_evaluation.json', baseline)
            inputs = [Path(p) for p in audit.load(out / 'protected_inputs.json')] + list(out.glob('*.json')) + list(run.glob('*.json')) + [run / 'initial.pt']
            backup = backup_before_training(run, inputs, {'operation': 'controlled_classifier_fit', 'recipe': recipe['name'], 'seed': seed})
            print('FIT BACKUP VERIFIED', recipe['name'], seed, flush=True)
            # No optimizer or fitting step is created before successful verification.
            optimizer = torch.optim.AdamW(head.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
            best, best_epoch, best_state = baseline['macro_f1'], 0, initial
            history = []
            for epoch in range(1, plan['epochs']+1):
                head.train(); optimizer.zero_grad(set_to_none=True)
                logits = collapse_logits(head(xt, zt), groups)/temp
                loss = (F.cross_entropy(logits, y, reduction='none')*w).mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 10, error_if_nonfinite=True); optimizer.step()
                p = scored(head, xv, zv, groups, temp); m = metrics(yp, p)
                if m['macro_f1'] > best:
                    best, best_epoch = m['macro_f1'], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                history.append({'epoch': epoch, 'loss': float(loss.detach()), 'planning_macro_f1': m['macro_f1'], 'planning_top1': m['top1']})
            torch.save({'model_state': best_state, 'config': cfg, 'classes': classes, 'encoder_sha256': plan['backbone_sha256'], 'epoch': best_epoch}, run / 'best.pt')
            torch.save({'model_state': head.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': plan['epochs'], 'config': cfg}, run / 'last.pt')
            save(run / 'history.json', history)
            head.load_state_dict(best_state)
            ps = scored(head, xs, zs, groups, temp)
            np.save(run / 'selection_probabilities.npy', ps)
            item = {'recipe': recipe['name'], 'seed': seed, 'epoch': best_epoch, 'planning_macro_f1': best,
                    'selection': compact(metrics(ys, ps)), 'checkpoint': str(run / 'best.pt'), 'sha256': sha256(run / 'best.pt'),
                    'backup': str(backup)}
            save(run / 'result.json', item); verify_backup(backup); fits.append(item)
            print('FIT COMPLETE', recipe['name'], seed, 'epoch', best_epoch, 'selection top1', item['selection']['top1'], flush=True)
            del head, optimizer; gc.collect(); torch.cuda.empty_cache()
    save(out / 'fits.json', fits)
    save(out / 'selection_lock.json', {'fits': fits, 'all_predeclared_recipes_retained': True,
        'ensemble': 'mean of three fixed seeds, not best seed', 'regression_not_yet_scored': True,
        'no_candidate_selected_for_deployment': True})
    verify_backup(out / 'pre_training_backup')


def evaluate(out):
    verify_backup(out / 'pre_training_backup')
    lock = audit.load(out / 'selection_lock.json'); plan = audit.load(out / 'plan.json')
    inference = Path(plan['inference_audit']); classes = audit.load(out / 'classes.json')
    complete = audit.load(inference / 'inference_complete.json')
    assert sha256(inference / 'features.npz') == complete['features_sha256']
    rows = audit.load(inference / 'rows.json')
    with np.load(inference / 'features.npz') as f:
        assert f['row_ids'].tolist() == [r['ablation_id'] for r in rows]
        xx, zz, ee = [torch.tensor(f[k], device='cuda') for k in ('x', 'raw', 'extra')]
    test, xt, zt = load_phase(audit.PROBE, 'test')
    temp, groups = plan['temperature'], classes['groups']
    old_metrics, modes, seeds = {}, {}, {}
    with torch.inference_mode():
        for recipe in RECIPES:
            heads = []
            for run in lock['fits']:
                if run['recipe'] != recipe['name']: continue
                assert sha256(run['checkpoint']) == run['sha256']
                s = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
                h = ResidualHead(recipe['architecture']).cuda().eval().requires_grad_(False); h.load_state_dict(s['model_state']); heads.append(h)
            oldp = np.mean([scored(h, xt, zt, groups, temp) for h in heads], axis=0)
            old_metrics[recipe['name']] = compact(metrics([r['target'] for r in test], oldp))
            for j, view in enumerate(('center', 'whole')):
                def output(h):
                    return (torch.cat([collapse_logits(h(xx[:,j], zz[:,j]), groups), ee[:,j]], -1)/temp).softmax(-1).cpu().numpy()
                ps = np.stack([output(h) for h in heads])
                modes[recipe['name'] + '_' + view] = ps.mean(0)
                seeds[recipe['name'] + '_' + view] = ps
            modes[recipe['name'] + '_context'] = .5*(modes[recipe['name']+'_center'] + modes[recipe['name']+'_whole'])
            seeds[recipe['name'] + '_context'] = .5*(seeds[recipe['name']+'_center'] + seeds[recipe['name']+'_whole'])
    np.savez_compressed(out / 'regression_predictions.npz', **modes)
    np.savez_compressed(out / 'regression_seed_predictions.npz', **seeds)
    raw_labels = complete['labels']
    grouped = {k: audit.grouped_array(p, raw_labels)[0] for k,p in modes.items()}
    labels = audit.grouped_array(next(iter(modes.values())), raw_labels)[1]
    subsets = {'common398': [i for i,r in enumerate(rows) if r['cohort']=='common'],
               'aquilaria_primary': [i for i,r in enumerate(rows) if r['cohort']=='rare' and r['source_kind']=='community_observation' and r['label']=='Aquilaria sinensis']}
    result = {'plan': plan, 'old_test_regression': old_metrics, 'subsets': {}, 'training_performed': True, 'deployment_changed': False}
    for name, ids in subsets.items():
        rr = [rows[i] for i in ids]; stats, correct = {}, {}
        for mode, p in grouped.items(): stats[mode], correct[mode] = audit.score(rr, p[ids], labels)
        pairs = {r['name'] + '_vs_control_mlp': audit.paired(rr, correct[r['name']+'_context'], correct['control_mlp_context']) for r in RECIPES[1:]}
        result['subsets'][name] = {'n': len(rr), 'modes': stats, 'paired': pairs}
    save(out / 'results.json', result)
    lines = ['# 初期分类头训练组件消融', '', '同一冻结编码器、同一训练/验证分割，80步、AdamW、lr=0.003、wd=0.01、三个固定种子。每次拟合独立备份并校验，验证集选轮次，不用回归集选方案。', '',
             '| 对照 | 常见树双取景 Top-1 | 土沉香正类召回率 | 旧测试集 Top-1 |', '| --- | ---: | ---: | ---: |']
    for r in RECIPES:
        name = r['name']; c = result['subsets']['common398']['modes'][name+'_context']; aq = result['subsets']['aquilaria_primary']['modes'][name+'_context']
        lines.append(f"| {CN[name]} | {c['correct']}/{c['n']} · {c['top1']:.2%} | {aq['correct']}/{aq['n']} · {aq['top1']:.2%} | {old_metrics[name]['top1']:.2%} |")
    lines += ['', '| 移除组件的损失：完整配方减去消融配方 | 百分点 | 改对 / 改错 | 成组95%区间 |', '| --- | ---: | ---: | --- |']
    for key, s in result['subsets']['common398']['paired'].items():
        ci = s['cluster_ci95_pp']; lines.append(f"| {key} | {s['gain_pp']:+.2f} | {s['corrected']} / {s['regressed']} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines += ['', '正数意味着该组件在这套配方/数据上有贡献，负数意味着完整配方反而更差。不能把这些留一消融差值直接相加：它们可能有交互，且不是从最终 H120 各阶段重新完整重训。', '',
              f"训练图片 {plan['train_count']}，其中当前人工审核通过标本 {plan['auxiliary_count']}；拒绝/待审材料没有进入训练。去类别均衡仍保留原标本x3权重，去标本保持其余类别均衡策略。线性头保持优化器/学习率/步数，但结构自身含层数、归一化和Dropout差异。", '',
              '常见树398张以及土沉香数据已看过，仅作为冻结回归，不是新的无偏泛化证据。旧测试集同样仅是历史回归。所有预声明方案和种子均报告，不挑有利方案，不改正式Demo。', '',
              f"每次拟合的备份位于 {out / 'runs'} 各 seed 子目录的 pre_training_backup；外层方案备份位于 {out / 'pre_training_backup'}。",
              '完整逐类指标、种子结果、训练配置、原标签/分割、初始和最终权重均保留。实验权重仅本地研究用途，没有公开发布。']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for run in lock['fits']: verify_backup(run['backup'])
    verify_backup(out / 'pre_training_backup')
    save(out / 'production_verification.json', {'deployed_files_unchanged': True, 'all_fit_backups_verified': True, 'fit_count': len(lock['fits'])})
    print('CONTROLLED ABLATION COMPLETE; production unchanged', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('prepare','fit','evaluate','all'))
    p.add_argument('--out', type=Path, required=True); p.add_argument('--inference', type=Path, required=True)
    args = p.parse_args(); out = args.out.resolve()
    if args.phase in ('prepare','all'): prepare(out, args.inference.resolve())
    if args.phase in ('fit','all'): fit(out)
    if args.phase in ('evaluate','all'): evaluate(out)
