"""One isolated, user-requested fine-tuning trial with immutable review snapshot."""
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from inventory_herbarium_images import ROOT, norm, write_json
from collect_herbarium_candidates import OUT
from training_safety import sha256, verify_backup

SCRIPTS = ROOT / 'full_image_species_baseline/scripts'
BASE = ROOT / 'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1'
INITIAL = BASE / 'pretrain/weights/selected_by_val_group_macro_f1.pt'


def canonical(query):
    if query in ('Araucaria columnaris', 'Araucaria heterophylla'):
        return norm('Araucaria columnaris(Araucaria heterophylla)')
    return norm(query)


def main():
    trial = OUT / 'experiments' / ('yolo_aux_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    trial.mkdir(parents=True, exist_ok=False)
    print('TRIAL=' + str(trial), flush=True)
    sys.path.insert(0, str(ROOT / 'tree_species_demo'))
    from herbarium_review_api import reviewed_rows
    reviewed = reviewed_rows()
    write_json(trial / 'human_review_snapshot.json', reviewed)
    accepted = [r for r in reviewed if r['human_review']['status'] == 'accepted']
    accepted_ids = {r['id'] for r in accepted}
    summary = json.loads((BASE / 'datasets/full/dataset_summary.json').read_text(encoding='utf-8'))
    names = summary['class_names']; lookup = {norm(n): i for i, n in enumerate(names)}
    with (BASE / 'datasets/full/inventory.csv').open(encoding='utf-8', newline='') as f:
        old = list(csv.DictReader(f))
    cross = json.loads((OUT / 'cross_dataset_duplicates.json').read_text(encoding='utf-8'))
    cross_matches = {r['id']: r['matches'] for r in cross['candidate_results']}
    included, excluded = [], []
    for r in accepted:
        why = None
        if r.get('duplicate_of') in accepted_ids:
            why = 'Same photograph already included via ' + r['duplicate_of']
        if r['id'] not in cross_matches or cross_matches[r['id']]:
            why = 'Missing or unresolved cross-dataset duplicate check'
        if canonical(r['query']) not in lookup:
            why = 'No existing output class; do not silently change classifier dimensions'
        if sha256(OUT / r['file']) != r['sha256']:
            raise RuntimeError('Accepted image changed: ' + r['id'])
        if why:
            excluded.append({'id': r['id'], 'reason': why}); continue
        included.append({**r, 'class_index': lookup[canonical(r['query'])]})
    if not included:
        raise RuntimeError('No eligible accepted images')
    # No accepted herbarium image is added to val/test. All old splits are preserved.
    dataset = trial / 'dataset'; dataset.mkdir()
    from_import = str(SCRIPTS)
    sys.path.insert(0, from_import)
    from materialize_yolo_cls import safe_class
    inventory = []
    for r in old:
        splitdir = 'held_out_test' if r['split'] == 'test' else 'training_view/' + r['split']
        dest = dataset / splitdir / safe_class(int(r['class_index']), r['species']) / Path(r['materialized_path']).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(r['materialized_path'], dest)
        inventory.append({**r, 'materialized_path': str(dest)})
    for r in included:
        idx = r['class_index']; src = OUT / r['file']
        dest = dataset / 'training_view/train' / safe_class(idx, names[idx]) / ('herbarium_' + r['id'] + '.jpg')
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
        inventory.append({'sample_id': 'herbarium_' + r['id'], 'split': 'train', 'class_index': idx, 'species': names[idx],
                          'group_id': 'herbarium_taxon_' + r['source_species_id'], 'source_image': str(src),
                          'materialized_path': str(dest), 'source_sha256': r['sha256']})
    with (dataset / 'inventory.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(old[0])); writer.writeheader(); writer.writerows(inventory)
    write_json(dataset / 'dataset_summary.json', {**summary, 'split_sizes': dict(Counter(r['split'] for r in inventory)),
                                               'source_manifest': str(trial / 'human_review_snapshot.json'),
                                               'source_manifest_sha256': sha256(trial / 'human_review_snapshot.json')})
    for filename in ['final_metrics.json', 'args.yaml', 'environment.json']:
        shutil.copy2(BASE / 'pretrain' / filename, trial / ('baseline_' + filename))
    config = {'status': 'prepared', 'trial': str(trial), 'review_counts': dict(Counter(r['human_review']['status'] for r in reviewed)),
              'accepted': len(accepted), 'new_training_images': len(included), 'excluded_accepted': excluded,
              'new_images_by_class': dict(Counter(names[r['class_index']] for r in included)),
              'initial_checkpoint': str(INITIAL), 'initial_sha256': sha256(INITIAL),
              'epochs': 3, 'learning_rate': 0.00005, 'seed': 20260907, 'auxiliary_sample_multiplier': 3,
              'original_split_manifest_sha256': sha256(BASE / 'datasets/full/inventory.csv'),
              'new_split_manifest_sha256': sha256(dataset / 'inventory.csv'),
              'use_scope': 'User-requested local experimental training; no distribution, no license grant inferred from human review.',
              'rights_caveat': 'Source notice does not explicitly address ML training; no commercial/distribution rights established.',
              'selection': 'Validation group macro-F1; test only after checkpoint selection; existing test is a regression set, not unseen field evidence.',
              'production_switch': False, 'point_or_fusion_training': False}
    write_json(trial / 'trial_config.json', config)
    write_json(trial / 'included_herbarium.json', included)
    write_json(trial / 'status.json', {'stage': 'training', 'trial': str(trial)})
    run = trial / 'candidate'
    env = dict(os.environ, TREE_HERBARIUM_SAMPLE_WEIGHT='3', PYTHONUTF8='1', MPLBACKEND='Agg')
    cmd = [sys.executable, str(SCRIPTS / 'train_yolo11_cls.py'), '--dataset-root', str(dataset), '--run-dir', str(run),
           '--model', str(INITIAL), '--epochs', '3', '--batch-size', '96', '--image-size', '224', '--seed', '20260907',
           '--learning-rate', '0.00005', '--warmup-epochs', '0.5', '--warmup-bias-lr', '0.00005']
    with (trial / 'training.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(trial / 'status.json', {'stage': 'training_failed', 'exit_code': result.returncode})
        raise RuntimeError('Training failed; inspect ' + str(trial / 'training.log'))
    finish(trial)


def finish(trial):
    trial = Path(trial)
    if (trial / 'comparison.json').exists():
        raise FileExistsError('Completed comparison already exists; do not overwrite')
    dataset, run = trial / 'dataset', trial / 'candidate'
    names = json.loads((dataset / 'dataset_summary.json').read_text(encoding='utf-8'))['class_names']
    included = json.loads((trial / 'included_herbarium.json').read_text(encoding='utf-8'))
    accepted = [r for r in json.loads((trial / 'human_review_snapshot.json').read_text(encoding='utf-8')) if r['human_review']['status']=='accepted']
    env = dict(os.environ, PYTHONUTF8='1', MPLBACKEND='Agg')
    sys.path.insert(0, str(SCRIPTS))
    verify_backup(run / 'pre_training_backup')
    write_json(trial / 'status.json', {'stage': 'validation_selection_and_regression_evaluation'})
    with (trial / 'evaluation.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(SCRIPTS / 'evaluate_yolo11_cls.py'), '--dataset-root', str(dataset), '--run-dir', str(run)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Recompute baseline on the exact same files and current software, not just historical scores.
    from evaluate_yolo11_cls import rows_for, predict, metric_values, group_metric_values, write_predictions
    baseline = {}
    for split in ['val', 'test']:
        selected_rows = rows_for(dataset, split)
        probs = predict(INITIAL, selected_rows, 224, 192)
        baseline[split + '_sample'] = metric_values(selected_rows, probs, names)
        baseline[split + '_group'] = group_metric_values(selected_rows, probs, names)
        write_predictions(trial / ('baseline_' + split + '_predictions.csv'), selected_rows, probs, names)
    new = json.loads((run / 'final_metrics.json').read_text(encoding='utf-8'))
    comparison = {'baseline': baseline, 'candidate': new['metrics'], 'delta': {
        scope: {k: new['metrics'][scope][k]-baseline[scope][k] for k in ['accuracy', 'macro_f1']} for scope in baseline},
        'selected_epoch': new['selection']['epoch'], 'production_unchanged': True,
        'backup': str(run / 'pre_training_backup'), 'deploy': False}
    verify_backup(run / 'pre_training_backup')
    write_json(trial / 'comparison.json', comparison)
    lines = ['# 圖像模型增量訓練對照', '', f'人工保留 {len(accepted)} 張；實際新增入訓 {len(included)} 張。原有訓練資料全部回放；驗證／測試切分不變。',
             '', '| 評估範圍 | 原模型準確率 | 候選準確率 | 原 Macro-F1 | 候選 Macro-F1 |', '|---|---:|---:|---:|---:|']
    for scope in baseline:
        a, b = baseline[scope], new['metrics'][scope]
        lines.append(f'| {scope} | {a["accuracy"]:.4%} | {b["accuracy"]:.4%} | {a["macro_f1"]:.4f} | {b["macro_f1"]:.4f} |')
    lines += ['', '候選僅依驗證集 group Macro-F1 選取，並未自動部署。既有測試集已有歷史使用，屬固定回歸對照，不是全新街景盲測；本輪没有點雲或融合精度提升證據。',
              '', '所有部署權重及相關設定已重新驗證雜湊，與訓練前相同。備份：`candidate/pre_training_backup/backup_manifest.json`。',
              '', '完整逐類 precision / recall / F1 與混淆矩陣見 comparison.json；不以新圖訓練命中率代替測試精度。']
    (trial / 'training_comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(trial / 'status.json', {'stage': 'complete', 'production_unchanged': True})
    print(json.dumps({'trial': str(trial), 'delta': comparison['delta'], 'backup': comparison['backup']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--evaluate-completed':
        trial = Path(sys.argv[2]).resolve()
        if not trial.is_relative_to((OUT / 'experiments').resolve()):
            raise ValueError('Recovery must target an isolated herbarium experiment')
        config = json.loads((trial / 'trial_config.json').read_text(encoding='utf-8'))
        with (trial / 'candidate/results.csv').open(encoding='utf-8', newline='') as f:
            epoch_rows = list(csv.DictReader(f))
        if [int(r['epoch']) for r in epoch_rows] != list(range(1, config['epochs']+1)):
            raise RuntimeError('Incomplete training epochs; cannot recover as completed')
        log = (trial / 'training.log').read_text(encoding='utf-8')
        if 'self.plot_metrics()' not in log or 'RecursionError' not in log:
            raise RuntimeError('Recovery only covers the inspected post-training plotting error')
        import torch
        checkpoints = []
        for epoch in range(config['epochs']):
            path = trial / 'candidate/weights' / f'epoch{epoch}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != epoch:
                raise RuntimeError('Checkpoint epoch mismatch')
            model = checkpoint.get('ema') or checkpoint['model']
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite checkpoint parameters')
            checkpoints.append({'path': str(path), 'epoch': epoch, 'sha256': sha256(path)})
            del checkpoint, model
        write_json(trial / 'postprocessing_recovery.json', {'training_repeated': False, 'reason': 'All 3 epochs completed; only final PySide/polars plotting recursed.', 'verified_checkpoints': checkpoints})
        write_json(trial / 'candidate/training_complete.json', {'status':'epochs_complete_postprocessing_recovered', 'epochs':config['epochs'], 'original_process_exit_code':1, 'details':'../postprocessing_recovery.json', 'test_loaded_during_training':False})
        finish(trial)
    else:
        main()
