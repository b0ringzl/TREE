"""Controlled native-148 continuation; immutable backups, no production writes.

Development selects one checkpoint/pipeline before historical regression runs.
Two views mean two deterministic crops of one image, not independent trees.
"""
import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
import torch
import torch.nn.functional as F
from distillation_core import read_json, save_json, digest, bootstrap_gain
from training_safety import backup_before_training, verify_backup, sha256
from import_artocarpus_user_20260918 import current_human_review_exclusions
from prepare_artocarpus_fit_20260918 import identities, row_ids
from train_artocarpus_user_20260918 import (
    protected, heads_from_deployment, logprob, predict, measure, cpu_state,
    SSTV1_MANIFEST, SOURCE, TEACHER)

PREV = ROOT / 'outputs/SSTv1_Artocarpus_User_20260918_v1'
AUDIT = ROOT / 'outputs/artocarpus_preprocess_audit_20260917_v1'
DEFAULT = ROOT / 'outputs/SSTv1_AQ_AR_Diagnostic_20260919_v1'
SEED = 20260919
RECIPES = [dict(name='A_small_lr', lr=3e-5, focus=1.),
           dict(name='B_larger_lr', lr=1e-4, focus=1.),
           dict(name='C_focus_weight', lr=1e-4, focus=4.)]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def tensor(a):
    return torch.tensor(np.asarray(a, dtype=np.float32), device='cuda')


def load_data(directory, phase):
    rows = read_json(directory / f'{phase}_rows.json')
    with np.load(directory / f'{phase}_features.npz', allow_pickle=False) as f:
        require(str(f['rows_hash']) == digest(rows), 'Cache ordering: ' + phase)
        return rows, tensor(f['x']), tensor(f['z'])


def verify_images(rows):
    def check(r):
        expected = r.get('sha256') or r.get('source_sha256') or r['audit_sha256']
        require(sha256(r['source_image']) == expected, 'Image content changed: ' + r['source_image'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, rows))


def evaluate_dev(rows, p):
    return dict(combined=measure(rows, p), old=measure(rows[:2640], p[:2640]),
                new=measure(rows[2640:], p[2640:]))


def eligible(candidate, baseline):
    for cohort in ('combined', 'old', 'new'):
        for species in ('aquilaria', 'white'):
            a, b = candidate[cohort][species], baseline[cohort][species]
            if a['tp'] < b['tp'] or a['fp'] > b['fp'] or a['f1'] < b['f1']-1e-12:
                return False
    a, b = candidate['combined']['all'], baseline['combined']['all']
    return a['correct'] >= b['correct'] and a['macro_f1'] >= b['macro_f1']-.001


def rank(m):
    a, b = m['combined']['aquilaria'], m['combined']['white']
    return ((a['f1'] + b['f1'])/2, min(a['recall'], b['recall']),
            m['combined']['all']['correct'], m['combined']['all']['macro_f1'])


def unchanged(protocol):
    for path, expected in protocol['protected_sha256'].items():
        require(sha256(path) == expected, 'Protected deployment changed: ' + path)


def user_training_replay(out, rows, before, after):
    usage = read_json(ROOT / 'outputs/user_folder_split_audit_20260919_v1/file_usage.json')
    # The SHA match avoids relying on unstable import IDs or file names.
    if isinstance(usage, dict):
        usage = usage.get('files', usage.get('rows', []))
    hashes = {r.get('sha256') for r in usage}
    ix = [i for i, r in enumerate(rows) if r['sha256'] in hashes]
    labels = read_json(out / 'labels.json')
    records = [dict(sample_id=rows[i]['sample_id'], source_image=rows[i]['source_image'],
                    target=labels[rows[i]['target']], baseline=labels[int(before[i].argmax())],
                    candidate=labels[int(after[i].argmax())],
                    baseline_true_score=float(before[i, rows[i]['target']]),
                    candidate_true_score=float(after[i, rows[i]['target']])) for i in ix]
    save_json(out / 'user_training_replay.json', dict(
        note='Training replay, not validation; letterbox input for both models, no confidence calibration claim',
        n=len(ix), baseline=measure([rows[i] for i in ix], before[ix]),
        candidate=measure([rows[i] for i in ix], after[ix]), images=records))


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    deployment = read_json(SSTV1_MANIFEST)
    review = current_human_review_exclusions()
    save_json(out / 'human_review_snapshot.json', review)
    old_backup = read_json(PREV / 'fit/pre_training_backup/backup_manifest.json')
    old_hashes = {str(Path(r['source']).resolve()): r['sha256'] for r in old_backup['files']}
    require(sha256(PREV/'train_features.npz') == old_hashes[str((PREV/'train_features.npz').resolve())],
            'Training feature cache differs from verified original backup')
    train = load_data(PREV, 'train')
    blocked = [r['sample_id'] for r in train[0]
               if r['sha256'] in review['blocked_sha256'] or row_ids(r) & set(review['blocked_ids'])]
    require(not blocked, 'Review firewall rejected train rows: ' + str(blocked))
    heldout = read_json(PREV / 'heldout_metadata.json')
    held_ids = set().union(*(set(r['identity_tokens']) for r in heldout['records']))
    require(not any(identities(r) & held_ids for r in train[0]), 'Train/heldout identity overlap')
    devrows = read_json(AUDIT / 'rows.json')
    require(len(devrows) == 2655, 'Changed development set')
    verify_images(train[0] + devrows)
    contract = read_json(PREV / 'feature_contract.json')
    require(contract['encoder_sha256'] == deployment['encoder_sha256'], 'Encoder cache differs')
    require(sha256(AUDIT / 'paired_features_predictions.npz') == contract['validation_cache_sha256'], 'Dev cache changed')
    for name in ('train_rows.json', 'heldout_metadata.json', 'labels.json', 'classes.json',
                 'teacher_choice.json', 'feature_contract.json', 'data_summary.json'):
        shutil.copy2(PREV / name, out / name)
    save_json(out / 'development_rows.json', devrows)
    with np.load(AUDIT / 'paired_features_predictions.npz', allow_pickle=False) as f:
        require(str(f['rows_hash']) == digest(devrows), 'Dev cache row mismatch')
        dev = (devrows, tensor(f['demo_letterbox_x']), tensor(f['demo_letterbox_z']))
        center = (devrows, tensor(f['center_crop_x']), tensor(f['center_crop_z']))
    with np.load(PREV / 'train_features.npz', allow_pickle=False) as f:
        params = dict(groups=read_json(out / 'classes.json')['groups'], temp=deployment['temperature'],
                      extra=tensor(f['extra_bank']), scale=float(f['scale']))
        require(str(f['encoder_sha256']) == deployment['encoder_sha256'], 'Train encoder differs')
    protocol = dict(created_at=stamp(), seed=SEED, recipes=RECIPES, steps=120, evaluate_every=10,
        deployment_manifest=deployment, temperature=deployment['temperature'],
        train_count=len(train[0]), review_blocked_train=blocked, weight_decay=.01, kl_weight=1.,
        trainable='Only existing three native148 residual MLP heads; encoder/text bank/temperature fixed',
        weighting='Equal class then source-group; C multiplies both focal class weights by4 then renormalizes',
        anchor='KL to starting mean ensemble on ALL training rows; focus multiplier affects CE only',
        views=['letterbox', 'fixed mean(.5 letterbox + .5 center crop probabilities)'],
        scope='Two crops of one image, not independent trees or physical multiview observations',
        gate='For combined/old/new dev cohorts both focal TP>=base FP<=base F1>=base; overall correct>=base macroF1>=base-.001',
        rank='Mean focal F1, minimum focal recall, overall correct, macro F1; strict improvement; earlier/simple tie wins',
        inference_cost='Dual view requires two encoder passes; single view unchanged',
        selection='Baseline is valid; one global checkpoint and view locked before regression. No automatic deployment.',
        evaluation='Historical regression and inspected dev, NOT fresh unbiased generalization; user photos are TRAIN only',
        protected_sha256={str(p.resolve()): sha256(p) for p in protected()},
        cache_sha256={str(p.resolve()): sha256(p) for p in [PREV/'train_features.npz', AUDIT/'paired_features_predictions.npz',
            PREV/'test_features.npz', SOURCE/'teacher_cache/bioclip25/test.npz']})
    save_json(out / 'protocol.json', protocol)
    heads = heads_from_deployment(protocol)
    initial = cpu_state(heads)
    torch.save(dict(model_state=initial), out / 'initial.pt')
    bt, bp, bc = [predict(heads, d, params) for d in (train, dev, center)]
    with np.load(PREV / 'fit/selected_predictions.npz') as f:
        require(float(np.max(np.abs(bt-f['train']))) < 5e-5, 'Current deployed/train baseline mismatch')
    baseline = evaluate_dev(devrows, bp)
    dualbase = evaluate_dev(devrows, (bp+bc)*.5)
    save_json(out / 'baseline_evaluation.json', dict(train=measure(train[0],bt), letterbox=baseline, dual=dualbase))
    np.savez_compressed(out / 'baseline_predictions.npz', train=bt, dev_pad=bp, dev_center=bc)
    print('BASELINE', json.dumps(baseline['combined']), flush=True)
    print('DUAL_BASELINE', json.dumps(dualbase['combined']), flush=True)
    best = dict(recipe='unchanged', epoch=0, view='letterbox', metrics=baseline, checkpoint=str(out/'initial.pt'))
    if eligible(dualbase, baseline) and rank(dualbase) > rank(baseline):
        best.update(view='dual', metrics=dualbase)
    gc = Counter((r['target'], r['group_id']) for r in train[0])
    cg = defaultdict(set)
    for r in train[0]:
        cg[r['target']].add(r['group_id'])
    baseweights = np.array([1/(len(cg[r['target']])*gc[(r['target'],r['group_id'])]) for r in train[0]])
    y = torch.tensor([r['target'] for r in train[0]], device='cuda')
    anchor = tensor(bt)
    histories = []
    for recipe in RECIPES:
        run = out / recipe['name']; run.mkdir(exist_ok=False)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
        heads = heads_from_deployment(protocol)
        save_json(run / 'config.json', dict(**recipe, seed=SEED, steps=120, initializer_sha256=sha256(out/'initial.pt')))
        inputs = [*protected(), *sorted(out.glob('*.json')), out/'initial.pt', out/'baseline_predictions.npz',
                  PREV/'train_features.npz', AUDIT/'paired_features_predictions.npz',
                  run/'config.json', Path(__file__), ROOT/'tools/training_safety.py']
        backup = backup_before_training(run, inputs, dict(recipe=recipe, seed=SEED, no_resume=True))
        print('BACKUP_VERIFIED', recipe['name'], backup, flush=True)
        weights = baseweights*np.where(np.isin(y.cpu().numpy(),[11,16]),recipe['focus'],1.)
        weights = tensor(weights/weights.sum())
        optimizer = torch.optim.AdamW(heads.parameters(), lr=recipe['lr'], weight_decay=.01)
        hist = []
        for step in range(1,121):
            heads.train(); optimizer.zero_grad(set_to_none=True)
            lp = logprob(heads, train[1], train[2], **params)
            ce = (F.nll_loss(lp,y,reduction='none')*weights).sum()
            kl = F.kl_div(lp,anchor,reduction='batchmean')
            loss = ce+kl
            require(bool(torch.isfinite(loss)), 'Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(heads.parameters(),5.,error_if_nonfinite=True); optimizer.step()
            if step%10:
                continue
            pad, cen = [predict(heads,d,params) for d in (dev,center)]
            item = dict(step=step, loss=float(loss), ce=float(ce), kl=float(kl), views={})
            for view,p in [('letterbox',pad),('dual',(pad+cen)*.5)]:
                m = evaluate_dev(devrows,p); allowed = eligible(m,baseline)
                item['views'][view] = dict(metrics=m,eligible=allowed)
                if allowed and rank(m)>rank(best['metrics']):
                    checkpoint = run/f'step_{step:03d}.pt'
                    if not checkpoint.exists():
                        torch.save(dict(model_state=cpu_state(heads),step=step,recipe=recipe),checkpoint)
                    best = dict(recipe=recipe['name'],epoch=step,view=view,metrics=m,checkpoint=str(checkpoint))
            hist.append(item)
            print('FIT', recipe['name'], step, 'AQ',item['views']['dual']['metrics']['combined']['aquilaria'],
                  'AR',item['views']['dual']['metrics']['combined']['white'], 'BEST',best['recipe'],best['epoch'],best['view'],flush=True)
        torch.save(dict(model_state=cpu_state(heads),step=120,recipe=recipe),run/'last.pt')
        save_json(run/'history.json',hist)
        save_json(run/'last_train_evaluation.json',measure(train[0],predict(heads,train,params)))
        histories.append(dict(recipe=recipe,history=hist,backup=str(backup)))
        verify_backup(backup); unchanged(protocol)
    best.update(locked_at=stamp(), sha256=sha256(best['checkpoint']), no_test_predictions_used=True, deployed=False)
    save_json(out/'selection_lock.json',best)
    print('SELECTION_LOCKED',best['recipe'],best['epoch'],best['view'],flush=True)
    base = heads_from_deployment(protocol)
    selected = heads_from_deployment(protocol)
    selected.load_state_dict(torch.load(best['checkpoint'],map_location='cpu',weights_only=False)['model_state'])
    user_training_replay(out,train[0],bt,predict(selected,train,params))
    test = load_data(PREV,'test')
    verify_images(test[0])
    rows_orig = read_json(SOURCE/'test_rows.json')
    require([r['sample_id'] for r in test[0]]==[r['sample_id'] for r in rows_orig], 'Test ordering differs')
    meta = read_json(SOURCE/'teacher_cache/bioclip25/metadata.json')
    require(meta['weight_sha256']==deployment['encoder_sha256'], 'Test center encoder differs')
    require(meta['class_names']==read_json(out/'classes.json')['original_names'], 'Test raw class mapping differs')
    choice = read_json(out/'teacher_choice.json')
    with np.load(SOURCE/'teacher_cache/bioclip25/test.npz',allow_pickle=False) as f:
        require(str(f['rows_hash'])==digest(rows_orig), 'Test center row digest')
        tcenter = (test[0],tensor(f['embeddings']),tensor(f['similarities'][choice['prompt']]*meta['logit_scale']))
    before, before_c, after, after_c = [predict(h,d,params) for h,d in [(base,test),(base,tcenter),(selected,test),(selected,tcenter)]]
    final = after if best['view']=='letterbox' else (after+after_c)*.5
    yt = np.array([r['target'] for r in test[0]])
    a,b = before.argmax(1),final.argmax(1)
    result = dict(baseline=measure(test[0],before,False), baseline_dual=measure(test[0],(before+before_c)*.5,False),
        candidate_single=measure(test[0],after,False),candidate=measure(test[0],final,False),
        corrected=int(((a!=yt)&(b==yt)).sum()),regressed=int(((a==yt)&(b!=yt)).sum()),
        bootstrap=bootstrap_gain(yt,before,final,[r['group_id'] for r in test[0]],seed=SEED,repeats=2000),
        note='Previously inspected historical regression, not fresh test; legacy center cache may have encoder batching/precision differences')
    save_json(out/'regression_results.json',result)
    np.savez_compressed(out/'regression_predictions.npz',baseline=before,baseline_center=before_c,candidate_pad=after,candidate_center=after_c,selected=final,targets=yt)
    labels = read_json(out/'labels.json')
    details=[]
    for i,r in enumerate(test[0]):
        if r['target'] in (11,16) or a[i] in (11,16) or b[i] in (11,16):
            details.append(dict(sample_id=r['sample_id'],source_image=r['source_image'],group_id=r['group_id'],
                truth=labels[r['target']],baseline=labels[int(a[i])],candidate=labels[int(b[i])],
                baseline_true_score=float(before[i,r['target']]),candidate_true_score=float(final[i,r['target']])))
    save_json(out/'focal_regression_audit.json',details)
    unchanged(protocol)
    save_json(out/'COMPLETE.json',dict(completed_at=stamp(),trained=True,optimizer_steps=360,
        selected=best,production_unchanged=True,deployed=False,backups=[x['backup'] for x in histories]))
    print('REGRESSION',json.dumps({k:result[k] for k in ('baseline','baseline_dual','candidate_single','candidate','corrected','regressed')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=DEFAULT)
    main(parser.parse_args().out.resolve())
