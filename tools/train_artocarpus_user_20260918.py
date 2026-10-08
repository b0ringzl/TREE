"""One warm-start, jointly selected three-MLP fit; no production writes.

Frozen BioCLIP encoder, unchanged 148-class contract. All image features use
the actual Demo letterbox/JPEG preparation. Epoch zero is a valid outcome.
"""
import argparse
import gc
import hashlib
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import numpy as np
import torch
import torch.nn.functional as F
from audit_artocarpus_preprocessing_20260917 import exact_demo_prepare, protected_paths
from bioclip_probe import ResidualHead, SOURCE, TEACHER
from distillation_core import (bootstrap_gain, collapse_logits, digest, metrics,
                               read_json, save_json)
from train_sstv1_incremental import prepare_pixels
from training_safety import backup_before_training, deployed_artifacts, sha256, verify_backup
from tree_species_teacher_demo.engine import Engine, SSTV1_MANIFEST, decode

OUT = ROOT / 'outputs/SSTv1_Artocarpus_User_20260918_v1'
AUDIT = ROOT / 'outputs/artocarpus_preprocess_audit_20260917_v1'
SEED = 20260918


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def protected():
    more = [ROOT / 'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib',
            ROOT / 'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json',
            ROOT / 'spacescan_trees_demo_v1/point_recognition.py']
    return list(dict.fromkeys([*protected_paths(), *deployed_artifacts(), *more]))


def unchanged(protocol):
    for path, expected in protocol['protected_sha256'].items():
        require(sha256(path) == expected, 'Protected artifact changed: ' + path)


def freeze_protocol(out):
    require(not (out / 'protocol.json').exists(), 'Protocol already frozen')
    required = ['train_rows.json', 'planning_rows.json', 'newdev_rows.json',
                'heldout_metadata.json', 'labels.json', 'classes.json',
                'teacher_choice.json', 'human_review_snapshot.json', 'contract.json',
                'selection_manifest.json', 'data_summary.json']
    require(all((out / x).is_file() for x in required), 'Preparation is incomplete')
    dep = read_json(SSTV1_MANIFEST)
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(), seed=SEED,
        epochs=60, lr=3e-5, weight_decay=.01, kl_weight=1., clip_norm=5.,
        architecture='Three existing residual MLPs, jointly continued and selected at the same epoch',
        trainable='Only MLP parameters; frozen BioCLIP 2.5 backbone, text banks and temperature',
        ensemble='Arithmetic mean of 3 calibrated member probabilities; logsumexp(log_softmax) for loss',
        weighting='Equal class weight, then equal source-group weight inside a class; no separate positive oversampling',
        loss='source/class-balanced ensemble negative log likelihood + unweighted replay KL to starting ensemble',
        preprocessing='Actual Demo prepare + decode, pixel equality with helper checked for EVERY re-encoded image',
        feature_batch_size=1, feature_x='normalized embedding e rounded float16 then float32 for MLP',
        feature_z='raw text logits computed from e.float BEFORE float16 rounding',
        temperature=dep['temperature'], deployed_version=dep['version'],
        encoder_sha256=dep['encoder_sha256'],
        selection_gates=['old planning white TP>=baseline, FP<=baseline, F1>=baseline',
                         'old planning correct count>=baseline, macroF1>=baseline-0.001',
                         'old planning Aquilaria TP>=baseline and FP<=baseline',
                         'fixed new-source development white TP>=baseline and FP<=baseline'],
        selection_rank='Among eligible epochs, strictly maximize (mean old/new white F1, old correct count, old macroF1), earlier tie wins; epoch0 allowed',
        checkpoint='One shared epoch for all three heads; no test/selection split tuning',
        heldout='Previously inspected old test is regression evidence, not fresh unbiased accuracy',
        user_data='User supplied target-species photos are train only, not proof of generalization; copies retain watermarks',
        no_resume='Any new fit/resume must create a NEW run directory and verified backup',
        prepared_files={x: sha256(out / x) for x in required},
        protected_sha256={str(p.resolve()): sha256(p) for p in protected()},
        deployment_manifest=dep, deployed=False)
    save_json(out / 'protocol.json', protocol)
    return protocol


def check_prepared(out, protocol):
    for name, expected in protocol['prepared_files'].items():
        require(sha256(out / name) == expected, 'Prepared input changed: ' + name)
    for artifact in read_json(out / 'contract.json')['artifacts']:
        require(sha256(artifact['path']) == artifact['sha256'], 'Preparation source changed: ' + artifact['path'])


def pixel_input(row, production):
    path = Path(row['source_image'])
    blob = path.read_bytes()
    expected = row.get('sha256') or row.get('source_sha256')
    require(hashlib.sha256(blob).hexdigest() == expected, 'Source image changed: ' + str(path))
    prepared, _, _ = production(blob)
    pixel = decode(prepared)[0]
    require(torch.equal(pixel, prepare_pixels(path)), 'Training/Demo pixel mismatch: ' + row['sample_id'])
    return pixel


def encode_rows(engine, rows, directory):
    directory.mkdir(exist_ok=False)
    production = exact_demo_prepare()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for start in range(0, len(rows), 64):
            part = rows[start:start+64]
            pixels = list(pool.map(lambda r: pixel_input(r, production), part))
            xx, zz = [], []
            for pixel in pixels:
                e = engine.model.encode_image(((pixel[None].cuda()-engine.mean)/engine.std).to(engine.dtype), normalize=True).float()
                z = e @ engine.bank.T * engine.model.logit_scale.exp().float()
                xx.append(e.half().cpu().numpy()); zz.append(z.cpu().numpy())
            np.savez_compressed(directory / f'{start:06d}.npz', x=np.concatenate(xx), z=np.concatenate(zz), rows_hash=digest(part))
            print('FEATURES', directory.name, start+len(part), '/', len(rows), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    xx, zz = [], []
    for start in range(0, len(rows), 64):
        with np.load(directory / f'{start:06d}.npz', allow_pickle=False) as cache:
            require(str(cache['rows_hash']) == digest(rows[start:start+64]), 'Chunk ordering differs')
            xx.append(cache['x']); zz.append(cache['z'])
    x, z = np.concatenate(xx), np.concatenate(zz)
    require(np.isfinite(x).all() and np.isfinite(z).all(), 'Nonfinite feature')
    return x, z


def engine_check(engine, protocol):
    require(engine.sstv1_state == 'ready' and engine.sstv1_version == protocol['deployed_version'], 'Deployed initialization changed')
    require(engine.encoder_sha256 == protocol['encoder_sha256'], 'Encoder changed')


def features(out):
    protocol = freeze_protocol(out)
    rows = read_json(out / 'train_rows.json')
    engine = Engine(); engine.load(); engine_check(engine, protocol)
    require(engine.labels == read_json(out / 'labels.json'), 'Labels differ')
    x, z = encode_rows(engine, rows, out / 'train_feature_chunks')
    np.savez_compressed(out / 'train_features.npz', x=x, z=z, rows_hash=digest(rows),
                        extra_bank=engine.extra_bank.detach().cpu().numpy(),
                        scale=float(engine.model.logit_scale.exp()), encoder_sha256=engine.encoder_sha256)
    audit_rows = read_json(AUDIT / 'rows.json')
    audit_protocol = read_json(AUDIT / 'protocol.json')
    require(audit_protocol['deployed_version'] == protocol['deployed_version'], 'Validation cache model mismatch')
    require(audit_protocol['batch_size'] == 1, 'Validation batch mismatch')
    require(audit_protocol['protected_sha256_before'][str(TEACHER)] == protocol['encoder_sha256'], 'Validation encoder mismatch')
    ids = {r['sample_id']: i for i, r in enumerate(audit_rows)}
    require(len(ids) == len(audit_rows), 'Validation IDs repeated')
    with np.load(AUDIT / 'paired_features_predictions.npz', allow_pickle=False) as cache:
        require(str(cache['rows_hash']) == digest(audit_rows), 'Validation rows digest mismatch')
        for phase in ('planning', 'newdev'):
            rr = read_json(out / f'{phase}_rows.json'); indices = []
            for r in rr:
                j = ids[r['sample_id']]; a = audit_rows[j]
                require(r['target'] == a['target'] and r['group_id'] == a['group_id'], 'Validation metadata changed')
                actual = sha256(r['source_image'])
                require(actual == a['audit_sha256'] == r['sha256'], 'Validation content changed')
                indices.append(j)
            np.savez_compressed(out / f'{phase}_features.npz',
                x=cache['demo_letterbox_x'][indices], z=cache['demo_letterbox_z'][indices],
                audit_baseline=cache['demo_letterbox_p148'][indices], rows_hash=digest(rr))
    save_json(out / 'feature_contract.json', dict(encoder_sha256=engine.encoder_sha256,
        preprocessing=protocol['preprocessing'], feature_batch=1, train_rows_hash=digest(rows),
        validation_cache=str(AUDIT / 'paired_features_predictions.npz'),
        validation_cache_sha256=sha256(AUDIT / 'paired_features_predictions.npz'),
        all_reencoded_pixels_equal_production=True, train_count=len(rows)))
    del engine; gc.collect(); torch.cuda.empty_cache()
    unchanged(protocol); check_prepared(out, protocol)
    print('FEATURE EXTRACTION COMPLETE', len(rows), flush=True)


def heads_from_deployment(protocol):
    heads = torch.nn.ModuleList()
    for item in protocol['deployment_manifest']['members']:
        path = SSTV1_MANIFEST.parent / item['file']
        require(sha256(path) == item['sha256'], 'Initialization changed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        require(state['config']['architecture'] == 'mlp', 'Not a warm-start MLP')
        h = ResidualHead('mlp').cuda(); h.load_state_dict(state['model_state'])
        heads.append(h)
    return heads


def logprob(heads, x, z, groups, temp, extra, scale):
    tail = x @ extra.T * scale
    member = torch.stack([torch.cat([collapse_logits(h(x, z), groups), tail], -1)/temp for h in heads])
    return torch.logsumexp(member.log_softmax(-1), dim=0)-math.log(len(heads))


def load_data(out, phase):
    rows = read_json(out / f'{phase}_rows.json')
    with np.load(out / f'{phase}_features.npz', allow_pickle=False) as cache:
        require(str(cache['rows_hash']) == digest(rows), 'Feature manifest changed')
        x = torch.tensor(cache['x'].astype(np.float32), device='cuda')
        z = torch.tensor(cache['z'], device='cuda')
    return rows, x, z


def predict(heads, data, params):
    heads.eval()
    with torch.no_grad():
        return torch.cat([logprob(heads, data[1][i:i+256], data[2][i:i+256], **params).exp()
                          for i in range(0, len(data[0]), 256)]).cpu().numpy()


def binary(y, p, target):
    truth = np.asarray(y) == target; pred = p.argmax(1) == target
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum()); tn = int((~truth & ~pred).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def measure(rows, p, compact=True):
    y = np.array([r['target'] for r in rows]); mask = y < 148
    m = metrics(y[mask], p[mask])
    if compact:
        m = {k: m[k] for k in ('n', 'top1', 'macro_f1', 'balanced_accuracy')}
    m['correct'] = int((p[mask].argmax(1) == y[mask]).sum())
    return dict(all=m, white=binary(y, p, 16), aquilaria=binary(y, p, 11))


def eligible(old, new, baseline, base_new):
    a, b = old['white'], baseline['white']; c, d = old['aquilaria'], baseline['aquilaria']
    return (a['tp'] >= b['tp'] and a['fp'] <= b['fp'] and a['f1'] >= b['f1']-1e-12
        and old['all']['correct'] >= baseline['all']['correct']
        and old['all']['macro_f1'] >= baseline['all']['macro_f1']-.001
        and c['tp'] >= d['tp'] and c['fp'] <= d['fp']
        and new['white']['tp'] >= base_new['white']['tp'] and new['white']['fp'] <= base_new['white']['fp'])


def rank(old, new):
    return ((old['white']['f1']+new['white']['f1'])/2., old['all']['correct'], old['all']['macro_f1'])


def cpu_state(heads):
    return {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}


def fit(out):
    protocol = read_json(out / 'protocol.json'); unchanged(protocol); check_prepared(out, protocol)
    require(not (out / 'fit').exists(), 'Fit exists: resumes require a new version and backup')
    run = out / 'fit'; run.mkdir()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
    heads = heads_from_deployment(protocol)
    classes = read_json(out / 'classes.json')
    with np.load(out / 'train_features.npz', allow_pickle=False) as f:
        params = dict(groups=classes['groups'], temp=protocol['temperature'],
            extra=torch.tensor(f['extra_bank'], device='cuda'), scale=float(f['scale']))
    train, val, new = [load_data(out, p) for p in ('train', 'planning', 'newdev')]
    bv = predict(heads, val, params); bn = predict(heads, new, params); bt = predict(heads, train, params)
    for name, p in [('planning', bv), ('newdev', bn)]:
        with np.load(out / f'{name}_features.npz') as f:
            require(np.max(np.abs(p-f['audit_baseline'])) < 5e-5, 'Baseline cache probability mismatch')
    baseval, basenew = measure(val[0], bv), measure(new[0], bn)
    save_json(run / 'baseline_evaluation.json', dict(planning=baseval, newdev=basenew, train=measure(train[0], bt)))
    np.savez_compressed(run / 'baseline_predictions.npz', train=bt, planning=bv, newdev=bn)
    initial = cpu_state(heads)
    torch.save(dict(model_state=initial, epoch=0, protocol=protocol, classes=classes), run / 'initial.pt')
    inputs = [*protected(), run / 'initial.pt', run / 'baseline_evaluation.json', run / 'baseline_predictions.npz',
              *sorted(out.glob('*.json')), *sorted(out.glob('*_features.npz')),
              *[Path(a['path']) for a in read_json(out / 'contract.json')['artifacts']],
              Path(__file__), ROOT / 'tools/prepare_artocarpus_fit_20260918.py',
              ROOT / 'tools/import_artocarpus_user_20260918.py', ROOT / 'tools/training_safety.py']
    backup = backup_before_training(run, inputs, dict(model='SSTv1-Artocarpus-user', seed=SEED, fit_attempt=1))
    print('BACKUP VERIFIED', backup, flush=True)
    for r in train[0]:
        require(sha256(r['source_image']) == r['sha256'], 'Train source changed before fit')
    anchor = torch.tensor(bt, device='cuda')
    replay_mask = torch.tensor([r['data_role'] == 'existing_replay' for r in train[0]], device='cuda')
    require(int(replay_mask.sum()) == len(train[0])-len(read_json(out / 'accepted_user_rows.json')), 'Replay anchor mask mismatch')
    y = torch.tensor([r['target'] for r in train[0]], device='cuda')
    group_counts = Counter((r['target'], r['group_id']) for r in train[0])
    class_groups = defaultdict(set)
    for r in train[0]: class_groups[r['target']].add(r['group_id'])
    weights = torch.tensor([1/(len(class_groups[r['target']])*group_counts[(r['target'],r['group_id'])]) for r in train[0]], device='cuda')
    weights /= weights.sum()
    heads.requires_grad_(True)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=protocol['lr'], weight_decay=protocol['weight_decay'])
    best_state, best_epoch, best_rank = initial, 0, rank(baseval, basenew)
    history = []
    for epoch in range(1, protocol['epochs']+1):
        heads.train(); optimizer.zero_grad(set_to_none=True)
        lp = logprob(heads, train[1], train[2], **params)
        ce = (F.nll_loss(lp, y, reduction='none')*weights).sum()
        kl = F.kl_div(lp[replay_mask], anchor[replay_mask], reduction='batchmean')
        loss = ce+protocol['kl_weight']*kl
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(heads.parameters(), protocol['clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        pv = predict(heads, val, params); pn = predict(heads, new, params)
        mv, mn = measure(val[0], pv), measure(new[0], pn)
        allowed = eligible(mv, mn, baseval, basenew)
        if allowed and rank(mv, mn) > best_rank:
            best_rank, best_epoch, best_state = rank(mv, mn), epoch, cpu_state(heads)
        history.append(dict(epoch=epoch, loss=float(loss), ce=float(ce), kl=float(kl),
                            planning=mv, newdev=mn, eligible=bool(allowed), selected_so_far=best_epoch))
        if epoch % 5 == 0:
            print('FIT', epoch, 'best', best_epoch, 'white', mv['white'], 'new_white', mn['white'], flush=True)
    last_state = cpu_state(heads)
    torch.save(dict(model_state=last_state, optimizer=optimizer.state_dict(), epoch=protocol['epochs'], protocol=protocol, classes=classes), run / 'last.pt')
    save_json(run / 'history.json', history)
    torch.save(dict(model_state=best_state, epoch=best_epoch, protocol=protocol, classes=classes), run / 'best.pt')
    heads.load_state_dict(best_state)
    cv, cn, ct = [predict(heads, d, params) for d in (val, new, train)]
    np.savez_compressed(run / 'selected_predictions.npz', train=ct, planning=cv, newdev=cn)
    save_json(run / 'selected_evaluation.json', dict(planning=measure(val[0],cv), newdev=measure(new[0],cn), train=measure(train[0],ct)))
    user_ids=set(read_json(out / 'selection_manifest.json')['new_user_ids'])
    ui=[i for i,r in enumerate(train[0]) if r['sample_id'] in user_ids]
    require(len(ui)==len(user_ids), 'User training accounting mismatch')
    labels=read_json(out / 'labels.json')
    save_json(out / 'user_training_replay.json', dict(
        interpretation='Fit replay of user-added training photos only; NOT validation/test or independent-tree accuracy',
        baseline=measure([train[0][i] for i in ui],bt[ui]),
        candidate=measure([train[0][i] for i in ui],ct[ui]),
        images=[dict(sample_id=train[0][i]['sample_id'],source_image=train[0][i]['source_image'],
            group_id=train[0][i]['group_id'],baseline_top1=labels[int(bt[i].argmax())],
            candidate_top1=labels[int(ct[i].argmax())],baseline_white_score=float(bt[i,16]),
            candidate_white_score=float(ct[i,16])) for i in ui]))
    delta_last = max(float((last_state[k]-initial[k]).abs().max()) for k in initial)
    require(delta_last > 0, 'No parameters changed: fitting did not happen')
    selection = dict(locked_at=datetime.now(timezone.utc).isoformat(), epoch=best_epoch,
        selected_checkpoint=str(run / 'best.pt'), sha256=sha256(run / 'best.pt'),
        validation_recommends_candidate=best_epoch>0, last_epoch=protocol['epochs'],
        actual_optimizer_steps=protocol['epochs'], max_abs_last_parameter_change=delta_last,
        shared_member_epoch=True, no_test_predictions_used=True, deployed=False,
        backup=str(backup), baseline_planning=baseval, baseline_newdev=basenew,
        selected_planning=measure(val[0],cv), selected_newdev=measure(new[0],cn))
    save_json(out / 'selection_lock.json', selection)
    members = out / 'selected_members'; members.mkdir(exist_ok=False)
    for i,h in enumerate(heads):
        torch.save(dict(model_state={k:v.detach().cpu() for k,v in h.state_dict().items()},
            epoch=best_epoch, config=dict(architecture='mlp', **{k:protocol[k] for k in ('seed','lr','epochs')}),
            classes=classes, encoder_sha256=protocol['encoder_sha256']), members / f'member_{i+1}.pt')
    verify_backup(backup); unchanged(protocol)
    print('SELECTION LOCKED', best_epoch, 'actual fit steps', protocol['epochs'], flush=True)


def evaluate(out):
    protocol = read_json(out / 'protocol.json'); lock = read_json(out / 'selection_lock.json')
    require(sha256(lock['selected_checkpoint']) == lock['sha256'], 'Selected checkpoint changed')
    require(not (out / 'test_features.npz').exists(), 'Evaluation already started; do not overwrite')
    unchanged(protocol); check_prepared(out,protocol)
    rows = read_json(SOURCE / 'test_rows.json')
    for r in rows: r['sha256'] = r.get('sha256') or r['source_sha256']
    train = read_json(out / 'train_rows.json'); trainsha = {r['sha256'] for r in train}; traingroup = {r['group_id'] for r in train}
    require(not any(r['sha256'] in trainsha or r['group_id'] in traingroup for r in rows), 'Heldout overlap')
    require(len(rows) == 5146, 'Old regression cohort changed')
    save_json(out / 'test_rows.json',rows)
    engine = Engine(); engine.load(); engine_check(engine, protocol)
    x,z = encode_rows(engine, rows, out / 'test_feature_chunks')
    np.savez_compressed(out / 'test_features.npz', x=x, z=z, rows_hash=digest(rows))
    del engine; gc.collect(); torch.cuda.empty_cache()
    base = heads_from_deployment(protocol); candidate = heads_from_deployment(protocol)
    candidate.load_state_dict(torch.load(lock['selected_checkpoint'],map_location='cpu',weights_only=False)['model_state'])
    with np.load(out / 'train_features.npz') as f:
        params=dict(groups=read_json(out / 'classes.json')['groups'], temp=protocol['temperature'],
                    extra=torch.tensor(f['extra_bank'],device='cuda'), scale=float(f['scale']))
    test = load_data(out,'test'); bp=predict(base,test,params); cp=predict(candidate,test,params)
    y=np.array([r['target'] for r in rows]); a=bp.argmax(1); b=cp.argmax(1)
    result=dict(baseline=measure(rows,bp,False), candidate=measure(rows,cp,False),
        corrected=int(((a!=y)&(b==y)).sum()), regressed=int(((a==y)&(b!=y)).sum()),
        corrected_ids=[r['sample_id'] for r,ok in zip(rows,(a!=y)&(b==y)) if ok],
        regressed_ids=[r['sample_id'] for r,ok in zip(rows,(a==y)&(b!=y)) if ok],
        paired_bootstrap=bootstrap_gain(y,bp,cp,[r['group_id'] for r in rows],seed=SEED,repeats=2000),
        interpretation='Previously inspected historical regression, not fresh independent accuracy; classification entry only')
    np.savez_compressed(out / 'test_predictions.npz', baseline=bp,candidate=cp,targets=y)
    save_json(out / 'test_results.json',result)
    # Full audit rows, including every white positive and white false positive.
    labels=read_json(out / 'labels.json'); records=[]
    for r,p,q in zip(rows,bp,cp):
        if r['target']==16 or p.argmax()==16 or q.argmax()==16:
            records.append(dict(sample_id=r['sample_id'],source_image=r['source_image'],sha256=r['sha256'],
                group_id=r['group_id'],truth=labels[r['target']],baseline=labels[int(p.argmax())],candidate=labels[int(q.argmax())],
                baseline_white_score=float(p[16]),candidate_white_score=float(q[16])))
    save_json(out / 'white_regression_audit.json',records)
    verify_backup(lock['backup']); unchanged(protocol)
    complete=dict(completed_at=datetime.now(timezone.utc).isoformat(), trained=True, optimizer_steps=60,
        selected_epoch=lock['epoch'], deployed=False, production_unchanged=True,
        backup_verified=True, backup=lock['backup'], test_cohort=5146)
    save_json(out / 'COMPLETE.json',complete)
    report(out,result)
    print('COMPLETE',out,'selected epoch',lock['epoch'],flush=True)


def report(out,result):
    lock=read_json(out/'selection_lock.json'); summary=read_json(out/'data_summary.json')
    before=read_json(out/'fit/baseline_evaluation.json'); after=read_json(out/'fit/selected_evaluation.json')
    rows=read_json(out/'train_rows.json')
    lines=['# 白桂木用户图片专项增量训练','',
        '在当前部署的 BioCLIP 2.5 + 三成员 MLP 上继续训练；冻结编码器、148类标签和温度；未部署。',
        f"固定配方60个优化步骤，三成员共用选定轮次：{lock['epoch']}。第0轮代表严格门槛下保留初始化，而不是没有训练。",'',
        f'训练图总数：{len(rows)}。新用户图片仅参与训练，不作为独立验证树木。清单和去重细节见 data_summary.json / selection_manifest.json。','',
        '| 对照数据 | 整体 Top-1：训练前 → 选定模型 | 白桂木 TP / 正样本 | 白桂木 FP | 白桂木 F1 |',
        '|---|---|---|---|---|']
    for name,a,b in [('旧开发集2640',before['planning'],after['planning']),('既有来源开发图15（多分类仅10张白桂木）',before['newdev'],after['newdev']),('旧回归集5146',result['baseline'],result['candidate'])]:
        w,v=a['white'],b['white']
        lines.append(f"| {name} | {a['all']['top1']:.2%} → {b['all']['top1']:.2%} | {w['tp']}/{w['tp']+w['fn']} → {v['tp']}/{v['tp']+v['fn']} | {w['fp']} → {v['fp']} | {w['f1']:.3f} → {v['f1']:.3f} |")
    aq0,aq1=result['baseline']['aquilaria'],result['candidate']['aquilaria']
    lines += ['',f"土沉香旧回归保护项：TP {aq0['tp']} → {aq1['tp']}；FP {aq0['fp']} → {aq1['fp']}。",
        f"回归集改正 {result['corrected']} 张，退步 {result['regressed']} 张。配对来源组 bootstrap：{result['paired_bootstrap']}。",'',
        '## 结论边界','',
        '- 选定模型只按固定开发集规则选择，选定权重 SHA 在读取测试预测前锁定。',
        '- 开发/测试均为此前看过的来源图或历史回归，不是新独立街景泛化试验；来源组也不是已证实的独立树。',
        '- 所有本次重编码图逐张检查与当前 Demo 补边/JPEG/224 像素一致；训练前后同样处理，无预处理偷换。',
        '- 检测器、全景选树及点云融合未在本轮改动；分类入口改善不能冒充端到端街景改善。',
        '- last.pt 记录实际训练60轮后的权重，best.pt 记录严格门槛选中的权重；若选中第0轮，不能宣称训练提高了可用模型精度。',
        '- user_data中同来源/近重复照片不计为新的独立树；保留原水印和版权，不新增Demo参考图。','',
        '## 可回退性','',f"备份：{lock['backup']}",
        '初始化权重、冻结编码器、相关线上图像/点云/融合模型、数据分组、人审快照、配置和原始评估均已物理备份并核验SHA。',
        'Demo与部署权重保持不变。','', '## 数据摘要','', '```json',
        __import__('json').dumps(summary,ensure_ascii=False,indent=2),'```','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['features','fit','evaluate']); parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args(); out=args.out.resolve()
    require(out.is_relative_to(ROOT/'outputs'),'Output outside project outputs')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    globals()[args.stage](out)


if __name__=='__main__':
    main()
