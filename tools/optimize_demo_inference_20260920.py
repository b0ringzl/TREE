"""Bounded validation-only inference selection; no fitting or production writes.

Current H120 is the baseline, not its predecessor. Freeze the candidate grid
before inspecting its development predictions. Lock once, then run regression.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
import torch
from bioclip_probe import ResidualHead
from distillation_core import read_json, save_json, digest, bootstrap_gain
from training_safety import sha256, backup_before_training, verify_backup
from train_artocarpus_user_20260918 import protected, predict, measure, binary
from train_aq_ar_diagnostic_20260919 import verify_images

OUT = ROOT / 'outputs/SSTv1_Inference_Optimization_20260920_v2'
CROPS = ROOT / 'outputs/SSTv1_Aquilaria_UserCrops_20260919_v2'
PREV = ROOT / 'outputs/SSTv1_Artocarpus_User_20260918_v1'
AUDIT = ROOT / 'outputs/artocarpus_preprocess_audit_20260917_v1'
CENTER = ROOT / 'outputs/SSTv1_AQ_AR_Diagnostic_20260919_v1/exact_regression/center_features.npz'
CURRENT = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/deployment.json'
ANCHOR = ROOT / 'tree_species_teacher_demo/models/SSTv1_Artocarpus_User_20260918_v1/deployment.json'
FOCAL = ['Aquilaria sinensis', 'Artocarpus hypargyreus']


def recipes():
    rows = [dict(name=f'views{int(c*100)}_anchor{int(a*100)}', center_weight=c,
                 anchor_weight=a, preserve_confident=None)
            for c in (.25, .5, .75) for a in (0., .25, .5)]
    rows += [dict(name=f'views50_keep{int(t*100)}', center_weight=.5,
                  anchor_weight=0., preserve_confident=t) for t in (.85, .95)]
    return rows


def prepare():
    OUT.mkdir(exist_ok=False)
    cache_paths = [AUDIT/'paired_features_predictions.npz', AUDIT/'rows.json',
                   PREV/'test_rows.json', PREV/'test_features.npz', CENTER,
                   PREV/'train_features.npz', CROPS/'train_rows.json', CROPS/'crop_manifest.json',
                   CROPS/'classes.json', CROPS/'labels.json', CROPS/'human_review_pre_fit.json']
    # Do not silently skip any expected manifest.
    paths = list(dict.fromkeys([*protected(), ROOT/'AGENTS.md', CURRENT, ANCHOR,
        *[ANCHOR.parent/m['file'] for m in read_json(ANCHOR)['members']],
        *cache_paths, Path(__file__),
        ROOT/'tree_species_teacher_demo/server.py',
        ROOT/'spacescan_trees_demo_v1/official_release.py',
        ROOT/'spacescan_trees_demo_v1/release.json',
        ROOT/'spacescan_trees_demo_v1/templates/index.html',
        ROOT/'spacescan_trees_demo_v1/static/apps.html']))
    assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
    backup = backup_before_training(OUT, paths, dict(
        purpose='Pre-experiment backup, inference-only; no training, no weight changes',
        user_authorization='2026-09-20: execute subsequent plans without per-plan approval; back up before every plan'))
    labels = read_json(CROPS/'labels.json')
    assert all(s in labels for s in FOCAL)
    save_json(OUT/'protocol.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        current_manifest=str(CURRENT), anchor_manifest=str(ANCHOR), recipes=recipes(),
        temperature=read_json(CURRENT)['temperature'], optimizer_steps=0, seed=20260920,
        development='Fixed 2655 rows (2650 in catalogue + 5 out-of-catalogue negatives); repeatedly inspected',
        regression='Fixed 5146 historically inspected images; not blind test; no regression-driven reselection',
        selection='Among eligible recipes maximize macro F1, then correct count, then mean focal F1; first wins exact ties.',
        eligibility='Relative to actual H120 single: overall Top1 >= +.01, macroF1 >= +.005; AQ/AR TP nondecreasing, FP nonincreasing, also protected separately within old and new dev.',
        deployment_gate='Locked candidate regression overall Top1 >= +.01, macroF1 nondecreasing; no decrease in TP or increase in FP for AQ/AR. Failed gate means retain current Demo, not choose another on regression.',
        scope='Target-confirmed single-tree classification only; no claim on end-to-end panorama detection, calibration or fresh HK generalization',
        views='Probability mixtures of exact existing letterbox and original-image center224; same photograph, not two independent observations',
        focal=FOCAL, labels=labels, backup=str(backup),
        absent_from_production_catalog=['Rhodoleia championii','Keteleeria fortunei'],
        prior_attempt='v1 backed up successfully then stopped before protocol/prediction: red-flower and Keteleeria absent from production 148; v2 explicitly retains catalogue and reports missing coverage.',
        protected_sha256={str(p):sha256(p) for p in paths},
        training_data='No fitting, no added/relabelled images, current accepted data/review snapshots preserved'))
    print('PREPARED_AND_BACKUP_VERIFIED', backup, flush=True)


def heads(path):
    manifest = read_json(path)
    hh = torch.nn.ModuleList()
    for member in manifest['members']:
        pp = path.parent/member['file']
        assert sha256(pp) == member['sha256']
        state = torch.load(pp, map_location='cpu', weights_only=False)
        assert state['classes'] == read_json(CROPS/'classes.json')
        h = ResidualHead('mlp').cuda().eval().requires_grad_(False)
        h.load_state_dict(state['model_state']); hh.append(h)
    return hh


def tensor(a):
    return torch.tensor(np.asarray(a, dtype=np.float32), device='cuda')


def get_predictions(phase, use_anchor=True):
    protocol = read_json(OUT/'protocol.json')
    if phase == 'dev':
        rr = read_json(AUDIT/'rows.json')
        with np.load(AUDIT/'paired_features_predictions.npz') as f:
            assert str(f['rows_hash']) == digest(rr)
            data = [tensor(f[k]) for k in ('demo_letterbox_x','demo_letterbox_z','center_crop_x','center_crop_z')]
    else:
        assert (OUT/'selection_lock.json').is_file()
        rr = read_json(PREV/'test_rows.json')
        with np.load(PREV/'test_features.npz') as f, np.load(CENTER) as c:
            assert str(f['rows_hash']) == str(c['rows_hash']) == digest(rr)
            data = [tensor(f['x']),tensor(f['z']),tensor(c['x']),tensor(c['z'])]
    verify_images(rr)
    with np.load(PREV/'train_features.npz') as f:
        params = dict(groups=read_json(CROPS/'classes.json')['groups'], temp=protocol['temperature'],
                      extra=tensor(f['extra_bank']), scale=float(f['scale']))
    result = {}
    for name,path in [('current',CURRENT), ('anchor',ANCHOR)]:
        if name == 'anchor' and not use_anchor:
            continue
        hh = heads(path)
        for view, arrays in [('whole',data[:2]),('center',data[2:])]:
            result[f'{name}_{view}'] = predict(hh, (rr,*arrays), params)
        del hh
    return rr,result


def mixture(pp, recipe):
    c,a = recipe['center_weight'],recipe['anchor_weight']
    result = (1-c)*pp['current_whole'] + c*pp['current_center']
    if a:
        result = (1-a)*result + a*((1-c)*pp['anchor_whole']+c*pp['anchor_center'])
    if recipe['preserve_confident'] is not None:
        keep = pp['current_whole'].max(1) >= recipe['preserve_confident']
        result[keep] = pp['current_whole'][keep]
    assert np.isfinite(result).all() and np.allclose(result.sum(1),1,atol=1e-5)
    return result


def score(rows, pp):
    result = measure(rows, pp)
    labels = read_json(OUT/'protocol.json')['labels']
    y = np.asarray([r['target'] for r in rows])
    result['focal'] = {s:binary(y,pp,labels.index(s)) for s in FOCAL}
    result['per_class'] = {s:binary(y,pp,i) for i,s in enumerate(labels)}
    return result


def failures(m,b,regression=False):
    bad=[]
    if m['all']['top1'] < b['all']['top1']+.01: bad.append('overall Top1 +1pp')
    if m['all']['macro_f1'] < b['all']['macro_f1']+(0 if regression else .005): bad.append('macro F1')
    for s in FOCAL:
        if m['focal'][s]['tp'] < b['focal'][s]['tp']: bad.append(s+' TP')
        if (not regression or s in FOCAL[:2]) and m['focal'][s]['fp'] > b['focal'][s]['fp']: bad.append(s+' FP')
    return bad


def select():
    assert not (OUT/'selection_lock.json').exists()
    protocol=read_json(OUT/'protocol.json'); verify_backup(protocol['backup'])
    rows,pp=get_predictions('dev'); baseline=score(rows,pp['current_whole'])
    save_json(OUT/'development_rows.json',rows)
    np.savez_compressed(OUT/'development_components.npz',**pp,rows_hash=digest(rows))
    results=[]; chosen=None; best=None
    for r in protocol['recipes']:
        p=mixture(pp,r); m=score(rows,p); bad=failures(m,baseline)
        for part,sl in [('old',slice(0,2640)),('new',slice(2640,None))]:
            for s in FOCAL[:2]:
                idx=protocol['labels'].index(s);y=[x['target'] for x in rows[sl]]
                bm=binary(y,pp['current_whole'][sl],idx);cm=binary(y,p[sl],idx)
                if cm['tp']<bm['tp'] or cm['fp']>bm['fp']:bad.append(part+' '+s)
        rank=(m['all']['macro_f1'],m['all']['correct'],np.mean([x['f1'] for x in m['focal'].values()]))
        results.append(dict(recipe=r,metrics=m,failed_gates=bad))
        print('DEV',r['name'],m['all'],m['focal'],'FAILED',bad,flush=True)
        if not bad and (best is None or rank>best):chosen=r;best=rank
    save_json(OUT/'development_results.json',dict(baseline=baseline,candidates=results))
    save_json(OUT/'selection_lock.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),
        selected=chosen,selected_on='development only',protocol_sha256=sha256(OUT/'protocol.json'),
        predictions_sha256=sha256(OUT/'development_components.npz'),training_performed=False))
    print('LOCKED',chosen,flush=True)


def regress():
    assert not (OUT/'regression_results.json').exists()
    lock=read_json(OUT/'selection_lock.json');recipe=lock['selected']
    assert recipe is not None, 'No validation candidate passed; do not inspect regression'
    rows,pp=get_predictions('test',use_anchor=bool(recipe['anchor_weight']))
    b=pp['current_whole'];p=mixture(pp,recipe)
    bm,cm=score(rows,b),score(rows,p);bad=failures(cm,bm,True)
    y=np.array([r['target'] for r in rows]);bc=b.argmax(1)==y;pc=p.argmax(1)==y
    changes=[dict(**r,baseline_label=read_json(OUT/'protocol.json')['labels'][int(b[i].argmax())],
                  candidate_label=read_json(OUT/'protocol.json')['labels'][int(p[i].argmax())],
                  kind='corrected' if pc[i] else 'regressed')
             for i,r in enumerate(rows) if bc[i]!=pc[i]]
    save_json(OUT/'regression_changed_cases.json',changes)
    save_json(OUT/'regression_results.json',dict(baseline=bm,candidate=cm,recipe=recipe,
        corrected=int((~bc&pc).sum()),regressed=int((bc&~pc).sum()),failed_deployment_gates=bad,
        deployment_eligible=not bad,selected_before_regression=True,
        bootstrap=bootstrap_gain(y,b,p,[r['group_id'] for r in rows],seed=20260920,repeats=1000)))
    np.savez_compressed(OUT/'regression_predictions.npz',baseline=b,candidate=p,rows_hash=digest(rows))
    verify_backup(read_json(OUT/'protocol.json')['backup'])
    print('REGRESSION',bm['all'],cm['all'],'CORRECTED',int((~bc&pc).sum()),'REGRESSED',int((bc&~pc).sum()),'FAILED',bad,flush=True)
    print('FOCAL',bm['focal'],cm['focal'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','select','regress']);args=parser.parse_args()
    globals()[args.phase]()
