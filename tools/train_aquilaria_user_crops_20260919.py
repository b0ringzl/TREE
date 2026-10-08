"""Explicit user-reviewed crop continuation. Physical backups, train lineage, no deployment."""
import argparse
import gc
import sys
import shutil
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
import cv2
from distillation_core import read_json,save_json,digest
from training_safety import sha256,backup_before_training,verify_backup
from import_artocarpus_user_20260918 import current_human_review_exclusions,perceptual_hash
from prepare_artocarpus_fit_20260918 import identities,row_ids
from train_aq_ar_diagnostic_20260919 import verify_images,evaluate_dev,rank,unchanged
from train_artocarpus_user_20260918 import protected,heads_from_deployment,predict,measure,cpu_state
from tree_species_teacher_demo.engine import Engine,SSTV1_MANIFEST,decode
from audit_artocarpus_preprocessing_20260917 import exact_demo_prepare
from aq_ar_dualview_next_20260919 import tensor,load_pair,paired_predict,paired_logprob,allowed

OUT=ROOT/'outputs/SSTv1_Aquilaria_UserCrops_20260919_v1'
OLD=ROOT/'outputs/SSTv1_AQ_AR_Dualview_20260919_v1'
PREV=ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1'
AUDIT=ROOT/'outputs/artocarpus_preprocess_audit_20260917_v1'
FOLDER=Path('C:/Users/25041725G/Desktop/Aquilaria sinensis')
NAMES=['B9A2B0F30924AEF1.jpg','622EB19F5DA7AA68.jpg','CM_Tree_20_W3583_01_286.jpg',
       'for MD blog Jun 2021 Citywalk 7MB jpeg IMG_8656.jpg','K13_KC13_W3581_006 (003).jpg',
       'K13_KC13_W3581_010 (002).jpg','large (15).jpg','large.jpg']
SEED=20260921
RECIPES=[dict(name='G_crops_group_balanced',crop_loss_weight=0.),
         dict(name='H_crops_targeted_002',crop_loss_weight=.02)]


def firewall(rows,review,held):
    for r in rows:
        assert r['sha256'] not in review['blocked_sha256'] and not row_ids(r)&set(review['blocked_ids']),r['sample_id']
        assert not identities(r)&held,r['sample_id']


def prepare(out):
    out.mkdir(exist_ok=False);(out/'user_crops').mkdir()
    train=read_json(OLD/'train_rows.json');verify_images(train)
    old_protocol=read_json(OLD/'protocol.json');unchanged(old_protocol)
    usage=read_json(ROOT/'outputs/user_folder_split_audit_20260919_v1/file_usage.json')['rows']
    review=current_human_review_exclusions()
    held=read_json(OLD/'heldout_metadata.json')
    held_tokens=set().union(*(set(r['identity_tokens']) for r in held['records']))
    firewall(train,review,held_tokens)
    crops=[]
    for number,name in enumerate(NAMES):
        previous=[r for r in usage if r['species']=='Aquilaria sinensis' and r['filename']==name]
        assert len(previous)==1 and previous[0]['lane']=='new_train_confirmed'
        parents=[r for r in train if r['sha256']==previous[0]['sha256']]
        assert len(parents)==1;parent=parents[0]
        path=FOLDER/name;h=sha256(path)
        assert h!=parent['sha256'],'Expected modified crop, not original'
        assert h not in {r['sha256'] for r in train+crops}
        dest=out/'user_crops'/name;shutil.copy2(path,dest)
        assert sha256(dest)==sha256(path)==h
        with Image.open(dest) as image:
            im=ImageOps.exif_transpose(image).convert('RGB');width,height=im.size;ph=perceptual_hash(im)
            gray=np.asarray(im.convert('L'))
        with Image.open(parent['source_image']) as image:
            pg=np.asarray(ImageOps.exif_transpose(image).convert('L'))
        match=None
        if gray.shape[0]<=pg.shape[0] and gray.shape[1]<=pg.shape[1]:
            result=cv2.matchTemplate(pg,gray,cv2.TM_CCOEFF_NORMED)
            _,value,_,pos=cv2.minMaxLoc(result)
            match=dict(correlation=float(value),top_left=list(pos),method='same-scale grayscale template check; user provides crop lineage')
        r=dict(id='user_aq_crop_'+h[:16],sample_id='user_aq_crop_'+h[:16],source_image=str(dest.resolve()),
            original_source=str(path),filename=name,sha256=h,width=width,height=height,phash=ph,
            parent_sha256=parent['sha256'],parent_sample_id=parent['sample_id'],parent_source_image=parent['source_image'],
            group_id=parent['group_id'],split_group=parent['split_group'],source_group=parent['source_group'],
            parent_identity_tokens=sorted(identities(parent)),target=11,source_species='Aquilaria sinensis',
            phase='train',training_eligible=True,data_role='user_confirmed_crop_training_only',
            label_authority='User explicitly confirmed these eight cropped photos for renewed learning in current request',
            user_confirmation_date='2026-09-19',crop_lineage='User supplied derivative of previously confirmed training photo; not independent tree',
            rights=parent.get('rights'),original_watermarks_retained=True,template_check=match,
            quality_flags=['short_side_below_224'] if min(width,height)<224 else [],
            quality=dict(dark_fraction=float((gray<25).mean()),bright_fraction=float((gray>245).mean())))
        crops.append(r)
    firewall(crops,review,held_tokens)
    assert len(crops)==8
    for name in ('labels.json','classes.json','teacher_choice.json','heldout_metadata.json'):
        shutil.copy2(OLD/name,out/name)
    save_json(out/'crop_manifest.json',crops);save_json(out/'train_rows.json',train+crops)
    save_json(out/'human_review_snapshot.json',review)
    save_json(out/'development_rows.json',read_json(AUDIT/'rows.json'))
    dep=read_json(SSTV1_MANIFEST)
    caches=[OLD/'train_features.npz',OLD/'train_center_features.npz',AUDIT/'paired_features_predictions.npz',
        PREV/'test_features.npz',ROOT/'outputs/SSTv1_AQ_AR_Diagnostic_20260919_v1/exact_regression/center_features.npz']
    done=read_json(OLD/'FEATURES_COMPLETE.json')
    assert sha256(caches[0])==done['pad_sha256'] and sha256(caches[1])==done['center_sha256']
    save_json(out/'protocol.json',dict(created_at=datetime.now(timezone.utc).isoformat(),seed=SEED,recipes=RECIPES,
        steps=120,evaluate_every=10,lr=3e-5,weight_decay=.01,kl_weight=.5,train_count=len(train)+len(crops),
        base_count=len(train),new_crops=len(crops),crop_groups=len({r['group_id'] for r in crops}),
        parent_files_retained=True,unspecified_low_quality_photos_removed=False,
        deployment_manifest=dep,temperature=dep['temperature'],
        views='Dual probability mean: exact Demo letterbox and exact center; both train and inference',
        sample_unit='8 crop derivatives of 8 training photos, 7 pre-existing groups; not 8 new independent trees',
        loss='Source/class-balanced NLL on replay plus crop derivatives, .5 KL anchor on original replay; H adds .02 group-balanced crop NLL',
        selection='Frozen prior allowed()/rank(): pass BOTH original single and dual focal dev and overall gates, then strict rank gain; initial_dual fallback. No crop replay or regression used for selection.',
        trainable='Three existing native148 MLP heads only; encoder, temperature, label map, point/fusion encoders unchanged',
        heldout='Old dev and test are previously inspected regression/development, not fresh blind validation',
        acceptance='Keep user labels; no predictions can relabel data; no pending/rejected review data included',
        protected_sha256={str(p.resolve()):sha256(p) for p in protected()},
        cache_sha256={str(p.resolve()):sha256(p) for p in caches},
        source_sha256={str(p.resolve()):sha256(p) for p in [Path(__file__),ROOT/'tools/aq_ar_dualview_next_20260919.py',
            ROOT/'tools/train_artocarpus_user_20260918.py',ROOT/'tools/train_aq_ar_diagnostic_20260919.py',
            ROOT/'tools/prepare_artocarpus_fit_20260918.py',ROOT/'tools/import_artocarpus_user_20260918.py']},deployed=False))
    print('PREPARED',[(r['filename'],r['width'],r['height'],r['template_check']) for r in crops],flush=True)


def check(out):
    protocol=read_json(out/'protocol.json');unchanged(protocol)
    for group in ('cache_sha256','source_sha256'):
        for path,h in protocol[group].items():assert sha256(path)==h,path
    return protocol


def features(out):
    protocol=check(out);assert not (out/'train_features.npz').exists()
    rows=read_json(out/'train_rows.json');crops=read_json(out/'crop_manifest.json');verify_images(rows)
    torch.set_num_threads(4);engine=Engine();engine.load()
    assert engine.sstv1_version==protocol['deployment_manifest']['version']
    production=exact_demo_prepare();x=[];z=[];xc=[];zc=[]
    with torch.inference_mode():
        for r in crops:
            blob=Path(r['source_image']).read_bytes()
            for content,xx,zz in ((production(blob)[0],x,z),(blob,xc,zc)):
                pixel=decode(content)[0]
                e=engine.model.encode_image(((pixel[None].cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
                xx.append(e.half().cpu().numpy());zz.append((e@engine.bank.T*engine.model.logit_scale.exp().float()).cpu().numpy())
    for filename,xx,zz in [('train_features.npz',x,z),('train_center_features.npz',xc,zc)]:
        with np.load(OLD/filename) as old:
            assert str(old['rows_hash'])==digest(rows[:-8])
            extras={k:old[k] for k in old.files if k not in ('x','z','rows_hash')}
            np.savez_compressed(out/filename,x=np.concatenate([old['x'],*xx]),z=np.concatenate([old['z'],*zz]),rows_hash=digest(rows),**extras)
    # Verify cache probabilities against live deployed API before any fitting.
    import requests
    heads=heads_from_deployment(protocol)
    params=dict(groups=read_json(out/'classes.json')['groups'],temp=protocol['temperature'],extra=engine.extra_bank.float(),scale=float(engine.model.logit_scale.exp()))
    pair=load_pair(out,'train')
    expected=predict(heads,pair[:3],params)[-8:];expected_c=predict(heads,(pair[0],pair[3],pair[4]),params)[-8:]
    errors=[]
    with requests.Session() as client:
        for i,r in enumerate(crops):
            raw=Path(r['source_image']).read_bytes()
            for view,content,cached in [('whole',production(raw)[0],expected[i]),('center',raw,expected_c[i])]:
                resp=client.post('http://127.0.0.1:8039/api/predict',data={'mode':'sstv1'},files={'image':('crop.jpg',content)},timeout=90)
                resp.raise_for_status();data=resp.json();assert data['deployment']==protocol['deployment_manifest']['version']
                p=np.array([data['sstv1_distribution'][label] for label in engine.labels]);err=float(np.abs(cached-p).max())
                assert err<5e-5,(r['filename'],view,err)
                errors.append(dict(filename=r['filename'],view=view,max_probability_error=err))
    save_json(out/'FEATURES_COMPLETE.json',dict(live_api_checks=errors,rows=len(rows),
        pad_sha256=sha256(out/'train_features.npz'),center_sha256=sha256(out/'train_center_features.npz'),
        encoder_sha256=engine.encoder_sha256,batch_size=1))
    unchanged(protocol);print('FEATURES_COMPLETE',len(rows),'live_checks',len(errors),flush=True)


def crop_replay(out,baseline,after,rows,name):
    labels=read_json(out/'labels.json');details=[]
    for i,r in enumerate(rows):
        def summary(p):return dict(top1=labels[int(p.argmax())],true_score=float(p[r['target']]),correct=int(p.argmax())==r['target'])
        details.append(dict(filename=r['filename'],source_image=r['source_image'],baseline=summary(baseline[i]),candidate=summary(after[i])))
    save_json(out/name,dict(scope='TRAIN replay only; not held-out accuracy',images=details,
        baseline_correct=sum(r['baseline']['correct'] for r in details),candidate_correct=sum(r['candidate']['correct'] for r in details)))


def fit(out):
    protocol=check(out);assert not (out/'baseline_evaluation.json').exists(),'Retries require a new version and backup'
    torch.set_num_threads(4)
    features=read_json(out/'FEATURES_COMPLETE.json')
    assert sha256(out/'train_features.npz')==features['pad_sha256'] and sha256(out/'train_center_features.npz')==features['center_sha256']
    train=load_pair(out,'train');dev=load_pair(out,'dev');rows=train[0]
    review=current_human_review_exclusions();save_json(out/'human_review_pre_fit.json',review)
    held=set().union(*(set(r['identity_tokens']) for r in read_json(out/'heldout_metadata.json')['records']))
    firewall(rows,review,held);verify_images(rows+dev[0])
    with np.load(out/'train_features.npz') as f:
        params=dict(groups=read_json(out/'classes.json')['groups'],temp=protocol['temperature'],extra=tensor(f['extra_bank']),scale=float(f['scale']))
    heads=heads_from_deployment(protocol);initial=cpu_state(heads)
    bt=paired_predict(heads,train,params);bts=predict(heads,train[:3],params)
    bd=paired_predict(heads,dev,params);bs=predict(heads,dev[:3],params)
    bases=[evaluate_dev(dev[0],p) for p in (bs,bd)]
    save_json(out/'baseline_evaluation.json',dict(single=bases[0],dual=bases[1],train=measure(rows,bt),
        crop_single_correct=int((bts[-8:].argmax(1)==11).sum()),crop_dual_correct=int((bt[-8:].argmax(1)==11).sum())))
    np.savez_compressed(out/'baseline_predictions.npz',train_dual=bt,train_single=bts,dev_single=bs,dev_dual=bd)
    torch.save(dict(model_state=initial),out/'initial.pt')
    print('BASELINE_CROPS single',int((bts[-8:].argmax(1)==11).sum()),'dual',int((bt[-8:].argmax(1)==11).sum()),flush=True)
    best=dict(recipe='initial_dual',step=0,checkpoint=str(out/'initial.pt'),metrics=bases[1])
    gc=Counter((r['target'],r['group_id']) for r in rows);groups=defaultdict(set)
    for r in rows:groups[r['target']].add(r['group_id'])
    weights=tensor([1/(len(groups[r['target']])*gc[(r['target'],r['group_id'])]) for r in rows]);weights/=weights.sum()
    cgroups=Counter(r['group_id'] for r in rows[-8:]);cw=tensor([1/cgroups[r['group_id']] for r in rows[-8:]]);cw/=cw.sum()
    y=torch.tensor([r['target'] for r in rows],device='cuda');anchor=tensor(bt[:-8]);parent_ix=[]
    for r in rows[-8:]:parent_ix.append(next(i for i,p in enumerate(rows[:-8]) if p['sha256']==r['parent_sha256']))
    for recipe in protocol['recipes']:
        run=out/recipe['name'];run.mkdir(exist_ok=False)
        save_json(run/'config.json',dict(**recipe,seed=SEED,steps=120,lr=protocol['lr'],initializer_sha256=sha256(out/'initial.pt')))
        inputs=[*protected(),*[Path(p) for p in protocol['source_sha256']],ROOT/'tools/training_safety.py',
            *sorted(out.glob('*.json')),*sorted(out.glob('*.npz')),out/'initial.pt',run/'config.json',
            *[Path(r['source_image']) for r in rows[-8:]],*[Path(r['parent_source_image']) for r in rows[-8:]],
            AUDIT/'paired_features_predictions.npz',AUDIT/'rows.json']
        backup=backup_before_training(run,inputs,dict(recipe=recipe,seed=SEED,training_parent_groups_unchanged=True))
        print('BACKUP_VERIFIED',recipe['name'],flush=True)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED)
        heads=heads_from_deployment(protocol);opt=torch.optim.AdamW(heads.parameters(),lr=protocol['lr'],weight_decay=.01)
        history=[]
        for step in range(1,121):
            heads.train();opt.zero_grad(set_to_none=True);lp=paired_logprob(heads,train,params)
            nll=F.nll_loss(lp,y,reduction='none');ce=(nll*weights).sum()
            kl=F.kl_div(lp[:-8],anchor,reduction='batchmean');cl=(nll[-8:]*cw).sum()
            loss=ce+.5*kl+recipe['crop_loss_weight']*cl
            assert bool(torch.isfinite(loss));loss.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),5.,error_if_nonfinite=True);opt.step()
            if step%10:continue
            dp=paired_predict(heads,dev,params);m=evaluate_dev(dev[0],dp);ok=allowed(m,bases)
            tp=paired_predict(heads,train,params);sp=predict(heads,train[:3],params)
            checkpoint=run/f'step_{step:03d}.pt';torch.save(dict(model_state=cpu_state(heads),step=step,recipe=recipe),checkpoint)
            item=dict(step=step,ce=float(ce.detach()),kl=float(kl.detach()),crop_loss=float(cl.detach()),eligible=ok,metrics=m,
                crop_single_correct=int((sp[-8:].argmax(1)==11).sum()),crop_dual_correct=int((tp[-8:].argmax(1)==11).sum()))
            history.append(item)
            if ok and rank(m)>rank(best['metrics']):best=dict(recipe=recipe['name'],step=step,checkpoint=str(checkpoint),metrics=m)
            print('FIT',recipe['name'],step,'CROPS',item['crop_single_correct'],item['crop_dual_correct'],
                'DEV_AQ',m['combined']['aquilaria'],'DEV_AR',m['combined']['white'],'allowed',ok,flush=True)
        save_json(run/'history.json',history)
        np.savez_compressed(run/'last_train_predictions.npz',dual=tp,single=sp)
        crop_replay(run,bt[-8:],tp[-8:],rows[-8:],'crop_replay_dual.json')
        crop_replay(run,bts[-8:],sp[-8:],rows[-8:],'crop_replay_single.json')
        verify_backup(backup);unchanged(protocol)
    best.update(sha256=sha256(best['checkpoint']),locked_at=datetime.now(timezone.utc).isoformat(),
        used_crop_replay_for_selection=False,used_regression_for_selection=False,deployed=False)
    save_json(out/'selection_lock.json',best);print('LOCKED',best['recipe'],best['step'],flush=True)


def evaluate(out):
    protocol=check(out);lock=read_json(out/'selection_lock.json');assert not (out/'regression_results.json').exists()
    assert sha256(lock['checkpoint'])==lock['sha256'];torch.set_num_threads(4)
    test=load_pair(out,'test');train=load_pair(out,'train');verify_images(test[0])
    with np.load(out/'train_features.npz') as f:
        params=dict(groups=read_json(out/'classes.json')['groups'],temp=protocol['temperature'],extra=tensor(f['extra_bank']),scale=float(f['scale']))
    base=heads_from_deployment(protocol);candidate=heads_from_deployment(protocol)
    candidate.load_state_dict(torch.load(lock['checkpoint'],map_location='cpu',weights_only=False)['model_state'])
    bs=predict(base,test[:3],params);bd=paired_predict(base,test,params)
    cs=predict(candidate,test[:3],params);cd=paired_predict(candidate,test,params)
    save_json(out/'regression_results.json',dict(baseline_single=measure(test[0],bs,False),baseline_dual=measure(test[0],bd,False),
        candidate_single=measure(test[0],cs,False),candidate_dual=measure(test[0],cd,False),
        interpretation='Historical regression after validation-only lock. Not fresh unbiased accuracy.'))
    np.savez_compressed(out/'regression_predictions.npz',baseline_single=bs,baseline_dual=bd,candidate_single=cs,candidate_dual=cd)
    ts=predict(candidate,train[:3],params);td=paired_predict(candidate,train,params)
    with np.load(out/'baseline_predictions.npz') as f:
        crop_replay(out,f['train_dual'][-8:],td[-8:],train[0][-8:],'selected_crop_replay_dual.json')
        crop_replay(out,f['train_single'][-8:],ts[-8:],train[0][-8:],'selected_crop_replay_single.json')
    np.savez_compressed(out/'selected_train_predictions.npz',single=ts,dual=td)
    unchanged(protocol)
    save_json(out/'COMPLETE.json',dict(steps=240,selected_recipe=lock['recipe'],selected_step=lock['step'],
        deployed=False,formal_files_unchanged=True,backups=[str(out/r['name']/'pre_training_backup') for r in RECIPES]))
    print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','features','fit','evaluate']);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();globals()[a.phase](a.out.resolve())
