"""Versioned SSTv1 continuation. No deployment, no fabricated human decisions."""
import argparse,gc,io,os,sqlite3,sys,shutil
from collections import Counter
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps,ImageEnhance
from train_sstv1_incremental import load_old,logits,probs,score
from bioclip_probe import ResidualHead,SOURCE,TEACHER
from distillation_core import read_json,save_json,digest,metrics,bootstrap_gain
from training_safety import sha256,backup_before_training,verify_backup,deployed_artifacts
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PREV=ROOT/'outputs/SSTv1_incremental_20260914_v1'
DATA=ROOT/'external_datasets/aquilaria_specialist_20260914_v1'
DEPLOY=ROOT/'tree_species_teacher_demo/models/SSTv1_20260914_v1/deployment.json'
SEED=20260915;AQ=11

def decisions(path):
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;return {r['id']:dict(r) for r in db.execute('SELECT * FROM decisions')}

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(PREV/'config.json');dep=read_json(DEPLOY);classes=read_json(PREV/'classes.json')
    assert classes['labels'][AQ]=='Aquilaria sinensis'
    cfg.update(model_name='SSTv1-Aquilaria',seed=SEED,epochs=45,lr=.00008,focus_targets=[AQ],
        selection='Per member: maximize mean(Aquilaria F1 on audited old planning, new-author planning recall); require old full macro-F1 drop <=0.005 and old non-Aquilaria false positives <= baseline+1. Epoch zero allowed.',
        loss='0.75 class-balanced full replay CE + 0.15 observation-balanced target CE + 0.10 hard-negative CE + 0.20 starting-head replay KL',
        review_authority='User delegated source/visual audit to agent; not human-reviewed or expert certification',
        inference='Unchanged 148 outputs and temperature; no detector or deployment edits',
        new_test_scope='3 newly collected photos, 2 observers; not verified independent street trees; possible encoder pretraining overlap',
        new_counts={},deployed=False)
    for name in ('classes.json','labels.json','teacher_choice.json','cache_metadata.json'):
        save_json(out/name,read_json(PREV/name))
    for phase in ('train','planning','selection','test','field','case'):
        rows=read_json(PREV/f'old_{phase}_rows.json');assert sha256(SOURCE/f'{phase}_rows.json')==cfg['source_hashes'][str(SOURCE/f'{phase}_rows.json')]
        save_json(out/f'old_{phase}_rows.json',rows)
    hd=decisions(ROOT/'external_datasets/herbarium_training_candidates_20260907/human_review.sqlite3')
    save_json(out/'old_human_review_snapshot.json',hd)
    source_audit=read_json(DATA/'old_author_audit.json')
    reliable={str(r['observation']) for r in source_audit if r['current_name']=='Aquilaria sinensis' and r['quality_grade']=='research'}
    oldtrain=read_json(out/'old_train_rows.json');keep=[];excluded=[]
    for i in read_json(PREV/'old_train_indices.json'):
        r=oldtrain[i];why=[]
        if r.get('auxiliary'):
            d=hd.get(r['sample_id'].removeprefix('herbarium_'),{})
            if d.get('status')!='accepted' or d.get('sha256')!=r['source_sha256']:why.append('not_current_human_accepted')
        if r['target']==AQ and r['group_id'].split('observation:')[-1] not in reliable:why.append('source_species_not_currently_verified')
        (excluded if why else keep).append({'row':r,'reasons':why} if why else i)
    save_json(out/'old_train_indices.json',keep);save_json(out/'excluded_replay.json',excluded)
    save_json(out/'accepted_replay_manifest.json',[oldtrain[i] for i in keep])
    # Previous SSTv1 new-data replay uses only the current human-accepted TRAIN partition.
    pd=decisions(ROOT/'external_datasets/sstv1_incremental_20260914_v1/human_review.sqlite3')
    save_json(out/'previous_incremental_human_review.json',pd)
    priorrows=read_json(PREV/'accepted_new_manifest.json')
    priorids=[i for i,r in enumerate(priorrows) if r['phase']=='train' and pd.get(r['id'],{}).get('status')=='accepted' and pd[r['id']]['sha256']==r['sha256']]
    save_json(out/'prior_replay_indices.json',priorids)
    save_json(out/'prior_replay_manifest.json',[priorrows[i] for i in priorids])
    # Explicit visual decisions after inspecting all 30 photos in contact sheets.
    exclude={'aq_inat_627446367':'Mixed ground foliage; target identity visually ambiguous',
        'aq_inat_82305304':'Isolated seed structures; too little target foliage for this training objective',
        'aq_inat_56579941':'Backlit dark foliage lacks useful detail',
        'aq_inat_56579937':'Underexposed fruit/foliage; clearer companion views available'}
    visual=[];new=[]
    for r in read_json(DATA/'manifest.json'):
        reason=exclude.get(r['id']);accepted=not r['quality_flags'] and not reason
        visual.append({'id':r['id'],'sha256':r.get('sha256'),'status':'agent_accepted' if accepted else 'agent_excluded',
            'reason':reason or 'Source exact-species consensus from >=2 identifiers; visual foliage/fruit structure consistent; not independent expert certification',
            'reviewer':'Codex agent','model_prediction_used_for_label':False})
        if not accepted:continue
        # Conservative observer blocks retain repeated dates/plants on one side.
        phase='test' if r['observer_id'] in (1133692,1714330) else 'planning' if r['observer_id']==1846136 else 'train'
        new.append({**r,'source_image':str(DATA/r['file']),'sample_id':r['id'],'target':AQ,
            'phase':phase,'group_id':r['source_group'],'split_group':'observer_'+str(r['observer_id']),
            'training_eligible':phase=='train','review_authority':'agent_source_and_visual_review'})
    save_json(out/'agent_review.json',visual);save_json(out/'accepted_new_manifest.json',new)
    train_obs={str(r['inat_observation_id']) for r in new if r['phase']=='train'}
    held_obs={r['group_id'].split('observation:')[-1] for p in ('planning','selection','test','field','case') for r in read_json(out/f'old_{p}_rows.json')}
    assert not train_obs&held_obs
    for phase in ('train','planning','test'):
        authors={r['observer_id'] for r in new if r['phase']==phase}
        assert not authors&{r['observer_id'] for r in new if r['phase']!=phase}
    save_json(out/'old_source_audit.json',source_audit)
    cfg['new_counts']=dict(Counter(r['phase'] for r in new));cfg['prior_replay_count']=len(priorids)
    cfg.update(new_source_taxa=1,accepted_compatible=len(new),excluded_unmapped=0,replay_count=len(keep),
        sparse_classes='Only 6 new observers; split by observer, report small-sample limitations',
        taxonomy='Exact Aquilaria sinensis target; preserve all other existing pooled output labels')
    cfg['audited_old_positive_train']=sum(oldtrain[i]['target']==AQ for i in keep)
    cfg['new_group_counts']={p:len({r['split_group'] for r in new if r['phase']==p}) for p in ('train','planning','test')}
    contract={'initial_heads':[{'path':str(DEPLOY.parent/r['file']),'sha256':r['sha256']} for r in dep['members']],
        'backbone':{'path':str(TEACHER),'sha256':dep['encoder_sha256']},
        'protected_artifacts':[{'path':str(p),'sha256':sha256(p)} for p in [DEPLOY,ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib',ROOT/'tree_species_teacher_demo/engine.py',ROOT/'spacescan_trees_demo_v1/server.py']]}
    for r in contract['initial_heads']+[contract['backbone']]:assert sha256(r['path'])==r['sha256']
    save_json(out/'base_contract.json',contract);save_json(out/'config.json',cfg)
    # Existing positive TRAIN data only; augmentations never enter validation/test.
    features=[]
    for r in [oldtrain[i] for i in keep if oldtrain[i]['target']==AQ]+new:
        phase=r.get('phase','train');source_image=r['source_image']
        expected=r.get('sha256') or r.get('source_sha256');assert sha256(source_image)==expected
        for brightness in ([1.,.7,1.2] if phase=='train' else [1.]):
            features.append({**r,'phase':phase,'brightness':brightness,'source_image':source_image,'sample_id':r['sample_id']+f'_brightness_{brightness}'})
    save_json(out/'feature_rows.json',features)
    print('PREPARED',cfg['new_counts'],'audited old positives',cfg['audited_old_positive_train'],'feature views',len(features),flush=True)

def extract(out):
    from tree_species_teacher_demo.engine import Engine,decode
    engine=Engine();engine.load();rows=read_json(out/'feature_rows.json');xx=[];zz=[]
    with torch.inference_mode():
        for i,r in enumerate(rows):
            with Image.open(r['source_image']) as src:
                im=ImageOps.exif_transpose(src).convert('RGB');im=ImageEnhance.Brightness(im).enhance(r['brightness'])
                edge=min(1024,max(im.size));im=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128));buf=io.BytesIO();im.save(buf,format='JPEG',quality=95)
            pixels=decode(buf.getvalue())[0][None].cuda()
            e=engine.model.encode_image(((pixels-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            z=e@engine.bank.T*engine.model.logit_scale.exp().float()
            xx.append(e.half().cpu().numpy());zz.append(z.cpu().numpy())
            if i%40==0:print('EMBED',i,len(rows),flush=True)
        extra=engine.extra_bank.cpu().numpy();scale=float(engine.model.logit_scale.exp())
    np.savez_compressed(out/'features.npz',x=np.concatenate(xx),z=np.concatenate(zz),extra_bank=extra,logit_scale=scale,rows_hash=digest(rows))
    del engine;gc.collect();torch.cuda.empty_cache()

def subset(d,ids):return ([d[0][i] for i in ids],d[1][ids],d[2][ids])
def join(a,b):return (a[0]+b[0],torch.cat([a[1],b[1]]),torch.cat([a[2],b[2]]))
def labels(d):return np.array([r['target'] for r in d[0]],dtype=np.int64)
def aqmetrics(d,p):
    y=labels(d);pred=p.argmax(1);tp=int(((y==AQ)&(pred==AQ)).sum());fn=int(((y==AQ)&(pred!=AQ)).sum());fp=int(((y!=AQ)&(pred==AQ)).sum())
    return {'positive_count':tp+fn,'correct':tp,'false_positive':fp,'recall':tp/(tp+fn) if tp+fn else 0,'precision':tp/(tp+fp) if tp+fp else 0,'f1':2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0}

def train(out):
    cfg=read_json(out/'config.json');contract=read_json(out/'base_contract.json');classes=read_json(out/'classes.json');groups=classes['groups'];temp=cfg['temperature']
    for r in contract['initial_heads']+[contract['backbone']]+contract['protected_artifacts']:
        if sha256(r['path'])!=r['sha256']:raise ValueError('Pinned initialization or deployed artifact changed before fitting')
    rows=read_json(out/'feature_rows.json')
    with np.load(out/'features.npz') as f:
        assert str(f['rows_hash'])==digest(rows)
        assert np.isfinite(f['x']).all() and np.isfinite(f['z']).all()
        features=(rows,torch.tensor(f['x'].astype(np.float32),device='cuda'),torch.tensor(f['z'],device='cuda'));extra=torch.tensor(f['extra_bank'],device='cuda');scale=float(f['logit_scale'])
    target=subset(features,[i for i,r in enumerate(rows) if r['phase']=='train'])
    newval=subset(features,[i for i,r in enumerate(rows) if r['phase']=='planning'])
    replay=load_old(out,'train')
    priorrows=read_json(PREV/'accepted_new_manifest.json');ids=read_json(out/'prior_replay_indices.json')
    with np.load(PREV/'new_features.npz') as f:
        assert str(f['rows_hash'])==digest(priorrows)
        replay=join(replay,([priorrows[i] for i in ids],torch.tensor(f['x'][ids].astype(np.float32),device='cuda'),torch.tensor(f['z'][ids],device='cuda')))
    reliable={str(r['observation']) for r in read_json(out/'old_source_audit.json') if r['current_name']=='Aquilaria sinensis' and r['quality_grade']=='research'}
    def audited(d):return subset(d,[i for i,r in enumerate(d[0]) if r['target']!=AQ or r['group_id'].split('observation:')[-1] in reliable])
    oldval=audited(load_old(out,'planning'));selection=audited(load_old(out,'selection'))
    save_json(out/'validation_assignment.json',{'planning':[r['sample_id'] for r in oldval[0]],'selection':[r['sample_id'] for r in selection[0]],'new':[r['sample_id'] for r in newval[0]]})
    initial=[]
    for member in contract['initial_heads']:
        state=torch.load(member['path'],map_location='cpu',weights_only=False);assert state['classes']==classes and state['encoder_sha256']==contract['backbone']['sha256']
        h=ResidualHead('mlp').cuda().eval();h.load_state_dict(state['model_state']);initial.append(h)
    predict=lambda h,d:probs(h,d,groups,temp,extra,scale)
    ensemble=lambda hs,d:np.mean([predict(h,d) for h in hs],axis=0)
    # Hard negatives are mined ONLY from fixed training rows; original labels retained.
    pr=ensemble(initial,replay);neg=np.where(labels(replay)!=AQ)[0];hardids=neg[np.argsort(-pr[neg,AQ])[:256]].tolist();hard=subset(replay,hardids)
    save_json(out/'hard_negative_manifest.json',[{**r,'starting_aquilaria_score':float(pr[i,AQ])} for r,i in zip(hard[0],hardids)])
    y=[torch.tensor(labels(d),device='cuda',dtype=torch.long) for d in (replay,target,hard)]
    count=Counter(labels(replay));rw=torch.tensor([1/count[r['target']] for r in replay[0]],device='cuda');rw/=rw.mean()
    counts=Counter(r['group_id'] for r in target[0]);tw=torch.tensor([1/counts[r['group_id']] for r in target[0]],device='cuda');tw/=tw.mean()
    candidates=[];runs=[]
    for n,base in enumerate(initial):
        seed=SEED+n;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        run=out/f'member_{n+1}_seed_{seed}';run.mkdir(exist_ok=False)
        head=ResidualHead('mlp').cuda();head.load_state_dict(base.state_dict())
        p0=predict(base,oldval);pn=predict(base,newval);b=score(oldval,p0);ba=aqmetrics(oldval,p0);bn=aqmetrics(newval,pn)
        save_json(run/'baseline_evaluation.json',{'old':b,'aq_old':ba,'new':bn,'new_predictions':pn.tolist()})
        save_json(run/'config.json',{**cfg,'seed':seed,'initial_head':contract['initial_heads'][n]})
        protected=[TEACHER,*[Path(r['path']) for r in contract['initial_heads']+contract['protected_artifacts']],*deployed_artifacts()]
        backup=backup_before_training(run,[*protected,Path(__file__),ROOT/'tools/train_sstv1_incremental.py',ROOT/'tools/training_safety.py',ROOT/'tools/bioclip_probe.py',ROOT/'tools/distillation_core.py',out/'features.npz',DATA/'manifest.json',DATA/'source_audit.json',DATA/'source_responses/inat_observations.json',DATA/'source_responses/old_aquilaria_observations.json',*[p for p in out.glob('*.json')],run/'baseline_evaluation.json',run/'config.json'],{'run':'SSTv1-Aquilaria','seed':seed,'human_review_not_fabricated':True})
        print('BACKUP VERIFIED',backup,flush=True)
        with torch.no_grad():anchor=logits(base,replay[1],replay[2],groups,temp,extra,scale).softmax(-1).detach()
        best=(ba['f1']+bn['recall'])/2;best_epoch=0;best_state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()};history=[]
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=.01)
        for epoch in range(1,cfg['epochs']+1):
            head.train();optimizer.zero_grad(set_to_none=True)
            a=logits(head,replay[1],replay[2],groups,temp,extra,scale);t=logits(head,target[1],target[2],groups,temp,extra,scale);h=logits(head,hard[1],hard[2],groups,temp,extra,scale)
            loss=.75*(F.cross_entropy(a,y[0],reduction='none')*rw).mean()+.15*(F.cross_entropy(t,y[1],reduction='none')*tw).mean()+.10*F.cross_entropy(h,y[2])+.20*F.kl_div(a.log_softmax(-1),anchor,reduction='batchmean')
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step()
            pv=predict(head,oldval);nv=predict(head,newval);m=score(oldval,pv);aq=aqmetrics(oldval,pv);nq=aqmetrics(newval,nv);objective=(aq['f1']+nq['recall'])/2
            if m['macro_f1']>=b['macro_f1']-.005 and aq['false_positive']<=ba['false_positive']+1 and objective>best:
                best=objective;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            history.append({'epoch':epoch,'loss':float(loss),'old_macro_f1':m['macro_f1'],'old_aq':aq,'new_aq':nq})
            if epoch%15==0:print('FIT',n+1,epoch,'best',best_epoch,'aq',aq,flush=True)
        torch.save({'model_state':head.state_dict(),'optimizer_state':optimizer.state_dict(),'epoch':cfg['epochs'],'config':{**cfg,'architecture':'mlp'},'classes':classes,'encoder_sha256':contract['backbone']['sha256']},run/'last.pt')
        torch.save({'model_state':best_state,'epoch':best_epoch,'config':{**cfg,'architecture':'mlp'},'classes':classes,'encoder_sha256':contract['backbone']['sha256']},run/'best.pt')
        save_json(run/'history.json',history);head.load_state_dict(best_state);candidates.append(head.eval())
        runs.append({'path':str(run/'best.pt'),'sha256':sha256(run/'best.pt'),'epoch':best_epoch,'backup':str(backup)})
        verify_backup(backup);del optimizer,anchor;gc.collect();torch.cuda.empty_cache()
    validation={}
    for name,d in [('selection',selection),('new_planning',newval)]:
        b=ensemble(initial,d);c=ensemble(candidates,d);validation[name]={'baseline':score(d,b),'candidate':score(d,c),'aq_baseline':aqmetrics(d,b),'aq_candidate':aqmetrics(d,c)}
    v=validation['selection'];nv=validation['new_planning'];qualified=v['candidate']['macro_f1']>=v['baseline']['macro_f1']-.005 and v['aq_candidate']['false_positive']<=v['aq_baseline']['false_positive']+1
    better=v['aq_candidate']['f1']+nv['aq_candidate']['recall']>v['aq_baseline']['f1']+nv['aq_baseline']['recall']
    save_json(out/'validation_results.json',validation)
    save_json(out/'selection_lock.json',{'model_name':'SSTv1-Aquilaria','members':runs,'validation_recommends_candidate':qualified and better,'deployed':False,'heldout_not_yet_scored':True})
    results={}
    test=load_old(out,'test');newtest=subset(features,[i for i,r in enumerate(rows) if r['phase']=='test'])
    for name,d in [('old_test_original_labels',test),('old_test_source_audited',audited(test)),('new_author_holdout',newtest),('field_regression',load_old(out,'field'))]:
        b=ensemble(initial,d);c=ensemble(candidates,d);r={'baseline':score(d,b),'candidate':score(d,c),'aq_baseline':aqmetrics(d,b),'aq_candidate':aqmetrics(d,c),
            'paired_gain':bootstrap_gain(labels(d),b,c,[r.get('split_group',r['group_id']) for r in d[0]],seed=SEED),
            'corrected':[r['sample_id'] for r,y,a,z in zip(d[0],labels(d),b.argmax(1),c.argmax(1)) if a!=y and z==y],
            'regressed':[r['sample_id'] for r,y,a,z in zip(d[0],labels(d),b.argmax(1),c.argmax(1)) if a==y and z!=y]}
        results[name]=r;save_json(out/f'{name}.json',r);save_json(out/f'{name}_rows.json',d[0]);np.savez_compressed(out/f'{name}_predictions.npz',baseline=b,candidate=c,targets=labels(d))
        print('RESULT',name,r['aq_baseline'],r['aq_candidate'],flush=True)
    for r in runs:verify_backup(r['backup'])
    for r in contract['initial_heads']+[contract['backbone']]+contract['protected_artifacts']:assert sha256(r['path'])==r['sha256']
    save_json(out/'COMPLETE.json',{'model':'SSTv1-Aquilaria','members':runs,'training_epochs':45*3,'validation_recommends_candidate':qualified and better,'deployed':False,'production_verified_unchanged':True})
    text=['# SSTv1-Aquilaria 加强训练结果','',f"新图自动来源＋目视审核通过 {sum(cfg['new_counts'].values())} 张，划分 {cfg['new_counts']}；没有伪造人工审核记录。",f"在当前 SSTv1 三个分类头基础上各训练 45 轮，保留原 148 类。验证集建议切换：{qualified and better}；实际未部署。",'','| 数据 | 土沉香样本数 | 原召回率 | 候选召回率 | 原误报数 | 候选误报数 | 全部类别准确率变化 |','|---|---:|---:|---:|---:|---:|---:|']
    for k,r in results.items():
        a,b=r['aq_baseline'],r['aq_candidate'];text.append(f"| {k} | {a['positive_count']} | {a['recall']:.2%} | {b['recall']:.2%} | {a['false_positive']} | {b['false_positive']} | {r['baseline']['top1']:.2%} → {r['candidate']['top1']:.2%} |")
    text+=['','网络图像结果不等同于街景精度；新留出只有少量观察者与图片。旧验证／测试已检查多次，均属回归诊断。源鉴定降为纲级的旧土沉香样本从审核版评价中排除，未改写原标签；原始固定标签结果也保留。','新图只取 CC0／CC BY／CC BY-SA，保留原文件与作者信息；非专家鉴定保证。原图未发布到 Demo。','仅图像分类训练；检测器、点云和融合模型未修改，未使用 991 三株作为新独立测试。','','## 可回退备份']+[r['backup'] for r in runs]
    (out/'REPORT.md').write_text('\n\n'.join(text),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','extract','train','all'],default='all');p.add_argument('--reuse-features',type=Path);a=p.parse_args();out=a.out.resolve()
    if a.stage in ('prepare','all'):prepare(out)
    if a.stage in ('extract','all'):
        if a.reuse_features:
            source=a.reuse_features.resolve()/'features.npz';target=out/'features.npz'
            if target.exists():raise FileExistsError(target)
            with np.load(source) as f:
                if str(f['rows_hash'])!=digest(read_json(out/'feature_rows.json')):raise ValueError('Feature reuse rows mismatch')
            source_hash=sha256(source);shutil.copy2(source,target)
            if sha256(target)!=source_hash:raise ValueError('Feature copy failed')
            save_json(out/'feature_reuse.json',{'source':str(source),'sha256':source_hash,'copy_verified':True,'fitting_reused':False})
        else:extract(out)
    if a.stage in ('train','all'):train(out)
