"""Versioned local conservation adaptation; no production mutation or relabelling.

Freeze current BioCLIP and all three deployed MLPs. Append two zero-shot labels
and learn a shared 3-output linear residual at the three exact target species'
final logits. Existing labels and pooled taxonomy remain immutable.
"""
import argparse
import gc
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from bioclip_probe import ResidualHead,SOURCE,TEACHER
from distillation_core import read_json,save_json,digest,collapse_logits,metrics
from train_sstv1_incremental import prepare_pixels
from training_safety import sha256,backup_before_training,verify_backup,deployed_artifacts

PRIOR=ROOT/'outputs/SSTv1_Aquilaria_User_20260914_v1'
PREV=ROOT/'outputs/SSTv1_incremental_20260914_v1'
DEPLOY=ROOT/'tree_species_teacher_demo/models/SSTv1_Aquilaria_User_20260915_v1/deployment.json'
DATA=ROOT/'outputs/conservation_browser_delivery_20260917_v1'
TARGETS=['Artocarpus hypargyreus','Rhodoleia championii','Keteleeria fortunei']
SEED=20260917

def current_decisions(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON');db.row_factory=sqlite3.Row
        return [dict(r) for r in db.execute('SELECT * FROM decisions')]

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    source=read_json(DATA/'source_consistent_candidates.json')
    assert len(source)==53
    dep=read_json(DEPLOY);classes=read_json(PRIOR/'classes.json')
    labels=read_json(PRIOR/'labels.json')
    assert len(labels)==148 and all(s not in labels for s in TARGETS[1:])
    labels=labels+TARGETS[1:];focus=[labels.index(s) for s in TARGETS]
    assert focus==[16,148,149]
    dbpaths=[ROOT/p for p in [
        'conservation_expansion_review/data/expansion_20260917_v1/human_review.sqlite3',
        'conservation_review/data/conservation_sources_20260917_v1/human_review.sqlite3',
        'external_datasets/herbarium_training_candidates_20260907/human_review.sqlite3',
        'external_datasets/sstv1_incremental_20260914_v1/human_review.sqlite3']]
    snapshots={str(p):current_decisions(p) for p in dbpaths}
    indexed=[r for rows in snapshots.values() for r in rows]
    rows=[]
    for original in source:
        assert original['final_lane']=='source_consistent_candidate'
        assert sha256(original['source_image'])==original['sha256']
        matches=[r for r in indexed if r.get('id')==original['id'] or r.get('sha256')==original['sha256']]
        assert not matches, 'Do not override a real human decision with batch authorization'
        site=original['source_record']['evidence']['source_site']
        assert site in {'HKHerbarium','IISc','Greening','TreesAndShrubsOnline'}, site
        # Entire source gallery stays together. Not proven independent trees.
        phase='train' if site=='HKHerbarium' else 'planning'
        rows.append(dict(original,phase=phase,target=labels.index(original['source_species']),
            sample_id=original['id'],group_id=original['source_record']['source_group'],
            split_group=original['source_record']['source_group'],
            training_eligible=phase=='train',
            training_authority='explicit_user_batch_experiment_authorization_20260917',
            label_authority='institutional_source_plus_agent_visual_review_not_human_certified',
            actual_human_review_match=False,rights_scope='local_noncommercial_research_only_no_distribution'))
    assert Counter(r['phase'] for r in rows)=={'train':38,'planning':15}
    assert not {r['group_id'] for r in rows if r['phase']=='train'} & {r['group_id'] for r in rows if r['phase']=='planning'}
    auth={'request':'那你就加入训练啊','interpreted_scope':'Use the 53 source-consistent candidates in a local non-commercial experimental training/validation pipeline; not blanket confirmation of 66 images or an assertion of expert species truth.',
          'included_candidates':53,'fit_images':38,'heldout_development_images':15,
          'human_pending_field_explanation':'Importer placeholder; no ID/SHA matched an actual decision in the inspected databases. Original files and databases remain unchanged.',
          'excluded_uncertain':12,'excluded_wrong_gallery':1,'automatic_deployment':False}
    rights={'purpose':'Local non-commercial research experiment only. No downloaded image or experimental weight publication.',
            'HKHerbarium':{'url':'https://www.herbarium.gov.hk/en/important-notices/index.html','basis':'Attributed non-commercial reproduction; commercial adaptation/distribution needs authorization. Not a blanket commercial or explicit AI license.'},
            'Greening':{'url':'https://www.greening.gov.hk/en/important-notice/index.html','basis':'Saved notice permits personal/non-commercial internal local storage; development evaluation only, no public images.'},
            'IISc':{'url':'https://indiaflora-ces.iisc.ac.in/herbsheet.php?cat=13&id=1045','license':'CC BY-NC-SA 4.0','author':'K. Sankara Rao'},
            'TSO':{'url':'https://www.treesandshrubsonline.org/about/licence/','license':'CC BY-SA 4.0','author':'John Anderson'},
            'Pukeiti':'Excluded entirely; unresolved terms and/or visual ambiguity',
            'restriction':'Keep original watermarks and attribution. No claim of unrestricted downstream use. Recheck intended use before any publication/deployment.'}
    cfg={'model':'SST-Conserve-Browser-v1','seed':SEED,'epochs':120,'lr':.005,'weight_decay':.01,
         'targets':TARGETS,'target_indices':focus,'output_count':150,'base_version':dep['version'],
         'temperature':dep['temperature'],'backbone_trainable':False,'existing_mlp_trainable':False,
         'architecture':'shared zero-initialized Linear(1024,3) adds to each frozen member calibrated final logits at indices16,148,149; probability mean across3 members',
         'text_prompts':['a photo of '+s for s in TARGETS[1:]],
         'extension_normalization':'F.normalize(float32 cached novel text vectors, dim=-1) at use time; raw cache immutable',
         'loss':'0.8 class-balanced old replay NLL + 0.2 source-and-class-balanced new NLL + 0.3 replay KL to initial 150-class baseline',
         'selection':'Validation only: maximize mean(new planning macro-F1, old planning white-species F1), then old planning macro-F1, then minimum residual norm. Require old macro-F1 no more than .005 below both native148 and expanded150 baselines; Aquilaria true positives cannot decrease and FP <= native+1. Epoch0 allowed.',
         'new_split':'HKHerbarium38 train; IISc6,Greening8,TSO1 source-heldout development15. No fresh test for new species; physical trees unknown and institutional sources can be related.',
         'old_test_scope':'Previously inspected historical regression, not fresh unbiased field accuracy.',
         'no_deployment':True}
    # Snapshot current old replay decisions, excluding actual pending/rejected auxiliary images.
    hd={r['id']:r for r in snapshots[str(dbpaths[2])]}
    old=read_json(PRIOR/'old_train_rows.json');ids=[]
    for i in read_json(PRIOR/'old_train_indices.json'):
        r=old[i];d=hd.get(r['sample_id'].removeprefix('herbarium_'),{})
        if r.get('auxiliary') and (d.get('status')!='accepted' or d.get('sha256')!=r['source_sha256']):continue
        ids.append(i)
    for p in ('train','planning','selection','test','field','case'):
        save_json(out/f'old_{p}_rows.json',read_json(PRIOR/f'old_{p}_rows.json'))
    pd={r['id']:r for r in snapshots[str(dbpaths[3])]}
    priorrows=read_json(PREV/'accepted_new_manifest.json')
    priorids=[i for i,r in enumerate(priorrows) if r['phase']=='train' and pd.get(r['id'],{}).get('status')=='accepted' and pd[r['id']]['sha256']==r['sha256']]
    aqrows=read_json(PRIOR/'feature_rows.json');aqids=[i for i,r in enumerate(aqrows) if r['phase']=='train' and r['brightness']==1.]
    replay=[old[i] for i in ids]+[priorrows[i] for i in priorids]+[aqrows[i] for i in aqids]
    held=rows+[r for p in ('planning','selection','test','field','case') for r in read_json(PRIOR/f'old_{p}_rows.json')]
    heldhash={r.get('sha256') or r.get('source_sha256') for r in held}
    heldgroups={r['group_id'] for r in held}
    keep=[];seen=set()
    for i,r in enumerate(replay):
        h=r.get('sha256') or r.get('source_sha256')
        if h in heldhash or r['group_id'] in heldgroups or h in seen:continue
        seen.add(h);keep.append(i)
    save_json(out/'replay_source_indices.json',{'old':ids,'previous_incremental':priorids,'aquilaria_originals':aqids,'concat_keep':keep})
    save_json(out/'accepted_replay_manifest.json',[replay[i] for i in keep])
    save_json(out/'new_manifest.json',rows)
    save_json(out/'user_training_authorization.json',auth)
    save_json(out/'rights_scope.json',rights)
    save_json(out/'human_review_snapshot.json',snapshots)
    save_json(out/'classes.json',classes);save_json(out/'labels.json',labels)
    save_json(out/'teacher_choice.json',read_json(PRIOR/'teacher_choice.json'))
    save_json(out/'cache_metadata.json',read_json(PRIOR/'cache_metadata.json'))
    cfg['old_source_hashes']=read_json(PRIOR/'config.json')['source_hashes']
    save_json(out/'config.json',cfg)
    protect=[TEACHER,DEPLOY,*[DEPLOY.parent/m['file'] for m in dep['members']],
             *deployed_artifacts(),ROOT/'spacescan_trees_demo_v1/server.py',ROOT/'tree_species_teacher_demo/engine.py',
             ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib',
             ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json',*dbpaths]
    save_json(out/'contract.json',{'artifacts':[{'path':str(p),'sha256':sha256(p)} for p in dict.fromkeys(protect)]})
    save_json(out/'data_summary.json',{'new':{s:dict(Counter(r['phase'] for r in rows if r['source_species']==s)) for s in TARGETS},'replay_photos':len(keep),'augmentation':'none; source images are not independent physical trees'})
    print('PREPARED',read_json(out/'data_summary.json'),flush=True)

def verify_contract(out):
    for a in read_json(out/'contract.json')['artifacts']:
        assert sha256(a['path'])==a['sha256'],a['path']

def extract(out):
    from tree_species_teacher_demo.engine import Engine
    import open_clip
    verify_contract(out)
    engine=Engine();engine.load()
    assert engine.sstv1_state=='ready'
    tokenizer=open_clip.get_tokenizer('ViT-H-14')
    cfg=read_json(out/'config.json');rows=read_json(out/'new_manifest.json')
    xs=[];zs=[]
    with torch.inference_mode():
        extension=engine.model.encode_text(tokenizer(cfg['text_prompts']).cuda(),normalize=True).float()
        for i,r in enumerate(rows):
            assert sha256(r['source_image'])==r['sha256']
            pixels=prepare_pixels(r['source_image'])[None].cuda()
            e=engine.model.encode_image(((pixels-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            z=e@engine.bank.T*engine.model.logit_scale.exp().float()
            xs.append(e.half().cpu().numpy());zs.append(z.cpu().numpy())
            if i%10==0:print('EMBED',i+1,len(rows),flush=True)
        file=out/'new_features.npz'
        assert not file.exists()
        np.savez_compressed(file,x=np.concatenate(xs),z=np.concatenate(zs),rows_hash=digest(rows),
                            extra_bank=engine.extra_bank.cpu().numpy(),extension_bank=extension.cpu().numpy(),
                            logit_scale=float(engine.model.logit_scale.exp()),encoder_sha256=engine.encoder_sha256)
    verify_contract(out)
    del engine;gc.collect();torch.cuda.empty_cache()

def load_cached(out,phase):
    if phase in ('selection','test','field','case'):
        lock=read_json(out/'selection_lock.json')
        assert sha256(lock['selected'])==lock['sha256'], 'Locked checkpoint changed'
    cfg=read_json(out/'config.json');source=SOURCE/f'{phase}_rows.json';cache=SOURCE/f'teacher_cache/bioclip25/{phase}.npz'
    assert sha256(source)==cfg['old_source_hashes'][str(source)]
    assert sha256(cache)==cfg['old_source_hashes'][str(cache)]
    rows=read_json(out/f'old_{phase}_rows.json')
    with np.load(cache) as f:
        assert str(f['rows_hash'])==digest(rows)
        x=f['embeddings'].astype(np.float32);z=f['similarities'][read_json(out/'teacher_choice.json')['prompt']].astype(np.float32)*read_json(out/'cache_metadata.json')['logit_scale']
    return rows,x,z

def replay(out):
    index=read_json(out/'replay_source_indices.json')
    old,x,z=load_cached(out,'train');parts=[(x[index['old']],z[index['old']])]
    for folder,rf,ff,key in [(PREV,'accepted_new_manifest.json','new_features.npz','previous_incremental'),(PRIOR,'feature_rows.json','features.npz','aquilaria_originals')]:
        rr=read_json(folder/rf)
        with np.load(folder/ff) as f:
            assert str(f['rows_hash'])==digest(rr)
            parts.append((f['x'][index[key]].astype(np.float32),f['z'][index[key]]))
    kept=index['concat_keep']
    return read_json(out/'accepted_replay_manifest.json'),np.concatenate([p[0] for p in parts])[kept],np.concatenate([p[1] for p in parts])[kept]

def models(out):
    verify_contract(out);dep=read_json(DEPLOY);hs=[]
    for m in dep['members']:
        state=torch.load(DEPLOY.parent/m['file'],map_location='cpu',weights_only=False)
        assert state['encoder_sha256']==dep['encoder_sha256']
        assert state['classes']==read_json(out/'classes.json'), 'Frozen head taxonomy differs'
        h=ResidualHead('mlp').cuda().eval().requires_grad_(False);h.load_state_dict(state['model_state']);hs.append(h)
    with np.load(out/'new_features.npz') as f:
        assert str(f['rows_hash'])==digest(read_json(out/'new_manifest.json'))
        assert str(f['encoder_sha256'])==dep['encoder_sha256']
        extra=torch.tensor(f['extra_bank'],device='cuda')
        # Normalize at use time, preserving the immutable raw bf16 extraction cache.
        novel=F.normalize(torch.tensor(f['extension_bank'],device='cuda',dtype=torch.float32),dim=-1)
        scale=float(f['logit_scale'])
    return hs,extra,novel,scale

def tensor_data(out,raw,model):
    rows,x,z=raw;hs,extra,novel,scale=model
    cfg=read_json(out/'config.json');groups=read_json(out/'classes.json')['groups']
    tensors=[]
    with torch.inference_mode():
        for start in range(0,len(rows),512):
            a=torch.tensor(x[start:start+512],device='cuda',dtype=torch.float32)
            b=torch.tensor(z[start:start+512],device='cuda',dtype=torch.float32)
            tensors.append(torch.stack([torch.cat([collapse_logits(h(a,b),groups),a@extra.T*scale,a@novel.T*scale],-1)/cfg['temperature'] for h in hs]))
    return {'rows':rows,'x':torch.tensor(x,device='cuda',dtype=torch.float32),'b':torch.cat(tensors,dim=1),
            'y':torch.tensor([r['target'] for r in rows],device='cuda',dtype=torch.long)}

def logp(head,d):
    residual=head(d['x'])
    delta=F.pad(residual[:,0:1],(16,133))+F.pad(residual[:,1:3],(148,0))
    return torch.logsumexp(F.log_softmax(d['b']+delta[None],-1),0)-np.log(3.)

def scores(head,d,native=False):
    with torch.inference_mode():
        return (d['b'][:,:,:148].softmax(-1).mean(0) if native else logp(head,d).exp()).cpu().numpy()

def compact(m):
    return {k:m[k] for k in ('n','top1','macro_f1','balanced_accuracy')}

def measure(head,d,native=False):
    return metrics(d['y'].cpu().numpy(),scores(head,d,native))

def aq_counts(d,p):
    y=d['y'].cpu().numpy();pred=p.argmax(1)
    return {'tp':int(((y==11)&(pred==11)).sum()),'fp':int(((y!=11)&(pred==11)).sum())}

def train(out):
    cfg=read_json(out/'config.json');torch.set_num_threads(4)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=models(out)
    tr=tensor_data(out,replay(out),model);va=tensor_data(out,load_cached(out,'planning'),model)
    nr=read_json(out/'new_manifest.json')
    with np.load(out/'new_features.npz') as f:
        nx=f['x'].astype(np.float32);nz=f['z'].copy()
    def ns(phase):
        ids=[i for i,r in enumerate(nr) if r['phase']==phase]
        return tensor_data(out,([nr[i] for i in ids],nx[ids],nz[ids]),model)
    nt,nv=ns('train'),ns('planning')
    run=out/'fit_seed_20260917';run.mkdir(exist_ok=False)
    head=torch.nn.Linear(1024,3).cuda();torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    torch.save({'model_state':head.state_dict(),'config':cfg},run/'initial.pt')
    baseold=measure(head,va);basenew=measure(head,nv);native=measure(head,va,True);aq0=aq_counts(va,scores(head,va,True))
    save_json(out/'baseline_evaluation.json',{'old_planning_150':baseold,'new_planning_150':basenew,'native148_old_planning':native,'native_aquilaria':aq0})
    save_json(out/'validation_assignment.json',{'old':[r['sample_id'] for r in va['rows']],'new':[r['sample_id'] for r in nv['rows']]})
    protected=[Path(r['path']) for r in read_json(out/'contract.json')['artifacts']]
    backup=backup_before_training(run,[run/'initial.pt',*protected,Path(__file__),ROOT/'tools/training_safety.py',ROOT/'tools/bioclip_probe.py',ROOT/'tools/distillation_core.py',ROOT/'tools/train_sstv1_incremental.py',out/'new_features.npz',*[p for p in out.glob('*.json')],DATA/'source_consistent_candidates.json',ROOT/'outputs/conservation_browser_training_audit_20260917_v1/audit.json',ROOT/'outputs/conservation_browser_training_audit_20260917_v1/prepared_audit.json',ROOT/'outputs/conservation_browser_training_audit_20260917_v1/prepared_queue_resolution.json'],{'purpose':cfg['model'],'seed':SEED,'initial_and_deployed_weights_physical_copies':True})
    print('BACKUP VERIFIED',backup,flush=True)
    def weights(d):
        counts=Counter(int(y) for y in d['y'].cpu().tolist())
        # Each source gallery gets equal mass within its class.
        groups=Counter((r['target'],r['group_id']) for r in d['rows'])
        perclass=Counter(y for y,g in groups)
        w=np.array([1/(groups[(r['target'],r['group_id'])]*perclass[r['target']]) for r in d['rows']],dtype=np.float32)
        return torch.tensor(w/w.sum(),device='cuda')
    rw,nw=weights(tr),weights(nt)
    with torch.no_grad():anchor=tr['b'].softmax(-1).mean(0).detach()
    opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    bestkey=(.5*(basenew['macro_f1']+baseold['per_class_f1'][16]),baseold['macro_f1'],0.)
    bestepoch=0;best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};history=[]
    for epoch in range(1,cfg['epochs']+1):
        head.train();opt.zero_grad(set_to_none=True)
        lp=logp(head,tr);ln=logp(head,nt)
        loss=.8*(-lp.gather(1,tr['y'][:,None]).squeeze(1)*rw).sum()+.2*(-ln.gather(1,nt['y'][:,None]).squeeze(1)*nw).sum()+.3*F.kl_div(lp,anchor,reduction='batchmean')
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);opt.step()
        head.eval();om=measure(head,va);nm=measure(head,nv);aq=aq_counts(va,scores(head,va))
        safe=om['macro_f1']>=max(native['macro_f1'],baseold['macro_f1'])-.005 and aq['tp']>=aq0['tp'] and aq['fp']<=aq0['fp']+1
        key=(.5*(nm['macro_f1']+om['per_class_f1'][16]),om['macro_f1'],-float(head.weight.detach().square().sum()))
        if safe and key>bestkey:
            bestkey=key;bestepoch=epoch;best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append({'epoch':epoch,'loss':float(loss.detach()),'old':compact(om),'new':compact(nm),'aq':aq,'safe':safe})
        if epoch%10==0:print('TRAIN',epoch,'selected',bestepoch,'old',om['macro_f1'],'new',nm['top1'],flush=True)
    torch.save({'model_state':head.state_dict(),'optimizer_state':opt.state_dict(),'epoch':cfg['epochs'],'config':cfg,'labels':read_json(out/'labels.json')},run/'last.pt')
    torch.save({'model_state':best,'epoch':bestepoch,'config':cfg,'labels':read_json(out/'labels.json'),'baseline_deployment':read_json(DEPLOY),'encoder_sha256':read_json(DEPLOY)['encoder_sha256']},run/'best.pt')
    save_json(run/'history.json',history)
    head.load_state_dict(best);head.eval()
    selected_old=measure(head,va);selected_new=measure(head,nv);selected_aq=aq_counts(va,scores(head,va))
    selected_safe=selected_old['macro_f1']>=max(native['macro_f1'],baseold['macro_f1'])-.005 and selected_aq['tp']>=aq0['tp'] and selected_aq['fp']<=aq0['fp']+1
    strict_gain=bestkey[0]>.5*(basenew['macro_f1']+baseold['per_class_f1'][16])+1e-12
    save_json(out/'selection_lock.json',{'selected':str(run/'best.pt'),'sha256':sha256(run/'best.pt'),'epoch':bestepoch,'key':bestkey,'backup':str(backup),'selection_uses_validation_only':True,'test_not_yet_opened':True,'deployment_authorized':False,'selected_safe':selected_safe,'strict_validation_gain':strict_gain,'validation_recommends_candidate':bool(selected_safe and strict_gain and bestepoch>0)})
    verify_backup(backup);verify_contract(out)
    def compare(name,d):
        zero=torch.nn.Linear(1024,3).cuda().eval();torch.nn.init.zeros_(zero.weight);torch.nn.init.zeros_(zero.bias)
        b=scores(zero,d);c=scores(head,d);y=d['y'].cpu().numpy()
        result={'baseline150':metrics(y,b),'candidate150':metrics(y,c),'aq_baseline':aq_counts(d,b),'aq_candidate':aq_counts(d,c),
                'corrected':int(((b.argmax(1)!=y)&(c.argmax(1)==y)).sum()),'regressed':int(((b.argmax(1)==y)&(c.argmax(1)!=y)).sum())}
        if int(y.max())<148:result['native148']=metrics(y,scores(head,d,True))
        save_json(out/f'{name}_evaluation.json',result)
        save_json(out/f'{name}_rows.json',d['rows'])
        np.savez_compressed(out/f'{name}_predictions.npz',baseline150=b,candidate150=c,target=y)
        print('RESULT',name,'base',compact(result['baseline150']),'candidate',compact(result['candidate150']),flush=True)
        return result
    results={}
    for name,d in [('new_training_replay',nt),('new_source_validation',nv),('old_planning',va)]:results[name]=compare(name,d)
    for phase in ('selection','test','field'):
        results['old_'+phase]=compare('old_'+phase,tensor_data(out,load_cached(out,phase),model))
    verify_backup(backup);verify_contract(out)
    save_json(out/'COMPLETE.json',{'trained':True,'model':cfg['model'],'selected_epoch':bestepoch,'selected_checkpoint':str(run/'best.pt'),'backup':str(backup),'deployed':False,'production_hashes_unchanged':True,'selected_safe':selected_safe,'validation_recommends_candidate':bool(selected_safe and strict_gain and bestepoch>0),'new_species_have_fresh_independent_test':False,'results':{k:{'baseline150':compact(v['baseline150']),'candidate150':compact(v['candidate150']),'corrected':v['corrected'],'regressed':v['regressed']} for k,v in results.items()}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','extract','train'],required=True)
    a=p.parse_args();out=a.out.resolve()
    {'prepare':prepare,'extract':extract,'train':train}[a.stage](out)
