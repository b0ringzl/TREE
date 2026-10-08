"""Read-only model verification and paired Demo-preprocessed regression, no fitting."""
import argparse,io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,requests
from bioclip_probe import ResidualHead
from train_sstv1_incremental import prepare_pixels,logits
from train_aquilaria_sstv1 import aqmetrics,labels,AQ
from distillation_core import read_json,save_json,metrics
from training_safety import sha256,verify_backup
from tree_species_teacher_demo.engine import Engine
from spacescan_trees_demo_v1.server import prepare

def main(run):
    complete=read_json(run/'COMPLETE.json');cfg=read_json(run/'config.json');contract=read_json(run/'base_contract.json');names=read_json(run/'labels.json')
    for r in complete['members']:verify_backup(r['backup'])
    for r in contract['initial_heads']+[contract['backbone']]+contract['protected_artifacts']:assert sha256(r['path'])==r['sha256']
    for phase in ('old_test_original_labels','old_test_source_audited','new_author_holdout','field_regression'):
        rows=read_json(run/f'{phase}_rows.json');saved=read_json(run/f'{phase}.json')
        with np.load(run/f'{phase}_predictions.npz') as f:
            for branch in ('baseline','candidate'):
                assert np.isfinite(f[branch]).all() and np.allclose(f[branch].sum(1),1,atol=1e-5)
                assert metrics(f['targets'],f[branch])['top1']==saved[branch]['top1']
                assert aqmetrics((rows,None,None),f[branch])==saved['aq_'+branch]
    engine=Engine();engine.load();heads=[]
    for member in complete['members']:
        assert sha256(member['path'])==member['sha256']
        state=torch.load(member['path'],map_location='cpu',weights_only=False)
        assert state['classes']==engine.classes and state['encoder_sha256']==engine.encoder_sha256
        h=ResidualHead('mlp').cuda().eval();h.load_state_dict(state['model_state']);heads.append(h)
    rows=[r for r in read_json(run/'old_test_source_audited_rows.json') if r['target']==AQ]
    rows += [r for r in read_json(run/'accepted_new_manifest.json') if r['phase']=='test']
    records=[];bp=[];cp=[];live_delta=[]
    with torch.inference_mode():
        for i,r in enumerate(rows):
            pixels=prepare_pixels(r['source_image'])[None].cuda()
            e=engine.model.encode_image(((pixels-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            raw=e@engine.bank.T*engine.model.logit_scale.exp().float();x=e.half().float()
            def p(hs):return torch.stack([logits(h,x,raw,engine.classes['groups'],cfg['temperature'],engine.extra_bank,engine.model.logit_scale.exp().float()).softmax(-1) for h in hs]).mean(0)[0].cpu().numpy()
            b=p(engine.sstv1_heads);c=p(heads);bp.append(b);cp.append(c)
            # Identical pixels through the still-deployed upstream (no model switch).
            if i==0:
                original=Path(r['source_image']).read_bytes();prepared,_,_=prepare(original)
                response=requests.post('http://127.0.0.1:8039/api/predict',files={'image':('audit.jpg',prepared,'image/jpeg')},data={'mode':'sstv1'},timeout=120);response.raise_for_status()
                live=response.json();livep=np.array([live['sstv1_distribution'][name] for name in names]);delta=float(np.max(np.abs(b-livep)));assert delta<1e-5;live_delta.append(delta)
            def top(p):return [{'species':names[j],'score':float(p[j])} for j in np.argsort(-p)[:3]]
            records.append({'id':r['sample_id'],'source_image':r['source_image'],'truth':'Aquilaria sinensis','baseline_top3':top(b),'candidate_top3':top(c),'baseline_correct':int(b.argmax())==AQ,'candidate_correct':int(c.argmax())==AQ,'source':'new' if r.get('review_authority')=='agent_source_and_visual_review' else 'old'})
            if i%10==0:print('DEMO REPLAY',i,len(rows),flush=True)
    paired={'input':'Exact Demo gray letterbox, JPEG95, single image per inference; automatic detector not evaluated','n':len(rows),
        'baseline_aquilaria':aqmetrics((rows,None,None),np.array(bp)),'candidate_aquilaria':aqmetrics((rows,None,None),np.array(cp)),
        'only_positive_photos':True,'not_a_specificity_test':True,'records':records,'live_baseline_max_abs_error':live_delta}
    save_json(run/'demo_preprocessing_replay.json',paired)
    save_json(run/'VERIFIED.json',{'passed':True,'physical_backups_verified':len(complete['members']),'production_verified_unchanged':True,'candidate_heads_compatible':True,'prediction_metrics_recomputed':True,'demo_preprocessing_positive_replay':len(rows),'live_baseline_matches':True,'no_training_or_deployment_in_verification':True})
    print('VERIFIED',paired['baseline_aquilaria'],paired['candidate_aquilaria'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
