"""Post-lock, inference-only source checks. Never fit or promote quarantine data."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import requests
from distillation_core import read_json,save_json,digest
from training_safety import sha256
from train_aq_ar_diagnostic_20260919 import unchanged
from prepare_artocarpus_fit_20260918 import identities,row_ids
from import_artocarpus_user_20260918 import current_human_review_exclusions
from spacescan_trees_demo_v1.server import prepare

RUN=ROOT/'outputs/SSTv1_AQ_AR_Dualview_20260919_v1'
NEW=ROOT/'external_datasets/aq_ar_external_check_20260919_v2'
OFFICIAL=ROOT/'outputs/conservation_authoritative_collection_20260918_v1'
OUT=RUN/'external_checks'


def main():
    OUT.mkdir(exist_ok=False)
    lock=read_json(RUN/'selection_lock.json')
    assert lock['recipe']=='initial_dual' and lock['step']==0
    assert sha256(lock['checkpoint'])==lock['sha256']
    protocol=read_json(RUN/'protocol.json');unchanged(protocol)
    review=current_human_review_exclusions()
    train=read_json(RUN/'train_rows.json')
    tokens=set().union(*(identities(r) for r in train))
    held=set().union(*(set(r['identity_tokens']) for r in read_json(RUN/'heldout_metadata.json')['records']))
    labels=read_json(RUN/'labels.json')
    organs={717499736:'fruit and leaves',715308898:'trunk, small shoots, overlaid common name',
        706992513:'fruit and leaves',704992602:'fruit on trunk',681333944:'leaves',
        678448082:'harvested fruit, mixed background',663822381:'fruit, backlit',
        651458451:'young fruit and foliage, dark, original watermark retained',
        619421523:'fruit and leaves',606067990:'fruit on trunk, mixed foliage background'}
    rows=[]
    for source in read_json(NEW/'manifest.json'):
        r=dict(source);pid=r['photo_id']
        r.update(cohort='new_source_community_negative',organ=organs[pid],
            agent_content_review='Direct visual inspection 2026-09-19; visible content not contradictory to source label; NOT expert certification',
            evaluate=pid!=715308898,
            evaluation_exclusion='Trunk-only target and species-name text overlay; excluded BEFORE inference' if pid==715308898 else None)
        rows.append(r)
    old_audit={r['id']:r for r in read_json(OFFICIAL/'audit/manifest.json')}
    for source in read_json(OFFICIAL/'hk_official/manifest.json')[:2]:
        r=dict(source);old=old_audit[r['id']]
        r.update(cohort='previously_collected_official_detail_illustration',organ='leaves and fruits',
            agent_content_review='Direct visual inspection 2026-09-19; leaves and fruits, source-consistent, not whole tree',
            evaluate=True,evaluation_exclusion=None,prior_source_audit={k:old.get(k) for k in
                ('historical_identity_matches','historical_cache_matches','latest_training_study_matches','historical_heldout_related_ids')},
            independence_warning='Collected 2026-09-18; only qualitative illustration checks, not new independent street-tree validation')
        rows.append(r)
    for r in rows:
        assert sha256(r['source_image'])==r['sha256']
        reasons=[]
        if identities(r)&tokens:reasons.append('Current training identity overlap')
        if identities(r)&held:reasons.append('Known held-out identity overlap')
        if r['sha256'] in review['blocked_sha256'] or row_ids(r)&set(review['blocked_ids']):
            reasons.append('Human rejected/pending decision')
        prior=r.get('prior_source_audit',{})
        if any(prior.values()):reasons.append('Prior source audit has overlap; retain only as quarantined reference')
        if reasons:r['evaluate']=False;r['evaluation_exclusion']='; '.join(reasons)
        r['training_eligible']=False
    # Immutable eligibility recorded before any classifier result is queried.
    save_json(OUT/'frozen_evaluation_manifest.json',dict(rows=rows,selection_lock_sha256=sha256(RUN/'selection_lock.json'),
        manifest_hash=digest(rows),filtered_using_predictions=False,training_used=0))
    results=[]
    version=protocol['deployment_manifest']['version']
    with requests.Session() as client:
        health=client.get('http://127.0.0.1:8039/api/health',timeout=15);health.raise_for_status()
        assert health.json()['sstv1_version']==version
        for r in rows:
            if not r['evaluate']:continue
            raw=Path(r['source_image']).read_bytes();pad=prepare(raw)[0]
            scores=[]
            for content in (pad,raw):
                response=client.post('http://127.0.0.1:8039/api/predict',data={'mode':'sstv1'},
                    files={'image':('image.jpg',content)},timeout=120)
                response.raise_for_status();p=response.json();assert p['deployment']==version
                scores.append(np.array([p['sstv1_distribution'][label] for label in labels]))
            t=labels.index(r['source_species'])
            def brief(p):
                return dict(top1=labels[int(p.argmax())],correct=int(p.argmax())==t,
                    true_score=float(p[t]),aquilaria_score=float(p[11]),white_score=float(p[16]),
                    top3=[dict(species=labels[int(j)],score=float(p[j])) for j in np.argsort(-p)[:3]])
            result=dict(id=r['id'],species=r['source_species'],cohort=r['cohort'],source_image=r['source_image'],
                source_page=r['source_page'],organ=r['organ'],single=brief(scores[0]),center=brief(scores[1]),dual=brief((scores[0]+scores[1])*.5))
            results.append(result);save_json(OUT/'partial_predictions.json',results)
            print(r['id'],result['single']['top1'],'->',result['dual']['top1'],flush=True)
    counts={}
    for cohort in sorted({r['cohort'] for r in results}):
        part=[r for r in results if r['cohort']==cohort]
        counts[cohort]={v:dict(n=len(part),correct=sum(r[v]['correct'] for r in part),
            predicted_aquilaria=sum(r[v]['top1']==labels[11] for r in part),
            predicted_white=sum(r[v]['top1']==labels[16] for r in part)) for v in ('single','center','dual')}
    unchanged(protocol)
    save_json(OUT/'results.json',dict(model_version=version,selected_is_original_weights=True,counts=counts,images=results,
        limitations='Small fruit/leaf-detail source checks. NOT street-tree accuracy, independent expert labels, or evidence of training improvement. No checkpoint or recipe chosen using these results.',
        excluded=[dict(id=r['id'],reason=r['evaluation_exclusion']) for r in rows if not r['evaluate']],
        formal_files_unchanged=True,training_used=0))
    print(counts,flush=True)


if __name__=='__main__':main()
