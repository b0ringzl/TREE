"""Read-only paired whole-image/center-crop audit on confirmed TRAIN photos."""
import json
from pathlib import Path
import requests
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/user_training_view_audit_20260919_v2'


def main():
    OUT.mkdir(exist_ok=False)
    usage=json.loads((ROOT/'outputs/user_folder_split_audit_20260919_v1/file_usage.json').read_text('utf-8'))
    prior=json.loads((ROOT/'outputs/user_folder_score_audit_20260919_v1/results.json').read_text('utf-8'))
    full={r['sha256']:r for r in prior['images']}
    train=json.loads((ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1/train_rows.json').read_text('utf-8'))
    labels=json.loads((ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1/labels.json').read_text('utf-8'))
    ids={r['sha256']:i for i,r in enumerate(train)}
    with np.load(ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1/fit/selected_predictions.npz') as cache:
        pad=cache['train'].copy()
    old=json.loads((ROOT/'outputs/user_training_view_audit_20260919_v1/results.json').read_text('utf-8'))
    rows=list(old['images'])
    done={r['sha256'] for r in rows}
    with requests.Session() as client:
        health=client.get('http://127.0.0.1:8039/api/health',timeout=10).json()
        assert health['sstv1_version']==usage['model_version']
        for item in usage['rows']:
            if item['lane'] not in ('new_train_confirmed','accepted_new_training') or item['sha256'] in done:
                continue
            path=Path(item['path']); data=path.read_bytes()
            import hashlib
            assert hashlib.sha256(data).hexdigest()==item['sha256']
            result=client.post('http://127.0.0.1:8039/api/predict',data={'mode':'sstv1'},
                files={'image':(path.name,data)},timeout=90)
            result.raise_for_status(); result=result.json()
            assert result['deployment']==usage['model_version']
            p=pad[ids[item['sha256']]]; q=np.array([result['sstv1_distribution'][label] for label in labels])
            t=labels.index(item['species']); mixed=(p+q)*.5
            assert abs(float(p[t])-full[item['sha256']]['expected_score'])<5e-5
            def summary(scores):
                return dict(top1=labels[int(scores.argmax())],correct=int(scores.argmax())==t,
                    true_score=float(scores[t]),rank=int((scores>scores[t]).sum()+1))
            rows.append(dict(**item,pad=summary(p),center=summary(q),dual=summary(mixed),quality=result['quality']))
            if len(rows)%20==0:
                print('VIEW AUDIT',len(rows),flush=True)
    summary=[]
    for species in sorted({r['species'] for r in rows}):
        part=[r for r in rows if r['species']==species]
        summary.append(dict(species=species,n=len(part),
            views={v:dict(correct=sum(r[v]['correct'] for r in part),
                          true_score_mean=float(np.mean([r[v]['true_score'] for r in part]))) for v in ('pad','center','dual')}))
    (OUT/'results.json').write_text(json.dumps(dict(
        interpretation='User training photos only. Frozen deployed model. Crop changes effective resolution, context and JPEG pixels; not a resolution-only intervention. No fitting, no deployment, no heldout selection.',
        model_version=usage['model_version'],summary=summary,images=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
