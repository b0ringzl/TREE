"""Experimental seven-class point-only inference. No fitting or Demo switch."""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tree_species_demo')]
from hk_three_species import Features,POINT
from rebuild_banyan_coverage_preview import sample_ids
from training_safety import sha256

def predict(model,cloud):
 bundle=joblib.load(model)
 assert bundle['encoder']=='ptv2' and bundle['encoder_sha256']==sha256(POINT),'Encoder/label compatibility mismatch'
 data=np.load(cloud,allow_pickle=False)
 if isinstance(data,np.lib.npyio.NpzFile):
  key=next((k for k in ['xyz','metric_xyz','metric'] if k in data),None)
  if key is None:raise ValueError('NPZ must contain measured XYZ: xyz, metric_xyz or metric')
  x=data[key].copy();data.close()
 else:x=data
 assert x.ndim==2 and x.shape[1]>=3,'Expected Nx3 measured XYZ'
 x=x[:,:3][np.isfinite(x[:,:3]).all(1)].astype('float32')
 _,unique=np.unique(x,axis=0,return_index=True);x=x[np.sort(unique)]
 assert len(x)>=64,'Too few finite points (minimum 64)'
 m=x[sample_ids(x,2048)];center=(x.min(0)+x.max(0))/2;scale=max(float(np.linalg.norm(x-center,axis=1).max()),1e-6);p=((m-center)/scale).astype('float32')
 torch.set_num_threads(4);torch.manual_seed(20260913);encoder=Features('cuda' if torch.cuda.is_available() else 'cpu');features=[]
 for degree in [0,90,180,270]:
  a=np.deg2rad(degree);r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],dtype='float32')
  with torch.inference_mode():features.append(encoder.encode_point(p@r.T))
 if bundle.get('aggregation')=='mean_probabilities_after_per_rotation_feature_normalization':
  f=np.asarray(features);f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)*10;scores=bundle['head'].predict_proba(f).mean(0)
 else:
  if bundle.get('aggregation') is not None:raise ValueError('Unsupported feature aggregation')
  f=np.mean(features,0)[None];f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)*10;scores=bundle['head'].predict_proba(f)[0]
 order=np.argsort(-scores)
 return dict(model=str(Path(model).resolve()),cloud=str(Path(cloud).resolve()),points=len(x),prediction=bundle['classes'][order[0]],candidates=[dict(species=bundle['classes'][i],score=float(scores[i])) for i in order],score_kind='Uncalibrated model score; seven-class candidate ranking, not confirmed identification',experimental=True,images_used=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cloud');p.add_argument('--model',default=str(ROOT/'outputs/aquilaria_ptv2_fewshot_20260913_44056021/runs/final_3shot/model.joblib'));a=p.parse_args();print(json.dumps(predict(a.model,a.cloud),ensure_ascii=False,indent=2))
