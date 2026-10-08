"""Real uploaded-XYZ inference. No file-name lookup, fitting or cached case scores."""
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from . import point_assist

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((Path(__file__).parent/'models/ptv2_seven_20260913_v1/deployment.json').read_text(encoding='utf-8'))
CLASSES = ['Albizia lebbeck','Aleurites moluccana','Aquilaria sinensis','Bauhinia x blakeana','Ficus microcarpa','Livistona chinensis','Wodyetia bifurcata']
LOCK = threading.Lock()
_engine = None
_bundle = None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def available():
    return all((ROOT/CONFIG[k]).is_file() for k in ('model','encoder'))

def infer_upload(data, name, units, z_up):
    global _engine, _bundle
    xyz, meta = point_assist.decode(data,name,units)
    evidence = point_assist.analyze(xyz,units,z_up)
    evidence.update(input_sha256=hashlib.sha256(data).hexdigest(),input_name=Path(name).name,normalization=meta)
    if evidence['unique_count']<256 or evidence['eigen_min_fraction']<.01 or not z_up:
        return evidence, None
    started=time.perf_counter()
    with LOCK:
        import joblib
        import torch
        sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tree_species_demo')]
        from hk_three_species import Features, POINT
        from rebuild_banyan_coverage_preview import sample_ids
        if _bundle is None:
            if sha(ROOT/CONFIG['model'])!=CONFIG['model_sha256'] or sha(POINT)!=CONFIG['encoder_sha256']:
                raise RuntimeError('Point model checksum mismatch')
            candidate=joblib.load(ROOT/CONFIG['model'])
            if candidate['classes']!=CLASSES or candidate['encoder']!='ptv2' or candidate['encoder_sha256']!=CONFIG['encoder_sha256'] or candidate.get('aggregation') is not None:
                raise RuntimeError('Incompatible point model')
            _bundle=candidate
            torch.set_num_threads(4)
            _engine=Features('cuda' if torch.cuda.is_available() else 'cpu')
        x=np.unique(xyz,axis=0)
        selected=x[sample_ids(x,2048)].astype('float32')
        center=(x.min(0)+x.max(0))/2
        scale=max(float(np.linalg.norm(x-center,axis=1).max()),1e-6)
        p=((selected-center)/scale).astype('float32')
        features=[]
        with torch.inference_mode():
            for degree in (0,90,180,270):
                a=np.deg2rad(degree)
                r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],dtype='float32')
                features.append(_engine.encode_point(p@r.T))
        f=np.mean(features,axis=0)[None]
        f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)*10
        scores=_bundle['head'].predict_proba(f)[0]
    evidence['classification_used']=True
    return evidence, dict(classes=CLASSES,scores=scores.tolist(),version=CONFIG['version'],model_sha256=CONFIG['model_sha256'],encoder_sha256=CONFIG['encoder_sha256'],inference_seconds=time.perf_counter()-started,sampled_points=len(selected),rotations=[0,90,180,270],feature_aggregation='mean_then_l2_times10',sampling='unique sorted XYZ; height/sector coverage sampling',metric_height_used_by_classifier=False)

def fuse(image_distribution, point):
    # The existing image model retains its historical merged Ficus class.
    mapping={('Ficus microcarpa' if s.startswith('Ficus microcarpa /') else s):float(v) for s,v in image_distribution.items()}
    if not all(c in mapping for c in CLASSES):raise RuntimeError('Full teacher distribution unavailable')
    image=np.array([mapping[c] for c in CLASSES])
    if not np.isfinite(image).all() or (image<0).any() or image.sum()<=0:raise RuntimeError('Invalid image distribution')
    mass=float(image.sum());image=image/mass
    cloud=np.array(point['scores'])
    if cloud.shape!=(len(CLASSES),) or not np.isfinite(cloud).all() or (cloud<0).any() or not np.isclose(cloud.sum(),1):raise RuntimeError('Invalid point distribution')
    w=CONFIG['fusion_point_weight']
    return image,cloud,(1-w)*image+w*cloud,mass
