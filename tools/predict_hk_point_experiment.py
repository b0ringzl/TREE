"""Offline inference for the experimental point heads; no image or label lookup."""
import argparse,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tree_species_demo'))
from hk_point_training_20260912 import OUT,clean,descriptors
from hk_point_model import predict_bundle
from training_safety import sha256
def main():
    import torch,joblib
    from hk_three_species import Features,POINT
    p=argparse.ArgumentParser();p.add_argument('--cloud',required=True);p.add_argument('--model',default=str(OUT/'runs/00_centroid_baseline/model.joblib'));a=p.parse_args();path=Path(a.cloud);modelpath=Path(a.model);torch.set_num_threads(4)
    # Only explicitly provided local experiment models; never fetch weights or labels remotely.
    if path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as f:
            if 'points_xyz' in f and 'metric_xyz' in f:points=f['points_xyz'];metric=f['metric_xyz'];q={'preprocessed_archive':True,'points':len(points)}
            else:
                key='xyz' if 'xyz' in f else 'points_xyz';x=f[key]
                if key=='points_xyz' and 'centroid_xyz' in f and 'scale' in f:x=x*f['scale']+f['centroid_xyz']
                _,metric,points,q=clean(x)
    elif path.suffix.lower() in ['.las','.laz']:
        import laspy
        las=laspy.read(path);_,metric,points,q=clean(np.c_[las.x,las.y,las.z])
    else:raise ValueError('Use a preprocessed NPZ or single-tree LAS/LAZ')
    if len(points)<512:raise ValueError('Fewer than 512 real points: insufficient experimental input quality')
    encoder=Features('cuda' if torch.cuda.is_available() else 'cpu')
    if modelpath.suffix=='.pt':
        ck=torch.load(modelpath,map_location='cpu',weights_only=False);assert sha256(POINT)==ck['encoder_sha256'];classes=ck['classes'];head=torch.nn.Sequential(torch.nn.Linear(384,128),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(128,len(classes)));head.load_state_dict(ck['state_dict']);head.eval();features=[]
        for degree in ck['config']['rotation_degrees']:
            rad=np.deg2rad(degree);rot=np.array([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]]);v=encoder.encode_point(points@rot.T);features.append(v/max(np.linalg.norm(v),1e-8))
        with torch.no_grad():probs=torch.softmax(head(torch.tensor(np.array(features),dtype=torch.float32)),1).mean(0).numpy()
    else:
        bundle=joblib.load(modelpath);assert sha256(POINT)==bundle['encoder_sha256'];classes=bundle['classes'];probs=predict_bundle(bundle,encoder.encode_point(points),descriptors(metric))[0]
    order=np.argsort(-probs);print(json.dumps(dict(model=str(modelpath),cloud=str(path),point_only=True,experimental=True,quality=q,confidence_calibrated=False,top3=[dict(species=classes[i],score=float(probs[i])) for i in order[:3]]),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
