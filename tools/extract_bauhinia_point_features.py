"""Frozen feature inference only; accepts Windows PTv2 and Linux PTv3 environments."""
import sys,json,argparse,time,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1'
sys.path.insert(0,str(ROOT/'tools'))
from training_safety import sha256
def main():
 parser=argparse.ArgumentParser();parser.add_argument('model',choices=['ptv2','ptv3']);parser.add_argument('--smoke',action='store_true');args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(20260912);np.random.seed(20260912)
 rows=json.loads((OUT/'manifest.json').read_text(encoding='utf8'));features=[];keys=[];start=time.time();path=OUT/(args.model+'_features.npz')
 if not args.smoke and path.exists():raise FileExistsError(path)
 if args.model=='ptv2':
  sys.path.insert(0,str(ROOT/'tree_species_demo'));from hk_three_species import Features,POINT;encoder=Features('cuda');weight=POINT
 else:
  sys.path.insert(0,str(ROOT/'models/vendor/sonata-official-fewshot-v1'));import sonata
  weight=ROOT/'models/sonata_official_v1/sonata.pth';encoder=sonata.load(str(weight),custom_config=dict(enable_flash=False,enc_patch_size=[256]*5,shuffle_orders=False)).cuda().eval()
  for p in encoder.parameters():p.requires_grad_(False)
 for row in rows[:1] if args.smoke else rows:
  f=OUT/row['derived'];assert sha256(f)==row['derived_sha256'];data=np.load(f);out=[]
  for degree in [0,90,180,270]:
   a=np.deg2rad(degree);rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],dtype='float32');xyz=data['points']@rot.T
   with torch.inference_mode():
    if args.model=='ptv2':feat=encoder.encode_point(xyz)
    else:
     normal=data['normal']@rot.T;grid=np.floor((xyz-xyz.min(0))/.02).astype('int32');_,idx=np.unique(grid,axis=0,return_index=True);idx=np.sort(idx);coord=xyz[idx]
     point=dict(coord=torch.from_numpy(coord).cuda(),grid_coord=torch.from_numpy(grid[idx]).cuda(),feat=torch.from_numpy(np.c_[coord,np.zeros_like(coord),normal[idx]]).cuda(),offset=torch.tensor([len(idx)],device='cuda',dtype=torch.long),grid_size=.02)
     result=encoder(point);layers=[]
     # Means at all five official hierarchy levels, no extra fitted transform.
     while True:
      layers.append(result.feat.mean(0))
      if 'pooling_parent' not in result:break
      result=result.pooling_parent
     feat=torch.cat(layers).float().cpu().numpy()
   out.append(feat)
  features.append(np.mean(out,axis=0));keys.append(row['key']);print(args.model,len(keys),len(rows),row['key'],len(feat),round(time.time()-start,1),flush=True)
 if not args.smoke:
  np.savez_compressed(path,features=np.array(features),keys=np.array(keys));(OUT/(args.model+'_feature_provenance.json')).write_text(json.dumps(dict(model=args.model,weight=str(weight),weight_sha256=sha256(weight),manifest_sha256=sha256(OUT/'manifest.json'),seconds=time.time()-start,dimensions=len(features[0]),images_used=False,backbone_frozen=True,torch=torch.__version__,cuda=torch.version.cuda),indent=2),encoding='utf8')
if __name__=='__main__':main()
