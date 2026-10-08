"""Three user-confirmed trees; immutable derived experiment, no source relabeling."""
import json,sys,itertools
from pathlib import Path
from collections import Counter
import numpy as np,laspy
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from training_safety import sha256
from rebuild_banyan_coverage_preview import sample_ids
OUT=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1'
SPECIES='Bauhinia x blakeana'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def normals(x):
 ids=cKDTree(x).query(x,k=min(20,len(x)))[1];a=x[ids]-x[ids].mean(1,keepdims=True);_,v=np.linalg.eigh(np.einsum('nki,nkj->nij',a,a));n=v[:,:,0];outward=x-x.mean(0);n[np.sum(n*outward,1)<0]*=-1;return n.astype('float32')
def main():
 OUT.mkdir(exist_ok=False);(OUT/'clouds').mkdir()
 source=ROOT/'outputs/single_tree_panorama_selection_v1/screened_manifest.json';pairs=read(source);new=[r for r in pairs if r['id'] in ['01_TreeID_13','01_TreeID_14','01_TreeID_15']]
 assert len(new)==3
 save(OUT/'user_species_confirmation.json',dict(user_statement='没错，前3株都是洋紫荆，你能不能就根据这三个样本进行小样本训练点云模型？你试一下ptv3模型？',confirmed_ids=[r['id'] for r in new],species=SPECIES,chinese='洋紫荊',scope='species confirmation and experimental training authorization; not new ground-truth geometry certification',source_manifest=str(source),source_manifest_sha256=sha256(source)))
 legacy=ROOT/'outputs/hk_point_training_20260912_v1';old=read(legacy/'manifest.json');plan=read(legacy/'plan.json');allowed=plan['evaluated_classes'];oldcatalog=read(legacy/'accepted_catalog_snapshot.json');live=read(ROOT/'outputs/hk_point_assist_20260911_v1/catalog.json');accepted={r['key'] for r in live if r['current_accepted'] and r['review_decision']=='accept'}
 save(OUT/'legacy_manifest_snapshot.json',old);save(OUT/'current_accepted_catalog_snapshot.json',live);save(OUT/'geometry_reviews_snapshot.json',read(ROOT/'outputs/hk_point_assist_20260911_v1/geometry_reviews.json'))
 newxyz={r['id']:np.c_[laspy.read(r['path']).x,laspy.read(r['path']).y,laspy.read(r['path']).z] for r in new}
 # Never let exact shared original points leak between the three trees.
 overlap=[];purged={k:np.zeros(len(x),bool) for k,x in newxyz.items()}
 for a,b in itertools.combinations(newxyz,2):
  xa,xb=newxyz[a],newxyz[b];ma=cKDTree(xb).query(xa,workers=4)[0]<.005;mb=cKDTree(xa).query(xb,workers=4)[0]<.005;purged[a]|=ma;purged[b]|=mb;overlap.append(dict(a=a,b=b,a_shared=int(ma.sum()),b_shared=int(mb.sum())))
 save(OUT/'new_tree_overlap_audit.json',overlap)
 records=[];exclude=[];seen=set();raws={}
 for r in sorted(old,key=lambda r:(r['group'],-r['quality']['clean_points'])):
  if r['species'] not in allowed or r['split'] not in ['train','validation','test_regression'] or r['exclusion_reasons']:continue
  if r['key'] not in accepted:exclude.append(dict(key=r['key'],reason='no longer accepted'));continue
  if r['survey']=='20250210-jianshazui' and any(np.linalg.norm(np.array(r['axis'])-n['axis_xy'])<12 for n in new):exclude.append(dict(key=r['key'],reason='12m buffer around new same-row positives'));continue
  if r['group'] in seen:continue
  assert sha256(r['source_assets']['cleaned'])==r['source_sha256']['cleaned']
  x=np.load(r['cleaned'])['xyz'];unsafe=False
  for n in new:
   if r['survey']=='20250210-jianshazui' and np.linalg.norm(np.median(x[:,:2],0)-n['axis_xy'])<35:
    fraction=float((cKDTree(newxyz[n['id']]).query(x[sample_ids(x,3000)],workers=4)[0]<.2).mean())
    if fraction>.01:unsafe=True;exclude.append(dict(key=r['key'],reason='overlap with positive tree',fraction=fraction))
  if unsafe:continue
  seen.add(r['group']);key=r['key'];records.append(dict(key=key,id=r['id'],species=r['species'],group=r['group'],split=r['split'],source=r['cleaned'],source_sha256=sha256(r['cleaned']),label_evidence='existing explicit accepted pair; current acceptance checked',legacy_key=key));raws[key]=x
 for r in new:
  assert sha256(r['path'])==r['sha256'];key=r['id'];raws[key]=newxyz[key][~purged[key]];records.append(dict(key=key,id=key,species=SPECIES,group=key,scene_group='01_same_row',split='fewshot_positive',source=r['path'],source_sha256=r['sha256'],shared_points_removed=int(purged[key].sum()),label_evidence='explicit user confirmation in current turn',pair_quality_caveat=r['visual_reason']))
 for r in records:
  x=np.unique(raws[r['key']][np.isfinite(raws[r['key']]).all(1)],axis=0);m=x[sample_ids(x,2048)].astype('float32');center=(x.min(0)+x.max(0))/2;scale=max(np.linalg.norm(x-center,axis=1).max(),1e-6);p=((m-center)/scale).astype('float32');normal=normals(p)
  path=OUT/'clouds'/(r['key']+'.npz');np.savez_compressed(path,points=p,metric=m,normal=normal);r.update(derived=path.relative_to(OUT).as_posix(),derived_sha256=sha256(path),points=len(p),class_index=sorted(allowed+[SPECIES]).index(r['species']));print('prepared',r['key'],flush=True)
 classes=sorted(allowed+[SPECIES]);config=dict(seed=20260912,classes=classes,positive_species=SPECIES,positive_ids=[r['id'] for r in new],model='Official Sonata pretrained PointTransformerV3 frozen encoder + fitted linear classification head',input='2048 measured XYZ points; unit sphere normalization; k20 PCA outward normals; all RGB channels zero (unavailable); no image features',augmentation='four fixed Z rotations for every individual (0,90,180,270deg); mean features; never count rotations as independent trees',head='L2 normalized features x10; LogisticRegression C=1 balanced max_iter=3000; fixed before evaluation',validation='3 predeclared leave-one-tree-out folds, 2 positive train + 1 positive held-out; old validation/test blocks fixed; no hyperparameter tuning using held-out positive',limitations=['Three positives from same row; not cross-site validation','No fully independent fourth positive for final 3-shot model evaluation','Old negative classes may retain partial-cloud bias','PTv3 backbone pretrained mainly outside tree-species domain; zero RGB is an input-domain mismatch','User confirmed species, not perfect point-level segmentation'],baseline='Existing WHU PTv2 frozen encoder with identical newly fitted six-class linear head; old deployed head lacks Bauhinia and is not a compatible six-class comparator',production_switch=False)
 save(OUT/'manifest.json',records);save(OUT/'excluded_legacy.json',exclude);save(OUT/'config.json',config);save(OUT/'label_map.json',dict(enumerate(classes)));save(OUT/'split_assignment.json',[dict(key=r['key'],group=r['group'],split=r['split'],species=r['species']) for r in records]);print(Counter((r['species'],r['split']) for r in records))
if __name__=='__main__':main()
