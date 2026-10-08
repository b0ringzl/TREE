"""Diagnostic copies of IDs present only in aggregate LAS; not acceptance/restoration."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,laspy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from rebuild_banyan_coverage_preview import orthographic,sample_ids
OUT=ROOT/'outputs/single_tree_panorama_selection_v1/total_only';SRC=ROOT/'lidar data/hk/data/data/04/04_single_tree_output/04_segmented_trees.las'
OUT.mkdir(exist_ok=True)
f=laspy.read(SRC);records=[]
for tid in [1,2,3,4,6,13,14,15,16,21]:
 d=OUT/('TreeID_'+str(tid));d.mkdir(exist_ok=True);out=laspy.LasData(f.header.copy());out.points=f.points[np.asarray(f.tree_id)==tid];p=d/'diagnostic_copy.las';out.write(p);x=np.column_stack([out.x,out.y,out.z]);v=d/'geometry.jpg';orthographic(x,sample_ids(x,6000),v,float(np.quantile(x[:,2],.25)))
 records.append(dict(id='04_TotalOnly_TreeID_'+str(tid),path=str(p),original_source=str(SRC),original_tree_id=tid,points=len(x),extent_m=np.ptp(x,axis=0).tolist(),geometry_preview=str(v),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),delivery_status='present_only_in_aggregate_reason_for_omission_unknown',species_verified=False,geometry_verified=False,training_eligible=False))
(OUT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8');print(len(records))
