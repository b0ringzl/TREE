"""Archive user-provided Street View evidence and render local alignment hypotheses.
No camera pose is claimed as a registration; no Google imagery is downloaded.
"""
import json,shutil,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from training_safety import sha256
OUT=ROOT/'outputs/tree991_streetview_alignment_20260913_v1'
def main():
 OUT.mkdir(exist_ok=False)
 source=Path('C:/Users/25041725G/AppData/Local/Temp/codex-clipboard-6595f11e-4f38-45b5-bbc7-e9334565bec2.png')
 dest=OUT/'user_streetview_original.png';shutil.copy2(source,dest);assert sha256(source)==sha256(dest)
 width,height=Image.open(source).size
 record=dict(source='user-supplied Google Maps URL and screenshot',screenshot=str(dest),screenshot_original=str(source),sha256=sha256(dest),dimensions=[width,height],
  panorama_id='Fcuy1asvGhsacnc_ayPC3A',url_camera_latitude=22.3703255,url_camera_longitude=114.1141614,url_heading_degrees=175.55,url_tilt_degrees=95.89,
  url_view_parameter_y=75,view_parameter_is_calibrated_intrinsics=False,
  streetview_capture_month='2025-11',streetview_date_evidence='Visible date in user screenshot',
  lidar_session_start='2026-09-09 13:43:38',lidar_session_end='2026-09-09 13:46:55',lidar_date_source=str(ROOT/'lidar data/hk/tree9/tree991/info.yaml'),
  map_lookup_link='https://www.google.com/maps/@?api=1&map_action=pano&pano=Fcuy1asvGhsacnc_ayPC3A&heading=175.55&pitch=5.89',
  notes=['Place coordinates and camera coordinates are different roles; no cloud origin assigned from either',
   'Screenshot includes browser/sidebar/compass UI; full screenshot is not an equirectangular panorama',
   '75y is retained as a URL parameter, not assumed calibrated horizontal FOV',
   'Google imagery retained with attribution; private local alignment review only',
   'Capture-area similarity is not individual-tree matching'],
  browser_access='Connection unavailable on this attempt',registration_verified=False,landmark_pairs=[],projection_generated=False)
 (OUT/'reference.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
 q=np.load(ROOT/'outputs/hk_fullscene_extract_20260912_v1/maps/recent_tree991/coarse.npy')
 rng=np.random.default_rng(991);q=q[rng.choice(len(q),min(len(q),90000),replace=False)]
 proposals=json.loads((ROOT/'outputs/tree991_single_tree_review_20260913_v1/manifest.json').read_text(encoding='utf8'))
 # Fixed geometric hypotheses, independent of image labels and not fitted to screenshot.
 center=np.array([0.,8.,5.]);fig,axes=plt.subplots(2,4,figsize=(20,10));hypotheses=[]
 for i,ax in enumerate(axes.flat):
  theta=i*np.pi/4;camera=np.array([35*np.cos(theta),8+35*np.sin(theta),1.6]);forward=center-camera;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
  def project(x):
   d=x-camera;z=d@forward;return np.column_stack([(d@right)/np.maximum(z,.01),(d@up)/np.maximum(z,.01)]),z
  uv,z=project(q);keep=(z>1)&(z<120)&(abs(uv[:,0])<1)&(abs(uv[:,1])<.75);order=np.flatnonzero(keep)[np.argsort(-z[keep])]
  ax.scatter(uv[order,0],uv[order,1],s=.15,c=np.clip(q[order,2],0,35),cmap='viridis',vmin=0,vmax=35)
  for r in proposals:
   x=np.array([*r['axis_xy'],5.]);p,d=project(x[None])
   if d[0]>1 and abs(p[0,0])<1 and abs(p[0,1])<.75:ax.text(*p[0],str(r['number']),fontsize=5,color='#a33')
  ax.set(xlim=(-1,1),ylim=(-.75,.75));ax.set_aspect('equal');ax.axis('off');label=chr(65+i)
  ax.set_title(f'{label} | local camera XY ({camera[0]:.1f}, {camera[1]:.1f})\nUNREGISTERED HYPOTHESIS',fontsize=10)
  hypotheses.append(dict(label=label,camera_local_xyz=camera.tolist(),look_at=center.tolist(),registered=False))
 fig.suptitle('991 measured cloud: orientation hypotheses ONLY — not Google Street View projections',fontsize=16);fig.tight_layout();fig.savefig(OUT/'orientation_hypotheses.png',dpi=150);plt.close(fig)
 (OUT/'orientation_hypotheses.json').write_text(json.dumps(hypotheses,indent=2),encoding='utf8')
 print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
