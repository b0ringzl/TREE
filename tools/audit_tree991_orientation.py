"""Read-only handedness and opposing-camera diagnostic; never mirrors data."""
import json,uuid,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tree991_box_service as s

def basis(yaw,elev):
 c,t=np.cos(yaw),np.sin(yaw);ce,se=np.cos(elev),np.sin(elev)
 return np.array([t,-c,0]),np.array([-c*se,-t*se,ce]),np.array([c*ce,t*ce,se])

def main():
 out=s.DATA/('orientation_audit_'+uuid.uuid4().hex[:10]);out.mkdir(exist_ok=False)
 meta=s.read(s.DATA/'source.json');state=s.read(s.DATA/'state.json')
 source_before=s.sha256(s.SOURCE);state_before=s.sha256(s.DATA/'state.json')
 q=np.load(s.COARSE);display=np.fromfile(s.DATA/'scene.bin',dtype='<f4').reshape(-1,3)
 determinants=[];errors=[]
 for yaw in np.linspace(-np.pi,np.pi,73):
  for elev in np.linspace(-np.pi/2,np.pi/2,11):
   r,u,f=basis(yaw,elev);rotation=np.vstack([r,u,-f]);determinants.append(np.linalg.det(rotation));errors.append(np.max(abs(rotation@rotation.T-np.eye(3))))
   # Standard right-handed lookAt: eye=center-D*f, right=f cross up.
   assert np.allclose(np.cross(f,u),r) and np.allclose(np.cross(r,u),-f)
 clouds={}
 for k in 'ABC':
  records=[(p,s.read(p)) for p in (s.DATA/'exports').glob('*/record.json')]
  p,r=max([(p,r) for p,r in records if r['slot']==k and r['selection']==state['slots'][k]],key=lambda v:v[1]['created_at'])
  with np.load(p.parent/'sample2048.npz') as z:clouds[k]=z['metric_xyz'].copy()
 allp=np.concatenate(list(clouds.values()));center=np.median(allp,axis=0)
 nearby=q[(abs(q[:,0]-center[0])<12)&(abs(q[:,1]-center[1])<13)&(q[:,2]>-3)&(q[:,2]<13)]
 fig,axes=plt.subplots(1,3,figsize=(17,7),gridspec_kw={'width_ratios':[1,1,1.1]})
 colors={'A':'#009bb7','B':'#b58116','C':'#965ace'};orders=[]
 for ax,yaw,title in zip(axes[:2],[0,np.pi],['Camera from -X, looking +X','Camera from +X, looking -X']):
  r,u,f=basis(yaw,0);d=nearby-center;ax.scatter(d@r,d@u,s=.4,c='#b1bcc2',alpha=.35)
  positions={}
  for k,x in clouds.items():
   d=x-center;ax.scatter(d@r,d@u,s=2,c=colors[k]);positions[k]=float(np.median(d@r));ax.text(positions[k],float((d@u).max())+.3,'old '+k,color=colors[k],ha='center',weight='bold')
  order=sorted(positions,key=positions.get);orders.append(order);ax.set_title(title+'\nLeft to right: '+' - '.join(order));ax.set_aspect('equal');ax.set_xlim(-9,9);ax.set_ylim(-4,10);ax.grid(alpha=.2);ax.set_xlabel('Screen right (m)');ax.set_ylabel('Screen up (m)')
 ax=axes[2];ax.scatter(nearby[:,0],nearby[:,1],s=.4,c='#b1bcc2',alpha=.35)
 for k,x in clouds.items():ax.scatter(x[:,0],x[:,1],s=2,c=colors[k]);ax.text(np.median(x[:,0]),np.median(x[:,1]),'old '+k,weight='bold')
 for sign in [-1,1]:
  eye=center[:2]+[sign*9,0];ax.annotate('',xy=center[:2]+[sign*4,0],xytext=eye,arrowprops={'arrowstyle':'->','color':'#bc4242','lw':2});ax.text(*eye,'Camera',ha='center',va='bottom',color='#bc4242')
 ax.set_aspect('equal');ax.set_title('Top view: same unchanged scene');ax.set_xlabel('Local X (m)');ax.set_ylabel('Local Y (m)');ax.grid(alpha=.2)
 fig.suptitle('Rotation changes left/right order without mirroring coordinates\nDiagnostic cameras only; NOT calibrated Google Street View viewpoints.',fontsize=14)
 fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(out/'opposing_views.png',dpi=140);plt.close(fig)
 image=s.Path('C:/Users/25041725G/AppData/Local/Temp/codex-clipboard-0f22977c-c654-4f12-a333-1351d4463230.png')
 if image.exists():shutil.copy2(image,out/'user_second_view.png')
 report=dict(saved_revision=state['revision'],source_hash_unchanged_and_matches_baseline=source_before==meta['source_sha256']==s.sha256(s.SOURCE),preview_identical_to_coarse=bool(np.array_equal(q,display)),right_handed_camera_rotation_determinant_range=[float(min(determinants)),float(max(determinants))],max_rotation_orthogonality_error=float(max(errors)),opposing_camera_orders=orders,user_confirmed_reference_mapping={'old_A':'original circled photograph RIGHT tree','old_B':'original circled photograph MIDDLE tree','old_C':'original circled photograph LEFT tree'},mapping_note='User explicitly confirms A/C reversed; retained as a correction record, not a rewrite of historical slots or exports.',findings=['No coordinate reflection in the inspected raw-to-preview/export pipeline.','Viewer right/up axes agree with standard right-handed lookAt. Positive forward depth is the depth mapping, not an XY mirror.','Preset front/side are local-axis views, not geographic or Street View camera directions.','Opposing camera headings reverse the visible A/B/C order without mirroring geometry.'],unresolved=['Raw mapping system handedness relative to physical surveyed landmarks has not been independently calibrated.','Screenshots lack calibrated camera pose; photographic alignment not proven.'],state_file_unchanged=state_before==s.sha256(s.DATA/'state.json'))
 s.write(out/'report.json',report);print(json.dumps({'output':str(out),**report},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
