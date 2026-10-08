"""Read-only source/annotation audit; writes only new diagnostic artifacts."""
import json,uuid
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tree991_box_service as s

def main():
 out=s.DATA/('audit_'+uuid.uuid4().hex[:10]);out.mkdir(exist_ok=False)
 state=s.read(s.DATA/'state.json');meta=s.read(s.DATA/'source.json')
 s.write(out/'state_snapshot.json',state)
 records=[(p,s.read(p)) for p in (s.DATA/'exports').glob('*/record.json')]
 selected={};report={'revision':state['revision'],'source_sha256_verified':s.sha256(s.SOURCE)==meta['source_sha256'],'slots':{},'overlaps':{},'original_annotations_unchanged':True}
 for k in 'ABC':
  matches=[(p,r) for p,r in records if r['slot']==k and r['selection']==state['slots'][k]]
  if not matches:raise ValueError('No exact current-state export for '+k)
  p,r=max(matches,key=lambda v:v[1]['created_at']);z=np.load(p.parent/'full.npz');x=z['xyz'];ids=z['source_indices'];selected[k]=(x,ids)
  report['slots'][k]={'export':str(p.parent),'points':len(x),'extent_m':np.ptp(x,axis=0).tolist(),'z_range_m':[float(x[:,2].min()),float(x[:,2].max())],'confirmed':r['selection']['confirmed'],'species':r['species'],'export_hashes_ok':all(s.sha256(p.parent/name)==v['sha256'] for name,v in r['files'].items()),'unique_source_indices':len(np.unique(ids))==len(ids),'finite':bool(np.isfinite(x).all()),'z_quantiles':np.quantile(x[:,2],[0,.01,.1,.5,.9,.99,1]).tolist()}
 for a,b in [('A','B'),('A','C'),('B','C')]:report['overlaps'][a+b]=int(len(np.intersect1d(selected[a][1],selected[b][1])))
 offset=0;matches={k:True for k in selected};total={k:0 for k in selected}
 for raw in s.chunks(s.SOURCE,s.header(s.SOURCE)):
  for k,(x,ids) in selected.items():
   mask=s.selection_mask(raw,s.validate_slot(state['slots'][k]));expected=np.flatnonzero(mask)+offset;actual=ids[(ids>=offset)&(ids<offset+len(raw))]
   matches[k] &= np.array_equal(expected,actual)
   rows=np.flatnonzero((ids>=offset)&(ids<offset+len(raw)))
   matches[k] &= np.array_equal(raw[actual-offset],x[rows]);total[k]+=len(expected)
  offset+=len(raw)
 report['original_point_scan']=offset;report['exact_source_coordinates_and_selection_match']=matches
 coarse=np.load(s.COARSE)
 fig,axes=plt.subplots(3,3,figsize=(14,15),facecolor='white')
 for row,k in enumerate('ABC'):
  x,ids=selected[k];sample=x[np.linspace(0,len(x)-1,min(45000,len(x)),dtype=int)];lo=x.min(0);hi=x.max(0)
  context=coarse[((coarse>=lo-1)&(coarse<=hi+1)).all(1)]
  for col,(i,j,title) in enumerate([(0,2,'XZ'),(1,2,'YZ'),(0,1,'XY / top')]):
   ax=axes[row,col];ax.set_facecolor('#f4f7f8');ax.scatter(context[:,i],context[:,j],s=.6,c='#b5bec5',alpha=.24,rasterized=True)
   ax.scatter(sample[:,i],sample[:,j],s=.5,c=sample[:,2],cmap='viridis',rasterized=True)
   ax.set_aspect('equal');ax.set_xlabel('XYZ'[i]+' (local m)');ax.set_ylabel('XYZ'[j]+' (local m)');ax.grid(alpha=.15)
   ax.set_title(f'{k} | {len(x):,} original points | {title}')
 fig.suptitle(f'991 manual selections | saved revision {state["revision"]}\nColour: selected measured points by Z; grey: nearby scene. Z span is NOT verified tree height.',fontsize=13)
 fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'three_views.png',dpi=150);plt.close(fig)
 s.write(out/'report.json',report)
 print(json.dumps({'output':str(out),**report},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
