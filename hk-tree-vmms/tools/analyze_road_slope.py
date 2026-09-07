"""Roadside-slope evidence report inputs. Trajectory grade is only a proxy."""
import csv
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from audit_hk_metric_height import DATA,read,write

WORK=DATA.parents[2];OUT=WORK/'reports/roadside_slope_20260907'
LABELS=['0–3°','3–8°','8–15°','≥15°']

def band(deg):return LABELS[int(np.digitize(abs(deg),[3,8,15]))]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((DATA.parents[1]/'outputs/coordinates/frame_coordinates.csv').open(encoding='utf-8-sig')))
    streams=defaultdict(list)
    for r in rows:streams[r['stream_id']].append(r)
    grades={};summary={}
    for stream,rs in streams.items():
        rs.sort(key=lambda r:int(r['frame_id']));xyz=np.array([[float(r['local_'+k]) for k in 'xyz'] for r in rs]);steps=np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1)
        # Discontinuities are not road segments; don't regress across them.
        segments=np.r_[0,np.cumsum(steps>30)];dist=np.r_[0,np.cumsum(np.minimum(steps,30))];samples=[];last=-100
        for i,r in enumerate(rs):
            left=np.searchsorted(dist,dist[i]-10);right=np.searchsorted(dist,dist[i]+10,side='right')
            ids=np.arange(left,right);ids=ids[segments[ids]==segments[i]]
            if len(ids)<5 or np.ptp(dist[ids])<10:continue
            x=dist[ids]-dist[i];coef=np.polyfit(x,xyz[ids,2],1);res=float(np.sqrt(np.mean((np.polyval(coef,x)-xyz[ids,2])**2)))
            if res>.3:continue
            deg=float(np.degrees(np.arctan(coef[0])));key=stream+'__'+r['frame_id']
            grades[key]={'road_grade_deg_proxy':deg,'bin':band(deg),'fit_rmse_m':res,'chainage_m':float(dist[i])}
            if dist[i]-last>=15:samples.append(abs(deg));last=dist[i]
        summary[stream]={'frames':len(rs),'valid_grade_frames':sum(k.startswith(stream+'__') for k in grades),
                         'sampled_15m_windows':len(samples),'grade_bins':dict(Counter(band(x) for x in samples)),
                         'abs_grade_quantiles_deg':np.quantile(samples,[.5,.9,.95,1]).tolist() if samples else []}
    manifest=read(DATA/'manifest.json');metrics=read(DATA/'metric_height_v1/measurements.json')
    byroute=defaultdict(Counter);bygrade=defaultdict(Counter)
    for r in manifest:
        if r['id'] not in metrics:continue
        m=metrics[r['id']];s=byroute[r['route']];s['point_pairs']+=1
        s['ground_plane_candidate']+=m['ground']['status']=='local_plane_candidate';s['height_gate_eligible']+=m['height_feature_eligible']
        if 'missing_basal_stem_or_wrong_ground' in m['flags']:s['basal_gap_or_wrong_ground']+=1
        g=grades.get(r['frame_key'])
        if g:
            v=bygrade[g['bin']];v['point_pairs']+=1;v['ground_plane_candidate']+=m['ground']['status']=='local_plane_candidate';v['height_gate_eligible']+=m['height_feature_eligible']
    batch=read(DATA/'banyan_tracking_batch_v2/results.json');qa=read(DATA/'banyan_tracking_batch_v2/agent_visual_qa.json');br=defaultdict(Counter)
    for r in batch:
        c=br[r['survey']];c['processed']+=1;c[r['status']]+=1
        c['priority_review']+=qa.get(r['id'],{}).get('disposition')=='review_candidate'
    report={'scope':'Roadside mobile panoramas and LiDAR; no UAV/forest-interior claim',
            'grade_method':'~20m trajectory elevation regression, residual<=0.3m; 15m sampled windows; INS-platform proxy, not surveyed road slope',
            'trajectory_by_stream':summary,'height_audit_by_route':dict(byroute),'height_audit_by_road_grade_proxy':dict(bygrade),
            'banyan_batch_by_survey':dict(br),'classification_accuracy_on_slope':'not_available_no_independent_accepted_slope_testset',
            'height_bias_examples':[{'grade_deg':a,'horizontal_tree_road_offset_m':5,'road_base_height_bias_m':float(5*np.tan(np.radians(a)))} for a in [5,10,15,20]]}
    write(OUT/'analysis.json',report);write(OUT/'frame_grade_proxy.json',grades)
    print(report)

if __name__=='__main__':main()
