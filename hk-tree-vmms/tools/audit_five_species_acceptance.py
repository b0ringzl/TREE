"""Read-only acceptance audit. Writes reports only, never source assets/reviews."""
import hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from build_hk_five_species_pairs import OUT as DATA,NAMES
from build_hk_lidar_pairs import read,write,project
from build_hk_dense_dataset import coordinates,polygon_inside

OUT=DATA.parents[2]/'reports/hk_expansion_review_audit_20260907'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=DATA/'reviews.json';before=hashlib.sha256(path.read_bytes()).hexdigest()
    reviews=read(path);manifest={r['id']:r for r in read(DATA/'manifest.json')};coords=coordinates()
    measurements=[];groups=defaultdict(list);reasons=Counter();corrections=[]
    for ident,review in reviews.items():
        r=manifest[ident];groups[r['tree_group']].append((ident,review))
        if review['species']!=r['species']:corrections.append({'id':ident,'original_species':r['species'],'corrected_species':review['species'],'pair_decision':review['decision'],'note':review['note']})
        if review['decision']=='reject':
            for token in ('未成功匹配','背景干扰','树冠缺失','不确定树种','树干欠分割','整体缺失'):
                if token in review['note']:reasons[token]+=1
        xyz=np.load(DATA/'instances'/ident/'cleaned_local.npz')['xyz']
        uv,_=project(xyz,coords[r['frame_key']]);poly=np.array(r['polygon']);q=r['quality']
        inside=polygon_inside(uv,poly,dilation=0);is_crown=xyz[:,2]>=q['crown_floor_z']
        # Projection support proxies only, not segmentation accuracy or full-tree coverage.
        span=np.quantile(uv[inside,1],[.02,.98]) if inside.any() else [0.,0.]
        tids=xyz[~is_crown];base=q['base_z'];floor=q['crown_floor_z']
        edges=np.arange(base,floor+.2,.2);hist=np.histogram(tids[:,2],edges)[0]
        longest=current=0
        for count in hist:
            current=current+1 if count<3 else 0;longest=max(longest,current)
        measurements.append({'id':ident,'species':review['species'],'decision':review['decision'],'note':review['note'],
            'projection_inside_fraction':float(inside.mean()),
            'crown_projection_inside_fraction':float(inside[is_crown].mean()) if is_crown.any() else 0.,
            'projected_vertical_span_ratio':float((span[1]-span[0])/max(np.ptp(poly[:,1]),1e-8)),
            'trunk_points':len(tids),'old_trunk_occupancy':q['trunk_bin_occupancy'],
            'longest_sparse_stem_interval_m':longest*.2,
            'image_label_agreement':r.get('image_evidence',{}).get('label_agreement',False)})
    write(OUT/'measurements.json',measurements)
    # Illustrative fixed cutoffs, not fitted or independently validated thresholds.
    gates={}
    for metric,threshold,side in [('projection_inside_fraction',.5,'low'),('projected_vertical_span_ratio',.4,'low'),('longest_sparse_stem_interval_m',1.0,'high')]:
        flagged=[x for x in measurements if (x[metric]<threshold if side=='low' else x[metric]>threshold)]
        gates[metric]={'threshold':threshold,'flagged_decisions':dict(Counter(x['decision'] for x in flagged)),'warning':'Descriptive resubstitution on reviewed candidates; not deployment validation.'}
    mixed=[{'tree_group':g,'items':[{'id':i,**r} for i,r in members]} for g,members in groups.items() if len({r['decision'] for i,r in members})>1]
    conflicted=[g for g,members in groups.items() if len({r['species'] for i,r in members if r['decision']=='accept'})>1]
    per_class={n:{'decisions':dict(Counter(r['decision'] for r in reviews.values() if r['species']==n)),
                  'accepted_algorithm_groups':len({manifest[i]['tree_group'] for i,r in reviews.items() if r['species']==n and r['decision']=='accept'})} for n in NAMES}
    qa=read(DATA/'agent_visual_qa.json')
    qa_cross=Counter((qa.get(i,{}).get('disposition','not_prechecked'),r['decision']) for i,r in reviews.items())
    summary={'review_count':len(reviews),'decisions':dict(Counter(r['decision'] for r in reviews.values())),
        'per_class':per_class,'rejection_notes_nonexclusive':dict(reasons),'species_corrections':corrections,
        'accepted_algorithm_groups':len({manifest[i]['tree_group'] for i,r in reviews.items() if r['decision']=='accept'}),
        'groups_with_both_accept_and_reject':len(mixed),'accepted_groups_with_species_conflict':conflicted,
        'image_agreement_by_decision':{d:sum(x['image_label_agreement'] for x in measurements if x['decision']==d) for d in ('accept','reject')},
        'assistant_precheck_vs_human':[{'precheck':a,'decision':d,'count':n} for (a,d),n in qa_cross.items()],
        'diagnostic_gates':gates,'source_review_sha256':before}
    write(OUT/'summary.json',summary);write(OUT/'mixed_decision_groups.json',mixed)
    # Inspect every rejection plus a few accepted controls with previously noted issues.
    controls=['jianshazui_pano_2__007376__1b38fea760','hewentian_pano__004287__9515e6f1a6','hewentian_pano__003181__078c5b92b6']
    ids=[i for i,r in reviews.items() if r['decision']=='reject']+controls
    for start in range(0,len(ids),5):
        batch=ids[start:start+5];sheet=Image.new('RGB',(1300,300*len(batch)),'white');draw=ImageDraw.Draw(sheet)
        for j,ident in enumerate(batch):
            draw.text((6,j*300+3),ident+' '+reviews[ident]['decision'],fill='black')
            for x,name,w in ((0,'projection.jpg',710),(720,'point_views.jpg',570)):
                im=Image.open(DATA/'instances'/ident/name);im.thumbnail((w,270));sheet.paste(im,(x,j*300+24))
        sheet.save(OUT/('visual_audit_'+str(start//5)+'.jpg'),quality=93)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    print(summary)

if __name__=='__main__':main()
