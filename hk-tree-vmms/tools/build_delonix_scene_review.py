"""All known Delonix panorama scenes, including failed extraction and rejected pairs.

Scene points are raw measured support, NOT species-segmented tree clouds.
No distance subsampling of frames and no source label/review mutation.
"""
import hashlib
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from build_hk_five_species_pairs import effective_records,canonical,OUT as PAIRS
from build_hk_dense_dataset import coordinates,polygon_inside,CACHE
from build_hk_lidar_pairs import read,write,VMMS,project
from expand_requested_five_species import DenserFrameMaps

OUT=PAIRS.parent/'delonix_scene_review_20260907'
COLORS=[(255,189,30),(24,229,208),(236,111,219),(139,206,66)]

def draw_labels(im,targets):
    draw=ImageDraw.Draw(im)
    for n,t in enumerate(targets):
        poly=np.array(t['polygon'])*[im.width,im.height]
        if len(poly)<3:continue
        color=COLORS[n%len(COLORS)];draw.line([tuple(v) for v in np.r_[poly,poly[:1]]],fill=color,width=3)
        x,y=poly.min(0);draw.rectangle((x,y,x+52,y+24),fill=color);draw.text((x+5,y+4),'T'+str(n+1),fill='black')
    return im

def overlay(im,xyz,row,targets,only_targets=False):
    uv,dist=project(xyz,row);valid=np.isfinite(uv).all(1)&(dist>2)&(dist<70)
    uv=uv[valid];dist=dist[valid]
    membership=np.full(len(uv),-1,int)
    for n,t in enumerate(targets):
        if len(t['polygon'])>=3:membership[polygon_inside(uv,np.array(t['polygon']),dilation=0)]=n
    if only_targets:
        keep=membership>=0;uv=uv[keep];dist=dist[keep];membership=membership[keep]
    # Closest observed surface per small display cell limits distant overlays.
    cell=np.floor(uv*[im.width/3,im.height/3]).astype(int)
    order=np.argsort(dist);_,first=np.unique(cell[order],axis=0,return_index=True);ids=order[first]
    canvas=im.copy();draw=ImageDraw.Draw(canvas)
    for i in ids:
        x,y=uv[i]*[im.width,im.height];c=COLORS[membership[i]%len(COLORS)] if membership[i]>=0 else (110,171,223)
        draw.ellipse((x-1,y-1,x+1,y+1),fill=c)
    return draw_labels(canvas,targets),len(ids)

def main():
    OUT.mkdir(exist_ok=True);(OUT/'scenes').mkdir(exist_ok=True)
    protected=[PAIRS/'reviews.json',PAIRS/'manifest.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    existing=read(PAIRS/'manifest.json');by_id={r['id']:r for r in existing};old_reviews=read(PAIRS/'reviews.json')
    effective,_=effective_records();scenes=defaultdict(dict);rows=coordinates()
    for key,r in effective.items():
        for label in r.get('labels',[]):
            ident=key+'__'+hashlib.sha256(str(label.get('label_id')).encode()).hexdigest()[:10]
            original=canonical(label.get('species',''))=='Delonix regia'
            reviewed=old_reviews.get(ident,{}).get('species')=='Delonix regia'
            predicted=by_id.get(ident,{}).get('image_evidence',{}).get('top1')=='Delonix regia'
            if not (original or reviewed or predicted):continue
            scenes[key][ident]={'id':ident,'polygon':label.get('points',[]),'original_species':label.get('species'),
               'selection_reasons':[s for s,b in [('existing_label',original),('pair_review_label',reviewed),('model_top1',predicted)] if b],
               'annotation_provenance':r['annotation_provenance'],'previous_pair_status':by_id.get(ident,{}).get('status'),
               'previous_pair_reason':by_id.get(ident,{}).get('reason'),'previous_review':old_reviews.get(ident)}
    maps=DenserFrameMaps(CACHE);manifest=[]
    for key,members in sorted(scenes.items()):
        folder=OUT/'scenes'/key;folder.mkdir(exist_ok=True);targets=list(members.values());row=rows.get(key)
        result={'id':key,'targets':targets,'status':'pending','projection_verified':False,'training_eligible':False}
        if row is None:result.update(status='missing_coordinates')
        else:
            result['source_panorama']=str(VMMS/row['source_image_relpath'])
            result['survey']=Path(row['source_image_relpath']).parts[0]
            with Image.open(result['source_panorama']) as src:im=src.convert('RGB');im.thumbnail((3072,1536))
            im.save(folder/'panorama.jpg',quality=95)
            draw_labels(im.copy(),targets).save(folder/'labels.jpg',quality=95)
            try:
                xyz,sources=maps.nearby(row)
                result.update(source_lidar=sources,source_sampling_m=maps.sampling_m,measured_nearby_points=len(xyz))
                for only,name in [(False,'scene_projection.jpg'),(True,'target_projection.jpg')]:
                    canvas,count=overlay(im,xyz,row,targets,only);canvas.save(folder/name,quality=94)
                    result[name+'_rendered_points']=count
                # Preserve old cleaned output for direct comparison, including rejected examples.
                cleaned=[]
                for t in targets:
                    path=PAIRS/'instances'/t['id']/'cleaned_local.npz'
                    t['has_previous_cleaned_cloud']=path.exists()
                    if path.exists():cleaned.append(np.load(path)['xyz'])
                if cleaned:
                    canvas,_=overlay(im,np.concatenate(cleaned),row,targets)
                    canvas.save(folder/'previous_cleaned_projection.jpg',quality=94)
                result.update(status='ready',has_previous_cleaned_projection=bool(cleaned))
            except (OSError,ValueError,IndexError) as exc:result.update(status='projection_failed',reason=str(exc))
        write(folder/'record.json',result);manifest.append(result);write(OUT/'manifest.progress.json',manifest)
        print(key,result['status'],len(targets),flush=True)
    write(OUT/'manifest.json',manifest)
    write(OUT/'summary.json',{'scenes':len(manifest),'target_instances':sum(len(r['targets']) for r in manifest),
         'statuses':dict(Counter(r['status'] for r in manifest)),'frame_sampling':'all known scenes; no 15m skipping',
         'scope':'Existing effective labels, current pair reviews and available five-class candidate model top1; not an exhaustive scan of unlabelled VMMS.',
         'projection_policy':'Measured scene points, nearest visible display cell; colored points inside polygons are NOT cleaned single-tree truth.',
         'protected_hashes':hashes})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'Source changed during generation'

if __name__=='__main__':main()
