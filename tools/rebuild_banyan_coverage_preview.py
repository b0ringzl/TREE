"""Rebuild full-cloud banyan evidence. Real point subset only; no geometry invention."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hk-tree-vmms/tools'))
from build_hk_dense_dataset import coordinates
from build_hk_lidar_pairs import project

OUT=ROOT/'hk-tree-vmms/derived/banyan_coverage_display_v3_20260912'
CATALOG=ROOT/'outputs/hk_point_assist_20260911_v1/catalog.json'
BASE=ROOT/'hk-tree-vmms/derived/hk_dualsource_three_species_20260907'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sample_ids(xyz,limit=2048):
    """Fixed quota across height and azimuth; extrema retained, no invented points."""
    if len(xyz)<=limit:return np.arange(len(xyz))
    lo,hi=xyz.min(0),xyz.max(0);center=(lo+hi)/2
    height=np.minimum(19,((xyz[:,2]-lo[2])/max(hi[2]-lo[2],1e-9)*20).astype(int))
    angle=np.minimum(7,((np.arctan2(xyz[:,1]-center[1],xyz[:,0]-center[0])+np.pi)/(2*np.pi)*8).astype(int))
    strata=height*8+angle
    chosen=set(np.r_[xyz.argmin(0),xyz.argmax(0)].tolist())
    quota=max(1,(limit-6)//160)
    for cell in np.unique(strata):
        ids=np.flatnonzero(strata==cell)
        # Order radially; spread sample over the central branches and outer crown.
        ids=ids[np.argsort(np.linalg.norm(xyz[ids,:2]-center[:2],axis=1),kind='stable')]
        chosen.update(ids[np.linspace(0,len(ids)-1,min(len(ids),quota),dtype=int)].tolist())
    remaining=np.setdiff1d(np.arange(len(xyz)),np.array(sorted(chosen)))
    need=limit-len(chosen)
    if need>0:chosen.update(remaining[np.linspace(0,len(remaining)-1,min(need,len(remaining)),dtype=int)].tolist())
    return np.array(sorted(chosen))

def projection(xyz,ids,row,source,path,floor,mode):
    with Image.open(source) as src:im=src.convert('RGB');im.thumbnail((1920,960))
    draw=ImageDraw.Draw(im);uv,_=project(xyz[ids],row)
    for (u,v),z in zip(uv,xyz[ids,2]):
        lower=z<floor
        if mode=='upper' and lower or mode=='lower' and not lower:continue
        x,y=u*im.width,v*im.height;draw.ellipse((x-1,y-1,x+1,y+1),fill='#ffa83d' if lower else '#37e5b0')
    im.save(path,quality=94)

def orthographic(xyz,ids,path,floor):
    im=Image.new('RGB',(1500,560),'#f3f6ee');draw=ImageDraw.Draw(im)
    center=(xyz.min(0)+xyz.max(0))/2;p=xyz-center;scale=430/max(np.ptp(xyz,axis=0).max(),1)
    for panel,(a,b,title) in enumerate([(0,2,'XZ'),(1,2,'YZ'),(0,1,'XY')]):
        draw.text((panel*500+15,15),title+' | ALL observed geometry | same scale',fill='#245741')
        for i in ids:
            x=panel*500+250+p[i,a]*scale;y=285-p[i,b]*scale
            draw.ellipse((x-1,y-1,x+1,y+1),fill='#c67f20' if xyz[i,2]<floor else '#28785a')
    im.save(path,quality=94)

def main():
    OUT.mkdir(exist_ok=False)
    rows=[r for r in read(CATALOG) if r['species']=='Ficus microcarpa']
    original={r['id']:r for r in read(BASE/'manifest.json')};coords=coordinates()
    hashes={str(CATALOG):sha(CATALOG),str(BASE/'manifest.json'):sha(BASE/'manifest.json')}
    for name in ['pair_reviews.json','banyan_tracking_pilot_v1/reviews.json','banyan_tracking_batch_v2/reviews.json']:
        p=BASE/name
        if p.exists():hashes[str(p)]=sha(p)
    results=[]
    for r in rows:
        folder=OUT/r['key'];folder.mkdir();orig=original[r['id']];source=Path(orig['source_panorama'])
        for kind,h in r['sha256'].items():
            p=Path(r['asset_paths'][kind]);assert sha(p)==h;hashes[str(p)]=h
        with np.load(r['asset_paths']['cleaned'],allow_pickle=False) as f:xyz=f['xyz'].astype(np.float64)
        floor=r['quality']['crown_floor_z'];ids=sample_ids(xyz);plot_ids=sample_ids(xyz,10000)
        assert len(ids)==2048 and len(np.unique(xyz[ids],axis=0))==2048
        np.testing.assert_array_equal(xyz[ids].min(0),xyz.min(0));np.testing.assert_array_equal(xyz[ids].max(0),xyz.max(0))
        center=(xyz.min(0)+xyz.max(0))/2;scale=max(np.linalg.norm(xyz-center,axis=1).max(),1e-9)
        preview=(xyz[ids]-center)/scale
        np.savez_compressed(folder/'point_coverage_2048.npz',points_xyz=preview.astype(np.float32),centroid_xyz=center,scale=scale,source_indices=ids,metric_units='m',geometry_changed=False)
        for mode in ['all','upper','lower']:
            projection(xyz,plot_ids,coords[orig['frame_key']],source,folder/(mode+'.jpg'),floor,mode)
        orthographic(xyz,plot_ids,folder/'views.jpg',floor)
        lower=xyz[:,2]<floor
        before_ids=np.linspace(0,len(xyz)-1,2048,dtype=int)  # old API sampled unique lexicographic XYZ
        before_xyz=np.unique(xyz,axis=0)[before_ids]
        hist=np.histogram(xyz[:,2],bins=np.linspace(xyz[:,2].min(),xyz[:,2].max()+1e-9,21))[0]
        item=dict(key=r['key'],id=r['id'],species=r['species'],display_version=OUT.name,geometry_version=r['version'],
            geometry_changed=False,original_geometry_sha256=r['sha256']['cleaned'],source_panorama=str(source),
            full_count=len(xyz),upper_count=int((~lower).sum()),lower_count=int(lower.sum()),
            full_extent_m=np.ptp(xyz,axis=0).tolist(),z_min_max=xyz[:,2].min().item(),
            observed_z_bounds_m=[float(xyz[:,2].min()),float(xyz[:,2].max())],
            historical_floor_z_m=floor,raw_height_bins=hist.tolist(),
            preview_count=len(ids),preview_lower_count=int((xyz[ids,2]<floor).sum()),
            old_preview_lower_count=int((before_xyz[:,2]<floor).sum()),
            preview_points=preview.round(6).tolist(),preview_lower_mask=(xyz[ids,2]<floor).tolist(),
            preview_sampling='20 height layers × 8 azimuth sectors, radial spread, XYZ extrema preserved; true point subset',
            point_source_membership_error_m=0.,total_height_verified=False,
            notes=['Archived projection drew repaired lower stems only; it was not a full-cloud coverage map.',
                'Green/orange are upper/lower observed regions split at the historical extraction floor, not semantic crown/stem truth.',
                'Full measured cloud is displayed; this does not certify whole-tree coverage, root visibility or single-tree purity.'],
            asset_paths={k:str(folder/v) for k,v in {'all':'all.jpg','upper':'upper.jpg','lower':'lower.jpg','views':'views.jpg','point':'point_coverage_2048.npz'}.items()})
        item['sha256']={k:sha(Path(v)) for k,v in item['asset_paths'].items()}
        results.append(item);print(r['id'],'full',len(xyz),'upper',item['upper_count'],'lower preview',item['old_preview_lower_count'],'->',item['preview_lower_count'],flush=True)
    # Audit nearby raw cached support without treating nearby background as tree recovery.
    from scipy.spatial import cKDTree
    for cache in sorted({p for r in rows for p in r['source_cache_paths']}):
        p=Path(cache);hashes[str(p)]=sha(p);meta=read(p.with_suffix('.json'));hashes[str(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
        with np.load(p,allow_pickle=False) as f:local=f['xyz']
        for r,item in zip(rows,results):
            if cache not in r['source_cache_paths']:continue
            axis=np.array(r['quality']['axis_local_xy']);roi=local[np.linalg.norm(local[:,:2]-axis,axis=1)<2]
            z=item['observed_z_bounds_m'][0]
            below=roi[(roi[:,2]<z-.15)&(roi[:,2]>z-3)]
            item.setdefault('raw_support_audit',[]).append(dict(cache=cache,raw_map=meta['source'],raw_map_exists=Path(meta['source']).is_file(),near_axis_points=len(roi),near_axis_below_old_base_points=len(below),
                caution='Unclassified nearby points may be ground, planter or neighbouring objects; NOT automatically added.'))
        del local
    for p,h in hashes.items():assert sha(Path(p))==h,p
    write(OUT/'manifest.json',results);write(OUT/'source_hashes.json',hashes)
    write(OUT/'verification.json',dict(created_at=datetime.now(timezone.utc).isoformat(),banyan_views=len(results),
        all_old_clouds_contain_upper_points=all(r['upper_count']>0 for r in results),all_preview_extrema_preserved=True,
        source_files_unchanged=len(hashes),other_species_modified=False,training=False,weights_modified=False,
        geometry_modified=False,synthetic_points_added=0,output_version=OUT.name))

if __name__=='__main__':main()
