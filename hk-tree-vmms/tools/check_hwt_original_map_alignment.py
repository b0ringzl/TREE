"""Read-only geometry diagnostic: compare original V/C map chunks with frame clouds.

Does not fit a transform or assume either coordinate variant is correct. The
comparison is a nearest-surface consistency check, not a calibrated registration.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from build_hk_dense_dataset import pcd_chunks
from build_hk_lidar_pairs import VMMS, read, write, voxel
from build_hk_five_species_pairs import OUT
from expand_requested_five_species import DenserFrameMaps
from build_hk_dense_dataset import coordinates,CACHE

def main():
    items=[r for r in read(OUT/'seeds.json') if r.get('survey')=='2023-10-27_hewentian' and r['status']=='seed_found']
    # First map chunk only, matched spatially against corresponding real frame clouds.
    results=[];rows=coordinates();maps=DenserFrameMaps(CACHE)
    for variant in ('DenseMap_Riegl_V_0.pcd','Opt_DenseMap_Riegl_C_0.pcd'):
        path=VMMS/'2023-10-27_hewentian/pointcloud/mapping'/variant
        stream=pcd_chunks(path);cloud=next(stream);stream.close()
        cloud=cloud[np.isfinite(cloud).all(1)];cloud=cloud[voxel(cloud,.10)]
        tree=cKDTree(cloud);xy_tree=cKDTree(cloud[:,:2]);centers=np.array([r['seed_center'] for r in items])
        close=np.argsort(xy_tree.query(centers[:,:2])[0]);seen=set();checks=[]
        for i in close:
            r=items[i]
            if r['frame_key'] in seen:continue
            if xy_tree.query(centers[i,:2])[0]>8:continue
            seen.add(r['frame_key']);frame,sources=maps.nearby(rows[r['frame_key']])
            pts=frame[np.linalg.norm(frame[:,:2]-centers[i,:2],axis=1)<8]
            distances=tree.query(pts)[0];near=distances[distances<1]
            checks.append({'frame_key':r['frame_key'],'frame_points':len(pts),'within_1m_points':len(near),
                           'median_nn_m_within_1m':float(np.median(near)) if len(near) else None,
                           'within_20cm_fraction':float((distances<.2).mean()) if len(pts) else 0,
                           'source_frame_files':sources})
            if len(checks)==3:break
        results.append({'source':str(path),'first_chunk_only':True,'checks':checks})
    write(OUT/'hwt_raw_map_alignment_diagnostic.json',results)
    print(results,flush=True)

if __name__=='__main__':main()
