"""Numerical checks and comparison sheets; does not mark samples accepted."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from refine_confirmed_delonix import OUT,PAIRS
from build_hk_lidar_pairs import read,write

def main():
    records=read(OUT/'manifest.json');checks=[]
    for r in records:
        if r['status']!='candidate_pending_review':continue
        root=OUT/'instances'/r['id'];asset=np.load(root/'point.npz');normalized=asset['points_xyz'];real=normalized*asset['scale']+asset['centroid_xyz'];full=np.load(root/'cleaned_local.npz')['xyz']
        assert 0<len(normalized)<=2048 and np.isfinite(normalized).all()
        assert len(np.unique(normalized,axis=0))==len(normalized)
        error=float(cKDTree(full).query(real)[0].max());assert error<1e-4
        assert np.linalg.norm(normalized,axis=1).max()<=1.00001
        checks.append({'id':r['id'],'sample_points':len(normalized),'max_roundtrip_error_m':error})
    hashes=read(OUT/'protected_hashes.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    write(OUT/'verification.json',{'assets_checked':len(checks),'checks':checks,'protected_files_unchanged':len(hashes),'new_auto_acceptances':0})
    for start in range(0,len(records),4):
        batch=records[start:start+4];sheet=Image.new('RGB',(1400,330*len(batch)),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(batch):
            y=j*330;draw.text((8,y+3),r['id']+'  stem_path='+str(r.get('quality',{}).get('stem_path_found')),fill='black')
            for x,name,w in ((0,'projection.jpg',730),(740,'point_views.jpg',650)):
                path=OUT/'instances'/r['id']/name
                if path.exists():
                    im=Image.open(path);im.thumbnail((w,295));sheet.paste(im,(x,y+25))
        sheet.save(OUT/('precheck_'+str(start//4)+'.jpg'),quality=93)
    print('Verified',len(checks),'unique real-point assets; protected files unchanged')

if __name__=='__main__':main()
