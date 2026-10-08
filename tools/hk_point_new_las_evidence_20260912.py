"""Projection evidence for exact-coordinate matches; no automatic species truth."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'hk-tree-vmms/tools'))
from hk_point_training_20260912 import OUT,read,write
from rebuild_banyan_coverage_preview import sample_ids,projection
from build_hk_dense_dataset import coordinates

def main():
    dest=OUT/'new_las_evidence';dest.mkdir(exist_ok=False);rows={r['key']:r for r in read(OUT/'manifest.json')};sources={}
    for name in ['hk_dualsource_three_species_20260907','hk_dualsource_five_species_20260907']:
        sources.update({r['id']:r for r in read(ROOT/'hk-tree-vmms/derived'/name/'manifest.json') if r.get('source_panorama')})
    coords=coordinates();evidence=[]
    for r in read(OUT/'new_las_associations.json'):
        if not r['matches']:continue
        match=rows[r['matches'][0]['key']];orig=sources[match['id']];folder=dest/r['id'];folder.mkdir();x=np.load(Path(r['point']).with_name('cleaned.npz'))['xyz']
        projection(x,sample_ids(x,8000),coords[orig['frame_key']],Path(orig['source_panorama']),folder/'projection.jpg',float(np.quantile(x[:,2],.25)),'all')
        e=dict(id=r['id'],matched_key=match['key'],suggested_species=match['species'],image=match['source_assets']['image'],source_panorama=orig['source_panorama'],frame_key=orig['frame_key'],projection=str(folder/'projection.jpg'),views=r['views'],exact_coordinate_fraction=r['matches'][0]['exact_coordinate_fraction'],training_eligible=False,coordinate_system_evidence='millimetre-level shared original points; empirical VMMS image projection, not independently calibrated extrinsics',status='projection_for_visual_verification_not_species_truth');evidence.append(e)
    write(dest/'manifest.json',evidence)
    for page in range((len(evidence)+3)//4):
        sheet=Image.new('RGB',(1800,1200),'white');d=ImageDraw.Draw(sheet)
        for i,e in enumerate(evidence[page*4:(page+1)*4]):
            xx=(i%2)*900;yy=(i//2)*600
            with Image.open(e['projection']) as im:sheet.paste(ImageOps.contain(im,(890,450)),(xx,yy))
            with Image.open(e['views']) as im:sheet.paste(ImageOps.contain(im,(660,140)),(xx,yy+440))
            d.text((xx+5,yy+580),e['id']+' '+e['suggested_species']+' overlap '+str(round(e['exact_coordinate_fraction'],3)),fill='black')
        sheet.save(dest/f'contact_{page+1}.jpg')
if __name__=='__main__':main()
