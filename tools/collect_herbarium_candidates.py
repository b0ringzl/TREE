"""Download the audited public photo manifest, preserving originals and provenance."""
import hashlib,json,io
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
import numpy as np
from scipy.fft import dctn
from inventory_herbarium_images import Client,ROOT,quality,write_json

INVENTORY=ROOT/'external_datasets/herbarium_inventory_20260907'
OUT=ROOT/'external_datasets/herbarium_training_candidates_20260907'

def perceptual_hash(im):
    g=np.array(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    values=dctn(g,type=2,norm='ortho')[:8,:8].flatten();bits=values>np.median(values[1:])
    return format(int(''.join('1' if x else '0' for x in bits),2),'016x')

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True);(OUT/'contact_sheets').mkdir(exist_ok=True)
    c=Client(INVENTORY);policy=c.policy();write_json(OUT/'access_and_rights.json',policy)
    source=json.loads((INVENTORY/'photo_manifest.json').read_text(encoding='utf-8'))
    existing=json.loads((OUT/'manifest.json').read_text(encoding='utf-8')) if (OUT/'manifest.json').exists() else []
    records=[];errors=[];tiles=[]
    for i,s in enumerate(source):
        ident=s['source_species_id']+'_'+s['media_id'];name='images/'+ident+'.jpg'
        record={**s,'id':ident,'file':name,'organ_review':'pending','split':'quarantine','training_eligible':False}
        try:
            raw,meta=c.get(s['image_url']);q,im=quality(raw)
            dest=OUT/name
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=meta['sha256']:raise RuntimeError('Existing image changed; refusing overwrite: '+ident)
            if not dest.exists():dest.write_bytes(raw)
            record.update(status='downloaded',sha256=meta['sha256'],bytes=len(raw),quality=q,phash=perceptual_hash(im))
            tile=Image.new('RGB',(360,300),'white');thumb=im.copy();thumb.thumbnail((350,245));tile.paste(thumb,((360-thumb.width)//2,0));draw=ImageDraw.Draw(tile)
            draw.text((7,249),f'{i+1:03d} / {ident}',fill='black');draw.text((7,269),s['query'],fill='black');tiles.append(tile)
        except RuntimeError:raise # Stop on access denial/rate limit/content conflict.
        except Exception as e:
            record.update(status='failed',error=type(e).__name__+': '+str(e));errors.append({'id':ident,'error':record['error']})
        records.append(record);write_json(OUT/'manifest.json',records);write_json(OUT/'download_errors.json',errors)
        if len(tiles)==20 or i==len(source)-1:
            sheet=Image.new('RGB',(1440,300*((len(tiles)+3)//4)),(232,236,231))
            for n,t in enumerate(tiles):sheet.paste(t,((n%4)*360,(n//4)*300))
            if tiles:sheet.save(OUT/'contact_sheets'/f'batch_{i//20+1:02d}.jpg',quality=92)
            tiles=[]
        if (i+1)%10==0:print(f'{i+1}/{len(source)} processed; failed={len(errors)}',flush=True)
    valid=[r for r in records if r['status']=='downloaded'];byhash={};exact=[];near=[]
    for r in valid:byhash.setdefault(r['sha256'],[]).append(r['id'])
    exact=[ids for ids in byhash.values() if len(ids)>1]
    for i,a in enumerate(valid):
        for b in valid[i+1:]:
            distance=(int(a['phash'],16)^int(b['phash'],16)).bit_count()
            if distance<=6:near.append({'a':a['id'],'b':b['id'],'hamming_distance':distance,'same_sha256':a['sha256']==b['sha256'],'status':'needs_visual_verification'})
    write_json(OUT/'within_source_duplicates.json',{'exact_groups':exact,'phash_candidates':near,'phash_threshold':6,'note':'Similarity is not identity; no photos deleted.'})
    summary={'discovered':len(source),'downloaded':len(valid),'failed':len(errors),'unique_sha256':len(byhash),'exact_duplicate_groups':len(exact),'phash_candidate_pairs':len(near),'bytes':sum(r['bytes'] for r in valid),'quality_flags':dict(Counter(f for r in valid for f in r['quality']['heuristic_flags'])),'organ_reviewed':0,'training_eligible':0,'models_modified':False}
    write_json(OUT/'collection_summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
