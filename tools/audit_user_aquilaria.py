"""Import user-identified images without modifying originals or old decisions."""
import argparse,json,re,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from collect_sstv1_incremental import phash
from training_safety import sha256
from distillation_core import read_json,save_json
ROOT=Path(__file__).resolve().parents[1]
def norm(p):return str(p).replace('\\','/').lower()
def main(src,out):
    src=src.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'originals').mkdir();(out/'contact_sheets').mkdir()
    phase_by_path={};phase_by_hash={}
    for phase in ('train','planning','selection','test','field','case'):
        for r in read_json(ROOT/f'outputs/distillation/20260908_0045/{phase}_rows.json'):
            for key in ('source_image','materialized_path'):
                if r.get(key):phase_by_path.setdefault(norm(r[key]),set()).add(phase)
            if r.get('source_sha256'):phase_by_hash.setdefault(r['source_sha256'],set()).add(phase)
    index=[]
    for line in (ROOT/'external_datasets/herbarium_training_candidates_20260907/existing_image_index.jsonl').read_text(encoding='utf8').splitlines():
        r=json.loads(line)
        if not r.get('phash'):continue
        phases=phase_by_path.get(norm(r['path']),set())|phase_by_hash.get(r.get('sha256'),set())
        index.append({'path':r['path'],'sha256':r.get('sha256'),'hash':int(r['phash'],16),'phases':sorted(phases) or ['existing_unassigned']})
    # Protect all previous supplementary collections, including human-pending/rejected entries.
    for root,manifest in [(ROOT/'external_datasets/sstv1_incremental_20260914_v1','manifest.json'),(ROOT/'external_datasets/aquilaria_specialist_20260914_v1','manifest.json')]:
        phases={}
        previous=ROOT/('outputs/SSTv1_incremental_20260914_v1/accepted_new_manifest.json' if 'sstv1_' in root.name else 'outputs/SSTv1_Aquilaria_20260914_v3/accepted_new_manifest.json')
        for r in read_json(previous):phases[r['id']]=r['phase']
        for r in read_json(root/manifest):
            if r.get('phash'):index.append({'path':str(root/r['file']),'sha256':r['sha256'],'hash':int(r['phash'],16),'phases':[phases.get(r['id'],'previous_pending_or_excluded')]})
    rows=[]
    for i,p in enumerate(sorted(src.rglob('*'))):
        if not p.is_file() or p.suffix.lower() not in ('.png','.jpg','.jpeg','.webp','.bmp'):continue
        rawhash=sha256(p);file=out/'originals'/(f'{len(rows):03d}_'+p.name);shutil.copy2(p,file);assert sha256(file)==rawhash
        r={'id':'user_aq_'+rawhash[:16],'filename':p.name,'original_source':str(p),'source_image':str(file),'sha256':rawhash,'target':11,'source_species':'Aquilaria sinensis',
            'label_authority':'explicit user identification, current request','rights':'User-supplied; original rights/attribution retained, not cleared for publication or redistribution','source_group':'user_batch_identity_unresolved','matches':[],'quality_flags':[]}
        try:
            im=ImageOps.exif_transpose(Image.open(file)).convert('RGB');gray=np.asarray(im.convert('L'));h=phash(im)
            r.update(width=im.width,height=im.height,phash=f'{h:016x}',dark_fraction=float(np.mean(gray<25)))
            if min(im.size)<160:r['quality_flags'].append('very_small')
            if r['dark_fraction']>.8:r['quality_flags'].append('mostly_dark')
            matches=[{k:v for k,v in old.items() if k!='hash'}|{'phash_distance':(h^old['hash']).bit_count()} for old in index if rawhash==old['sha256'] or (h^old['hash']).bit_count()<=6]
            r['matches']=sorted(matches,key=lambda q:q['phash_distance'])[:30]
        except Exception as e:r['quality_flags'].append('unreadable');r['error']=str(e)
        # Priority: keep heldout separation, then avoid double-counting training copies.
        held=any(set(m['phases'])&{'planning','selection','test','field','case'} for m in r['matches'])
        r['lane']='excluded_heldout_overlap' if held else 'excluded_existing_duplicate' if r['matches'] else 'quality_review' if r['quality_flags'] else 'new_candidate'
        rows.append(r)
    # Within-folder duplicates stay on one side and do not inflate sample count.
    for i,r in enumerate(rows):
        if r['lane']!='new_candidate':continue
        near=[x for x in rows[:i] if x.get('phash') and (x['sha256']==r['sha256'] or (int(x['phash'],16)^int(r['phash'],16)).bit_count()<=6)]
        if near:
            r['within_folder_matches']=[x['filename'] for x in near]
            r['lane']='excluded_heldout_overlap' if any(x['lane']=='excluded_heldout_overlap' for x in near) else 'excluded_within_folder_duplicate'
    save_json(out/'manifest.json',rows)
    for start in range(0,len(rows),16):
        sheet=Image.new('RGB',(1440,1440),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[start:start+16]):
            im=Image.open(r['source_image']);im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((350,285));x=j%4*360;y=j//4*360;sheet.paste(im,(x+(350-im.width)//2,y))
            draw.text((x+3,y+288),str(start+j)+' '+r['filename'][:41],fill='black');draw.text((x+3,y+309),r['lane'],fill='black')
        sheet.save(out/'contact_sheets'/f'{start//16:02d}.jpg',quality=94)
    summary={'original_folder':str(src),'total':len(rows),'lanes':dict(Counter(r['lane'] for r in rows)),'originals_verified':True,'source_folder_modified':False,'species_confirmation':'user-provided','not_independent_trees':True}
    save_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.source,a.out)
