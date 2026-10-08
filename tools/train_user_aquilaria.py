"""New user-confirmed data continuation; old immutable training scripts reused."""
import argparse,sys,gc
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_aquilaria_sstv1_conservative as base
from distillation_core import read_json,save_json,digest
from training_safety import sha256
ROOT=Path(__file__).resolve().parents[1]
USERDATA=ROOT/'external_datasets/aquilaria_user_confirmed_20260914_v1'

def prepare(out):
    base.prepare(out)
    imported=read_json(USERDATA/'manifest.json');accepted=[];decisions=[]
    for r in imported:
        assert sha256(r['original_source'])==r['sha256']==sha256(r['source_image'])
        reason=r['lane'] if r['lane']!='new_candidate' else 'montage_not_single_photo' if r['filename']=='webphoto.jpg' else None
        decisions.append({'filename':r['filename'],'id':r['id'],'sha256':r['sha256'],'species_confirmation':'user, 2026-09-14 explicit folder identification',
            'training_decision':'included' if reason is None else 'not_added','reason':reason or 'user species confirmed; visually readable photo',
            'previous_human_database_modified':False})
        if reason:continue
        name=r['filename'];group=r['id'];unit='image group, not verified independent tree'
        for prefix in ('093_','K13_','K7_','KC007_','Snipaste_'):
            if name.startswith(prefix):group='user_series_'+prefix
        if name in ('005.jpg','006.jpg','007.jpg'):group='user_series_005_007'
        if name in ('45FB32753B20D3CE.jpg','52E7A3CA99B792B2.jpg','FD5F743D05D8BE19.jpg'):group='user_shared_whole_tree_reference'
        # Same training-side source composition shown as a screenshot.
        if name in ('Fam_197.jpg','Snipaste_2026-09-14_17-47-50.png'):group='user_shared_flower_reference'
        accepted.append({**r,'sample_id':r['id'],'phase':'train','group_id':group,'group_unit':unit,
            'split_group':group,'training_eligible':True,'review_authority':'user_confirmed_species_current_request'})
    previous=read_json(out/'accepted_new_manifest.json')
    save_json(out/'accepted_new_manifest.json',previous+accepted)
    save_json(out/'user_import_manifest.json',imported);save_json(out/'user_confirmation_decisions.json',decisions);save_json(out/'accepted_user_manifest.json',accepted)
    cfg=read_json(out/'config.json');cfg.update(model_name='SSTv1-Aquilaria-User',accepted_compatible=len(previous)+len(accepted),
        new_counts=dict(Counter(r['phase'] for r in previous+accepted)),user_folder_total=len(imported),user_added_photos=len(accepted),
        user_label_authority='User explicitly identified the named local folder as Aquilaria sinensis; not relabeled by model',
        user_evaluation_policy='All non-overlapping qualified supplied photos enter TRAIN only. Their replay is fit evidence, never generalization accuracy.',
        review_authority='Previous agent-reviewed corpus plus separately recorded current user-confirmed folder; old human database unmodified',
        rights='New local files are user-supplied, some third-party watermarked; preserve originals and attribution, no publication or demo reference upload')
    save_json(out/'config.json',cfg)
    rows=read_json(out/'feature_rows.json')
    for r in accepted:
        for brightness in (1.,.7,1.2):rows.append({**r,'brightness':brightness,'sample_id':r['id']+f'_brightness_{brightness}'})
    save_json(out/'feature_rows.json',rows)
    contract=read_json(out/'base_contract.json')
    for p in (Path(__file__).resolve(),ROOT/'tools/audit_user_aquilaria.py'):
        contract['protected_artifacts'].append({'path':str(p),'sha256':sha256(p)})
    save_json(out/'base_contract.json',contract)
    assert len({r['sha256'] for r in accepted})==len(accepted)
    assert not any(set(m['phases'])&{'planning','selection','test','field','case'} for r in accepted for m in r['matches'])
    save_json(out/'user_data_preflight.json',{'user_originals_checked':len(imported),'user_added':len(accepted),'decisions':dict(Counter(d['reason'] for d in decisions)),
        'known_heldout_overlap_excluded':True,'user_files_train_only':True,'no_user_folder_changes':True,'model_labels_unchanged':True})
    print('USER PREPARED',len(accepted),'added;',len(rows),'feature views',flush=True)

def extract(out):
    # Reuse only identical, frozen, hash-verified old views, then embed added views.
    import io,torch
    from PIL import Image,ImageOps,ImageEnhance
    from tree_species_teacher_demo.engine import Engine,decode
    previous=ROOT/'outputs/SSTv1_Aquilaria_20260914_v3'
    oldrows=read_json(previous/'feature_rows.json');rows=read_json(out/'feature_rows.json')
    assert rows[:len(oldrows)]==oldrows
    with np.load(previous/'features.npz') as f:
        assert str(f['rows_hash'])==digest(oldrows)
        xx=[f['x'].copy()];zz=[f['z'].copy()];extra=f['extra_bank'].copy();scale=float(f['logit_scale'])
    e=Engine();e.load()
    assert np.allclose(e.extra_bank.cpu().numpy(),extra) and np.isclose(float(e.model.logit_scale.exp()),scale)
    with torch.inference_mode():
        for i,r in enumerate(rows[len(oldrows):]):
            assert sha256(r['source_image'])==r['sha256']
            with Image.open(r['source_image']) as src:
                im=ImageOps.exif_transpose(src).convert('RGB');im=ImageEnhance.Brightness(im).enhance(r['brightness'])
                edge=min(1024,max(im.size));im=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128));b=io.BytesIO();im.save(b,format='JPEG',quality=95)
            x=decode(b.getvalue())[0][None].cuda();feature=e.model.encode_image(((x-e.mean)/e.std).to(e.dtype),normalize=True).float();z=feature@e.bank.T*e.model.logit_scale.exp().float()
            xx.append(feature.half().cpu().numpy());zz.append(z.cpu().numpy())
            if i%40==0:print('USER EMBED',i,len(rows)-len(oldrows),flush=True)
    np.savez_compressed(out/'features.npz',x=np.concatenate(xx),z=np.concatenate(zz),extra_bank=extra,logit_scale=scale,rows_hash=digest(rows))
    save_json(out/'feature_reuse.json',{'source':str(previous/'features.npz'),'sha256':sha256(previous/'features.npz'),'reused_views':len(oldrows),'added_views':len(rows)-len(oldrows),'row_identity_verified':True})
    del e;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','extract','train','all'],default='all');a=p.parse_args();out=a.out.resolve()
    if a.stage in ('prepare','all'):prepare(out)
    if a.stage in ('extract','all'):extract(out)
    if a.stage in ('train','all'):base.train(out)
