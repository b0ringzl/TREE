"""Post-lock positive-only street classification audit; no fit and no deployment."""
import argparse
import base64
import html
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from focus8_data import ROOT, LABELS, TAXA
from bioclip_probe import ResidualHead, SOURCE, TEACHER
from distillation_core import read_json, save_json, metrics
from training_safety import sha256


def main(experiment):
    out = experiment / 'vmms_positive_audit'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'crops').mkdir()
    lock = read_json(experiment / 'selection_lock.json')
    if sha256(experiment / 'best.pt') != lock['checkpoint_sha256']:
        raise ValueError('Locked weight changed')
    derived = ROOT / 'hk-tree-vmms/derived'
    current_path = derived / 'exhaustive_species_frame_labeler/runtime/frame_review_state.json'
    old_path = derived / 'exhaustive_species_frame_labeler/runtime/backups/banyan_merge_20260906_073128/runtime/frame_review_state.json'
    current, old = read_json(current_path), read_json(old_path)
    refined = derived / 'delonix_refined_v1_20260907'
    reviews = read_json(refined / 'reviews.json')
    manifest = {r['id']: r for r in read_json(refined / 'manifest.json')}
    rows = []
    accepted_frames = set()
    for ident, decision in reviews.items():
        if decision.get('decision') != 'accept' or decision.get('species') != 'Delonix regia':
            continue
        r = manifest[ident]
        accepted_frames.add(r['frame_key'])
        rows.append({'id': ident, 'species': 'Delonix regia', 'source_crop': str(refined / 'instances' / ident / 'image.jpg'),
            'frame_key': r['frame_key'], 'review': decision, 'provenance': 'explicit_refined_pair_accept', 'tree_id': None})
    for key, frame in current.items():
        if frame.get('frame_status') != 'annotated' or not frame.get('updated_at_utc'):
            continue
        old_labels = {l['label_id']: l for l in old.get(key, {}).get('labels', [])}
        for label in frame.get('labels', []):
            species = label['species']
            target, provenance = None, None
            if species.startswith('Delonix regia') and key not in accepted_frames:
                target, provenance = 'Delonix regia', 'current_submitted_frame_positive_user_confirms_species'
            elif species.startswith('Bauhinia x blakeana') or species.startswith('Bauhinia × blakeana'):
                target, provenance = 'Bauhinia x blakeana', 'current_submitted_frame_positive_user_confirms_species'
            elif species.startswith('Ficus microcarpa'):
                target, provenance = 'Ficus microcarpa', 'current_explicit_raw_species'
            elif species in ('榕树', '榕樹'):
                previous = old_labels.get(label['label_id'], {})
                if previous.get('species', '').startswith('Ficus microcarpa') and previous.get('points') == label.get('points'):
                    target, provenance = 'Ficus microcarpa', 'premerge_raw_microcarpa_current_retained_same_geometry_user_confirms_species'
            if target:
                rows.append({'id': key + '_' + label['label_id'], 'frame_key': key, 'species': target,
                    'source_image': frame['source_image'], 'points': label['points'], 'provenance': provenance,
                    'submitted_at': frame['updated_at_utc'], 'tree_id': None})
    if not rows:
        raise RuntimeError('No auditable accepted positives found')
    training_hashes = {r['source_sha256'] for r in read_json(experiment / 'train_rows.json')}
    seen = set()
    selected = []
    for row in rows:
        source = Path(row.get('source_crop') or row['source_image'])
        row['source_sha256'] = sha256(source)
        if row['source_sha256'] in training_hashes:
            raise ValueError('Project source overlaps this training run')
        with Image.open(source) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            if 'points' in row:
                xy = np.asarray(row['points'])
                x0,y0 = xy.min(0); x1,y1 = xy.max(0)
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ValueError('Unexpected panorama coordinate system')
                box = [int(x0*im.width), int(y0*im.height), int(np.ceil(x1*im.width)), int(np.ceil(y1*im.height))]
                row['crop_box_xyxy'] = box
                im = im.crop(box)
            raw = im.tobytes()
            import hashlib
            key = hashlib.sha256(raw).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            row['crop_size'] = list(im.size)
            path = out / 'crops' / (str(len(selected)).zfill(4) + '.jpg')
            im.save(path, quality=95)
        row['crop'] = str(path)
        row['target'] = LABELS.index(row['species'])
        selected.append(row)
    save_json(out / 'manifest.json', selected)
    save_json(out / 'protocol.json', {'data_role': 'positive-only diagnostic; never exhaustive detection ground truth',
        'source_hashes': {str(p): sha256(p) for p in (current_path,old_path,refined/'reviews.json',refined/'manifest.json')},
        'sampling': 'All eligible persisted positives, exact-pixel dedup; no selection by model confidence or correctness',
        'independence': 'Not used in this fit. Historical review set, adjacent frames may show same tree; not an independent tree-count benchmark',
        'geometry': 'User polygons can contain multiple trees. Crop classification is diagnostic, not verified single-tree AP',
        'preprocessing': 'Whole annotated rectangle letterboxed to 224; same pixels for baseline and candidate; differs from web cache center crop',
        'checkpoint_sha256': lock['checkpoint_sha256'], 'counts': dict(Counter(r['species'] for r in selected)),
        'missing_species': [s for s in LABELS if not any(r['species']==s for r in selected)]})
    print('Frozen positive audit: ' + str(len(selected)) + ' crops ' + str(dict(Counter(r['species'] for r in selected))), flush=True)
    sys.path.insert(0, str(ROOT / 'external_datasets/distillation_runtime'))
    import open_clip
    meta = read_json(SOURCE / 'teacher_cache/bioclip25/metadata.json')
    if sha256(TEACHER) != meta['weight_sha256']:
        raise ValueError('Encoder changed')
    model, _, preprocess = open_clip.create_model_and_transforms('local-dir:' + str(TEACHER.parent), precision=meta['precision'], device='cuda')
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    aliases = {r['original_label']: r for r in read_json(SOURCE/'aliases.json')}
    bank = []
    with torch.inference_mode():
        for s in LABELS:
            texts = ['a photo of ' + term for term in aliases[s]['prompt_terms']]
            f = model.encode_text(tokenizer(texts).cuda(), normalize=True).float()
            bank.append(F.normalize(f.mean(0), dim=0))
        bank = torch.stack(bank)
        features = []
        dtype = torch.bfloat16 if meta['precision']=='bf16' else torch.float16
        for start in range(0,len(selected),8):
            batch=[]
            for row in selected[start:start+8]:
                with Image.open(row['crop']) as im:
                    square=ImageOps.pad(im.convert('RGB'), (224,224), method=Image.Resampling.BICUBIC, color=(124,116,104))
                    batch.append(preprocess(square))
            features.append(model.encode_image(torch.stack(batch).cuda().to(dtype), normalize=True).float())
        x = torch.cat(features)
        z = x @ bank.T * meta['logit_scale']
        checkpoint = torch.load(experiment / 'best.pt', map_location='cpu', weights_only=False)
        head=ResidualHead('linear',classes=8).cuda()
        head.load_state_dict(checkpoint['model_state']);head.eval()
        temperature=checkpoint['config']['temperature']
        base=(z/temperature).softmax(-1).cpu().numpy()
        candidate=(head(x,z)/temperature).softmax(-1).cpu().numpy()
    results={'baseline':metrics([r['target'] for r in selected],base), 'candidate':metrics([r['target'] for r in selected],candidate)}
    save_json(out/'results.json',results)
    np.savez(out/'predictions.npz',baseline=base,candidate=candidate)
    # Read-only evidence gallery. No automatic label acceptance or deployment controls.
    cards=[]
    for i,row in enumerate(selected):
        before=int(base[i].argmax());after=int(candidate[i].argmax())
        correct=after==row['target']
        with Image.open(row['crop']) as im:
            im.thumbnail((360,280));buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=85)
        uri='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
        cards.append('<article class="'+('correct' if correct else 'error')+'"><img src="'+uri+'"><h3>'+html.escape(TAXA[row['target']][1])+'</h3><p>'+html.escape(row['frame_key'])+'</p><p>Baseline / 原始教師: '+html.escape(TAXA[before][1])+f' {base[i,before]:.1%}</p><p>Candidate / 專項候選: '+html.escape(TAXA[after][1])+f' {candidate[i,after]:.1%}</p></article>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Focus-eight audit / 八類驗證</title><style>body{font:16px system-ui;background:#f3f6f1;color:#163b31;margin:30px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px}article{background:white;padding:16px;border:2px solid #cad9ca;border-radius:12px}article.error{border-color:#be6b45}img{width:100%;height:280px;object-fit:contain}button{padding:12px;margin-bottom:20px}</style><h1>Positive-crop audit / 正樣本裁片驗證</h1><p>Closed eight-class experiment, not detection accuracy. Model scores are not calibrated probabilities. / 限定八類的分類實驗，不是偵測精度；模型分數未經校準。</p><p>Repeated views may show the same tree. Missing classes have no street accuracy claim. / 相鄰影像可能是同一棵樹；缺少真值的類別不報街景準確率。</p><button onclick="document.querySelectorAll('.correct').forEach(x=>x.hidden=!x.hidden)">Toggle correct cases / 顯示或隱藏正確項目</button><main>'''+''.join(cards)+'</main></html>'
    (out/'gallery.html').write_text(page,encoding='utf8')
    print({k:round(v['top1'],4) for k,v in results.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    main(parser.parse_args().experiment.resolve())
