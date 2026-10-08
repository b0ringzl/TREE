"""Inference-only BioCLIP service. Never fits, writes weights, or queries example labels."""
import base64
import io
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms import functional as TF, InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'external_datasets/distillation_runtime'))
from bioclip_probe import ResidualHead, SOURCE
from distillation_core import read_json, collapse_logits
from training_safety import sha256
from tree_species_demo.demo_locale import TRAD

EXPERIMENT = ROOT / 'outputs/distillation/bioclip_probe_20260908_0227'
MODES = {'teacher': 'BioCLIP 2.5 · Original teacher / 原始教師',
         'adapted': 'BioCLIP 2.5 + MLP · Adapted teacher / 適配教師',
         'sstv1': 'SSTv1 · Adapted teacher / 增量適配教師'}
SSTV1_MANIFEST = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/deployment.json'
MAX_BYTES = 20 * 1024 * 1024
CONTEXT_PIPELINE = 'sstv1-confirmed-target-whole-center-mean-v2'


def display_name(label):
    if label.startswith('Ficus microcarpa / Ficus benjamina'):
        return 'Ficus microcarpa / Ficus benjamina · 榕樹'
    translation = TRAD.get(label, '')
    if label.startswith('Araucaria columnaris'):
        translation = '南洋杉'
    return label + (' / ' + translation if translation else '')


def decode(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Provide an image up to 20 MB. / 請提供不超過 20 MB 的影像。')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ('JPEG', 'PNG', 'WEBP') or source.width * source.height > 40_000_000:
                    raise ValueError('Use JPG, PNG or WEBP, up to 40 million pixels. / 請使用 JPG、PNG 或 WEBP，上限四千萬像素。')
                im = ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Image cannot be decoded. / 無法讀取影像，請重新匯出。') from exc
    crop = TF.center_crop(TF.resize(im, 224, interpolation=InterpolationMode.BICUBIC), [224, 224])
    gray = np.asarray(crop.convert('L'), dtype=np.float32)
    blur = float(np.var(gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]))
    notices = []
    if min(im.size) < 224:
        notices.append('Low resolution: supply a sharper image at least 224 × 224. / 解像度偏低：請提供至少 224 × 224 的清晰影像。')
    if float((gray < 15).mean()) > .4:
        notices.append('Dark input: improve lighting. / 影像偏暗：請改善光線。')
    if float((gray > 245).mean()) > .4:
        notices.append('Bright input: reduce overexposure. / 影像偏亮：請降低過度曝光。')
    if blur < 30:
        notices.append('Little fine detail: refocus or move closer. / 細節偏少：請重新對焦或靠近目標。')
    retained = min(im.size) / max(im.size)
    if retained < .75:
        notices.append('Center crop omits much of the image; crop one tree before uploading. / 中央裁切略去較多畫面；請先裁切單一樹木再上傳。')
    encoded = io.BytesIO()
    crop.save(encoded, format='JPEG', quality=92)
    return TF.to_tensor(crop), {'width': im.width, 'height': im.height, 'crop_retained_fraction': retained,
        'detail_variance': blur, 'warnings': notices}, 'data:image/jpeg;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii')


class Engine:
    def __init__(self):
        self.state = 'unloaded'
        self.lock = threading.Lock()
        self.classes = read_json(EXPERIMENT / 'classes.json')
        # Keep the original head mapping immutable; extend presentation separately.
        self.labels = self.classes['labels'] + ['Wodyetia bifurcata']
        self.sstv1_heads = []
        self.sstv1_state = 'not_loaded'
        self.flora_state = 'not_loaded'
        self.flora_bank = None

    def load(self):
        with self.lock:
            if self.state == 'ready':
                return
            self.state = 'loading'
            try:
                self._load()
                self.state = 'ready'
            except Exception:
                self.state = 'error'
                raise

    def _load(self):
        import open_clip
        if not torch.cuda.is_available():
            raise RuntimeError('This installation requires CUDA')
        protocol = read_json(EXPERIMENT / 'protocol.json')
        meta = read_json(EXPERIMENT / 'cache_metadata.json')
        self.choice = read_json(EXPERIMENT / 'teacher_choice.json')
        selection = read_json(EXPERIMENT / 'selection_lock.json')
        weight = Path(protocol['teacher_weight'])
        if sha256(weight) != protocol['teacher_sha256'] or sha256(SOURCE / 'aliases.json') != meta['aliases_sha256']:
            raise ValueError('Teacher or prompt mapping checksum mismatch')
        if meta['class_names'] != self.classes['original_names']:
            raise ValueError('Class ordering mismatch')
        self.encoder_sha256 = protocol['teacher_sha256']
        self.model, _, _ = open_clip.create_model_and_transforms('local-dir:' + str(weight.parent), precision=meta['precision'], device='cuda')
        self.model.eval().requires_grad_(False)
        self.dtype = torch.bfloat16 if meta['precision'] == 'bf16' else torch.float16
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
        bank = []
        templates = ['{}', 'a photo of {}', 'a photo of the tree species {}']
        with torch.inference_mode():
            for row in read_json(SOURCE / 'aliases.json'):
                if self.choice['prompt'] < 3:
                    texts = [templates[self.choice['prompt']].format(n) for n in row['prompt_terms']]
                else:
                    texts = [t.format(n) for n in row['prompt_terms'] for t in ['{}', 'a photo of {}', 'a street tree of {}', 'a photo of the tree species {}']]
                features = self.model.encode_text(tokenizer(texts).cuda(), normalize=True).float()
                bank.append(F.normalize(features.mean(0), dim=0))
        self.bank = torch.stack(bank)
        with torch.inference_mode():
            self.extra_bank = F.normalize(self.model.encode_text(
                tokenizer(['a photo of Wodyetia bifurcata']).cuda(), normalize=True).float(), dim=-1)
        self.heads = []
        for run in selection['selected_runs']:
            path = Path(run['run']) / 'best.pt'
            if sha256(path) != selection['weights_sha256'][run['run']]:
                raise ValueError('Head checksum mismatch')
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state['encoder_sha256'] != self.encoder_sha256 or state['classes'] != self.classes:
                raise ValueError('Incompatible classifier head')
            head = ResidualHead(state['config']['architecture']).cuda().eval().requires_grad_(False)
            head.load_state_dict(state['model_state'])
            self.heads.append(head)
        self.mean = torch.tensor([.48145466, .4578275, .40821073], device='cuda')[None, :, None, None]
        self.std = torch.tensor([.26862954, .26130258, .27577711], device='cuda')[None, :, None, None]
        if SSTV1_MANIFEST.is_file():
            try:
                self._load_sstv1()
            except Exception:
                import logging
                self.sstv1_state = 'error'
                logging.exception('SSTv1 unavailable; original teacher modes remain available')

    def _load_sstv1(self):
        manifest = read_json(SSTV1_MANIFEST)
        if (manifest['encoder_sha256'] != self.encoder_sha256 or
                manifest['output_count'] != len(self.labels) or
                manifest['temperature'] != self.choice['calibration_temperature'] or
                len(manifest['members']) != 3):
            raise ValueError('Incompatible SSTv1 deployment')
        heads = []
        for member in manifest['members']:
            path = (SSTV1_MANIFEST.parent / member['file']).resolve()
            if not path.is_relative_to(SSTV1_MANIFEST.parent.resolve()) or sha256(path) != member['sha256']:
                raise ValueError('SSTv1 member checksum mismatch')
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state['classes'] != self.classes or state['encoder_sha256'] != self.encoder_sha256:
                raise ValueError('Incompatible SSTv1 label mapping')
            head = ResidualHead(state['config']['architecture']).cuda().eval().requires_grad_(False)
            head.load_state_dict(state['model_state'])
            heads.append(head)
        self.sstv1_heads = heads
        self.sstv1_version = manifest['version']
        self.sstv1_state = 'ready'

    def predict(self, data, mode='adapted'):
        if mode not in MODES:
            raise ValueError('Select a supported model. / 請選擇支援的模型。')
        pixels, quality, thumbnail = decode(data)
        if self.state != 'ready':
            raise RuntimeError('Model is not ready')
        if mode == 'sstv1' and self.sstv1_state != 'ready':
            raise RuntimeError('SSTv1 is not ready; no silent fallback')
        with self.lock, torch.inference_mode():
            started = time.perf_counter()
            embedding = self.model.encode_image(((pixels[None].cuda() - self.mean) / self.std).to(self.dtype), normalize=True).float()
            raw = embedding @ self.bank.T * self.model.logit_scale.exp().float()
            groups, temperature = self.classes['groups'], self.choice['calibration_temperature']
            extra = embedding @ self.extra_bank.T * self.model.logit_scale.exp().float()
            probs = {'teacher': (torch.cat([collapse_logits(raw, groups), extra], -1) / temperature).softmax(-1)[0]}
            feature = embedding.half().float()
            # Match the evaluated expansion baseline: no new trained palm layer.
            adapted_extra = feature @ self.extra_bank.T * self.model.logit_scale.exp().float()
            probs['adapted'] = torch.stack([(torch.cat([collapse_logits(h(feature, raw), groups), adapted_extra], -1) / temperature).softmax(-1) for h in self.heads]).mean(0)[0]
            if mode == 'sstv1':
                probs['sstv1'] = torch.stack([(torch.cat([collapse_logits(h(feature, raw), groups), adapted_extra], -1) / temperature).softmax(-1) for h in self.sstv1_heads]).mean(0)[0]
            output = {}
            for key, tensor in probs.items():
                scores, ids = tensor.topk(3)
                output[key] = [{'species': self.labels[i], 'name': display_name(self.labels[i]), 'score': score}
                               for i, score in zip(ids.cpu().tolist(), scores.cpu().tolist())]
            elapsed = time.perf_counter() - started
        return {'model': MODES[mode], 'mode': mode, 'top3': output[mode], 'comparison': output,
                'quality': quality, 'model_input': thumbnail, 'inference_seconds': elapsed,
                'class_count': len(self.labels), 'experimental': True,
                'teacher_distribution': dict(zip(self.labels, probs['teacher'].detach().cpu().tolist())),
                'adapted_distribution': dict(zip(self.labels, probs['adapted'].detach().cpu().tolist())),
                **({'sstv1_distribution': dict(zip(self.labels, probs['sstv1'].detach().cpu().tolist()))} if mode == 'sstv1' else {}),
                'deployment': self.sstv1_version if mode == 'sstv1' else 'teacher-148-foxtail-zero-shot-extension', 'palm_correction_loaded': False,
                'encoder_sha256': self.encoder_sha256,
                'score_notice': 'Model scores, not guaranteed correctness. / 模型分數，不代表正確率。',
                'scope': 'Single-tree image classification; no point-cloud input or botanical explanation model. / 單木影像分類；不使用點雲，亦非植物學解釋模型。'}

    def predict_context(self, whole_data, original_data):
        """Two batch=1 views of ONE confirmed target, not independent observations.

        Whole is the exact existing Demo letterbox/JPEG95. Detail is decoded
        directly from the original target bytes, never from its padded JPEG.
        Keep legacy predict() unchanged for point fusion and regression replay.
        """
        whole = self.predict(whole_data, 'sstv1')
        detail = self.predict(original_data, 'sstv1')
        if whole['deployment'] != detail['deployment']:
            raise RuntimeError('Classifier changed between views')
        result = dict(whole)
        result['comparison'] = {}
        for mode in ('teacher', 'adapted', 'sstv1'):
            key = mode + '_distribution'
            mixed = {label: .5 * whole[key][label] + .5 * detail[key][label]
                     for label in self.labels}
            result[key] = mixed
            ranked = sorted(mixed, key=mixed.get, reverse=True)[:3]
            result['comparison'][mode] = [dict(species=s, name=display_name(s), score=mixed[s]) for s in ranked]
        result['top3'] = result['comparison']['sstv1']
        result['inference_seconds'] = whole['inference_seconds'] + detail['inference_seconds']
        result['model_input_detail'] = detail['model_input']
        result['inference_pipeline'] = CONTEXT_PIPELINE
        result['view_evidence'] = dict(
            whole_top1=whole['top3'][0], detail_top1=detail['top3'][0],
            agreement=whole['top3'][0]['species'] == detail['top3'][0]['species'],
            whole_weight=.5, detail_weight=.5, encoder_passes=2,
            independent_images=1, detail_retained_fraction=detail['quality']['crop_retained_fraction'])
        result['quality'] = dict(detail['quality'])
        # Detail cropping is intentional: the whole-tree view remains present.
        result['quality']['warnings'] = [w for w in detail['quality']['warnings']
                                         if not w.startswith('Center crop omits')]
        if not result['view_evidence']['agreement']:
            result['quality']['warnings'].append(
                'Whole-image and center-detail predictions differ. Check the target or add a clear leaf view. / 全圖與中心細節判斷不同，請核對目標或補充清晰葉片。')
        result['scope'] = 'One target image, two deterministic crops; no multi-photo or point-cloud fusion. / 同一目標影像的兩種取景；並非多張影像或點雲融合。'
        return result

    def predict_flora(self, whole_data, detail_data=None):
        """Independent original-encoder broad-catalogue query; no trained heads."""
        if self.state != 'ready':
            raise RuntimeError('Original BioCLIP is not ready')
        views = [decode(whole_data)]
        if detail_data is not None:
            views.append(decode(detail_data))
        with self.lock, torch.inference_mode():
            started = time.perf_counter()
            if self.flora_bank is None:
                from .flora import FloraBank
                try:
                    aliases = read_json(SOURCE / 'aliases.json')
                    hk_names = self.classes['original_names'] + ['Wodyetia bifurcata']
                    hk_names += [name for row in aliases for name in row['prompt_terms']]
                    self.flora_bank = FloraBank(self.encoder_sha256, hk_names)
                    self.flora_state = 'ready'
                except Exception:
                    self.flora_state = 'error'
                    raise RuntimeError('Broad plant catalogue unavailable; Hong Kong model is unchanged')
            logits = []
            for pixels, _, _ in views:
                feature = self.model.encode_image(((pixels[None].cuda() - self.mean) / self.std).to(self.dtype), normalize=True).float()
                logits.append((feature @ self.flora_bank.bank.T * self.model.logit_scale.exp().float())[0])
            result = self.flora_bank.summarize(logits)
            result.update(inference_seconds=time.perf_counter() - started,
                          model_input=views[0][2], model_input_detail=views[1][2] if len(views) == 2 else None,
                          quality=views[-1][1], view_count=len(views))
            return result
