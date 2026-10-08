"""Read-only broad plant classifier using official BioCLIP 2.5 text vectors.

The HK head and original HK prompt bank are never altered. Scope routing is an
advisory, not a botanical range verdict or cross-model confidence comparison.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch

from tools.training_safety import sha256

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'external_datasets/bioclip25_flora_20261008_v1'


def normalize_name(name):
    return re.sub(r'\s+', ' ', name.replace('×', 'x').replace('_', ' ').strip()).casefold()


class FloraBank:
    def __init__(self, encoder_sha, hk_names):
        manifest = json.loads((DIRECTORY / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['encoder_sha256'] != encoder_sha or manifest['embedding_dimension'] != 1024:
            raise RuntimeError('Broad catalogue is incompatible with the image encoder')
        for row in manifest['derived_files']:
            path = (DIRECTORY / row['file']).resolve()
            if not path.is_relative_to(DIRECTORY.resolve()) or sha256(path) != row['sha256']:
                raise RuntimeError('Broad catalogue failed checksum')
        self.records = json.loads((DIRECTORY / 'plants.json').read_text(encoding='utf-8'))
        values = np.load(DIRECTORY / 'plant_embeddings.npy')
        if values.shape != (len(self.records), 1024) or not np.isfinite(values).all():
            raise RuntimeError('Broad catalogue dimensions or values are invalid')
        self.bank = torch.from_numpy(values).cuda()
        self.hk_names = {normalize_name(name) for name in hk_names}
        self.hk_mask = torch.tensor([normalize_name(r['species']) in self.hk_names for r in self.records],
                                    dtype=torch.bool, device='cuda')
        if int(self.hk_mask.sum()) < 100:
            raise RuntimeError('Too few Hong Kong taxa matched; scope check cannot be trusted')
        self.hk_genera = {name.split()[0] for name in self.hk_names if name}
        self.manifest = manifest

    def summarize(self, logits):
        # Logits are computed in the SAME broad catalogue and scale for both
        # best-in-scope and best-outside. Never compare SST percentages to this.
        logits = torch.stack(logits)
        pooled = logits.mean(0)
        probabilities = pooled.softmax(-1)
        scores, ids = probabilities.topk(3)
        top3 = []
        for index, score in zip(ids.cpu().tolist(), scores.cpu().tolist()):
            record = self.records[index]
            top3.append({**record, 'score': score, 'in_hk_catalog': bool(self.hk_mask[index])})
        best_hk = int(pooled.masked_fill(~self.hk_mask, -torch.inf).argmax())
        best_out = int(pooled.masked_fill(self.hk_mask, -torch.inf).argmax())
        view_ids = logits.argmax(-1).cpu().tolist()
        view_records = [self.records[i] for i in view_ids]
        margin = float(pooled[best_out] - pooled[best_hk])
        top = top3[0]
        # A deliberately conservative, predeclared pilot gate. Unseen species
        # in an already-covered genus receive no strong automatic scope claim.
        outside_genus = normalize_name(top['genus']) not in self.hk_genera
        agreeing = len({normalize_name(r['genus']) for r in view_records}) == 1
        suggest = (not top['in_hk_catalog'] and outside_genus and agreeing and margin >= 2.0)
        return {'available': True, 'top3': top3, 'version': self.manifest['version'],
                'class_count': len(self.records), 'matched_hk_taxa': int(self.hk_mask.sum()),
                'score_calibrated': False, 'temperature': 1.0,
                'scope_policy': 'flora-advisory-v1-margin2-same-genus-no-auto-switch',
                'suggest_broader_scope': suggest, 'outside_vs_hk_logit_margin': margin,
                'best_hk_species': self.records[best_hk]['species'],
                'best_hk_score': float(probabilities[best_hk]),
                'hk_probability_mass': float(probabilities[self.hk_mask].sum()),
                'view_top1': [r['species'] for r in view_records], 'view_genus_agreement': agreeing,
                'reason': 'outside_catalog_evidence' if suggest else 'no_strong_scope_evidence',
                'weights_changed': False, 'encoder_sha256': self.manifest['encoder_sha256'],
                'catalogue_sha256': self.manifest['derived_files'][1]['sha256']}
