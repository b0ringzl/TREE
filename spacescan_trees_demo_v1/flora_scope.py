"""Independent original BioCLIP results and conservative, reversible scope advice."""
import requests

from .species_names import bilingual_names

SCOPES = {'hong_kong', 'broad_plants'}
SCOPE_VERSION = 'flora-advisory-v1-margin2-same-genus-no-auto-switch'


def query(upstream, pixels, data, view_policy):
    files = {'image': ('whole.jpg', pixels, 'image/jpeg')}
    if view_policy == 'context':
        files['detail'] = ('detail', data)
    response = requests.post(upstream + '/api/predict-flora', files=files, timeout=120)
    response.raise_for_status()
    result = response.json()
    if result.get('scope_policy') != SCOPE_VERSION or not result.get('available') or len(result.get('top3', [])) != 3:
        raise ValueError('Invalid broad model response')
    return result


def candidates(raw, references, digest):
    rows = []
    for record in raw['top3']:
        species = record['species']
        score = float(record['score'])
        if not 0 <= score <= 1:
            raise ValueError('Invalid broad model score')
        try:
            names = bilingual_names(species)
            refs = references(species, digest)
        except KeyError:
            common = record.get('common_name') or species
            names = {'scientific_name': species, 'chinese_name': common, 'name': species + ' / ' + common}
            refs = []
        rows.append({**names, 'species': species, 'score': score, 'references': refs,
                     'family': record['family'], 'in_hk_catalog': record['in_hk_catalog']})
    return rows


def advisory(raw, sst_top, quality_warnings):
    if any('Dark input' in note or 'Bright input' in note or 'Little fine detail' in note
           for note in quality_warnings):
        return {'action': 'improve_image', 'message': '影像細節不足，請更換角度或補充清晰葉／花近照；暫不能據此判定超出香港範圍。',
                'reason': 'poor_quality', 'auto_switched': False}
    if raw.get('suggest_broader_scope'):
        species = raw['top3'][0]['species']
        return {'action': 'suggest_broad', 'message': '可能不在香港常見樹木範圍內。世界模型的首位候選為 ' + species + '。要查看世界模型候選嗎？',
                'reason': 'outside_catalog_evidence', 'auto_switched': False}
    if float(sst_top['score']) < .14:
        return {'action': 'uncertain', 'message': '該樹種暫不能確定。請更換觀察角度或補充葉／花近照，也可嘗試世界模型。',
                'reason': 'sst_score_below_14_percent', 'auto_switched': False}
    return {'action': 'stay', 'message': '', 'reason': 'no_strong_scope_evidence', 'auto_switched': False}


def broad_result(raw, width, height, references, digest, target_digest, padded_digest):
    return {'brand': 'SpaceScan Trees', 'scenario': 'street', 'source': 'image',
            'mode': 'bioclip_broad', 'recognition_scope': 'broad_plants',
            'model': '世界模型', 'image_model_version': raw['version'],
            'display_class_count': raw['class_count'], 'top3': candidates(raw, references, digest),
            'raw_model_top3': raw['top3'], 'image_distribution': {}, 'image_display_distribution': {},
            'warnings': ['世界模型分數是此植物目錄內的候選排序，不能與 SST 分數直接比較。',
                         '世界模型零樣本推理；目錄覆蓋不等於每個物種均已驗證可準確識別。'] +
                        (['該樹種暫不能確定，請更換觀察角度或補充清晰葉／花近照。'] if raw['top3'][0]['score'] < .14 else []),
            'width': width, 'height': height, 'inference_seconds': raw['inference_seconds'],
            'model_input': raw['model_input'], 'model_input_detail': raw.get('model_input_detail'),
            'preprocessing': '世界模型圖像特徵＋植物文本特徵；' +
                             ('完整裁片單次推理，無中心細節投票。' if raw['view_count'] == 1 else '完整影像＋中心細節，等權平均廣域 logits。'),
            'inference_pipeline': 'bioclip25-official-plantae-' + ('single-v1' if raw['view_count'] == 1 else 'whole-center-logit-mean-v1'),
            'single_view': raw['view_count'] == 1, 'multiview_fusion': False, 'tree_detection': False,
            'view_evidence': None, 'comparison_review': None, 'specialist_model': False,
            'flora_check': {k: v for k, v in raw.items() if k not in {'model_input', 'model_input_detail', 'quality'}},
            'release_provenance': {'actual_image_classifier': raw['version'], 'classifier_alignment': 'separate-original-bioclip-broad-plant-branch',
                                   'target_bytes_sha256': target_digest, 'padded_jpeg_sha256': padded_digest,
                                   'encoder_sha256': raw['encoder_sha256'], 'catalogue_sha256': raw['catalogue_sha256'],
                                   'weights_changed': False, 'automatic_scope_switch': False}}
