"""Persist model-free organ/quality notes separately from immutable raw intake."""
from collections import Counter
from pathlib import Path
import json

from PIL import Image, ImageOps
from import_conservation_browser_images_20260917 import digest, phash, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'external_datasets/artocarpus_whole_source_20260917_v1'
NOTES = {
    644765603: ('trunk_and_partial_crown', 'high', True,
        'Near trunk and partial crown; top and sides extend beyond the frame, with nearby trees and a finger at the bottom. Confirm the labelled individual. / 近距樹幹及局部樹冠，頂部和兩側超出畫面，鄰樹及手指入鏡；請確認來源標註的是哪株樹。'),
    476475668: ('leaf_detail', 'normal', False,
        'Leaf close-up only, not a whole-tree/crown view. Species remains source-labelled only. / 只有葉片近照，非整株或樹冠影像；樹種仍只有來源標註。'),
    167334042: ('leaf_and_branch_detail', 'normal', False,
        'Leaf and branch detail from the same observation as the two crown views; do not split across partitions. / 與兩張樹冠照屬同一觀察的葉枝細節，不可拆往不同資料分組。'),
    167334054: ('partial_crown', 'high', True,
        'Upward partial-crown view with visible leaves; does not show the complete crown or trunk. Confirm species with the paired detail. / 仰視局部樹冠，可見葉片，未包括完整樹冠及樹幹；請配合同觀察細節確認樹種。'),
    167334063: ('trunk_and_partial_crown', 'high', True,
        'Trunk and lower crown view, top outside frame. Pair with photos 167334042 and 167334054; source label is not independently confirmed. / 樹幹與下部樹冠，樹頂超出畫面；配合167334042及167334054核對，来源樹種未經獨立確認。'),
    125820639: ('leaf_detail_blurred', 'high', False,
        'Visible leaf detail is substantially out of focus; unsuitable for a whole-tree demonstration. / 葉片細節明顯失焦，不適合整株展示。'),
    125820830: ('bark_detail', 'normal', False,
        'Bark-only close-up from the same observation as a blurred leaf photo; no whole/crown structure. / 只有樹皮近照，與失焦葉照同一觀察，沒有整株或樹冠結構。'),
    9699477: ('mixed_understory_ambiguous_target', 'high', False,
        'Several trunks and foreground foliage; the source-labelled target is not visually unambiguous. Confirm target before using. / 多株樹幹及前景葉叢，無法由畫面明確判定來源標註的目標，使用前須確認。'),
}


def main():
    raw_path = OUT / 'manifest.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    rows = []
    for row in raw:
        organ, priority, relevant, note = NOTES[row['photo_id']]
        rows.append({**row, 'visual_qa': {
            'reviewer': 'agent_visible_content_only_no_model_predictions',
            'organ': organ, 'complete_whole_tree': False,
            'whole_crown_relevant_candidate': relevant,
            'species_confirmation': 'not_performed',
        }, 'review_priority': priority, 'review_note': note,
            'agent_recommendation': 'human_confirmation_required_not_accepted'})
    save(OUT / 'visual_review_manifest.json', rows)
    save(OUT / 'agent_visual_notes.json', {
        'raw_manifest_sha256': digest(raw_path), 'human_decisions_written': False,
        'count': len(rows), 'organ_counts': dict(Counter(r['visual_qa']['organ'] for r in rows)),
        'complete_whole_tree_count': 0, 'partial_crown_relevant_count': 3,
        'partial_crown_observations': [353356916, 100271692],
        'notes': [{'id': r['id'], 'photo_id': r['photo_id'], 'visual_qa': r['visual_qa'],
                   'review_note': r['review_note']} for r in rows],
    })
    cuhk = OUT / 'source_documents/cuhk_IMG_5086.jpg'
    with Image.open(cuhk) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        metadata = {'sha256': digest(cuhk), 'phash': format(phash(rgb), '016x'),
                    'dimensions': list(im.size), 'bytes': cuhk.stat().st_size}
    save(OUT / 'institution_source_leads.json', [
        {'source': 'Shiu Ying Hu Herbarium, CUHK', 'source_species': 'Artocarpus hypargyreus',
         'source_page': 'https://syhuherbarium.sls.cuhk.edu.hk/collections/plant-portfolio/',
         'image_url': 'https://syhuherbarium.sls.cuhk.edu.hk/collections/wp-content/uploads/sites/2/IMG_5086_v1024-1024x768.jpg',
         'source_image': str(cuhk.resolve()), **metadata,
         'organ': 'fruit_and_leaf_detail', 'whole_crown_relevant': False,
         'source_group': 'cuhk_plant_portfolio_artocarpus_hypargyreus_unknown_individual',
         'attribution': 'Shiu Ying Hu Herbarium, School of Life Sciences, CUHK; embedded watermark preserved',
         'license': 'not_verified', 'rights_status': 'rights_unverified',
         'training_eligible': False, 'demo_reference_eligible': False,
         'history_dedup': 'pending', 'human_review_status': 'not_accepted'},
        {'source': 'EPD public environmental monitoring report / Urban Arborist Ltd appendix',
         'source_species': 'Artocarpus hypargyreus',
         'source_page': 'https://www.epd.gov.hk/eia/files/applications/en/pp_130/vep_4230/progress/action_17226/femar2.pdf',
         'local_document': str((OUT / 'source_documents/epd_femar2.pdf').resolve()),
         'document_sha256': digest(OUT / 'source_documents/epd_femar2.pdf'),
         'attribution': 'Urban Arborist Ltd; environmental monitoring report hosted by Environmental Protection Department, Hong Kong',
         'source_tree_ids': ['AT261', 'AT262'], 'location': 'Victoria Road, Works Area MA, West Island Line temporary magazine',
         'observation_period': 'September 2009 to November 2011',
         'source_group': 'epd_wil_femar2_urban_arborist_same_report',
         'reported_individuals': 2, 'related_photo_pages_one_based': list(range(55, 67)),
         'most_clear_tree_id_evidence_pages': [60, 66],
         'organ': 'trunk_and_partial_crown_multi_temporal',
         'captions_caution': 'AT262 section pages 62-65 repeat AT261 in captions; do not infer independent trees from these repeated captions.',
         'native_photo_dimensions_examples': {'page60': [401, 515], 'page66': [393, 650]},
         'photos_extracted_for_training': 0, 'license': 'not_CC; third_party_rights_not_cleared',
         'rights_status': 'third_party_rights_unverified',
         'rights_source': 'https://www.epd.gov.hk/epd/english/environmentinhk/notices/notices.html',
         'rights_scope': 'Government material: personal/noncommercial internal local storage with copyright notice. Third-party material requires copyright-owner permission. Report contractor photographs cannot automatically inherit government permission.',
         'training_eligible': False, 'demo_reference_eligible': False,
         'history_dedup': 'not_certified; document/tree/photo evidence only',
         'human_review_status': 'not_accepted'},
        {'source': 'EPD Fo Tan Nullah Revitalisation EIA Appendix 9.4',
         'source_species': 'Artocarpus hypargyreus',
         'source_page': 'https://www.epd.gov.hk/eia/files/applications/en/pp_319/eia_1964/progress/action_1512/EIA/Appendices/Appendix%209.4.pdf',
         'local_document': str((OUT / 'source_documents/epd_FoTan_appendix9_4.pdf').resolve()),
         'organ': 'juvenile_leaf_and_branch_detail', 'whole_crown_relevant': False,
         'evidence': 'Page 1 Plate 4 explicitly labelled Silver-back Artocarpus; July 2021 AECOM/Drainage Services Department plate.',
         'license': 'not_verified', 'rights_status': 'third_party_rights_unverified',
         'training_eligible': False, 'demo_reference_eligible': False},
    ])


if __name__ == '__main__':
    main()
