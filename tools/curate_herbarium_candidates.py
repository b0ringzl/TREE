"""Materialize assistant visual triage; never authorize training or alter originals.

Annotations were made by inspecting all ten dated contact sheets. These coarse
organ tags are not botanical re-identification or human acceptance. The frozen
input digest prevents positional annotations being applied to a changed manifest.
"""
import hashlib
import json
from collections import Counter

from collect_herbarium_candidates import OUT
from inventory_herbarium_images import write_json

EXPECTED_MANIFEST_SHA256 = 'e3fb92ff7eb88430dd4580de838e565c17aeb174b92894540292eace348c0e69'
LEGEND = {
    'T': 'tree_or_crown_context',
    'L': 'foliage_or_leafy_branch',
    'F': 'flower_or_inflorescence',
    'R': 'fruit_seed_or_cone',
    'B': 'bark_trunk_or_roots',
    'M': 'macro_or_staged_detail',
}
# Each line is one inspected contact sheet, in canonical manifest order.
OBSERVATIONS = '''
T RL TB LR L B T RL L RL RM RM RM RM RM R RL TL RL L
L FL RL TF F RL F FL RL T FL RL FRL LF FL L RL T RL RL
RL TRL FL FL FL RL T L LR L T LR FL FL FL F TF F FL FL
FL TF F FL T FL T L TL RL FL FL FL RL L RL RL T B FL
FL B FM RM FM RM RM F FL LF LF FL FL FL RL L T FL RL T
B TFL L R T R RL R FM FL FL FL RM B T L B R L RL
RL FL FL FL FL R L L RL RL RL RL L T FL RL F TL TB FL
FL FL L RL RL TL RL RL TF FL L FL RL F F T RL FRL F F
F L FL RL FL RL T RL FL FL FL T T FL FL RL RL FL RL F
L TF RL L B T LF RL RL L RL FL FL FL R TL T
'''

DUPLICATE_OF = {'7984_114584': '7984_86872'}


def organ_tags():
    tokens = OBSERVATIONS.split()
    if len(tokens) != 197:
        raise ValueError(f'Expected 197 visual annotations, found {len(tokens)}')
    return [sorted({LEGEND[c] for c in token}) for token in tokens]


def candidate_lane(tags, duplicate=False, cross_match=False):
    if duplicate:
        return 'duplicate_reference_only'
    if cross_match:
        return 'cross_dataset_match_quarantine'
    if LEGEND['M'] in tags:
        return 'macro_reference_only'
    if LEGEND['T'] in tags:
        return 'tree_context_candidate'
    if LEGEND['B'] in tags:
        return 'trunk_root_auxiliary_candidate'
    return 'organ_auxiliary_candidate'


def main():
    raw = (OUT / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError('Manifest changed; positional visual annotations require review')
    manifest = json.loads(raw)
    cross = json.loads((OUT / 'cross_dataset_duplicates.json').read_text(encoding='utf-8'))
    matched = {r['id']: r['matches'] for r in cross['candidate_results']}
    if set(matched) != {r['id'] for r in manifest}:
        raise ValueError('Incomplete cross-dataset check')
    annotations, curated = [], []
    for i, (source, tags) in enumerate(zip(manifest, organ_tags()), 1):
        image = OUT / source['file']
        if hashlib.sha256(image.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Original photo checksum mismatch: ' + source['id'])
        duplicate = DUPLICATE_OF.get(source['id'])
        lane = candidate_lane(tags, bool(duplicate), bool(matched[source['id']]))
        annotation = {
            'id': source['id'], 'sha256': source['sha256'], 'ordinal': i,
            'organ_tags': tags, 'reviewer': 'assistant_visual_triage',
            'review_basis': f'contact_sheets/batch_{(i-1)//20+1:02d}.jpg',
            'review_resolution': 'contact_sheet_coarse',
            'species_label_basis': 'official_source_metadata_not_independent_identification',
            'human_accepted': False, 'whole_tree_completeness': 'not_established',
            'individual_tree_identity': 'unknown', 'lidar_pair_available': False,
            'duplicate_of': duplicate, 'candidate_lane': lane,
            'quality_note': ('Dark background is a staged macro view, not automatically an unusable subject.'
                             if source['quality']['heuristic_flags'] == ['dark'] else None),
        }
        if source['id'] in ('7984_86872', '7984_114584'):
            annotation['duplicate_review'] = 'Both full-resolution images inspected; same photograph with encoding/tone differences.'
        annotations.append(annotation)
        curated.append({
            **source, 'organ_review': annotation, 'review_status': 'assistant_triaged_not_training_accepted',
            'candidate_lane': lane, 'duplicate_of': duplicate,
            'content_candidate': lane.endswith('_candidate'),
            'split': 'quarantine', 'training_eligible': False,
            'training_gates': ['intended_use_and_training_rights_confirmation', 'experiment_manifest_acceptance'],
            # Source taxon grouping is deliberately conservative, NOT an individual-tree ID.
            'leakage_group': 'herbarium_taxon_' + source['source_species_id'],
            'split_policy': 'Use only as auxiliary training source after gates; never randomly split this small source for claimed street-scene test accuracy.',
        })
    write_json(OUT / 'organ_annotations.json', annotations)
    write_json(OUT / 'curated_manifest.json', curated)
    counts = Counter(r['candidate_lane'] for r in curated)
    summary = {
        'photos': len(curated), 'source_taxa': len({r['source_species_id'] for r in curated}),
        'assistant_organ_triaged': len(annotations), 'human_accepted': 0,
        'visually_redundant_photos': len(DUPLICATE_OF), 'distinct_photo_candidates': len(curated)-len(DUPLICATE_OF),
        'by_candidate_lane': dict(counts), 'content_candidates': sum(r['content_candidate'] for r in curated),
        'training_eligible': 0, 'cross_indexed_files': cross['indexed_unique_files'],
        'cross_missing_files': len(cross['unreadable_or_missing']),
        'cross_candidate_matches': sum(bool(v) for v in matched.values()),
        'training_started': False, 'models_modified': False, 'lidar_pairs_added': 0,
    }
    write_json(OUT / 'curation_summary.json', summary)
    lines = [
        '# 植物標本室影像採集與清理報告', '',
        '日期：2026-09-07。此批為獨立候選資料，不改寫既有人工驗收、資料集或 Demo 模型。', '',
        '## 結果', '',
        '- 已下載 197 / 197 張公開照片，涉及 28 個來源分類單元；全部可解碼，無下載失敗。',
        '- 197 個不同 SHA-256；另有一對同照片的不同編碼／色調版本，完整尺寸檢視後只保留一份作候選，原檔均保留。',
        f'- 助理已對全部 197 張縮圖逐圖做粗粒度器官分流，不等於使用者驗收或獨立物種鑑定。內容候選共 {summary["content_candidates"]} 張。',
        '- 5 張觸發「偏暗」警告，均為黑底微距／擺拍細節；不能僅用全圖亮度認定主體不可用。',
        '- 狐尾椰子本次公開檢索仍無匹配，不能宣稱八個重點分組全部得到新圖。', '',
        '| 分流 | 張數 | 用途 |', '|---|---:|---|',
    ]
    purposes = {
        'tree_context_candidate': ('樹形／樹冠環境候選', '較接近街景，但不保證根部、樹幹或整樹完整'),
        'trunk_root_auxiliary_candidate': ('樹幹／根部輔助候選', '補充樹皮及根幹紋理；不能補出點雲'),
        'organ_auxiliary_candidate': ('花果葉器官輔助候選', '與樹形樣本分層使用，控制來源及器官採樣權重'),
        'macro_reference_only': ('微距／擺拍參考', '暫不混入街景主訓練集，另作器官參考'),
        'duplicate_reference_only': ('重複參考', '不重複入訓；兩份原檔保留以供追溯'),
        'cross_dataset_match_quarantine': ('跨資料集疑似重複隔離', '複核前不入訓'),
    }
    for lane, (label, purpose) in purposes.items():
        lines.append(f'| {label} | {counts[lane]} | {purpose} |')
    lines += ['', '## 去重範圍與限制', '',
        '- 現有整圖基線 33,717、裁剪基線 2,946、五類試驗 45、三類雙源試驗 236，共 36,944 個圖像檔案，全部成功讀取。',
        '- 新圖逐一進行 SHA-256 及 64-bit pHash（漢明距離 ≤ 6）比對，未發現匹配。未掃描上述清單外的專案資料，亦不能排除重裁剪、翻轉或大幅改色的近重複。',
        '- 來源未提供可靠的每株樹識別碼，197 張照片並非 197 棵獨立樹。按來源分類單元保守分組，避免在本批內隨機拆分後聲稱街景測試準確率。', '',
        '## 標籤與後續試驗', '',
        '- 原始學名、來源頁、圖片 ID、水印與 SHA-256 完整保留；來源種級標籤與現有模型合併標籤分開。銀合歡與大葉合歡不合併。',
        '- 整圖與器官分流是助理視覺初篩，樹形照片不作整樹完整性／樹高真值。沒有增加 LiDAR 配對或訓練點雲。',
        '- 目前 training_eligible 全為 false；尚未啟動訓練，尚無精度提升結論。正式入訓前確認用途與訓練授權範圍，再接受實驗清單。',
        '- 建議做候選版本對照：A 為既有資料，B 為加入樹形與根幹，C 再加入低權重器官特寫；保持舊資料回放、固定超參數與驗證集。先在驗證集選版本，再對鎖定且未參與選圖的街景測試集做一次分類 precision / recall / F1 及混淆矩陣對比。',
        '- 不使用本批小量器官照片的訓練命中率作成效；不因候選版本較新而直接替換正式 Demo。', '',
        '## 來源與用途', '',
        '來源：Hong Kong Herbarium, Agriculture, Fisheries and Conservation Department（香港植物標本室，漁農自然護理署）。圖片保持公開水印版本及原始位元組。', '',
        '[官方重要告示](https://www.herbarium.gov.hk/en/important-notices/index.html)列有非商業複製署名要求及商業用途書面許可要求，但未明確涵蓋機器學習訓練；此清單不是授權書。未將圖片發布到 GitHub。', '',
        '## 檢視資料', '',
        '- [逐圖清單](review_index.md)',
        '- [清理後機器可讀清單](curated_manifest.json)',
        '- [跨資料集檢查](cross_dataset_duplicates.json)', '',
    ]
    lines += ['## 各來源分類單元', '', '| 來源學名 | 照片 | 內容候選 |', '|---|---:|---:|']
    for name in dict.fromkeys(r['source_species'] for r in curated):
        rows = [r for r in curated if r['source_species'] == name]
        lines.append(f'| {name} | {len(rows)} | {sum(r["content_candidate"] for r in rows)} |')
    (OUT / 'collection_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    index = ['# 逐圖初篩清單', '', '僅助理初篩，尚未入訓。點擊編號檢視原圖；原始來源物種名稱未改写。', '',
             '| 縮圖批次 |', '|---|']
    index += [f'| [Batch {n:02d}](contact_sheets/batch_{n:02d}.jpg) |' for n in range(1, 11)]
    index += ['', '| 編號 | 來源學名 | 器官／場景標記 | 分流 |', '|---|---|---|---|']
    for r in curated:
        a = r['organ_review']
        index.append(f'| [{a["ordinal"]:03d}]({r["file"]}) | {r["source_species"]} | {", ".join(a["organ_tags"])} | {r["candidate_lane"]} |')
    (OUT / 'review_index.md').write_text('\n'.join(index)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
