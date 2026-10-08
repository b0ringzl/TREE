"""Build a <=1.5 GB image-only selection from a verified, read-only snapshot.

The set is a storage/curation selection, not a valid estimate of classifier accuracy.
Wrong-but-clear images are not relabeled, and a fraction of hard examples remains.
"""

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/urban27_image_quality_20260928_v1'
BACKUP = RUN / 'pre_curation_backup'
LEGACY_PROVENANCE = ROOT / 'outputs/Urban27_Image_Package_Backup_20260928_v1/來源與審核狀態.csv'
OUTPUT = ROOT / 'deliverables/香港市區常見27樹種_現有圖片精簡版_20260928_v1'
LIMIT_BYTES = 1_500_000_000
IMAGE_BUDGET = 1_470_000_000


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path, records):
    rows = list(records)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_key(row):
    return row['relative_path'].split('\\', 1)[0]


def audit_rows():
    verification = json.loads((RUN / 'pre_curation_backup_verification.json').read_text(encoding='utf-8'))
    if not verification['sha256_verified'] or Path(verification['backup']) != BACKUP:
        raise RuntimeError('Run-specific source backup is not verified')
    predictions = read_jsonl(RUN / 'predictions.jsonl')
    pixels = read_jsonl(RUN / 'pixel_quality.jsonl')
    if len(predictions) != verification['files'] or len(pixels) != verification['files']:
        raise RuntimeError(f'Incomplete per-image audit: model={len(predictions)}, pixels={len(pixels)}, expected={verification["files"]}')
    pred = {row['relative_path']: row for row in predictions}
    pixel = {row['relative_path']: row for row in pixels}
    if len(pred) != len(predictions) or len(pixel) != len(pixels) or set(pred) != set(pixel):
        raise RuntimeError('Duplicate or mismatched image identities')
    provenance = defaultdict(list)
    with LEGACY_PROVENANCE.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            provenance[row['sha256'].lower()].append(row)
    rows = []
    for relative, p in pred.items():
        q = pixel[relative]
        path = BACKUP / relative
        if (not path.is_file() or path.stat().st_size != p['bytes'] or
                p['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise RuntimeError(f'Input changed after verification: {relative}')
        if q['status'] != 'ok':
            quality_flags = ['decode_error']
        else:
            quality_flags = []
            if min(q['width'], q['height']) < 224:
                quality_flags.append('small_native_image')
            if q['dark_fraction'] > .40 or q['bright_fraction'] > .40:
                quality_flags.append('extreme_exposure')
            if q['entropy_bits'] < 5.0:
                quality_flags.append('low_tonal_detail')
            if q['laplacian_variance'] < 60:
                quality_flags.append('little_edge_detail')
        stem = Path(relative).stem
        source_set = stem.split('_', 1)[1] if '_' in stem else 'unclassified'
        original_rows = provenance[p['sha256'].lower()]
        page = next((r['source_page'] for r in original_rows if r['source_page']), '')
        license_name = next((r['license'] for r in original_rows if r['license']), '')
        attribution = next((r['attribution'] for r in original_rows if r['attribution']), '')
        original_source = next((r['source_file'] for r in original_rows if r['source_file']), '')
        review = next((r['source_status'] for r in original_rows if r['source_status']), '')
        user_confirmed = source_set.startswith('user_confirmed') or source_set == 'user_identified'
        record = {
            'relative_path': relative,
            'species_folder': group_key(p),
            'source_taxon': p['source_taxon'],
            'source_set': source_set,
            'source_page': page,
            'source_status': review,
            'license': license_name or ('user-provided; redistribution not assessed' if user_confirmed else ''),
            'attribution': attribution,
            'original_source_file': original_source,
            'bytes': p['bytes'],
            'sha256': p['sha256'],
            'evaluation_lane': p['evaluation_lane'],
            'expected_model_label': p['expected_model_label'],
            'prediction_status': p['status'],
            'top1': p.get('top1', ''),
            'top1_score': p.get('top1_score'),
            'expected_score': p.get('expected_score'),
            'expected_rank': p.get('expected_rank'),
            'source_label_agreement': p.get('source_label_agreement'),
            'view_agreement': p.get('view_agreement'),
            'width': q.get('width'),
            'height': q.get('height'),
            'dhash64': q.get('dhash64'),
            'quality_flags': quality_flags,
            'protected_user_image': user_confirmed,
            'keep': True,
            'reason': 'retained',
        }
        rows.append(record)
    return rows


def priority(row, dhash_count, page_count):
    """Higher value means lower marginal value per byte, not an invalid label."""
    badness = 0.0
    if row['prediction_status'] != 'ok':
        badness += 25
    badness += 12 * len(row['quality_flags'])
    if row['source_status'] in {'rejected', 'quarantine'}:
        badness += 6
    elif row['source_status'] in {'pending', 'assistant_triaged_not_training_accepted'}:
        badness += 2
    if row['evaluation_lane'] != 'unsupported' and row['source_label_agreement'] is False:
        badness += 3
        if row['expected_rank'] and row['expected_rank'] > 5:
            badness += 3
        if row['expected_score'] is not None and row['expected_score'] < .14:
            badness += 2
    if row['view_agreement'] is False:
        badness += 1
    if row['dhash64'] and dhash_count[(row['species_folder'], row['dhash64'])] > 1:
        badness += 2
    if row['source_page'] and page_count[(row['species_folder'], row['source_page'])] > 2:
        badness += 1
    # Large files are preferred for removal only after signal/quality flags.
    badness += 2.5 * math.sqrt(row['bytes'] / 1_000_000)
    return badness


def choose(rows):
    counts = Counter(r['species_folder'] for r in rows)
    kept_counts = counts.copy()
    hard_counts = Counter(r['species_folder'] for r in rows if r['source_label_agreement'] is False)
    kept_hard = hard_counts.copy()
    dhash_count = Counter((r['species_folder'], r['dhash64']) for r in rows if r['dhash64'])
    page_count = Counter((r['species_folder'], r['source_page']) for r in rows if r['source_page'])
    for row in rows:
        row['removal_priority'] = priority(row, dhash_count, page_count)
    total = sum(r['bytes'] for r in rows)
    if total <= IMAGE_BUDGET:
        return
    order = sorted(rows, key=lambda r: (-r['removal_priority'], -r['bytes'], r['relative_path']))
    for row in order:
        if total <= IMAGE_BUDGET:
            break
        species = row['species_folder']
        # Retain scarce and user-confirmed observations regardless of the model result.
        if counts[species] <= 50 or row['protected_user_image']:
            continue
        if kept_counts[species] <= math.ceil(.60 * counts[species]):
            continue
        # Preserve at least one quarter of legible wrong-source-label examples
        # for honest regression inspection and future human review.
        if row['source_label_agreement'] is False and not row['quality_flags']:
            floor = max(5, math.ceil(.25 * hard_counts[species]))
            if kept_hard[species] <= floor:
                continue
        row['keep'] = False
        if row['prediction_status'] != 'ok':
            row['reason'] = 'inference_or_decode_failure'
        elif row['quality_flags']:
            row['reason'] = 'image_quality_flag'
        elif row['source_status'] in {'rejected', 'quarantine'}:
            row['reason'] = 'source_review_status'
        elif row['source_label_agreement'] is False and (row['expected_rank'] or 0) > 5:
            row['reason'] = 'source_label_model_disagreement_needs_review'
        elif row['dhash64'] and dhash_count[(species, row['dhash64'])] > 1:
            row['reason'] = 'visually_redundant_or_large'
        else:
            row['reason'] = 'size_budget_lower_marginal_value'
        total -= row['bytes']
        kept_counts[species] -= 1
        if row['source_label_agreement'] is False:
            kept_hard[species] -= 1
    if total > IMAGE_BUDGET:
        raise RuntimeError(f'Cannot reach byte budget while preserving class and hard-example floors: {total}')


def summary(rows):
    result = []
    for folder in sorted({r['species_folder'] for r in rows}):
        all_rows = [r for r in rows if r['species_folder'] == folder]
        kept = [r for r in all_rows if r['keep']]
        lane = all_rows[0]['evaluation_lane']
        before_ok = sum(r['source_label_agreement'] is True for r in all_rows)
        after_ok = sum(r['source_label_agreement'] is True for r in kept)
        result.append({
            'species_folder': folder,
            'scientific_name': all_rows[0]['source_taxon'],
            'evaluation_lane': lane,
            'before_images': len(all_rows),
            'after_images': len(kept),
            'removed_images': len(all_rows) - len(kept),
            'before_bytes': sum(r['bytes'] for r in all_rows),
            'after_bytes': sum(r['bytes'] for r in kept),
            'before_source_label_top1_matches': before_ok if lane != 'unsupported' else '',
            'after_source_label_top1_matches': after_ok if lane != 'unsupported' else '',
            'before_source_label_match_fraction': round(before_ok / len(all_rows), 4) if lane != 'unsupported' else '',
            'after_source_label_match_fraction': round(after_ok / len(kept), 4) if lane != 'unsupported' else '',
        })
    result.append({
        'species_folder': '27_二喬木蘭', 'scientific_name': 'Yulania x soulangeana',
        'evaluation_lane': 'unsupported', 'before_images': 0, 'after_images': 0,
        'removed_images': 0, 'before_bytes': 0, 'after_bytes': 0,
        'before_source_label_top1_matches': '', 'after_source_label_top1_matches': '',
        'before_source_label_match_fraction': '', 'after_source_label_match_fraction': '',
    })
    return sorted(result, key=lambda r: r['species_folder'])


def main():
    if OUTPUT.exists():
        raise RuntimeError(f'Refusing to overwrite existing delivery: {OUTPUT}')
    rows = audit_rows()
    choose(rows)
    stats = summary(rows)
    write_csv(RUN / '逐张测试与筛选结果.csv', ({
        **{key: value for key, value in row.items() if key != 'quality_flags'},
        'quality_flags': '|'.join(row['quality_flags'])
    } for row in sorted(rows, key=lambda r: r['relative_path'])))
    write_csv(RUN / '每树种筛选统计.csv', stats)
    write_csv(RUN / '移出清单.csv', ({
        'relative_path': row['relative_path'], 'species_folder': row['species_folder'],
        'bytes': row['bytes'], 'sha256': row['sha256'], 'reason': row['reason'],
        'backed_up_at': str(BACKUP / row['relative_path']),
    } for row in rows if not row['keep']))

    OUTPUT.mkdir(parents=True)
    for folder in sorted({r['species_folder'] for r in rows} | {'27_二喬木蘭'}):
        (OUTPUT / folder).mkdir()
    for index, row in enumerate((r for r in rows if r['keep']), 1):
        source = BACKUP / row['relative_path']
        target = OUTPUT / row['relative_path']
        shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError(f'Copied image SHA-256 mismatch: {target}')
        if index % 500 == 0:
            print(f'Copied and verified {index} images', flush=True)
    write_csv(OUTPUT / '每树种统计.csv', stats)
    write_csv(OUTPUT / '图片来源索引.csv', ({
        'relative_path': r['relative_path'], 'sha256': r['sha256'],
        'source_set': r['source_set'], 'source_status': r['source_status'],
        'source_page': r['source_page'], 'license': r['license'],
        'attribution': r['attribution'], 'original_source_file': r['original_source_file'],
    } for r in rows if r['keep']))
    note = (
        '香港市區常見 27 樹種：圖片精簡版\n'
        '圖片由目前資料夾的已驗證快照逐張測試、篩選而來；原圖沒有重新編碼。\n'
        '目標是控制容量和整理展示/研究用圖片，不是獨立的模型準確率測試。\n'
        '原資料包含訓練、參考和待審來源；模型與來源標籤一致不等於專家鑑定。\n'
        '柱狀南洋杉、垂葉榕、細葉榕只能核對合併模型類別，不能聲稱種級識別。\n'
        '樂昌含笑、紅花風鈴木和二喬木蘭目前無獨立輸出類別；其圖片不計入匹配率。\n'
        '移出的圖片以及完整逐張記錄、原始來源均在項目的驗證備份中，可恢復。\n'
        '圖片授權及出處見「圖片來源索引.csv」；勿直接公開分發待審或版權不明的圖片。\n'
    )
    (OUTPUT / '说明.txt').write_text(note, encoding='utf-8')
    final_files = [p for p in OUTPUT.rglob('*') if p.is_file()]
    final_bytes = sum(p.stat().st_size for p in final_files)
    if final_bytes > LIMIT_BYTES:
        raise RuntimeError(f'Final folder exceeds 1.5 GB: {final_bytes}')
    metrics = {
        'original_snapshot': str(BACKUP),
        'curated_folder': str(OUTPUT),
        'image_count_before': len(rows),
        'image_count_after': sum(r['keep'] for r in rows),
        'folder_bytes_after': final_bytes,
        'folder_size_limit_bytes': LIMIT_BYTES,
        'removed_reasons': dict(Counter(r['reason'] for r in rows if not r['keep'])),
        'source_label_agreement_is_not_independent_accuracy': True,
        'training_performed': False,
        'deployed_weights_changed': False,
    }
    (RUN / 'selection_summary.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    report = [
        '# 香港市區常見樹種圖片逐張回放與容量整理',
        '',
        f'- 原快照：{len(rows)} 張，{sum(r["bytes"] for r in rows):,} bytes。',
        f'- 精簡副本：{metrics["image_count_after"]} 張，整個文件夾 {final_bytes:,} bytes（上限 {LIMIT_BYTES:,} bytes）。',
        '- 正式網頁 H120 whole+center 推理：逐張原圖，以 Photo 模式且已選定目標；不含自動找樹、全景投影、SAM 或多圖融合。',
        '- 來源混有訓練圖、參考圖、待審圖。下表是「來源標籤與模型首選是否一致」，**不是**獨立測試準確率；精簡後數字又受篩選影響，不能作性能宣稱。',
        '- 模型認錯並不證明來源圖片錯；移出檔案可從已校驗備份恢復，亦保留了部分高品質難例。',
        '',
        '| 樹種 | 模型輸出層級 | 原圖 | 保留 | 移出 | 原始標籤匹配 | 精簡後匹配 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for stat in stats:
        before = (f'{stat["before_source_label_top1_matches"]}/{stat["before_images"]}'
                  if stat['evaluation_lane'] != 'unsupported' else '不能種級計算')
        after = (f'{stat["after_source_label_top1_matches"]}/{stat["after_images"]}'
                 if stat['evaluation_lane'] != 'unsupported' else '不能種級計算')
        report.append(f'| {stat["species_folder"]} | {stat["evaluation_lane"]} | {stat["before_images"]} | {stat["after_images"]} | {stat["removed_images"]} | {before} | {after} |')
    report += [
        '',
        'merged = 只能評估合併類（南洋杉或榕樹），不能宣稱分清各樹種；unsupported = 尚無獨立輸出類別。',
        '',
        f'完整逐張結果：`{RUN / "逐张测试与筛选结果.csv"}`',
        f'移出清單：`{RUN / "移出清单.csv"}`',
        f'可恢復備份：`{BACKUP}`',
        '沒有重新訓練或修改部署模型。',
    ]
    (RUN / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
