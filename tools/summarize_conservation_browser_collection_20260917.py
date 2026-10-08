"""Join browser provenance, historical deduplication and actual visual inspection.

Creates a new report, not a training manifest or human-review decision.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / 'outputs/conservation_browser_visual_qa_20260917_v1'
SPECIES = {'Artocarpus hypargyreus': '白桂木', 'Rhodoleia championii': '紅花荷', 'Keteleeria fortunei': '油杉'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(data, out):
    data, out = data.resolve(), out.resolve()
    raw = read(data/'manifest.json')
    qa = {}
    for row in read(QA/'white/visual_qa.json'):
        qa[row['image_url']] = dict(row, decision=row['status'])
    for row in read(QA/'oil/oil_visual_qa.json')['images']:
        qa[row['image_url']] = dict(row, decision=row['review_status'])
    red_index = {r['number']: r for r in read(QA/'red/image_index.json')}
    for row in read(QA/'red/visual_decisions_by_number.json')['items']:
        qa[red_index[row['number']]['url']] = row
    assert len(raw) == 66 and len(qa) == 66, 'All 66 browser files require actual visual review'
    out.mkdir(parents=True, exist_ok=False)
    joined = []
    for row in raw:
        review = qa[row['image_url']]
        if row['status'] != 'imported_pending':
            lane = 'excluded_duplicate_or_invalid'
        elif review['decision'] == 'exclude_visual_mismatch':
            lane = 'excluded_visual_mismatch'
        elif review['decision'] == 'agent_source_consistent':
            lane = 'source_consistent_candidate'
        else:
            lane = 'requires_targeted_corroboration'
        enriched = dict(row, visual_qa=review, final_lane=lane, training_eligible=False,
                        human_acceptance=False, individual_identity_verified=False)
        if row['status'] == 'imported_pending':
            file = Path(row['source_image'])
            assert file.is_file() and sha(file) == row['sha256'], file
            enriched['original_byte_copy_reverified'] = True
        joined.append(enriched)
    save(out/'triage_manifest.json', joined)
    candidates = [r for r in joined if r['final_lane']=='source_consistent_candidate']
    uncertain = [r for r in joined if r['final_lane']=='requires_targeted_corroboration']
    excluded = [r for r in joined if r['final_lane'].startswith('excluded_')]
    save(out/'source_consistent_candidates.json', candidates)
    save(out/'uncertain_candidates.json', uncertain)
    save(out/'excluded_manifest.json', excluded)
    counts = {}
    for name in SPECIES:
        selected = [r for r in joined if r['source_species']==name]
        counts[name] = dict(downloaded=len(selected), new_after_dedup=sum(r['status']=='imported_pending' for r in selected), **Counter(r['final_lane'] for r in selected))
    # Compare human DB against its read-only history-scan hash; never mutate it.
    audit = read(data/'prior_index_audit.json')
    db_name = ROOT/'conservation_expansion_review/data/expansion_20260917_v1/human_review.sqlite3'
    entries = [r for r in audit['files'] if Path(r['path']).resolve()==db_name.resolve()]
    assert len(entries)==1, 'Required historical human DB missing from dedup audit'
    unchanged = entries[0]['sha256'].lower()==sha(db_name).lower()
    assert unchanged, 'Human-review DB changed during collection; reconcile before summary'
    summary = {'downloaded':len(joined), 'by_species':counts, 'new_after_dedup':sum(r['status']=='imported_pending' for r in joined),
               'source_consistent_candidates':len(candidates), 'uncertain_new':len(uncertain),
               'excluded_counts':dict(Counter(r['final_lane'] for r in excluded)),
               'human_review_db_hash_unchanged_since_index':unchanged, 'human_review_db_sha256':sha(db_name),
               'training_performed':False, 'training_eligible':0, 'demo_modified':False,
               'individual_tree_count':'unknown', 'source_terms_clearance':'Per-source records; not a blanket commercial/ML licence grant.'}
    save(out/'summary.json', summary)
    # Local research QA sheets preserve full-frame source imagery and watermarks.
    for name in SPECIES:
        group = [r for r in candidates if r['source_species']==name]
        for start in range(0, len(group), 9):
            sheet = Image.new('RGB', (1440,1320), 'white')
            draw = ImageDraw.Draw(sheet)
            for offset, row in enumerate(group[start:start+9]):
                with Image.open(row['source_image']) as image:
                    preview = ImageOps.exif_transpose(image).convert('RGB')
                    preview.thumbnail((460,370))
                x,y = (offset%3)*480,(offset//3)*440
                sheet.paste(preview,(x+(460-preview.width)//2,y))
                draw.text((x+6,y+375),name,fill='black')
                draw.text((x+6,y+393),row['id'],fill='black')
                draw.text((x+6,y+411),'Source-consistent candidate / NOT training-approved',fill='black')
            sheet.save(out/f"{name.replace(' ','_')}_candidates_{start//9+1}.jpg",quality=93)
    lines = ['# 浏览器采集：保育树种图片（2026-09-17）','',
             '本轮通过浏览器打开真实物种页面和图库，下载来源页面图片，不使用搜索引擎缩略图，不把截图或增强版本算新增。以下为图片数，不是独立植株数。','',
             '| 目标树种 | 保存 | 去重后新增 | 来源与视觉一致候选 | 需补证 | 明显不相关 |',
             '|---|---:|---:|---:|---:|---:|']
    for name,zh in SPECIES.items():
        c=counts[name]
        lines.append(f"| {name} / {zh} | {c['downloaded']} | {c['new_after_dedup']} | {c.get('source_consistent_candidate',0)} | {c.get('requires_targeted_corroboration',0)} | {c.get('excluded_visual_mismatch',0)} |")
    lines += ['',f"下载合计 **{len(joined)} 张**；与历史图库及本批去重后 **{summary['new_after_dedup']} 张**；其中 **{len(candidates)} 张**视觉/来源一致候选、**{len(uncertain)} 张**需补证。并未把候选自动当作训练合格图。",'',
              '## 发现与处理','',
              '- Pukeiti 图库的 9376.jpg 是地被和草坪，并非可见紅花荷目标，已排除。保存原始下载记录用于审计，不进入候选清单。',
              '- 油杉只有 1 张近全株图，冠顶触边；大部分为枝叶、雄球花及球果。不能据此夸大整株街景识别能力。',
              '- 白桂木包含整冠和整株图，但部分树皮、幼态裂叶、林下混杂图缺少足够鉴别信息。紅花荷也有幼苗/树皮待补证。',
              '- 同摄影系列、同枝条和同机构相关图片不能拆入训练与测试两侧；独立树身份尚未建立。',
              '- 所有 66 张均有逐图视觉检查；“来源一致”仅表示未见明显冲突，不是假冒专家鉴定或人工确认。',
              '- 下载原件保留水印、原始字节、来源 URL、署名、尺寸、SHA-256 与去重依据。','',
              '去重范围：读取历史清单、已有 SHA/pHash/URL 与人工拒绝记录；缺失的视觉哈希仅为本轮目标树种及相关拒绝图片补算。不声称所有无关旧图均重新完成视觉去重；大量裁剪、镜像和未记录的同株关系仍可能漏检。中断的 v1/v2 保留运行记录，最终使用 v3。','',
              '## 来源与权利记录','',
              '- [香港植物标本室图库](https://www.herbarium.gov.hk/en/hk-plant-database/index.html)：三种均有专属页面；[条款](https://www.herbarium.gov.hk/en/important-notices/index.html)要求非商业使用时署名。',
              '- [Greening 白桂木](https://www.greening.gov.hk/en/community-outreach/qrcode-tree-labels/index_id_184.html)、[紅花荷](https://www.greening.gov.hk/en/community-outreach/qrcode-tree-labels/index_id_1193.html)：本地个人/非商业内部用途条件，版权声明随目录保存；不公开再发布。',
              '- [IISc 白桂木图库](https://indiaflora-ces.iisc.ac.in/herbsheet.php?cat=13&id=1045)：K. Sankara Rao，CC BY-NC-SA 4.0，保留署名与水印。',
              '- [Trees and Shrubs Online 油杉](https://www.treesandshrubsonline.org/articles/keteleeria/keteleeria-fortunei/)：John Anderson；[网站许可](https://www.treesandshrubsonline.org/about/licence/)为 CC BY-SA 4.0。',
              '- [Pukeiti 紅花荷](https://pukeiti.gardenexplorer.org/taxon-148030.aspx)：未核实使用许可，隔离保存，不能自动进入训练或公开展示。','',
              '## 文件与状态','',
              f'- [66 张下载原件存档（含重复及排除图，仅审计）](<{(ROOT/"outputs/conservation_browser_collection_20260917_v1/raw_browser_downloads").as_posix()}>).',
              f'- [原图与完整来源目录](<{data.as_posix()}>).',
              f'- [去重＋视觉核验总清单](<{(out/"triage_manifest.json").as_posix()}>).',
              f'- [来源一致候选](<{(out/"source_consistent_candidates.json").as_posix()}>).',
              f'- [疑难补证清单](<{(out/"uncertain_candidates.json").as_posix()}>).',
              f'- [排除记录](<{(out/"excluded_manifest.json").as_posix()}>).',
              '- 原 8052 审核数据库与既有拒绝决定未覆盖；当前目录是新采集批次，不表示旧页面已切换。',
              '- 未训练、未切换模型、未改 Demo。尚无新增精度结论；当前工作是收集、去重与质检。',
              '- 后续如训练，先落实允许用途、样本与同源分组、固定验证集，并按照项目要求新建权重备份；不以本批图库内部随机切分冒充街景泛化验证。','']
    with (out/'REPORT.md').open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    main(args.data,args.out)
