"""Verify fresh collection provenance and write truthful pending-review summary."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main(out):
    out=out.resolve()
    read=lambda name:json.loads((out/name).read_text(encoding='utf8'))
    save=lambda name,obj:(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    rows=read('manifest.json'); clean=read('review_manifest.json'); totals=read('totals.json'); idx=read('prior_index_audit.json')
    assert len(clean)==totals['pending_clean']
    assert all(r['training_eligible'] is False and r['split']=='quarantine' and r['human_review_status']=='pending' for r in rows)
    assert all(not r['quality_flags'] for r in clean)
    assert len({r['id'] for r in rows})==len(rows)
    assert len({r['sha256'] for r in clean})==len(clean)
    original_hashes=[]
    for r in rows:
        if r['status']!='downloaded': continue
        path=Path(r['source_image']).resolve()
        assert (out/'images') in path.parents
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==r['sha256'],path
        original_hashes.append({'id':r['id'],'source_image':str(path),'sha256':actual,'bytes':path.stat().st_size})
    old_verified=[]
    for prior in idx['manifests']:
        path=Path(prior['path']); assert hashlib.sha256(path.read_bytes()).hexdigest()==prior['sha256'],path
        old_verified.append(str(path))
    grouping=[]
    for species in sorted({r['source_species'] for r in clean}):
        selected=[r for r in clean if r['source_species']==species]
        counts=Counter(r['observation_id'] for r in selected)
        grouping.append({'species':species,'photos':len(selected),'observations':len(counts),
            'observers':len({r['observer_id'] for r in selected}),
            'multiview_observations':sum(n>1 for n in counts.values()),
            'public_location_grid_count':len({r['location_grid_0p01deg'] for r in selected if r['location_grid_0p01deg']}),
            'warning':'Observations/photographers/grids are grouping proxies, not verified independent trees or complete sites.',
            'observation_groups':[{'observation_id':oid,'ids':[r['id'] for r in selected if r['observation_id']==oid]} for oid in counts]})
    save('grouping_summary.json',grouping)
    save('VERIFIED.json',{'source_files_checked':original_hashes,'prior_manifests_unchanged':old_verified,
         'rows':len(rows),'pending_clean':len(clean),'training_eligible':0,'human_accepted':0,
         'training_or_deployment_performed':False,'source_label_not_prediction':True,
         'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
         'review_manifest_sha256':hashlib.sha256((out/'review_manifest.json').read_bytes()).hexdigest()})
    aq=next(x for x in grouping if x['species']=='Aquilaria sinensis')
    strong=sum(r['source_species']=='Aquilaria sinensis' and r['source_evidence']['tier']=='multiple_independent_exact_ids' for r in clean)
    lines=['# 保育专项网络补充数据 — 待人工复核','','日期：2026-09-17。未训练、未部署；所有新图保持 pending。',
        '',f"本批共下载 {totals['downloaded']} 张，去重与基础质量门后保留 {len(clean)} 张供复核；土沉香来源候选 {aq['photos']} 张，非土沉香来源候选 {totals['negative_pending']} 张。",
        '',f"土沉香共有 {aq['observations']} 个观察记录、{aq['observers']} 位拍摄者；其中 {aq['multiview_observations']} 个观察各有两张图。**这些不是已确认的独立树数量。**",
        '',f"新增土沉香达到‘至少两位当前精确鉴定、其中至少一位非拍摄者、无当前种级冲突’的数量为 **{strong}**。既有强证据来源大多已在旧项目使用；本批土沉香均需用户复核，不能作为已确认正样本直接训练。",
        '', '## 来源及去重', '',
        '- 使用 iNaturalist 公开 API，串行访问并限速；仅下载 CC0 / CC BY / CC BY-SA 图像，逐图保留作者、源观察页面、照片编号、许可及原响应。下载的是服务提供的 large 图像，字节原样保留，不声称为相机原始文件。',
        '- 排除既有所有 split 的观察/照片 ID，以及 external_datasets 的全部 manifest（包括 pending/rejected 和用户 84 张图）；SHA-256 与感知哈希距离 ≤ 6 近重复隔离。',
        '- 基础质量门：最短边至少 224、暗像素占比不超过 65%。通过这些检查不等于树种正确，也不等于整株构图。',
        '- 负例类别来自既有 training-only hard_negative_manifest 的易混淆类；本轮没有从测试错误挑选训练图或使用模型分数生成标签。负例也全为 pending。',
        '- 未改动任何旧人审决定、旧图、权重或 Demo；prior_index_audit.json 与 VERIFIED.json 记录已核对的旧 manifest 哈希。',
        '', '## 待复核数量', '', '| 来源树种 | 图像 | 多独立鉴定图 | 仅来源标签图 |','|---|---:|---:|---:|']
    for row in totals['by_species']:
        lines.append(f"| {row['species']} | {row['pending_clean']} | {row['strong_evidence_pending']} | {row['weak_source_pending']} |")
    lines += ['', '## 用户复核要求', '',
        '选择“土沉香 / 非土沉香 / 无法判断”。建议先复核土沉香来源候选，再检查外观相近负例。看不清、混有多种植物、只有无诊断力局部的图保留为无法判断；不要为了凑数量接受。',
        '同观察两张图可互相补充，但用户还需确认确属同一株；未确认前不能当作可靠多视角组合。',
        '通过复核后，再按观察、作者和可用地点联合分组并锁定划分。与旧库作者重叠及未知原树身份的情况必须保留限制说明；本批尚未分配 train/validation/test。',
        '', '## 文件', '',
        '- `review_manifest.json`：仅质量/重复门合格的待审记录，供界面使用。',
        '- `manifest.json`：完整下载审计，含隔离项。',
        '- `source_responses/`：原始 API 元数据。',
        '- `contact_sheets/`：待审接触表，不是人工验收证据。',
        '- `grouping_summary.json`：观察级多图组、作者和公开位置分组统计。',
        '- `EXTERNAL_POINT_DATA_FINDINGS.md`：外部点云检索线索与不适用原因。',
        '- `VERIFIED.json`：新图字节哈希、旧 manifest 未改动校验。',
        '', '## 鉴定参考与边界', '',
        '以[香港植物标本室土沉香资料](https://www.herbarium.gov.hk/tc/publications/books/book2/text/aquilaria-sinensis/index.html)作形态核验参考；其网页照片本轮未采集。',
        'iNaturalist 社区标签和研究级状态不是植物专家鉴定的保证。照片许可仅按源记录保留；本批不上传到公开 Demo 或发布实验权重。',
        '检索未确认可立即投入独立全树图像＋LiDAR验证的公开土沉香配对数据；不能凭已下载图像扩大点云测试数量。']
    if totals['errors']:
        lines+=['','## 采集错误','',json.dumps(totals['errors'],ensure_ascii=False)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'pending_clean':len(clean),'positive':aq,'negative':totals['negative_pending'],'verified_downloads':len(original_hashes),'old_manifests_unchanged':len(old_verified)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
