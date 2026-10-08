"""Reproducible coverage report, conservative status counts and safety checks."""
import sys,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT))
from hk_fullscene_extract_20260912 import OUT,read,write
from hk_fullscene_refine_20260912 import DEST

def main():
    sources=read(OUT/'sources.json');summary=read(DEST/'summary.json');v1=read(OUT/'summary.json');rows=read(DEST/'candidates.json');evidence=[read(p) for p in (DEST/'image_evidence').glob('*/record.json')];dense=[r for p in (DEST/'dense_recovery').glob('*/manifest.json') for r in read(p)];scans=[read(OUT/'maps'/s['id']/'coarse.json') for s in sources]
    counts=Counter(r['status'] for r in evidence);regions={}
    for s in sources:
        region='近期採集' if s['id'].startswith('recent') else '何文田' if s['id'].startswith('hewentian') else '尖沙咀' if s['id'].startswith('jianshazui') else '司徒拔道'
        d=regions.setdefault(region,dict(maps=0,points=0,files=0));d['maps']+=1;d['points']+=s['declared_points'];d['files']+=next(c['candidates'] for c in summary['coverage'] if c['id']==s['id'])
    checks=dict(all_28_scans_complete=len(scans)==28,scanned_counts_match=all(r['scanned_points']==r['source']['declared_points'] for r in scans),original_metadata_unchanged=all(Path(s['coarse_source']).stat().st_size==s['source_bytes'] and Path(s['coarse_source']).stat().st_mtime_ns==s['mtime_ns'] for s in sources),unique_candidate_ids=len({r['id'] for r in rows})==len(rows),no_candidate_assigned_training=all(r['training_eligible'] is False and r['species'] is None for r in rows),all_exports_exist=all(Path(r[k]).is_file() for r in rows for k in ['point','cloud','views']))
    sampled=[]
    for s in sources:
        rs=[r for r in rows if r['map_id']==s['id']]
        for r in rs[::max(1,len(rs)//3)][:3]:
            with np.load(r['point'],allow_pickle=False) as f:q=f['points_xyz'];valid=q.ndim==2 and q.shape[1]==3 and len(q)<=2048 and np.isfinite(q).all()
            if not valid:raise ValueError('Invalid point sample '+r['id'])
            sampled.append(r['id'])
    from spacescan_trees_demo_v1.tree_detection import WEIGHT,CLIP_WEIGHT
    old=read(DEST/'image_evidence/method.json')['weights'];hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [WEIGHT,CLIP_WEIGHT]};checks['inference_weights_sha256_unchanged']=hashes==old
    checks['finite_sample_exports_checked']=len(sampled);checks['fit_or_training_performed']=False;write(DEST/'verification.json',checks|{'inference_weights':hashes,'sampled_export_ids':sampled})
    if not all(v for k,v in checks.items() if isinstance(v,bool) and k!='fit_or_training_performed'):raise ValueError('Verification failed')
    write(DEST/'image_evidence/summary.json',dict(records=len(evidence),status_counts=dict(counts),training_eligible=0));write(DEST/'image_evidence/manifest.json',evidence)
    lines=['# 香港全場景點雲擴展：階段報告','', '日期：2026-09-12。這是全場景提取與質量核對，不是新模型訓練完成報告。','', '## 核心結論','',f'已對 28 份代表地圖完成全範圍讀取，共 {sum(s["declared_points"] for s in sources):,} 個輸入點。首輪 {v1["raw_candidate_files"]:,} 個粗候選，經合併近距離種子及形狀檢查後保留 {len(rows):,} 份幾何候選，跨地圖暫時分為 {summary["provisional_groups"]:,} 組。**這些數字不是已確認單木數、樹種數或可訓練樣本數。**','', '之前 100 多份結果主要來自既有單木與配對；本輪改為直接掃描場景，不以已有人工標籤的位置限定搜索範圍。','', '## 實際資料覆蓋','', '| 區段 | 地圖數 | 本輪首掃輸入點 | 幾何候選檔案 |','|---|---:|---:|---:|']
    for name,v in regions.items():lines.append(f'| {name} | {v["maps"]} | {v["points"]:,} | {v["files"]:,} |')
    lines+=['','近期 7 份 map_dense.pcd 和何文田 5 份優化稠密圖均完整讀取。尖沙咀 12 份、司徒拔 4 份首掃使用覆蓋全場景的既有 DS 地圖，**不代表這 16 份對應的全部稠密原始文件已處理完畢**。本輪另完整回讀尖沙咀第 1 份稠密 PCD（105,997,651 點）作首批恢復。不同表示與重疊路段不相加當作不重複實體量。','', '## 清洗與自我檢查','', '首輪存在立面、燈杆與過密種子造成的碎片；未直接交作訓練集。第二輪以 3 m 非傳遞種子抑制、冠部雙向跨度、三維厚度、平面佔比、尺寸與點數檢查剔除部分失敗候選。兩版均保留。此流程是幾何啟發式，不是已驗證的單木分割模型；可能漏掉細小、稀疏或遮擋樹木，也可能仍混入非樹物體。','', '跨圖分組要求同一坐標系、軸位置接近和點雲支持重合；新採集不同會話未經配準不強行去重。分組仍非實體真值，不能用作最終獨立測試分組而不復核。','', '## 影像交叉核對','',f'本輪共記錄 {len(evidence):,} 份候選的影像查找／核對狀態；其中 {sum(bool(r["views"]) for r in evidence):,} 份有視角結果。尖沙咀第 1 圖完整檢查 220 個候選，其餘以按 ID 等距選取的 12／36 份分層試跑；不是全體影像核驗、隨機精度估計或獨立測試集。','']
    status_names={'one_view_tree_support':'單視角自動樹木支持','two_view_tree_support':'雙視角自動樹木支持','no_registered_panorama':'無對應或無 7–45 m 鄰近已配準影像','image_not_correlated_or_background_veto':'影像核對未通過（不能直接判定非樹）','projection_or_crop_unusable':'投影或裁片不可用'}
    lines+=['| 狀態 | 候選檔案數 |','|---|---:|']+[f'| {status_names.get(k,k)} | {v} |' for k,v in counts.items()]
    lines+=['', '使用既有相機投影關係生成去畸變透視裁片，兩視角觀察方位至少相差 10°，再用本地 YOLO 樹木框與 CLIP 樹木／背景比對作保守門檻。經驗外參未重新標定；鄰近樹木、暗區、錯位均可能影響結果。**影像支持不等於證明這份點雲是純淨單木，更不能單憑它確定樹種。**','', '具體反例：jianshazui_1_r0001 原圖可見深陰影樹冠，但自動檢查未通過；保留待復核，不刪除原始點雲。jianshazui_1_r0180 兩視角可見樹木，但點雲仍不完整。','', '## 稠密點雲回讀與抽樣','', '| 候選 | 初步實測點 | 回讀後保留點 | 輕量抽樣點 |','|---|---:|---:|---:|']
    for r in dense:lines.append(f'| {r["id"]} | {r["coarse_points"]:,} | {r["dense_retained_points"]:,} | {r["sample_points"]:,} |')
    lines+=['', '只恢復初步幾何支持 0.6 m 內的原始實測鄰點，以 0.08 m 體素保留代表點，再抽樣最多 2048 點。沒有生成或補造點，沒有把密度增加當作完整度提高。自查上述 6 份仍有杆件、低位異物、斷幹或離散樹冠；詳見 agent_visual_audit.json。它們均沒有被確認為可訓練單木。','', '## 新採集資料的實際障礙','', '- tree991、tree993、tree9111、tree9112、tree9113 的 BAG 只有 LiDAR、IMU、里程計，沒有相機或 GPS 話題；目前無法可靠對上 Google 街景或既有 VMMS 樹種。需要照片／視頻或采集位置線索。','- tree981 的 BAG 索引讀取失敗；map_dense 可用，原 BAG 未改寫或修復覆蓋。','- tree981／tree982 的 INSV 可抽出原始魚眼畫面，但尚未拼接和做相機—雷達配準。tree982 現有視頻容器顯示約 5.64 秒，明顯短於約 183 秒點雲軌跡，不能假定全程有對應影像。','', '## 訓練與安全','', '本輪沒有新增分類器擬合、微調或模型部署。新增候選全部 training_eligible=false、species=null；未將未知／待核驗資料悄悄貼模型預測標籤後訓練。原正式 Demo、既有實驗權重均未替換。使用的 YOLO／CLIP 權重前後 SHA-256 一致，原始首掃文件大小與修改時間一致；詳見 verification.json。因本輪無訓練，沒有新的訓練前權重備份；將來每次訓練仍需先新建並驗證實體副本。','', '## 可查看的成果與後續','', '- 全場景核驗頁：http://127.0.0.1:8045/ ，按地圖、影像支持及稠密回讀篩選，可看原全景、透視影像、三視圖與 2048 點交互預覽。這是只讀核驗頁，不是已驗收訓練集。','- 原點雲訓練實驗頁 8044 與正式 Demo 8043 保持不變。','- 下一階段先修正圖像—點雲對齊與完整單木邊界，剔除杆件／立面／鄰樹；按可靠影像給樹種證據，避免直接用全景分類結果替代標註。新採集位置補充後才能拓展其樹種核驗。','- 只有通過幾何與樹種核驗的獨立單木才進入新的固定分組訓練對比；目前不能聲稱已擴展出數千棵合格訓練樹木。','']
    (DEST/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(dict(coverage=regions,image_status=dict(counts),dense=len(dense),verification=checks),flush=True)
if __name__=='__main__':main()
