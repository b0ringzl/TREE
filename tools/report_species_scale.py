"""Report catalog-size experiments without conflating closed-set and field accuracy."""
import sys
from pathlib import Path
import numpy as np
from species_scale_experiment import load_json,write_json,metrics,PRIORITY


def main(out):
    out=Path(out);coarse=load_json(out/'scale_selection.json');full=load_json(out/'full_validation_results.json');final=load_json(out/'final_test.json');lock=load_json(out/'final_choice_locked.json')
    lines=['# 樹種類別規模：循環訓練與驗證報告','',
           f'本輪暫選 **{final["k"]} 個已知樹種標籤＋1 個清單外類別**。這是所測方案中的候選，不是對所有類別組合的全域最優證明；未替換正式 Demo。','',
           '**建議先以約 20 個標籤作核心清單，30 個作後續擴展方向；本次縮類重訓沒有足夠的綜合收益，繼續保留原模型。**同清單測試的已知 Top-1 幾乎相同，而新模型通過可信門檻的已知影像覆蓋率較低；不能把「建議清單規模」誤解為「已證明新模型值得部署」。','',
           '## 如何比較','',
           '- 20、30、40、60 個已知標籤各訓練 3 個隨機種子，共 12 次成功的 YOLO 凍結特徵分類頭擬合；每次 80 個全批次訓練輪次。另有一次首步損失計算前的整數型別錯誤，未更新參數，其備份保留，重試使用新目錄。',
           '- 由驗證結果挑出 20 和 30 類，各做 3 輪完整 YOLO11s-cls 微調；允許微調前模型在驗證較佳時勝出，不強行採用更新版。',
           '- 榕樹（細葉榕／垂葉榕合併）、蒲葵、鳳凰木、石栗、大葉合歡、南洋杉為所有方案的固定重點標籤。其餘標籤按規劃驗證集表現與支持數排序，形成巢狀清單；沒有按測試結果刪掉難種。',
           '- 既有驗證圖按 group 分成規劃／調參 2,640 張、規模選擇 2,476 張，兩者無 group 交集。規劃集選類、選 epoch、設定門檻；選擇集比較 K；最後才使用固定測試集。',
           '- 148 類原模型中的兩種榕樹併成一個使用者標籤。狐尾椰子、銀合歡不是該 148 類模型的輸出，原先獨立 Demo 分支未變，不能宣稱本輪已評估它們。',
           '- 本輪仍包含 23,455 張原訓練圖與 127 張已驗收新圖；未選入 K 的既有類別作清單外訓練資料，沒有假裝它們不存在。', '',
           '## 首輪分類頭：規模選擇驗證集（三種子中位數）','',
           '| 已知標籤數 | 已知 Top-1 | 已知 Macro-F1 | 通過門檻的已知圖比例 | 清單外誤接納率 |',
           '|---:|---:|---:|---:|---:|']
    for row in coarse['scales']:
        m=row['median_selection'];lines.append(f'| {row["k"]} | {m["known_closed_accuracy"]:.2%} | {m["known_closed_macro_f1"]:.4f} | {m["known_coverage"]:.2%} | {m["other_false_accept_rate"]:.2%} |')
    lines+=['','**已知 Top-1 是封閉集指標**：只對真實樹種在名單內的圖片，取已知 K 類中排名第一者計算。它忽略拒識，不能當成任意上傳圖片的可靠判定率。表中覆蓋率單独顯示有多少已知圖真正通過可信門檻。不同 K 的已知樣本集合不同，不能把全部提升歸因於重新訓練。不能將此處約 85% 直接與原 148 類約 57% 相減，兩者題目與樣本範圍不同；應看下方同一清單、同一圖片的原模型對照。','',
            '## 完整 YOLO 微調後：規模選擇驗證集','',
            '| 已知標籤數 | 已知 Top-1 | 已知 Macro-F1 | 已知覆蓋率 | 清單外誤接納率 | 所有接納結果的精確率 | 採用完整微調 |',
            '|---:|---:|---:|---:|---:|---:|---|']
    for r in full:
        m=r['selection'];lines.append(f'| {r["k"]} | {m["known_closed_accuracy"]:.2%} | {m["known_closed_macro_f1"]:.4f} | {m["known_coverage"]:.2%} | {m["other_false_accept_rate"]:.2%} | {m["accepted_precision_all_sources"]:.2%} | {r["full_fine_tune_selected"]} |')
    if (out/'masked_baseline_validation.json').exists():
        lines+=['','### 不重訓、只限制同一清單的驗證對照','', '| 清單大小 | 原模型已知 Top-1 | 原模型已知覆蓋率 | 原模型清單外誤接納率 |', '|---:|---:|---:|---:|']
        for r in load_json(out/'masked_baseline_validation.json'):
            m=r['selection'];lines.append(f'| {r["k"]} | {m["known_closed_accuracy"]:.2%} | {m["known_coverage"]:.2%} | {m["other_false_accept_rate"]:.2%} |')
        lines+=['','同一個 K 下，縮類訓練可能改善已知樹種排名卻降低可直接判定的覆蓋率。這個對照不需要替換原部署權重，亦未直接修改 Demo 的展示清單。']
    lines+=['',f'規模選擇驗證是否同時達到所有預設門檻：**{lock["target_met"]}**。門檻為已知 Top-1 ≥85%、已知 Macro-F1 ≥0.80、已知覆蓋率 ≥60%、清單外誤接納率 ≤10%、所有接納結果精確率 ≥85%。未達標不會以降低門檻後的結果冒充原目標達成。','',
            '## 鎖定後的固定回歸測試：與同一清單的舊模型比較','',
            f'只測試事前鎖定的 {final["k"]} 類候選，不再用測試分數重新選 K。對照模型為原 148 類模型，合併榕樹並將其他輸出概率加總為清單外；使用完全相同的測試圖片，且其置信門檻也只由規劃驗證集設定。','',
            '| 模型 | 已知 Top-1 | 已知 Macro-F1 | 已知覆蓋率 | 清單外誤接納率 | 所有接納結果精確率 |',
            '|---|---:|---:|---:|---:|---:|']
    for label,key in [('原模型・相同清單','baseline_same_catalog'),('循環訓練候選','candidate')]:
        m=final[key];lines.append(f'| {label} | {m["known_closed_accuracy"]:.2%} | {m["known_closed_macro_f1"]:.4f} | {m["known_coverage"]:.2%} | {m["other_false_accept_rate"]:.2%} | {m["accepted_precision_all_sources"]:.2%} |')
    m=final['candidate'];lines+=['',f'測試包含 {m["known_count"]} 張名單內、{m["other_count"]} 張名單外圖片。候選在本次回歸測試是否通過所有門檻：**{final["target_met_on_regression_test"]}**。','',
            '## 局限與部署建議','',
            '- 舊權重的歷史選擇使用過既有驗證集，固定測試也已有歷史報告；本輪不是全新盲測。需要未參與模型、名單或門檻選擇的新街景資料才能估計外場表現。',
            '- 清單外測試只覆蓋原資料中的其餘樹種，且原編碼器曾學過這些類別；不是任意未知物種測試。',
            '- 若已知 Top-1 良好但覆蓋率偏低，應返回候選或提示補拍，不能隱藏拒識後聲稱所有輸入都達到該準確率。',
            '- 未改點雲、融合模型，也未把新編碼器強接到舊雙源分類頭。原部署檔案在全部成功訓練後重新驗證雜湊，未變更。每次訓練在各自 pre_training_backup 下有獨立權重備份。',
            '- 此輪沒有加入 21–29 類等更細網格、更多種子、長週期訓練或不同骨幹架構，不能聲稱找到普適最優樹種數。','',
            '## 候選清單','', '| 編號 | 樹種標籤 | 固定重點 |','|---:|---|---|']
    for i,n in enumerate(final['class_labels'][:-1],1):lines.append(f'| {i} | {n} | {"是" if n in PRIORITY else ""} |')
    lines+=['','另有 1 個 Other / 清單外類別，**不計作樹種**。','',
            '完整數值：[測試結果](final_test.json)；[規模選擇紀錄](final_choice_locked.json)；[分組與試驗協議](protocol.json)。']
    (out/'species_count_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'recommended_catalog.json',{'known_labels':final['class_labels'][:-1],'other_label':final['class_labels'][-1],'deployment':False,'target_met':lock['target_met'],'source':'validation_locked_catalog_size_experiment'})
    print('REPORT='+str(out/'species_count_report.md'))


if __name__=='__main__':main(sys.argv[1])
