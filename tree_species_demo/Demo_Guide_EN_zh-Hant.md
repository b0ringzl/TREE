# Tree Atlas / 樹識

## Models / 模型

- General image: 19 known classes; point clouds are used only for quality checks and preview. / 通用影像：19 個已知類別；點雲僅用於品質檢查與預覽。
- Paired multimodal: 17 selected classes, with image, point-cloud and fusion inference. / 配對多模態：17 個選定類別，支援影像、點雲及融合推理。
- Street-tree multimodal: banyan, fan palm and foxtail palm; experimental transfer heads. / 街景多模態：榕樹、蒲葵及狐尾椰子；使用實驗遷移分類頭。
- Extended multimodal: flame tree, candlenut, white leadtree, lebbeck and the original merged Araucaria class. / 擴展多模態：鳳凰木、石栗、銀合歡、大葉合歡及原訓練合併的南洋杉類。

Model names describe capabilities. Renaming the interface does not change weights, training labels or historical review records. / 模型名稱描述功能；介面更名不會改變權重、訓練標籤或歷史驗收紀錄。

## Use / 使用方式

1. Select a model and upload a single-tree image or cloud. The public demo does not show an example selector. / 選擇模型並上傳單木影像或點雲；公開演示介面不顯示示例選擇區。
2. For fusion, confirm that both sources depict the same tree. Use upright Z-axis cloud coordinates. / 使用融合前，確認兩種來源屬於同一棵樹；點雲須以 Z 軸朝上。
3. Review quality warnings and separate source candidates. Improve blurred images or incomplete scans before relying on predictions. / 檢視品質警告及各來源候選；影像模糊或掃描不完整時，請先改善輸入。
4. Export the bilingual result JSON. Reference labels appear only after inference and are not used to choose the prediction. / 匯出雙語結果 JSON；參考標籤只在推理完成後附加，不用於選擇預測。

## Interpretation / 判讀限制

Confidence values are uncalibrated model scores, not measured accuracy. When evidence is insufficient, the main card shows up to three ranked candidates with their original scores, without renormalizing them. Quality and disagreement warnings remain visible. / 置信度為未校準模型分數，並非實測正確率。證據不足時，主卡片顯示最多三個排序候選及原始分數，不重新正規化；品質及分歧警告仍然保留。

If no class scores exist for an input, the demo requests a compatible input rather than fabricating candidates. Exported results include the candidate presentation and retain the underlying decision for traceability. / 若輸入沒有可用的類別分數，演示會要求相容輸入，不會編造候選。匯出結果包含候選顯示資料，並保留底層判定以供追溯。

The five-class experiment prioritizes images and reports fusion separately. Flame tree and white leadtree lack independent class evaluation; their point-cloud reliability remains unverified. / 五類實驗優先採用影像並獨立展示融合結果。鳳凰木及銀合歡尚無獨立類別評估，其點雲識別可靠性仍未驗證。

Vertical extent is not verified tree height when roots or crowns are missing. Neither height alone nor agreement between sources establishes species identity. / 根部或樹冠缺失時，垂直跨度並非已驗證的樹高。單憑高度或來源一致，均不能確定樹種。

Training examples are explicitly marked and must not be counted as independent tests. Inspect the species table for validation scope. / 訓練示例有明確標示，不可計入獨立測試；請在樹種表查閱驗證範圍。
