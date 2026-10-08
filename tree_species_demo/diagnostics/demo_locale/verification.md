# Bilingual presentation verification / 雙語顯示驗證

The active recognition page, catalog, quality messages, inference explanations and result export use English plus Traditional Chinese. Model names describe capabilities without geographic names. / 目前識別頁、樹種清單、品質提示、推理說明及結果匯出使用英文加繁體中文；模型名稱僅描述功能，不顯示地名。

The presentation adapter does not change weights, class IDs, raw inference outputs or review records. Legacy review pages and historical reports are outside this UI-only change. / 顯示轉接層不更改權重、類別編號、原始推理輸出或驗收紀錄；舊驗收頁面及歷史報告不在此次識別介面修改範圍內。

- 3 localization unit tests passed; 36 existing tests passed. / 3 項語系單元測試及 36 項既有測試通過。
- 11 real inference presentations across four model scopes passed neutral-name and score-preservation checks. Catalog labels, sample labels, errors and static assets were checked. / 四種模型的 11 組實際推理顯示通過中性名稱及分數保留檢查；亦檢查清單、示例名稱、錯誤訊息及靜態資產。
- Existing integration verification passed: 15 extended-class calls, quality/error cases, old-model regressions and protected-source hashes. / 既有整合驗證通過：15 次擴展類別呼叫、品質與錯誤案例、既有模型回歸及受保護來源雜湊檢查。
- Browser accessibility inspection confirmed the new page loads with four bilingual model options and neutral example names. Full browser click automation was not performed. / 瀏覽器可存取性檢查確認新頁面載入四個雙語模型選項及中性示例名稱；未執行完整瀏覽器點擊自動化。

These are software verification results, not a new species-accuracy evaluation. / 以上為軟體驗證結果，並非新的樹種精度評估。
