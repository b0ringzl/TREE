# Local assets / 本地資產

No asset downloads or credentials are embedded. Restore authorized files into the original relative layout, or adapt path configuration before running. / 未內嵌資產下載或憑證；請恢復已獲授權檔案至原相對目錄，或執行前調整路徑設定。

| Location / 位置 | Required content / 所需內容 |
| --- | --- |
| `full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1/` | Image weights, labels and metadata referenced by engine.py / 影像權重、標籤及中繼資料 |
| `experiments/whu18_yolo11_ptv2_fusion_20260826/` | Paired image, PTv2 and fusion checkpoints and class metadata / 配對影像、PTv2、融合權重及類別資料 |
| `hk-tree-vmms/derived/hk_dualsource_three_species_20260907/` | Three-class model, pair index and assets / 三類模型、配對索引與資產 |
| `hk-tree-vmms/derived/reviewed_five_species_pilot_v1_20260907/` | model.joblib, plan, split, encoder hashes and associated assets / 五類分類頭、計畫、劃分、編碼器雜湊及資產 |
| Other paths under `hk-tree-vmms/derived/` | Legacy review assets referenced by API modules / API 模組引用的歷史驗收資產 |
| `tree_species_demo/samples.json`, `tree_species_demo/test_data/` | Local manifests and files needed by legacy routes/static mount / 舊路由及靜態掛載所需本地清單與檔案 |
| Training/cleaning dataset roots | Authorized images, LiDAR, poses, annotations and split manifests / 已獲授權影像、LiDAR、位姿、標註及劃分清單 |

Only load trusted checkpoints: model formats may execute code during loading. / 僅載入可信權重；部分模型格式可在載入時執行程式。

For publication, six copied launch/package scripts replace personal absolute paths with project-relative defaults. The original working directory is unchanged. Historical example paths and experiment-specific assumptions elsewhere still require adaptation for a new installation. / 六個複製的啟動及打包腳本已改用專案相對路徑；原工作目錄保持不變。其他歷史示例路徑及實驗假設仍須依新環境調整。
