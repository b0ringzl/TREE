# Tree Atlas — Core source snapshot / 樹識核心原始碼

Current recognition demo and supporting image, point-cloud, fusion, cleaning and review code. The demo uses English + Traditional Chinese and capability-based model names. / 目前識別演示及配套影像、點雲、融合、清洗與驗收程式；演示採用英文加繁體中文，模型以功能命名。

This is a **source-only research archive, not a ready-to-run distribution**. This upload excludes datasets, panorama photos, LiDAR assets, human review records, trained weights, environments and private credentials. / 本版本**僅為研究原始碼歸檔，並非可直接執行的獨立安裝包**；本次上傳不包含資料集、全景照片、LiDAR 資產、人工驗收紀錄、權重、環境及私人憑證。

## Code map / 程式碼導覽

| Path / 路徑 | Purpose / 用途 |
| --- | --- |
| `tree_species_demo/` | FastAPI inference, quality checks, bilingual UI, top-three candidates / 推理、質檢、雙語介面及前三候選 |
| `hk-tree-vmms/tools/` | Panorama/LiDAR pairing, tree cleaning, stem recovery, metric-height audits, review replay and training / 全景與點雲配對、清洗、樹幹恢復、高度檢查、驗收回放及訓練 |
| `hk-tree-vmms/annotation_app/` | Annotation and review frontend/backend / 標註驗收前後端 |
| `Lidar/tscmdl_whu_sandbox/src/tscmdl_whu/` | Projection, sampling, PTv2 and fusion / 投影、採樣、PTv2 及融合 |
| `Lidar/tscmdl_whu_sandbox/scripts/` | Training, evaluation and experiment preparation / 訓練、評估及實驗準備 |
| `full_image_species_baseline/scripts/` | Image classification and segmentation baselines / 影像分類及分割基線 |
| `yolo11_species_baseline/scripts/` | YOLO data preparation, training and evaluation / YOLO 資料準備、訓練及評估 |
| `tools/tree_species_label_merge_map.csv` | Label configuration, not sample data / 標籤設定，並非樣本資料 |

The existing `hk-tree-labeler/` on the main branch is unchanged. Historical source names remain for compatibility, but are not displayed as model names in the active demo. / 主分支既有標註工具保持不變；歷史原始碼名稱保留以維持相容性，但演示不以此顯示模型名稱。

## Setup / 環境與執行

Use Python 3.10+; first install GPU-compatible torch and torchvision. The working local environment used torch 2.11.0+cu128 and torchvision 0.26.0+cu128. `requirements-core.txt` records non-PyTorch package versions, not a portable lockfile. / 使用 Python 3.10 以上版本，先安裝適合顯示卡的 torch／torchvision；原驗證環境使用上述版本。需求檔記錄非 PyTorch 套件版本，並非跨平台鎖定檔。

```sh
python -m pip install -r requirements-core.txt
```

Restore authorized assets described in [ASSETS_REQUIRED.md](ASSETS_REQUIRED.md) before launching. Some historical experiments also require separately obtained third-party backbones. / 啟動前恢復資產說明列出的已獲授權資產；部分歷史實驗亦須另行取得第三方骨幹模型。

```sh
cd tree_species_demo
python server.py
```

Visit http://127.0.0.1:8037 on the same computer; keep the server running. This upload does not deploy a public website. / 在同一台電腦開啟該網址，並保持服務執行；本次上傳不會部署公開網站。

The UI accepts uploads without an example selector. Low-confidence decisions show up to three candidates with unchanged scores; names have no catalog-status suffix. Quality and conflict diagnostics remain visible. / 介面直接接受上傳，不顯示示例選擇器；低置信度結果顯示最多三個候選及原始分數，名稱不附加清單狀態；品質及分歧診斷仍然保留。

## Verification and limits / 驗證與限制

```sh
cd tree_species_demo
python -m unittest test_demo_locale
```

Most other tests require local assets or a running service. A source checkout alone does not reproduce accuracy experiments. Scores are uncalibrated, and experimental fusion is not guaranteed to improve identification. Flame tree and white leadtree lack independent class evaluation in the five-class pilot. / 多數其他測試需要本地資產或執行中的服務；下載原始碼不等於重現精度實驗。分數未校準，實驗融合不保證改善識別；五類實驗的鳳凰木及銀合歡尚無獨立類別評估。

This snapshot grants no new blanket license to third-party code, weights or datasets. Respect upstream licenses and data permissions. / 此歸檔不會对第三方程式、權重或資料集授予新的概括授權；請遵守原授權及資料使用權限。
