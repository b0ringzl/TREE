# SpaceScan Trees — Official local Demo / 正式本機 Demo

Local URL / 本機網址：http://127.0.0.1:8043/

## Two Demo editions / 兩個展示版本（2026-09-19）

- **Image-only / 單源影像版**: `http://127.0.0.1:8043/image-only`; Windows launcher `START_SPACESCAN_IMAGE.cmd`. No source selector, cloud upload controls or point-cloud JavaScript is rendered. / 不顯示資料源選擇或點雲輸入，也不載入點雲功能腳本。
- **Image + Cloud / 完整版**: `http://127.0.0.1:8043/multimodal`; Windows launcher `START_SPACESCAN_MULTIMODAL.cmd`. Choose image-only or image + cloud. `/` remains a backward-compatible complete edition. / 可切換單源影像或影像＋點雲，舊首頁保持完整版。

Both use one server-rendered template (Jinja2) and the same image inference APIs, automatic tree selection, polygon cropping, paste upload, panorama correction and SSTv1 weights. Existing seven-class fusion is unchanged; these editions do not imply 148-class point fusion. They are separate entry points, not separately trained models or new Android packages. `/apps` links both. / 兩版共用模板、圖像流程及模型；七類點雲融合不變，並非重新訓練或安卓重新打包。

## OneDrive image import / OneDrive 影像匯入（配置後可用）

The image-only edition has a read-only OneDrive picker. It uses Microsoft Authentication Library 5.22.0 and Microsoft Graph delegated `Files.Read`; selecting a JPG/PNG/WEBP downloads that file into the browser and passes it to the same `addFiles` validation as local upload (4 images maximum, 20 MB each, 40 MP each). The chosen image then follows the ordinary tree-selection and SSTv1 inference flow. Access tokens stay in browser memory; the Demo backend receives only the selected image bytes when recognition is requested. The OneDrive original is not changed. Internet access is required. This is **not** part of the fully offline Android app or the image+point edition.

To enable it, register a Microsoft Entra single-page application supporting your intended Microsoft account types (personal, work/school, or both), add Microsoft Graph delegated `Files.Read`, and register the SPA redirect URI `http://localhost:8043/static/onedrive-redirect.html`. Set `SPACESCAN_ONEDRIVE_CLIENT_ID` to that app's public Application (client) ID **before launching** the 8043 server, then open `http://localhost:8043/image-only`. No client secret is used or stored. If using `127.0.0.1` instead of `localhost`, register the exact URI displayed in the OneDrive dialog; Microsoft's portal requires a manifest edit for an HTTP `127.0.0.1` loopback URI. Until a valid client ID is configured, the button explains the missing setup and does not pretend to import.

The login/account flow still requires a real Entra app registration and user consent for end-to-end verification. Build sources are in `onedrive/`; `npm ci` and `npm run build` regenerate the two local bundles. The redirect page must remain a dedicated page without `Cross-Origin-Opener-Policy` headers.

## Dropbox image import / Dropbox 影像匯入（配置後可用）

The image-only edition also has a Dropbox Chooser button. Create a Dropbox **Chooser** web app at <https://www.dropbox.com/developers/apps/create>, register the Demo domain `localhost` in its Chooser/Drop-ins domains, and copy its public **App key**. Set `SPACESCAN_DROPBOX_APP_KEY` to that key before launching the 8043 server, then open `http://localhost:8043/image-only`. Do not enter the App secret. If using a different hostname, register that hostname and open the Demo through it consistently. The Dropbox app and account can be personal; Microsoft Entra and a school tenant are not involved.

The chooser permits JPG/JPEG/PNG/WEBP files of at most 20 MB. Selected direct links are consumed immediately in the browser (they expire), and only the downloaded image bytes enter the existing `addFiles` workflow, which also checks the four-image and 40 MP limits. Links are not sent to the Demo server or stored. Dropbox originals are not modified. Internet and browser pop-ups are required. This is **not** in the offline Android app or the image+point edition. Without a configured App key, the button explains setup rather than opening a fake chooser. A live Dropbox sign-in and selection cannot be verified until an App key and user account are supplied.

Edition verification: 36 Python regression tests passed (editions, preview, release, point recognition, JSON errors), plus JavaScript API-response and clipboard tests. Both checked Windows launcher modes succeed. In-browser, the same botanical test photo with Flower selected returned the same displayed top three (Lagerstroemia speciosa 99.7%, Lagerstroemia indica 0.1%, Cassia fistula 0.1%) and the pinned classifier version in both editions, with no console warnings/errors. Complete-edition cloud controls toggle correctly. This is a functional smoke test, not an accuracy measurement; point-fusion weights were not changed or retrained. / 兩版同圖實測結果一致；36 項 Python 測試及 JavaScript 測試通過，無控制台錯誤，點雲切換正常；僅為功能核驗，不代表精度評測。

## Current official entry points / 目前正式入口（2026-09-19）

`/apps` lists the pinned classifier version and the verified Android APK/model ZIP. Windows starts the same web service through the project-root `START_SPACESCAN_TREES.cmd`; there is no second Windows classifier. Startup checks the active deployment against `release.json` rather than trusting an old `?v=` URL. / `/apps` 提供版本及安卓檔案；Windows 與網頁共用服務，啟動時核對實際模型版本，不以網址中的 `v` 參數判定版本。

Image-only: `sstv1-artocarpus-user-20260918-v1`, BioCLIP 2.5 + three adapted MLP heads, 148 candidates. Android v0.2 uses the same classifier, label order, heads and prompt logic in staged ONNX FP32, fully offline. It does **not** include web automatic detection or panorama correction. Full pipeline parity is not claimed. / 單源影像使用同版 148 類 SSTv1；安卓完整自動選樹流程尚未同步，不能宣稱所有照片必然得到相同分數。

The historical 21/21 fixed-RGB conversion agreement is not 100% species accuracy. Select the same tree and crop for comparisons; different crops, JPEG decoders and BF16/FP32 arithmetic can change scores. Result JSON now records the actual classifier, target-image hash and prepared-JPEG hash. / 歷史 21／21 轉換首選一致並非樹種準確率；結果紀錄保留實際模型及輸入雜湊。

After automatic/user tree selection, **Export target crop / 匯出目標切片** downloads the exact native JPEG that was sent to the classifier, not the preview thumbnail. Use it as the same-target input on Android. The signed selection must still be valid; expired selections require identifying again. This is a comparison aid, not an automatic detector port to Android. / 選樹後可匯出本次分類使用的原生切片，供安卓以相同目標核對；簽章過期需重新識別。此為核對工具，不代表已移植安卓自動偵測。

Research/review tools and old Android packages are retained, not silently upgraded or deleted. Image + point remains the separately evaluated original BioCLIP + seven-class PTv2 fusion. No training or weight replacement is part of this entry-point update. / 保留研究存檔及舊包；七類融合不變，本次入口整理不訓練、不替換權重。

## Panorama perspective correction / 全景透視校正（v2）

The image-geometry selector supports Auto (about 2:1 implies an uncropped
360°×180° sphere), Photo, or explicit spherical panorama. Ordinary wide photos
must use Photo. Panoramas are reprojected into 16 overlapping perspective views
before detection; the modal and classifier both use the corrected geometry.
After identification, the main preview shows the selected perspective crop;
View still opens the original upload. / 全景先重投影再偵測及識別；可切換普通照片
以避免誤將寬幅照片視為球面圖。結果主預覽顯示校正切片，仍可檢視原圖。

This corrects spherical unfolding, not lens calibration, stitching parallax,
missing detail, or metric tree height. Results remain candidates. See the paired
royal-palm diagnosis in `outputs/spacescan_panorama_audit_20260910_153714/REPORT.md`.
No model weights changed; 30 unit tests pass. / 不把重投影當成鏡頭標定、樹高量測
或新增細節；未改模型權重，已通過 30 項單元測試。

## Try it / 試用

1. Choose Image only or Image + cloud. The scenario selector has been removed; the sidebar shows only the model version and date. / 選擇單源影像或影像＋點雲；已移除場景選擇，側欄模型資訊僅顯示版本與日期。
2. Keep Image only selected. Upload a tree photo or panorama, then click Identify tree. An ambiguous/off-center target opens a crop chooser; confirm one crop to identify. / 選擇單源影像，上傳樹木照片或全景；目標不居中或不明確時，選擇彈窗切片後識別。
3. Quick panorama test: About → Load panorama test → Identify tree. No boxes or results are prefilled. For the botanical detail test photo, choose its visible organ (Leaf / Flower) before identification. / 全景試用：關於 → 載入全景測試圖片 → 識別樹木；沒有預填框或結果。花葉測試照片請先選擇相應部位。
4. Click a reference thumbnail to compare images side by side; zoom, switch references or close the dialog. / 點擊參考縮略圖可左右對照、放大、切換或關閉。
5. Up to four image views can be held in the browser. Choose a thumbnail to identify that view separately. Removing an image can be undone. / 可保留四個視角，選擇縮略圖後分別識別；移除可復原。
6. Copy an image or screenshot, then press Ctrl+V (Mac: Cmd+V) on the main page. Pasted images follow the same JPG/PNG/WEBP, 20 MB, 40 MP and four-view limits. Text/links are not downloaded; no clipboard permission or background access is requested. Paste is ignored during recognition, in dialogs or in editable fields. / 複製圖片或截圖後，在主頁按 Ctrl+V；格式、大小及視角上限與原上傳方式相同。不下載貼上的文字連結，不在背景讀取剪貼簿；識別中、彈窗及文字欄位內不匯入圖片。

## What works / 已接通

- SpaceScan Trees branding, bilingual interface and two input-source selectors. / SpaceScan Trees 品牌、雙語介面及兩種資料源選項。
- Real image-only inference via the latest deployed SSTv1 adapted teacher on local port 8039. / 透過本機 8039 已部署的 SSTv1 適配教師進行真實單源影像推理。
- Aspect-preserving padding retains the whole selected rectangle; native dimensions are recorded. / 等比例補邊保留完整所選影像，並記錄原始尺寸。
- Top-three raw model scores, quality warnings, actual model-input preview, JSON export. / 三候選原始分數、品質提醒、實際模型輸入预覽及 JSON 匯出。
- Reviewed whole-tree/crown references from licensed iNaturalist photographs, with author, source and license links. Byte-identical input/reference images are excluded. Clear flower/leaf details are allowed for gaps under the revised user policy. / 採用已篩選、附授權的 iNaturalist 整株或樹冠照片，保留作者、來源及許可連結；排除與輸入相同的圖片，缺圖時可使用清晰花、葉細節。
- Browse `/references` for all 148 bilingual classes. Missing references remain pending; coverage is recorded in `external_datasets/spacescan_whole_tree_references_20260910_v2/mixed_v3/coverage.json`. / 參考圖頁列出全部 148 類；缺圖保持待補，實際進度存於覆蓋記錄。

Reference labels are not independent ground truth. No claim of same-tree identity is made. Files and watermarks are unchanged; no image rights are granted by this preview. No Pl@ntNet pictures or code were copied into this app. / 參考標籤並非獨立真值，未確認同株關係。原圖與水印未修改；本預覽不授予影像權利，亦未複製 Pl@ntNet 圖片或程式。

## Limits / 限制

- This is a local evaluation Demo, not a validated botanical identification service. / 此為本機評估 Demo，並非經認證的植物鑑定服務。
- Real tree selection uses local YOLO11m-seg plus conditional Grounding DINO Base and a CLIP background filter. A 0.40 detection threshold is not a calibrated probability. Closely overlapping trees may still share a crop; seam-wrapping trees are not reconstructed. / 真實選樹採用本地 YOLO11m-seg、條件式 Grounding DINO Base 補檢及 CLIP 背景過濾；0.40 門檻不是校準機率。鄰樹仍可能合框，全景拼接縫兩端尚未合併。
- Image + cloud performs local seven-class PTv2 score fusion after same-tree confirmation and quality/axis checks. It retains the independently evaluated original-teacher image branch; this is not 148-class fusion or a measured accuracy gain for every input. / 雙源已接通本機七類融合，需確認同株及品質／座標軸；不宣稱 148 類融合或每次均提升準確率。
- Multiple images are evaluated individually, not fused. Whole-tree / crown views enter detection; explicit leaf / flower / fruit / bark details bypass whole-tree detection. / 多圖逐一識別，未融合；全株／樹冠先偵測，明確選定的葉／花／果實／樹皮細節直接分類。
- The removed scenario selector did not choose separate models. The API's legacy scenario field is retained for compatibility; conservation status still needs independent verification. / 已移除的場景選擇不對應不同模型；API 保留舊場景欄位以維持相容，保育身分仍需另行核實。
- The current classifier includes prior incremental training; this entry-point update itself does not train or modify weights. Historical demos are preserved for research. / 目前分類器包含先前增量訓練；本次入口整理本身不訓練、不修改權重，保留歷史研究 Demo。

## Start / 啟動

## Manual reference editor / 手動補圖

Open `/references/add`, or select **Add / replace photos** on any reference-gallery species card. Choose a display slot, preview an existing same-species library image or upload JPG/PNG/WEBP, check species and clear whole-tree/crown or flower/leaf framing, and save. Existing-library selections autofill source and author. / 開啟補圖頁或點擊樹種卡片的補圖／替換；選擇位置後可從同樹種图库挑選或上傳圖片，核對後儲存。

The existing-library browser currently covers the original 1,128-photo reference collection plus licensed supplements, not every historic training dataset. Previous exclusions are explicitly marked; selecting a photo does not silently change its earlier screening decision. / 現有圖庫入口目前接入原 1,128 張參考候選及已附許可的補充圖，不代表所有歷史訓練資料；先前排除狀態會標示，不會因選用而竄改既有判定。

Records persist under `external_datasets/spacescan_manual_references_v1/references.sqlite3`; original bytes and display copies are retained in its `images` directory. Replacement and withdrawal are reversible from history. User references take precedence in the chosen display slots. Training manifests, weights and human-review databases are untouched. Unconfirmed licenses are local-reference-only, not publication or training permission. / 手動補圖獨立保存，保留原圖與歷史，替換、撤回可復原；不修改模型、訓練資料或原驗收資料庫。

Verified with 16 unit tests, including persistence, replace/withdraw/restore, exact-input exclusion, slot handling, file/type limits and cross-origin write rejection. / 已通過 16 項單元測試。

Run the project-root `START_SPACESCAN_TREES.cmd` or this directory's `START_LIVE_DEMO.cmd`. Both use `start-live.ps1`, start missing loopback-only services on 8039 and 8043, wait for model/version checks, then open the same current Demo. `start.ps1` delegates to this checked launcher. Existing unrelated port owners are never killed automatically. / 兩個 Windows 入口共用同一啟動腳本，啟動缺失服務並核對模型版本後開啟目前 Demo；不自動終止無關程序。

No external network is used for local inference. Reference source links are optional external links and require internet access when opened. / 本機推理不使用外部網路；選擇開啟外部參考來源時需要連網。

## Verification / 驗證

- Seven unit tests passed: assets, padding, bad inputs, unsupported fusion, unmodified score forwarding, offline response and identical-reference exclusion. / 七項單元測試通過。
- JavaScript syntax and Python compilation checks passed. / JavaScript 語法與 Python 編譯檢查通過。
- Live browser: blank input state; local test image → teacher response; candidate references; comparison dialog; image removal → disabled run; undo → image restored; slope and dual-source selection. / 已透過瀏覽器操作上述流程。
- Narrow homepage checked at requested viewport override 430×844 (content/client width 415 with scrollbar), no horizontal overflow. / 430×844 窄屏首頁檢查無水平溢出。
- This is functional verification, not a model-accuracy evaluation. / 此為功能驗證，並非模型準確率評測。

## Revised reference policy / 更新參考圖規則

Whole-tree/crown images remain preferred and the original 136 selections remain in place. Clear flower and leaf details fill gaps in `mixed_v3/accepted_manifest.json`; the strict manifest and historic decisions are preserved. New manual uploads use the same relaxed policy. This is presentation-only and leaves weights, training labels and user manual choices unchanged. / 保留原 136 張整株／樹冠圖，缺圖改用清晰花、葉細節；新版清單獨立保存，不修改權重、訓練標籤或使用者選圖。
