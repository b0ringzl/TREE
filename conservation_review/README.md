# Conservation review / 保育樹種人工復核

Independent, localhost-only review tool. Does not modify the Demo, models, source photographs, collection manifests, or any prior human-review database. No model scores or suggested labels are exposed. Source species are available in collapsed provenance details, not presented as a verified answer.

## Start

Run from the TREE project root (use an unused port; the server does not kill/reuse another process):

```powershell
& 'C:/Users/25041725G/AppData/Local/miniconda3/envs/yolo/python.exe' conservation_review/server.py --manifest external_datasets/conservation_sources_20260917_v1/review_manifest.json --port 8051
```

If the collection only supplies `manifest.json`, pass that instead. Environment alternative: `CONSERVATION_REVIEW_MANIFEST`; database override: `--database` / `CONSERVATION_REVIEW_DATABASE`. The startup JSON gives the exact database path. Default is a new `conservation_review/data/<manifest-parent>_<path-hash>/human_review.sqlite3`, never an older project's DB. Reuse the same manifest path to resume decisions. The manifest is loaded at startup: restart to load new samples.

Open `http://127.0.0.1:8051/?filter=pending`. Optional query filters: `source_species=Aquilaria%20sinensis`, `priority=high`. `filter=all`, `reviewed`, `aquilaria`, `not_aquilaria`, `uncertain`, `unusable` are also supported.

Only image paths inside the manifest directory are readable by default. If an intentional manifest points to other existing data, explicitly pass repeatable `--image-root` directories. Do not supply a broad home/workspace root unnecessarily. Unknown IDs, file path traversal, source SHA mismatches, and non-local Host headers are rejected. APIs return JSON errors. Writes require a per-process token and same-origin requests.

## Review

- `A`: Aquilaria sinensis / 土沉香
- `N`: Not Aquilaria sinensis / 非土沉香
- `U`: Uncertain / 無法判斷
- `X`: Image unusable / 圖片不可用
- `←` / `→`: previous/next, without assigning a label
- `Ctrl+Z` / `Cmd+Z`, or Undo: undo the latest decision made in this browser tab. It will not overwrite a newer decision from another tab.

Type an optional note **before** selecting a decision. Each decision is automatically saved; a saved indicator confirms success. Navigation asks before discarding an unsaved note. Keyboard shortcuts do not intercept text fields, selects, or contenteditable areas. Default save-and-next can be unchecked. Zoom and original-image links preserve the source bytes.

Decisions are four-way binary species/quality judgments; `not_aquilaria` **does not assign a replacement fine-grained species**. A positive/negative human identification is **not** a copyright/license grant, a guaranteed quality acceptance, or automatic approval for supervised training. Pending/uncertain/unusable images require separate policy handling. Keep previous human exclusions authoritative. Training code must join on exact image SHA and read the SQLite decisions rather than infer decisions from a preview or export age.

## Manifest

JSON list (or `items`, `rows`, `records`, `images` array) with:

```json
[{"id":"observation_photo_id","source_image":"images/photo.jpg","sha256":"<64 hex digits>","source_species":"Aquilaria sinensis","source_url":"https://...","license":"CC BY 4.0","attribution":"Photographer","observation_id":"123","source":"iNaturalist","review_priority":"high"}]
```

`source_image` may be absolute. Missing supplied hashes are computed from the original bytes; supplied hashes must match. Supported original files: JPEG, PNG, WebP. Source URLs are HTTP(S) only. Additional fields (including model predictions and training eligibility) are not shown or altered.

Optional `review_note` is an English + Traditional Chinese plain string for source/quality-check observations (for example, a visible tree sign conflicts with source metadata). It appears separately above the original image under **Source/quality check note / 來源及品質核驗提示**, using `textContent` only; absent, empty or non-string values remain hidden. It does not preselect a decision, rewrite the source species, or show model scores. Keep these notes evidence-based rather than predictions.

`GET /api/export` downloads JSON with current decisions, source identity, exact image hashes, manifest hash, timestamps, human reviewer identity, and append-only event/undo history; it does not overwrite the input manifest. Decision events do not imply that metadata-only source labels are human confirmed.

## Tests

```powershell
& 'C:/Users/25041725G/AppData/Local/miniconda3/envs/yolo/python.exe' -m unittest discover -s conservation_review -p 'test_*.py' -v
node --check conservation_review/static/app.js
& 'C:/Users/25041725G/AppData/Local/miniconda3/envs/yolo/python.exe' conservation_review/check_browser.py
```

Tests use generated temporary fixture files and an ephemeral port; no collection images/decisions are changed. Tests cover persistence, revisions/concurrent edits, multi-step undo and audit history, source byte preservation, image root and SHA guards, hidden model fields, CSRF, Host and origin checks, JSON errors, static allowlist and export.

`check_browser.py` uses installed headless Edge through the bundled Playwright library. It verifies shortcuts, note handling, save-and-next, navigation, pending completion, undo, export and browser exceptions against a temporary fixture DB. Pass an actual manifest path as its optional positional argument for a read-only collection screenshot using a temporary empty review DB; that mode never submits decisions.
