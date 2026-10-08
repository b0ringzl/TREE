# Multi-species conservation review / 多樹種保育復核

This independent local-only review entry handles unresolved source-identification questions for **Artocarpus hypargyreus / 白桂木**, **Rhodoleia championii / 紅花荷**, and **Keteleeria fortunei / 油杉** (and additional species supplied by a manifest). It does not start a service automatically, change the current Demo, mutate source images, or write the Aquilaria review database.

## Start only when the unresolved manifest is ready

```powershell
& 'C:/Users/25041725G/AppData/Local/miniconda3/envs/yolo/python.exe' conservation_expansion_review/server.py --manifest external_datasets/conservation_expansion_20260917_v1/human_review_required.json --database conservation_expansion_review/data/expansion_20260917_v1/human_review.sqlite3 --port 8052
```

The collection version/path above is an example; pass the **actual** unresolved manifest produced by the curation step. Use only `human_review_required` samples for this user queue; agent-accepted samples are not automatically copied into the human decision database. Do not label automatic/agent decisions as human decisions. Reuse the same explicit database path when changing manifest order so already reviewed images remain reviewed.

Environment alternatives: `EXPANSION_REVIEW_MANIFEST`, `EXPANSION_REVIEW_DATABASE`. The server binds only `127.0.0.1`; startup fails rather than stopping an occupied port. Default port is 8052, independent of the existing 8051 Aquilaria tool. A database not marked as this expansion tool is rejected **before writes**, including any old Aquilaria database. Default DBs are created under this new directory only.

## Controls

- **A** — Confirm the displayed source species / 確認正在顯示的來源樹種.
- **N** — Reject the displayed source species / 否定來源樹種. This is not a replacement species label.
- **U** — Uncertain / 無法判斷.
- **X** — Image unusable / 圖片不可用.
- **Left / right** — Previous / next; does not assign a label.
- **Ctrl+Z / Cmd+Z** — Undo the latest decision in this tab, unless superseded by another tab.

Type an optional note before choosing a decision. Decisions save immediately, with explicit saved/error status. Text fields, selects, and contenteditable areas are excluded from shortcuts. Save-and-next can be disabled. Original preview, zoom, source attribution/license/SHA, species filtering and JSON decision export are provided. Pending/uncertain/unusable samples are not training approvals; species confirmation is not an image license grant.

## Manifest and decisions

Use the previous collection schema: a list or `items`/`rows`/`records`/`images` list. Each row has `id`, `source_image`, `sha256`, `source_species`, `source_url`, `license`, `attribution`, and optional `review_priority`. Images must be below the manifest directory, or in explicit repeatable `--image-root` allowlisted directories. Hashes are checked both during load and before review. Original bytes and source labels are never rewritten.

Additional optional fields:

- `source_species_zh_hant`: Traditional Chinese name; the three target species above have built-in defaults. `source_names[0]` is a fallback for other species.
- `agent_question` or `review_note`: English + Traditional Chinese plain text explaining the unresolved source/quality issue. Rendered with `textContent`, never HTML; it does not preselect a decision.
- `plantnet_reference`: explicit optional object `{scientific_name, score, source_url, checked_at}`. `score` must be a finite number from 0 to 1, and `source_url` must be HTTP(S). It is shown in a separate collapsed section labelled **Automatic model reference, not ground truth / 自動模型參考，並非真值**. Other arbitrary model scores/predictions do not pass through. Missing results do not create fake model references.

SQLite `decisions` stores `decision` (`confirm_source`, `reject_source`, `uncertain`, `unusable`), exact `source_species`, `source_species_zh_hant`, original `sha256`, `reviewer_identity=human`, note, UTC timestamp, session and revision. `history` keeps save/undo events and the manifest SHA. `GET /api/export` downloads current human reviews plus audit history, without replacing the input manifest. The original source species remains the same for rejected items; no fine-grained alternative species is inferred.

## Shared code and isolation

`server.py` imports `conservation_review.server` for image path/SHA guards, SQLite connection helpers, source URL filtering, JSON errors, localhost/CSRF checks, and HTTP responses. Its new store implements distinct choices and species-bound decisions; it never changes the old module's global `CHOICES`. It reuses the old stylesheet through read-only serving plus a small new stylesheet. No edits to the old review directory are required.

## Tests

```powershell
& 'C:/Users/25041725G/AppData/Local/miniconda3/envs/yolo/python.exe' -m unittest discover -s conservation_expansion_review -p 'test_*.py' -v
node --check conservation_expansion_review/static/app.js
```

Tests only use temporary synthetic images, manifests and SQLite databases; the HTTP test listens on an automatically allocated temporary port and shuts down. They cover bilingual species, safe external references, decisions with species/SHA, unchanged source files, rejection semantics, persistence, undo/concurrency, rejection of old database paths, unchanged old choices, hash/path guards, static/API responses, CSRF/origin checks and JSON export. No real source image is reviewed and no persistent service is started by the tests.
