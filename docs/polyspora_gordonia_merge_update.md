# Polyspora/Gordonia Merge Update

Date: 2026-06-07

`Polyspora axillaris` and `Gordonia axillaris` are now treated as one project label:

`Polyspora axillaris(Gordonia axillaris)`

Local source coordinates have already been consolidated under:

`D:\TREE\选取树种\Polyspora axillaris(Gordonia axillaris)`

Project updates:

- Backend/frontend species selection uses `D:\TREE\选取树种`, so it now exposes only the merged folder after restart.
- `tools/tree_species_label_merge_map.csv` maps both old names to the merged label.
- `tools/inaturalist_parent_name_supplement.csv` queries iNaturalist with `Polyspora axillaris` for the merged label.
- `tools/train_tree_species_classifier.py` now uses the label merge map by default.
- `tools/catalog_inaturalist_batches.py` now prefers `original_species` when present, then writes both original and merged species fields:
  - `original_species`: source directory/metadata label.
  - `target_species` and `label_species`: merged project label.
- `D:\TREE\external_datasets\inat_max150\Gordonia axillaris` was removed after confirming it largely duplicated the Polyspora folder.
- `D:\TREE\external_datasets\inat_max150\Polyspora axillaris` was renamed to `D:\TREE\external_datasets\inat_max150\Polyspora axillaris(Gordonia axillaris)`.
- The merged legacy folder metadata was rewritten so paths point to the renamed folder, `target_species` uses the merged project label, and `original_species` preserves `Polyspora axillaris`.
- `D:\TREE\external_datasets\inaturalist_master_catalog.csv` and summary were rebuilt with the merged project label.
- The duplicate visual review page was rebuilt after removing the old duplicate Gordonia legacy folder.

Verification snapshot:

- Backend species list contains the merged folder and not the two old standalone names.
- Merged coordinate records loaded by backend: 2117.
- Master catalog rows after later Ficus cleanup: 33417.
- Master summary:
  - `2026-05-24`: 150 images from original `Polyspora axillaris`.
  - `2026-06-07`: 150 images from original `Polyspora axillaris`.
- Legacy duplicate visual review:
  - 199 remaining duplicate pairs.
  - 0 remaining Polyspora/Gordonia duplicate rows.
- YOLO pretrain smoke export:
  - Merged class present.
  - Old standalone classes absent.
  - Smoke export classes: 150.

Historical model output folders under `D:\TREE\models` are left as snapshots. Re-export/retrain to produce new class ids with the merged label.
