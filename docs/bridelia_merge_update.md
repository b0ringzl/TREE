# Bridelia Merge Update

Date: 2026-06-23

`Bridelia tomentosa` and `Bridelia insulana` are now treated as one project label:

`Bridelia tomentosa(Bridelia insulana)`

Local source coordinates have already been consolidated under:

`D:\TREE\选取树种\Bridelia tomentosa(Bridelia insulana)`

Project updates:

- Backend/frontend species selection scans `D:\TREE\选取树种`, so it now exposes only the merged folder after restart.
- `tools/tree_species_label_merge_map.csv` maps both old names to the merged label.
- `tools/inaturalist_parent_name_supplement.csv` queries iNaturalist with `Bridelia tomentosa` for the merged label.
- Existing iNaturalist source folders were kept physically separate for source audit, while catalog/export labels are merged.
- `D:\TREE\external_datasets\inaturalist_master_catalog.csv` and summary were rebuilt with the merged project label.
- `D:\TREE\external_datasets\inaturalist_dated\2026-06-07\species_counts_canonical.csv` and deficit report were rebuilt with the merged label.
- `D:\TREE\external_datasets\inat_max150\duplicate_photo_review\index.html` was rebuilt.
- No Street View accepted dataset folders or rejection buckets existed for these old Bridelia names, so no Street View migration was needed.

Verification snapshot:

- Backend `list_species()` returns `Bridelia tomentosa(Bridelia insulana)` and not the two old standalone names.
- Master catalog rows: 33417.
- Master catalog missing absolute paths: 0.
- Master catalog old Bridelia target rows: 0.
- Master catalog merged Bridelia rows: 371.
- Master summary:
  - `2026-05-24`: 221 images from original `Bridelia insulana` and `Bridelia tomentosa`.
  - `2026-06-07`: 150 images from original `Bridelia tomentosa`.
- Dated canonical species count for `2026-06-07`: 152.
- Legacy duplicate visual review:
  - 62 remaining duplicate pairs.
  - 0 Bridelia duplicate rows.
- iNaturalist YOLO pretrain smoke export:
  - Merged class present.
  - Old standalone classes absent.
  - Smoke export classes: 152.

Historical model output folders under `D:\TREE\models` are left as snapshots. Re-export/retrain to produce new class ids with the merged label.
