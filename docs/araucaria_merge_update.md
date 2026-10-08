# Araucaria Merge Update

Date: 2026-06-23

`Araucaria columnaris` and `Araucaria heterophylla` are now treated as one project label:

`Araucaria columnaris(Araucaria heterophylla)`

Local source coordinates have already been consolidated under:

`D:\TREE\选取树种\Araucaria columnaris(Araucaria heterophylla)`

Project updates:

- Backend/frontend species selection scans `D:\TREE\选取树种`, so it now exposes only the merged folder after restart.
- `tools/tree_species_label_merge_map.csv` maps both old names to the merged label.
- `tools/inaturalist_parent_name_supplement.csv` queries iNaturalist with `Araucaria columnaris` for the merged label.
- Existing iNaturalist source folders were kept physically separate for source audit, while catalog/export labels are merged.
- `D:\TREE\external_datasets\inaturalist_master_catalog.csv` and summary were rebuilt with the merged project label.
- `D:\TREE\external_datasets\inaturalist_dated\2026-06-07\species_counts_canonical.csv` and deficit report were rebuilt with the merged label.
- `D:\TREE\external_datasets\inat_max150\duplicate_photo_review\index.html` was rebuilt.
- Street View accepted samples were moved from the two old dataset folders into:
  `D:\TREE\dataset\Araucaria columnaris(Araucaria heterophylla)`.
- Street View sample `metadata.json` files were rewritten to use the merged species label.
- `hk-tree-labeler/backend/rejected_views.json` old Araucaria buckets were merged into the new species bucket.

Verification snapshot:

- Backend `list_species()` returns `Araucaria columnaris(Araucaria heterophylla)` and not the two old standalone names.
- Master catalog rows: 33417.
- Master catalog missing absolute paths: 0.
- Master catalog old Araucaria target rows: 0.
- Master catalog merged Araucaria rows: 535.
- Master summary:
  - `2026-05-24`: 300 images from original `Araucaria columnaris` and `Araucaria heterophylla`.
  - `2026-06-07`: 235 images from original `Araucaria columnaris` and `Araucaria heterophylla`.
- Dated canonical species count for `2026-06-07`: 153.
- Street View merged dataset samples: 22.
- Street View merged rejected trees/views: 34 trees, 39 views.
- Legacy duplicate visual review:
  - 62 remaining duplicate pairs.
  - 0 Araucaria duplicate rows.
- iNaturalist YOLO pretrain smoke export:
  - Merged class present.
  - Old standalone classes absent.
  - Smoke export classes: 153.

Historical model output folders under `D:\TREE\models` are left as snapshots. Re-export/retrain to produce new class ids with the merged label.
