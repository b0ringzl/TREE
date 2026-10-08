# Ficus Variegata Legacy Cleanup

Date: 2026-06-07

The legacy `2026-05-24` iNaturalist batch had many images stored under `Ficus variegata` whose iNaturalist taxon was actually `Ficus benjamina`.

Cleanup performed:

- User removed the misfiled legacy images from `D:\TREE\external_datasets\inat_max150\Ficus variegata\images`.
- `D:\TREE\external_datasets\inat_max150\Ficus variegata\metadata.jsonl` was rewritten to keep only records whose image files still exist.
- `D:\TREE\external_datasets\inaturalist_master_catalog.csv` and `D:\TREE\external_datasets\inaturalist_master_summary.csv` were rebuilt.
- `D:\TREE\external_datasets\inat_max150\duplicate_photo_review\index.html` was rebuilt.

Verification snapshot:

- Legacy `Ficus variegata` metadata rows: 7.
- Legacy `Ficus variegata` remaining taxon values: 7 `Ficus variegata`, 0 `Ficus benjamina`.
- Legacy `Ficus benjamina` / `Ficus variegata` shared photo ids: 0.
- Full master catalog rows: 33417.
- Full master catalog missing absolute paths: 0.
- Legacy duplicate visual review pairs: 62.
- Legacy duplicate visual review `Ficus benjamina` / `Ficus variegata` rows: 0.

The dated `2026-06-07` batch was checked separately and did not show this Ficus mix-up:

- `Ficus benjamina`: 150 images, all `Ficus benjamina`.
- `Ficus variegata`: 150 images, all `Ficus variegata`.
- Cross-species shared photo ids in the dated batch: 0.
