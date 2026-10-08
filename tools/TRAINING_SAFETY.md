# Training backups and rollback / 訓練備份與回退

User requirement recorded in root `AGENTS.md`: back up before every training
attempt, including resumed training and fitted classifier heads.

## Current image training entry

`full_image_species_baseline/scripts/train_yolo11_cls.py` invokes
`backup_before_training` before constructing/training the model. A local initial
checkpoint is mandatory. Missing inputs, copy corruption, or a reused backup
directory abort the attempt. Older point-cloud/head trainers must use the same
guard before future runs; they were not executed or modified in this trial.

Each run stores `pre_training_backup/backup_manifest.json` with physical copies
of the initialization and deployed model files, original paths, sizes, SHA-256,
and training arguments. Copies are not hard links. Training output lives beside
the backup, never at the original deployed checkpoint path.

## Safe handoff

1. Verify the backup hashes and compare baseline/candidate classification metrics
   on identical fixed data. Keep the reviewed-data snapshot and class mapping.
2. A candidate does not become production automatically, even if one aggregate
   score improves. Check per-class regressions and encoder/head compatibility.
3. The original Demo requires **no rollback when it was never switched**.
4. If the user later explicitly requests a switch or restoration, resolve exact
   original paths from the manifest, verify the backup, preserve the version
   being replaced, and switch only that authorized model/configuration set.
   Do not restore unrelated files or mix incompatible encoder/head versions.

The current helper is `tools/training_safety.py`; call
`verify_backup(path, verify_sources=True)` for a read-only check that both the
copies and original deployment remain identical. `verify_sources=False` checks
backup integrity after an intentional, authorized deployment change.
