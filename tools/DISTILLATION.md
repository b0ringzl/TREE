# Isolated BioCLIP-to-YOLO experiments

`run_distillation.py` uses the original YOLO11s-cls architecture and physical
run-specific backups from `training_safety.py`. It never deploys a candidate.
The 148 original logits are grouped to 147 user labels using logsumexp to merge
Ficus microcarpa and Ficus benjamina. The original output-index mapping is checked
against dataset class folders, not guessed from English display names.

Use the existing `yolo` environment. OpenCLIP 3.3.0 and ftfy 6.3.1 are installed
with `--target external_datasets/distillation_runtime --no-deps`; this does not
change the Demo environment. BioCLIP safetensors are cached under
`external_datasets/distillation_models/bioclip25`, pinned to the recorded official
revision. Experimental data, models and reports are ignored by Git under
`outputs/distillation/`.

Stages (each accepts `--out <versioned directory>`):

1. `--stage audit`: creates a new directory; checks every image SHA, source-group
   splits, model class order and current read-only SQLite human decisions.
2. `--stage features`: frozen YOLO features and matched-preprocessing baseline.
3. `--stage teacher`: frozen BioCLIP, prompt banks, raw scores and embeddings;
   chunk caches permit inference continuation after interruption.
4. `--stage calibrate`: planning-only prompt and temperature selection; separate
   selection-set and small street diagnostic results. Never uses test for tuning.
5. `--stage screen`: actual GT-only and six KD frozen-head fits with equal budgets.
6. `--stage full`: smoke fit, matched GT-only continuation and selected KD full
   backbone fit. Each has new independent backup and output directory.
7. `--stage final`: fixed historical regression comparison after final_lock.json.
8. `--stage report`: refreshes honest completion/blocked status and source hashes.

`--stage pipeline` runs stages 2–7 after a successful audit. A dependent stage
failure stops its downstream stages, writes FAILURES.md, and leaves a partial
report; it cannot claim unavailable KD results. Previously completed inference
caches remain usable. Re-running training stages starts new fits with new backups,
not silent in-place continuation. Full checkpoints contain epoch-level recovery
state. `--stage resume --resume-checkpoint <incomplete last.pt>` checks the old
backup and dataset/class hashes, makes a new independent backup, and restores
the epoch-boundary optimizer/scheduler/scaler/RNG state in a new fit directory.
Completed checkpoints are not silently retrained as a "resume". Resume does not
automatically change the previously locked candidate selection.

## Important limits

- Same spatial crop for student/teacher. No cached global-feature target is paired
  with a different random resized crop.
- Calibration T is applied to grouped raw teacher logits. KD T separately softens
  `log(calibrated teacher probability)`; KL direction is teacher || student.
- Ground-truth labels are never changed by teacher predictions.
- Source observation groups are not guaranteed single-tree identities. VMMS
  conservative overlap blocks are not identity labels. Multi-view training is
  withheld if independently verified identities are insufficient.
- DINOv3 is optional and is blocked when official weights require unavailable
  authorized access. No mirrors or access bypass are used.
- Web test was historically inspected; teacher pretraining overlap is unknown.
- A small VMMS diagnostic cannot establish general field accuracy.
- Pending/rejected herbarium images are excluded; attribution/watermarks remain.

Tests: `python -m unittest discover -s tools -p test_distillation_core.py -v`.
