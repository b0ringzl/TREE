# TSCMDL WHU-STree Reproduction Sandbox

This directory contains only source code, configuration, tests, and stage
reports for the TSCMDL reproduction. The original WHU-STree dataset remains
read-only at `D:\TREE\lidar data\whu`.

The generated cross-conversation status entry is `CURRENT_STATUS.md`.
Machine-readable experiment paths, metrics, validation states, and SHA-256
values are stored in `experiment_registry.json`. Regenerate both with
`scripts/consolidate_project_status.py` after a completed experiment.

## Working rules

- Work proceeds through the approval gates in `STAGE_GATES.md`.
- Each stage ends with a Chinese report under `reports/`.
- The next stage does not start until the user approves it.
- The `yolo` and `pdf-reading` conda environments must not be modified.
- Heavy derived data belongs under
  `D:\TREE\lidar data\whu\derived\tscmdl`, not in this code directory.
- Raw PLY, NPY, panoramic images, and trajectory files are read-only inputs.
- Git staging, if explicitly approved later, must name individual source files.

## Current state

- Completed: A0, sandbox and reproduction baseline lock.
- Completed: A1, isolated environment creation and PointMLP dependency
  validation.
- Completed: A1b, native Windows/CUDA 12.8 compatibility for the legacy
  `pointnet2_ops` extension. GPU smoke tests passed at 1024 and 8192 points.
- Completed: A2, memory-mapped PLY/NPY trajectory reader with synchronized
  embedded, external-reference, and unlabeled modes.
- Completed: A3, chunked individual-tree profiling and extraction,
  reproducible 8192-point sampling, and reversible unit-sphere normalization.
- Completed: A4, coarse equirectangular projection and an eight-pair Codex
  visual precheck across four tree instances and two nearest panoramas per
  tree; user visual review remains pending.
- Completed: B1, a deterministic balanced 600-tree multimodal dataset for
  three Nanjing/Hong Kong shared species, with full pair validation.
- Completed: B2, a three-class PointMLP point-cloud baseline trained for 40
  epochs, with checkpoints, predictions, curves, confusion matrices, and a
  reproducible artifact validator.
- Completed: B3, a three-class ImageNet-pretrained ResNet50 image baseline
  trained for 300 epochs, with test accuracy 80.00% and macro F1 80.19%.
- Completed: B4a, frozen-backbone feature alignment, gradient, memory, and
  checkpoint-restore smoke validation.
- Completed: B4b, a 300-epoch frozen-backbone TSCMDL fusion run with cached
  aligned features, complete predictions, plots, and independent validation.
- Completed: B4c, validation-locked three-seed regularization selection and
  fusion/image/point modality ablations with post-lock test evaluation.
- Completed: B5, a native-PyTorch PTv2 mode-1 classification encoder with
  cached full-resolution kNN, 8192-point inputs, independent validation, and
  fixed-split baseline comparison.
- Completed: B5b, locked three-seed PTv2 repeatability, probability ensemble,
  prediction agreement, and fixed-road domain diagnostics.
- Completed: B5c, a three-fold road-grouped cross-validation protocol with
  whole-road train/validation/test separation and balanced evaluation subsets.
- Completed: C1a, the full 19-class inventory and exact road-partition
  feasibility audit. No point clouds or images were exported.
- Completed: C1b, a shared-asset plan for the 19-class benchmark track and the
  strict 16-class road-domain track, including deterministic manifests,
  machine validation, and a conservative disk budget. No assets were exported
  and no training was started.
- Completed: C1, all 17,134 shared point-cloud/image pairs, dual-track
  manifests, exhaustive content validation, and a Codex visual precheck;
  user visual review remains pending.
- Completed: C2a, a resumable 4.18 GiB exact-kNN cache for all 17,134
  instances plus a real 19-class PTv2 batch-16 GPU forward, backward, optimizer
  update, and validation-forward smoke test.
- Completed: C2, the full 19-class PTv2 baseline. The validation-selected
  epoch-48 checkpoint reached 69.82% validation accuracy / 68.80% macro F1 and
  79.06% test accuracy / 62.29% macro F1.
- Completed: C2b, independent checkpoint, prediction, probability, metric,
  plot, and class-level diagnostic validation. C3 has not started.

## Run the user-owned visual review

Double-click `START_USER_VISUAL_REVIEW.bat` to open the localhost review UI.
The default scope contains 38 class-stratified C1 samples; the UI can also be
switched to all 17,134 samples. Codex visual prechecks are reference material
only. A formal review is complete only after the user has submitted every
sample in the selected scope and clicked the completion control.

Use `GENERATE_USER_REVIEW_ACTION_LISTS.bat` to produce non-destructive point
and multimodal training manifests, image quarantine lists, alternate-view or
manual-recrop queues, and whole-sample exclusions. Source point clouds and
images are never deleted, moved, or overwritten. The Chinese operating guide
is `reports/点云图像配对_用户视觉复核说明.md`.

## Prepare and validate the C2a PTv2 cache

Create the standard PTv2 view over the C1 shared assets without copying point
clouds or images:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\prepare_c2a_ptv2_dataset_view.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset"
```

Build or resume the split-wise exact-kNN cache:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_c2a_knn_cache.ps1
```

Display a live snapshot from another terminal:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\monitor_c2a_knn_cache.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\c2a_ptv2\knn_memmap_k8"
```

The cache contains `train`, `val`, and `test` memory maps with shape
`[samples, 8192, 8]` and `int32` indices. C2a proved recovery from a controlled
128-sample partial run, validated all cache bytes and indices, and ran one
batch-16 GPU optimization step without writing a model checkpoint.

## Validate the completed C2 full PTv2 baseline

The authoritative received run is stored at:

```text
D:\TREE\lidar data\whu\derived\tscmdl\c2_ptv2\received_24gb_20260803\run
```

Regenerate the readable class diagnostics, then validate the full result:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\analyze_c2_results.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\c2_ptv2\received_24gb_20260803\run" `
  --output-dir .\reports

D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_c2_run.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\c2_ptv2\received_24gb_20260803\run" `
  --cache-index "D:\TREE\lidar data\whu\derived\tscmdl\c2a_ptv2\knn_memmap_k8\index.json" `
  --diagnostics-dir .\reports `
  --output .\reports\C2_validation.json
```

The validator reconstructs the 9,394,097-parameter model, loads both
checkpoints, verifies the local C2a cache hash, recomputes all validation and
test metrics from 4,018 prediction rows, checks the delayed-test policy, and
validates the readable diagnostic files. Use `best.pt` for downstream
inference; `last.pt` is the final recovery checkpoint, not the selected model.

## Plan the C1b dual-track dataset

Generate the manifests and storage budget without exporting point clouds or
images:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\plan_c1b_dual_track.py `
  --inventory "D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\inventory.json" `
  --species-csv "D:\TREE\lidar data\whu\metadata\whu_benchmark_species.csv" `
  --audit "D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\audit\c1a_audit.json" `
  --reference-dataset "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --knn-cache "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\knn_cache_k8.pt" `
  --feature-cache "D:\TREE\lidar data\whu\derived\tscmdl\b4_frozen_features\b2e19_b3e166_center224\features.pt" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan" `
  --benchmark-val-per-class 30 `
  --road-eval-per-class 20 `
  --seed 20260731 `
  --safety-factor 1.25
```

The 19-class benchmark manifest references all 17,134 instances and retains
all official reference instances for test. The 16-class road-domain manifests
use only development data and keep train, validation, and test roads disjoint
in every run. Both tracks reference one physical asset store, so they do not
duplicate point-cloud or image files. The conservative shared-asset estimate
is 5.72 GiB including a 25% safety margin.

## Smoke-test the C1 shared-asset exporter

The C1c smoke plan selects one development sample from every benchmark class
while minimizing the number of raw trajectories. Run one controlled partial
export first:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\export_c1c_shared_asset_smoke.py `
  --dataset-root "D:\TREE\lidar data\whu" `
  --inventory "D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\inventory.json" `
  --shared-assets "D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan\shared_assets.csv" `
  --benchmark-manifest "D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan\benchmark_19.csv" `
  --road-manifest "D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan\road_domain_16_samples.csv" `
  --classes "D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan\classes_19.json" `
  --output-root "D:\TREE\lidar data\whu\derived\tscmdl\c1c_shared_asset_smoke" `
  --samples-per-class 1 --max-trajectories 3 `
  --max-new-trajectories 1 --seed 20260731
```

Run the same command again without `--max-new-trajectories` to resume and
finish. The exporter verifies hashes and modification times before reusing a
completed trajectory. Validate all point/image contracts and render the
19-class visual contact sheet with:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_c1c_shared_asset_smoke.py `
  --smoke-root "D:\TREE\lidar data\whu\derived\tscmdl\c1c_shared_asset_smoke" `
  --output "D:\TREE\lidar data\whu\derived\tscmdl\c1c_shared_asset_smoke\validation.json"
```

The completed smoke set contains 19 point/image pairs from three development
trajectories. Eight pairs from the controlled partial run were verified and
reused unchanged. The smoke output is about 7.6 MiB; it is not a full export
or a training dataset.

## Export and validate the full C1 shared dataset

Run or resume the 17,134-sample export:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_c1_full_export.ps1
```

The exporter writes one atomic checkpoint shard per trajectory. It uses four
image workers by default while keeping trajectory extraction sequential.
Monitor it from another terminal:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\monitor_c1_export.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset"
```

Perform full content validation, including every point archive and JPEG:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_c1_full_validation.ps1

D:\TREE\envs\whu-tscmdl\python.exe .\scripts\monitor_c1_validation.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset"
```

The completed dataset contains 17,134 point files, 17,134 image files, 110
trajectory shards, a 19-class benchmark manifest, and a three-run 16-class
road-domain manifest. Physical shared assets use 3.896 GiB and the complete
directory uses 4.005 GiB. This stage does not create training caches or model
checkpoints.

## Build and audit the C1a full inventory

Build the resumable inventory for benchmark labels 0 through 18:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\build_b1_inventory.py `
  --dataset-root "D:\TREE\lidar data\whu" `
  --trajectory-manifest "D:\TREE\lidar data\whu\metadata\trajectory_manifest.csv" `
  --output "D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\inventory.json" `
  --chunk-size 2000000 `
  --label 0 --label 1 --label 2 --label 3 --label 4 --label 5 `
  --label 6 --label 7 --label 8 --label 9 --label 10 --label 11 `
  --label 12 --label 13 --label 14 --label 15 --label 16 `
  --label 17 --label 18
```

Audit exact per-class and joint whole-road split feasibility:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\audit_c1a_road_coverage.py `
  --inventory "D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\inventory.json" `
  --species-csv "D:\TREE\lidar data\whu\metadata\whu_benchmark_species.csv" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\audit" `
  --folds 3 --eval-per-class 30 --seed 20260730
```

The inventory contains all 17,134 published instances. A strict three-fold
30-sample-per-class road-domain protocol is jointly feasible for 13 classes.
A three-fold 20-sample protocol is feasible for 16 classes. All 19 classes
become feasible together only at two folds and 10 samples per class.

## Build the B5c road-grouped protocol

The protocol uses only non-official-test instances for cross-validation.
Official reference trajectories remain withheld for a separate
benchmark-style evaluation. Each cross-validation run keeps complete roads in
exactly one of train, validation, or test:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\plan_road_group_cv.py `
  --inventory "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\inventory.json" `
  --classes "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset\classes.json" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\b5c_road_group_cv" `
  --folds 3 --eval-per-class 30 --min-train-per-class 30 `
  --validation-road-fraction 0.2 --seed 20260730
```

The primary validation and road-unseen test subsets contain 30 deterministic
samples per class. Full assigned validation/test pools are retained for
supplementary metrics. The protocol does not itself retrain a model.

## Repeat and diagnose the B5 PTv2 baseline

Run the two additional seeds sequentially. The command reuses a completed run
when its artifacts already pass validation, and resumes from `last.pt` when a
run was interrupted:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\run_b5b_repeats.py `
  --baseline-run "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728" `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --knn-cache "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\knn_cache_k8.pt" `
  --output-root "D:\TREE\lidar data\whu\derived\tscmdl\b5b_ptv2_repeats" `
  --seeds 20260729 20260730
```

Aggregate the baseline and repeat runs, then diagnose road composition and
point-structure differences:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\analyze_b5b_domain.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --run "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728" `
  --run "D:\TREE\lidar data\whu\derived\tscmdl\b5b_ptv2_repeats\run_batch16_seed20260729" `
  --run "D:\TREE\lidar data\whu\derived\tscmdl\b5b_ptv2_repeats\run_batch16_seed20260730" `
  --output ".\reports\B5b_validation.json" `
  --sample-csv ".\reports\B5b_sample_diagnostics.csv" `
  --metrics-plot ".\reports\B5b_seed_metrics.png" `
  --roads-plot ".\reports\B5b_road_domain.png" `
  --structure-plot ".\reports\B5b_point_structure.png"
```

## Train, monitor, and validate the B5 PTv2 baseline

Build the deterministic full-resolution neighbour cache once:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\cache_b5_ptv2_knn.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --output "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\knn_cache_k8.pt" `
  --progress-every 10
```

Run the locked B5 configuration:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\train_b5_ptv2.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --knn-cache "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\knn_cache_k8.pt" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728" `
  --epochs 200 --batch-size 16 --eval-batch-size 16 `
  --learning-rate 0.001 --min-learning-rate 0.00001 `
  --weight-decay 0.05 --label-smoothing 0.1 --patience 40 `
  --seed 20260728 --workers 0 --progress-every 27
```

View live progress from a second Codex Terminal. The monitor is read-only and
exits automatically after a completed run:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\monitor_training.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728"
```

Validate the formal artifacts and reproduce the four-model comparison:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_b5_run.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728" `
  --output ".\reports\B5_validation.json"

D:\TREE\envs\whu-tscmdl\python.exe .\scripts\compare_b5_baselines.py `
  --b2-run "D:\TREE\lidar data\whu\derived\tscmdl\b2_pointmlp\run_batch3_seed20260714" `
  --b3-run "D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723" `
  --b4-run "D:\TREE\lidar data\whu\derived\tscmdl\b4_tscmdl\run_batch8_l2_lr001_seed20260727" `
  --b5-run "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728" `
  --output ".\reports\B2_B3_B4_B5_prediction_comparison.json" `
  --plot ".\reports\B2_B3_B4_B5_metric_comparison.png"
```

## Run and validate the B4c suite

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\run_b4c_regularization_ablation.py `
  --feature-cache "D:\TREE\lidar data\whu\derived\tscmdl\b4_frozen_features\b2e19_b3e166_center224\features.pt" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\b4c_regularization_ablation\run_3seed_20260727" `
  --epochs 50 --patience 20 --batch-size 8 --eval-batch-size 32 `
  --seeds 20260727 20260728 20260729 --workers 0

D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_b4c_run.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b4c_regularization_ablation\run_3seed_20260727" `
  --output ".\reports\B4c_validation.json"
```

## Train and validate the B4b fusion model

Cache the deterministic frozen features once:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\cache_b4_frozen_features.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --b2-checkpoint "D:\TREE\lidar data\whu\derived\tscmdl\b2_pointmlp\run_batch3_seed20260714\best.pt" `
  --b3-checkpoint "D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723\best.pt" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\b4_frozen_features\b2e19_b3e166_center224" `
  --batch-size 8 --workers 0
```

Train the fusion head and validate all formal artifacts:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\train_b4_tscmdl.py `
  --feature-cache "D:\TREE\lidar data\whu\derived\tscmdl\b4_frozen_features\b2e19_b3e166_center224\features.pt" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\b4_tscmdl\run_batch8_l2_lr001_seed20260727" `
  --epochs 300 --batch-size 8 --eval-batch-size 32 `
  --learning-rate 0.01 --normalization l2 --no-amp `
  --checkpoint-every 5 --seed 20260727 --workers 0

D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_b4_run.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b4_tscmdl\run_batch8_l2_lr001_seed20260727" `
  --output ".\reports\B4_validation.json"
```

## Run and validate the B4a fusion smoke

The smoke run uses the B2 and B3 best checkpoints, freezes both backbones, and
checks the paper-style `1024 + 1024 = 2048` fusion interface. Its one-batch
metrics are diagnostics only and must not be reported as model performance.

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\smoke_b4a_fusion.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --b2-checkpoint "D:\TREE\lidar data\whu\derived\tscmdl\b2_pointmlp\run_batch3_seed20260714\best.pt" `
  --b3-checkpoint "D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723\best.pt" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\b4a_fusion_smoke\run_seed20260727" `
  --steps 2 --batch-size 2 --workers 0

D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_b4a_run.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b4a_fusion_smoke\run_seed20260727" `
  --output ".\reports\B4a_validation.json"
```

## Train and validate the B2 PointMLP baseline

Run the 40-epoch baseline from a new output directory:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\train_b2_pointmlp.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\b2_pointmlp\run_batch3_seed20260714" `
  --epochs 40 `
  --batch-size 3 `
  --eval-batch-size 8 `
  --learning-rate 0.009375 `
  --min-learning-rate 0.00046875 `
  --patience 25 `
  --seed 20260714 `
  --workers 0 `
  --progress-every 5
```

Validate checkpoints, predictions, metrics, and plots:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_b2_run.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b2_pointmlp\run_batch3_seed20260714" `
  --output ".\reports\B2_validation.json"
```

The completed run selected epoch 19 by validation macro F1. Its test accuracy
is 66.67% and test macro F1 is 66.11%. See the Chinese B2 stage report for the
resume boundary, split limitations, and per-class results.

## Build and validate the B1 dataset

Build the resumable all-trajectory inventory:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\build_b1_inventory.py `
  --dataset-root "D:\TREE\lidar data\whu" `
  --trajectory-manifest "D:\TREE\lidar data\whu\metadata\trajectory_manifest.csv" `
  --output "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\inventory.json"
```

Create the deterministic `140/30/30` per-class split plan:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\plan_b1_dataset.py `
  --inventory "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\inventory.json" `
  --species-csv "D:\TREE\lidar data\whu\metadata\whu_hk_shared_species.csv" `
  --output-json "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\selection_plan.json" `
  --output-csv "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\selection_plan.csv"
```

Export and validate all 600 point-cloud/image pairs:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\export_b1_multimodal_dataset.py `
  --dataset-root "D:\TREE\lidar data\whu" `
  --selection-plan "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\selection_plan.json" `
  --output-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset"

D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_b1_dataset.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --output "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset\validation.json"
```

Class indices are local to this three-class experiment: class 0 is WHU label
1 (`Cinnamomum camphora`), class 1 is WHU label 2 (`Lagerstroemia indica`),
and class 2 is WHU label 13 (`Magnolia grandiflora`).

## Build A4 panorama pairing previews

Create raw 768x512 crops, point overlays, panorama previews, and a manifest
from the selected A3 samples:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\prepare_a4_pairing_preview.py `
  --dataset-root "D:\TREE\lidar data\whu" `
  --road 26 `
  --trajectory 1 `
  --sample-dir "D:\TREE\lidar data\whu\derived\tscmdl\a3_samples\26_1" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\a4_pairing\26_1" `
  --views-per-tree 2
```

The public trajectory columns are treated as `roll, pitch, heading`; this was
inferred by comparing all three orientation columns with consecutive camera
path bearings. The projection remains a coarse prototype because the local
public data package does not include a complete camera lever-arm/extrinsic
calibration model.

## Profile and prepare tree instances

Profile one labeled trajectory without exporting its points:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\profile_whu_instances.py `
  --dataset-root "D:\TREE\lidar data\whu" `
  --road 26 `
  --trajectory 1
```

Prepare explicitly selected 8192-point samples under the derived-data tree:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\prepare_a3_tree_samples.py `
  --dataset-root "D:\TREE\lidar data\whu" `
  --road 26 `
  --trajectory 1 `
  --tree-id 3959 --tree-id 3963 --tree-id 3966 --tree-id 4015 `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\a3_samples\26_1"
```

Raw labels greater than 17 are preserved in each archive and mapped to
benchmark class 18 (`Others`). Instances with raw label -1 are excluded from
classification samples.

## Inspect one WHU-STree trajectory

The inspector reads only the PLY header plus bounded samples from the beginning
and end of a trajectory. It automatically uses a matching
`reference_data/<road>_<trajectory>.npy` file when present.

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\inspect_whu_trajectory.py `
  --dataset-root "D:\TREE\lidar data\whu" `
  --road 02 `
  --trajectory 1
```

Run the synthetic format and boundary tests with:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe -m unittest discover -s .\tests -v
```

## Monitor any standard training run

Every training stage writes the same atomic progress artifacts:
`run_config.json`, `training_history.json`, `last.pt`, and `train.log`.
The shared monitor is read-only and works with B2, B3, and future runs:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\monitor_training.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723"
```

It refreshes every three seconds and shows epoch progress, recent metrics,
best validation score, elapsed time, ETA, artifact freshness, and live GPU
utilization. Use `--once` for a single snapshot or `--json` for integration
with another dashboard. It exits automatically when `final_metrics.json`
reports a complete run; pass `--no-exit-on-complete` to keep watching.

## Train and validate the B3 ResNet50 image baseline

The B3 image branch follows the paper protocol: ImageNet-pretrained ResNet50,
random 224x224 training crops, horizontal flips, ImageNet normalization, batch
size 8, SGD at learning rate 0.1, and cosine annealing over 300 epochs.

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\train_b3_resnet50.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --output-dir "D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723"

D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_b3_run.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723" `
  --output ".\reports\B3_validation.json"
```

## Rebuild PointNet2 on this machine

The installed Visual Studio 2026 compiler is newer than CUDA 12.8 officially
supports. The compatibility switch is therefore explicit rather than enabled
by default:

```powershell
& .\scripts\build_pointnet2_windows.ps1 -AllowUnsupportedCompiler
```

Run the GPU smoke tests after rebuilding:

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\validate_pointmlp_gpu.py
```

The upstream source changes are reproducible from
`patches/pointnet2_ops_windows_cuda128.patch`.
