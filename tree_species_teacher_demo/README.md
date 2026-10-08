# Tree Atlas — teacher image Demo

## Active deployment: 148 classes (2026-09-08)

User-approved expansion adds Wodyetia bifurcata to both modes. Original encoder,
148-output MLP checkpoint mappings, and three-head ensembling remain unchanged.
147 existing merged labels plus the new zero-shot candidate give 148 displayed
classes. The new class logit is added before each softmax; the experimental
two-row palm correction is NOT loaded. No weights were rewritten or retrained.
Earlier 147-class metrics below describe the prior scope. Expanded baseline web
regression: 90.23% Top-1; six small accepted palm diagnostic images were correct,
not evidence of stable 100% accuracy. Silver leadtree remains outside this scope.
Pre-switch source copies and hashes:
`outputs/distillation/teacher_deploy_148_20260908_v1/pre_training_backup/`.
Rollback requires restoring the source files named by that manifest and restarting
8039; teacher/head weight files do not change. Earlier training/inference tools
continue to use the original `Engine.classes` checkpoint mapping.

Independent local image-only app: http://127.0.0.1:8039/
Original multimodal app remains at http://127.0.0.1:8037/ and is not modified.

Start by double-clicking `start.cmd`. It uses the existing `yolo` Python environment
and CUDA GPU. Close that launcher window to stop this version. The development
launch may already be active; do not start a duplicate on port 8039.

## Inference

- Original teacher: frozen BioCLIP 2.5 ViT-H/14 with existing taxonomy prompts.
- Adapted teacher (default): the same encoder plus the three previously selected
  residual MLP heads. This is NOT the failed YOLO distilled student.
- Both modes share one image encoding; switching mode updates the ranking without
  rerunning inference. 148 original outputs collapse to 147 labels, merging the
  two previously specified Ficus classes via logsumexp before softmax.
- Uses the locked prompt, temperature 1.8877486253633864, BF16 encoder, and stored
  FP16-rounded image features for compatibility with the trained heads.
- Checkpoint SHA-256, prompt mapping and head class compatibility are verified
  at service startup. All modules are frozen/eval, with no fitting path.
- EXIF-correct RGB, shortest side 224, bicubic resize, center crop 224. The actual
  crop is shown. Upload a single-tree crop; this is not panorama detection.
- Uploads are not retained after each request or sent to external services.
  The HTTP framework may spool large uploads to temporary disk storage while
  processing; its upload handle is closed immediately after reading. No images or
  weights are served by the static file mount. Do not bind publicly without an
  access-control and upload-rate review.

## Evidence and limitations

Scores are model ranking scores, not guaranteed correctness. Input checks are
heuristics, not plant detection. The interface offers comparison, top-two margin,
actual crop, and quality warnings; it does not invent botanical explanations.
Untranslated species retain scientific names rather than fabricated common names.
The existing merged Araucaria class definition is retained.

The historical web regression set (5,146 images) measured 84.05% original teacher
and 90.30% adapted Top-1. These are not fresh street-scene accuracy estimates.
The small field diagnostic was only 5/13 versus 6/13 and includes correlated tree
views. Closed-set 147-class ranking may force an incorrect species for absent
classes, including Wodyetia bifurcata and Leucaena leucocephala. Use the original
multimodal Demo when its relevant specialized model is needed.

Model assets and existing training backups stay in
`outputs/distillation/bioclip_probe_20260908_0227/runs/*/pre_training_backup/`.
This app adds inference only; no new training or new training backup is needed.
It reads weights in place, never overwrites or republishes them.

## Verification

From the project root with the existing Python environment:

```
python -m unittest tree_species_teacher_demo.test_teacher_demo -v
python tree_species_teacher_demo/verify_live.py
```

The live test requires both services and checks one real image against the prior
offline result, both modes, invalid input, the original app health, and protected
file hashes. It is an integration test, not an accuracy evaluation.
