# Project training safety — persistent user requirement

The user requires a recoverable pre-training backup for EVERY training run.

- Before any training, fine-tuning, classifier-head fitting or resumed training,
  copy the initialization weights and all potentially affected deployed weights
  into a new run-specific backup directory. Do not use hard links for weights.
- Use `tools/training_safety.py` or an equivalent verified backup. Record original
  absolute paths, backup paths, sizes and SHA-256; verify the copies before fitting.
  If backup verification fails, stop before training.
- Snapshot the accepted-data manifest, label mapping, split assignment, training
  configuration, seed and original evaluation results. Never silently include
  rejected/pending human-review images or relabel them using model predictions.
- Every attempt (including a resume) must use a new versioned output directory;
  never overwrite an earlier run, its backups, or production weights in place.
- Evaluate baseline and candidate on the same fixed classification validation/test
  data. Select on validation only; report regressions and per-class metrics honestly.
  Previously inspected test sets are regression sets, not fresh unbiased evidence.
- On 2026-09-20 the user explicitly authorized subsequent accuracy-improvement
  plans and Demo updates without asking for approval each time, provided EVERY
  plan starts with a verified, recoverable backup. This supersedes the earlier
  requirement for separate approval of each model switch. Select improvements
  on validation only, disclose regressions, validate the actual inference path,
  and retain rollback copies before updating the Demo. Do not silently switch
  to an unvalidated candidate or inflate confidence to imply better accuracy.
  Preserve related image/point/fusion encoders and label-map compatibility.
- Report the backup location and whether deployed files remained unchanged.

Herbarium image review uses `human_review.sqlite3` as the source of human decisions.
Image-quality acceptance is not a license grant. Preserve source attribution and
watermarked originals; do not publish downloaded images or experimental weights
without checking the applicable rights and intended distribution.
