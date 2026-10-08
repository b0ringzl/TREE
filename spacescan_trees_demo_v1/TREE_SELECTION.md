# Image-only target selection — historical v1, 2026-09-10

**Superseded for spherical panoramas:** v2 now reprojects the original 2:1 sphere
into 16 overlapping pinhole views BEFORE detection. No raw ERP detections/tiles
are used in that branch. Photo mode retains the original path. See `panorama.py`
and `outputs/spacescan_panorama_audit_20260910_153714/REPORT.md` for geometry,
paired diagnosis, current limits, and verification. The v1 record below is kept
as implementation history, not the description of the current panorama branch.

## Flow

The home page stays empty until the user uploads or explicitly loads a local test.
Whole-tree/crown → local detection → confident central dominant box automatically
selected, otherwise a modal displays up to 12 actual candidate crops → explicit
confirmation → original native-resolution crop → existing BioCLIP 2.5 on port 8039.
Leaf/flower/fruit/bark selection explicitly bypasses the whole-tree detector.
Image+cloud is unchanged and remains disabled pending real fusion integration.

Cancel/Escape does not run classification. Controls lock while detecting, selecting
or classifying. No-target and detector-failure cases do not silently classify the
whole panorama. “Choose another tree” reopens selection on the original image.

## Models and fixed policy

- YOLO11m-seg original project checkpoint (`yolo11m_seg_1024_20260901_104757/weights/best.pt`), input 1024, tree class only, score >=0.40, IoU NMS 0.5.
- For width/height >=1.8: full image + 3 overlapping horizontal tiles. Source for detection is bounded to 4096; final crops are always from the uploaded native pixels.
- Existing local Grounding DINO Base supplements panoramas and cases with no YOLO output; prompt `tree. palm tree.`, score >=0.40. This is a new inference pipeline, not a newly trained species model.
- Existing frozen CLIP ViT-B/32 compares tree/background prompts on letterboxed and center-cropped views. If background wins in both, withhold the candidate. This conservative heuristic can also miss real trees.
- Reject tiny boxes (<48px wide or <72px high), nearly entire-panorama boxes (>85% area), duplicates and coarse containing boxes supported by substantial separate detections. A tightly framed non-panorama tree may legitimately fill the photo. Small background detections alone must not delete a large foreground tree.
- Automatic central target: largest retained box, center X within 0.12 of image center; width >=0.12 (<=0.80 for panoramas), height >=0.30, covering the center Y; area >=1.6 times the runner-up. These are heuristics, not calibrated reliability or guaranteed single-instance segmentation.
- Each crop adds 4% context per side and preserves aspect ratio. Classifier padding never center-crops the chosen rectangle. Detection and species scores remain distinct, uncalibrated outputs; no fabricated confidences.

## Data / model safety

Uploads are processed in memory, not added to training or reference databases.
Signed selection tokens bind image SHA-256, coordinates, detector score and weight
hashes; expire after 30 minutes or server restart. The identify endpoint verifies
the token and original image digest before cropping. The exported result includes
these records and the selection policy; embedded previews are omitted from JSON.
Detector calls are serialized; simultaneous work is rejected with retry guidance.
No new model downloads, training, fitting or weight modification was performed.
No pre-training weight backup is required for this inference-only change; existing
teacher weights and original demos remain unchanged.

## Verification and honest limits

24 unit tests pass (existing preview/reference tests + 8 selection tests): source
guarding, invalid uploads, EXIF orientation, native crop/padding, token mismatch,
expiry/tampering, central/ambiguous routing, duplicate/group/tiny filtering, busy
handling and forwarding the actual selected crop.

Real inference records: `outputs/spacescan_tree_selection_20260910_152011/`.

- Palm panorama: 8 candidate crops >=0.40; popup required. Inspection finds some
  near palms still share a crop. This is not eight verified single trees.
- Reported bridge frame: three architectural candidates vetoed; one actual tree
  region at the right edge retained by the supplement. Other real trees are missed.
- Roof-only crop: no target; no automatic species classification.
- Botanical detail: no whole-tree target; user chooses its organ to classify.
- Tight, centered crop `palms_1.jpg`: live `/api/detect-trees` returned `auto`, score
  0.630, native crop 825×898 after fixing rejection of nearly full-frame trees in
  non-panorama photos. The classifier is unchanged.

Browser end-to-end verification on port 8043: initial empty home; explicitly load
panorama; real eight-crop popup; cancel without classification; retry; select crop
4; identify; existing top-three bilingual result/reference layout rendered with
native input 1514×1811. Its top model score was only 40.3%, and the application
retained the ambiguity warning rather than claiming a correct species. The result
page's “Choose another tree” action returns to detection on the original upload.

Earlier smoke runs are retained. The first run exposed over-aggressive containing-box
removal when only tiny background trees existed inside a foreground group box; the
policy was corrected and rerun. These inspected cases are regression checks, not
an independent precision/recall evaluation. Threshold 40% does not eliminate false
positives, group boxes or panorama seam problems.
