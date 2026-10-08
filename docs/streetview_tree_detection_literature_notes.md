# Street View Tree Detection Paper Notes

Source PDF:
`C:\Users\57680\Desktop\树种识别\论文\街景\利用深度学习和谷歌街景图像自动检测和估算城市街头树种类和剖面估计方法.pdf`

Working text extract:
`D:\TREE\tmp\pdfs\streetview_tree_detection_paper.txt`

## Paper

Choi et al. (2022), *An automatic approach for tree species detection and profile estimation of urban street trees using deep learning and Google Street View images*, ISPRS Journal of Photogrammetry and Remote Sensing.

## Useful Method Summary

The paper builds a street-tree inventory pipeline from Google Street View panoramas:

1. Download Google Street View 360 images with geolocation, capture date, heading, and camera metadata.
2. Filter out images without usable trees, very young/unrecognizable trees, or leafless/poor-condition examples.
3. Manually annotate visible individual trees in YOLO format.
4. Train a YOLO detector to locate individual trees and classify species.
5. Run semantic segmentation to isolate vegetation and ground/surface pixels.
6. Combine detection boxes, segmentation masks, camera height, field of view, and panorama geometry to estimate tree height, DBH proxy, crown width, and location.

Important reported results:

- YOLOv3 detector: mAP 0.564, precision 0.727, recall 0.634 at IoU 0.5.
- High confidence threshold favored precision over recall.
- Species with strong morphology or phenology were easier to classify.
- Tree profile estimation worked much better on "ideal" images where the tree base and stem were visible and not heavily occluded.
- DBH estimation was much less reliable than height unless image quality was good.

## What Applies To This Project

Our current project already has three pieces that match the paper:

- Google Street View collection by tree coordinate.
- Manual tree-box review in the labeler UI.
- YOLO-style `.txt` label files saved beside accepted street-view images.

The highest-value improvement is therefore not another whole-image classifier first. It is to train a street-view tree detector from our reviewed labels:

```text
Street View image -> tree bounding boxes -> crop/instance-level species prediction -> optional profile estimation
```

This should reduce the noise from whole-image classification, because each image may contain multiple trees, cars, buildings, signs, shadows, and non-target vegetation.

## Recommended Project Roadmap

Phase 1 - Detection dataset export:

- Export accepted `dataset/<species>/<tree_id>/*.jpg` and matching `.txt` files into YOLO layout.
- Rewrite class ids from local single-class labels to global species ids.
- Keep parent-species label merging for variants.
- Produce `data.yaml`, train/val manifests, and a JSON report.

Phase 2 - Train detector:

- Use YOLOv8/YOLO11 or another modern detector instead of YOLOv3.
- Start with parent-species labels.
- Evaluate mAP@0.5, precision, recall, per-class AP.

Phase 3 - Classifier refinement:

- Run detector on street-view images.
- Crop detected tree boxes.
- Use the existing MobileNetV3 parent-label classifier on cropped trees.
- Compare crop-based classification against current whole-image classification.

Phase 4 - Quality filters:

- Score detected images for visibility of tree base, overlap, box size, and confidence.
- Prioritize "ideal" tree images for profile estimation and model training.

Phase 5 - Optional profile estimation:

- Add segmentation only after detection works.
- Estimate height first; treat DBH as experimental because the paper shows it is much more sensitive to occlusion and segmentation errors.

## Immediate Implementation Decision

Add a YOLO export tool that turns existing reviewed street-view samples into a clean training dataset under `D:\TREE\models`, without modifying `D:\TREE\dataset`.
