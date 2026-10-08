# SpaceScan Trees — Android offline preview

Native Java Android application with bundled ONNX Runtime CPU inference. No WebView,
Python server, remote recognition endpoint, account, telemetry or INTERNET permission.
This is a **mobile feasibility build**, not a port of the current desktop BioCLIP teacher.

## Included

- Local photo selection and external camera capture, with bounded decoding and EXIF orientation handling.
- Manual polygon target selection, undo vertex, apply, cancel, restore original.
- Real on-device inference, scientific + Traditional Chinese top-three candidates.
- Image-quality heuristics, explicit model-input centre-square preview.
- Private local result history (100 records), JSON export, integrity-checked model, synthetic inference self-test.
- One evaluation APK for Android 8+ (API 26), arm64-v8a / x86_64. Not 32-bit Android or iOS.

## Important limitations

The compact model is YOLO11s-cls with an already fitted, validation-locked KD head:
147 displayed labels, 22.6 MB FP32 graph. No fitting was performed for this app.
The current desktop's 148-class BioCLIP 2.5 + MLP and conservation specialists were
not replaced or silently labelled as mobile models.

Historical planning regression (2,640 images), with the original preprocessing:
Top-1 57.39%, Top-5 80.68%, macro-F1 0.5475. This is not fresh street-scene or phone
accuracy. The scores next to species are uncalibrated model probabilities, not accuracy.
Aquilaria sinensis recall: 36.36% (22 images); Artocarpus hypargyreus recall: 33.33%
(6 images). This build is NOT sufficient evidence of conservation-specialist performance.

No point-cloud fusion, automatic tree detector or panorama reprojection is included.
Wodyetia bifurcata, Rhodoleia championii and Keteleeria fortunei are not output classes.
The historical Ficus pair is combined; the displayed Ficus microcarpa name does not
establish discrimination from Ficus benjamina.

Preprocessing is EXIF/RGB → shortest-side antialiased bicubic resize 224 → centre
224 square → RGB /255. The preview explicitly marks that square. Long panoramas
need target cropping first. Polygon selection does not eliminate all perspective distortion.

## Build

Use JDK 17, Gradle 8.9, Android SDK 35 and AGP 8.7.3. Build dependencies are pinned in
the Gradle files; building requires Internet, running the APK does not.

On the current workstation, run `powershell -File spacescan_android/build.ps1`.
Else open this project in Android Studio, point to an installed SDK, and run
`gradle assembleDebug`. The model and JSON assets must come from the supplied source
package, or be reproduced by the explicitly versioned export/validation scripts.
Do not rerun the export on existing versioned directories.

For development-device diagnostics:

```text
adb install app-debug.apk
adb shell am start -n org.spacescan.trees.offline/.MainActivity --ez selftest true
adb shell run-as org.spacescan.trees.offline cat files/selftest_result.json
```

This APK has a development signing certificate, not a production/app-store key.
Never disable device security controls wholesale to install it. Only authorize
installation from your chosen file manager if you decide to test this package.

## Data, licences and rollback

Photographs remain in private local cache; Clear removes the app's current image
copy, not the source photo. Historical records contain predictions and a tensor hash,
not image files. Exports go only to the destination selected by the user. Android's
camera/file-picker apps have their own behaviour and permissions.

No training/reference photographs are bundled. License texts and upstream notices
are under `app/src/main/assets`. Application source: AGPL-3.0-or-later; ONNX Runtime:
MIT; AndroidX: Apache-2.0; Pillow conventions: HPND notice included. Project-weight
public redistribution still requires a training-data rights review. This delivery is
for the user's local evaluation, not publication or app-store release.

The desktop image/point/fusion artifacts were physically backed up before export:
`outputs/SpaceScan_Android_Offline_20260918_v1/pre_training_backup` in the main project.
Their hashes are reverified during packaging. No desktop rollback is needed because
no deployed model is changed.
