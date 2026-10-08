# SpaceScan Trees — Android offline v0.4

Native Android app (Java, API 26+) using the deployed SSTv1-H120 BioCLIP 2.5 teacher and three adapted heads. The class order, 148 raw probabilities, prompt bank, encoder stages, and temperature are retained. H120 replaces only the three classifier heads in the final ONNX stage; no training or fitting was performed for this port.

The app uses two equal-weight views of one image for original photos and manual polygon crops, matching the current web H120 recognition policy. User-confirmed EfficientSAM-Ti crops use a tight rectangle of original pixels with one classifier pass and no grey mask fill. This is a selection aid, not a species classifier; web and Android SAM mask proposals may differ. Android has no automatic tree detector, spherical panorama correction, or point-cloud fusion.

There are 148 raw model classes. For display only, three Bauhinia labels are summed as `Bauhinia group / 洋紫荊`, yielding 146 visible classes/groups. The in-app offline catalog has 147 licensed reference photos covering all raw classes except `Photinia benthamiana`; every photo retains its original bytes, attribution, license, and source page. References are not training images and are not confirmed identifications of a user's photo.

If the highest displayed group score is below 14%, all results are below 14%; the UI says the species is undetermined and suggests another angle. This threshold is an uncalibrated UX prompt, not a calibrated probability or species-accuracy guarantee. Raw probabilities remain in exported JSON.

The separate local model ZIP is imported once through Android's document picker. All files are checked against the bundled size/SHA-256 manifest; incomplete imports are not activated. The application has no INTERNET permission, server, or analytics. It uses sequential ONNX Runtime CPU FP32 inference to bound memory. The v0.4 debug-signed APK retains the official package ID and signing certificate, permitting upgrade from v0.2; the older model pack remains in app storage until separately removed.

Verification: Gradle build, emulator self-test, emulator classification of one Terminalia and one Aquilaria image, tight unmasked SAM crop, offline catalog, and low-score UI tests passed. Fixed-photo H120 ONNX-versus-desktop top-1 agreement was 2/2 (maximum raw-score drift 0.0094 when RGB inputs were held fixed). These are inspected conversion regressions, not an independent accuracy estimate. Android camera and ARM64 physical-phone performance remain unmeasured.

Build with `build.ps1` and the local Android toolchain. The export and gallery scripts are `tools/export_android_h120_tail_20260927.py` and `tools/package_android_reference_gallery_20260927.py`. The app source contains no redistributed adapted model pack. See the versioned delivery folder for installation and rights notes.
