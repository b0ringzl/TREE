# HK-TreeLabel VMMS

This directory contains the restart work for HK-TreeLabel using the local VMMS
panorama and LiDAR captures. The source data under `receive/vmms` is treated as
read-only. Generated indexes and derived data are written below this project.

## Coordinate solver

Run from PowerShell:

```powershell
.\run_coordinate_solver.ps1
```

The launcher prefers the existing `miniconda3\envs\yolo` Python because that
environment already contains `pyproj`.

Outputs are written to `outputs/coordinates`:

- `frame_coordinates.csv`: one row per Ladybug panorama frame, with UTC/Hong
  Kong time, interpolated INS pose, HK80 coordinates and WGS84 coordinates.
- `route_trajectories.geojson`: lightweight map-ready route lines and start/end
  points.
- `pointcloud_assets.csv`: mapped point-cloud inventory, PCD header information,
  coordinate origins, integrity checks and bounds for downsampled ASCII clouds.
- `coordinate_summary.json`: coordinate-system, transformation, coverage,
  uncertainty and route summaries.

## Tree-route overlay

Run after coordinate solving:

```powershell
.\run_tree_route_overlay.ps1
```

This preserves the route-only GeoJSON and creates:

- `route_with_nearby_trees.geojson`: routes plus tree inventory points within
  100 m, with the nearest route and HK80 distance on every tree feature.
- `nearby_trees.csv`: the same selected trees in a table.
- `tree_route_overlay_summary.json`: counts by source, route and distance band.

Coordinate convention:

```text
HK80 easting  = survey origin easting  + local x
HK80 northing = survey origin northing + local y
HK80 z        = survey origin z        + local z
WGS84         = EPSG:2326 -> EPSG:4326
```

The resulting position is the interpolated INS/platform position. It is not yet
the optical center of an individual Ladybug camera. Camera/LiDAR lever-arm and
boresight calibration must be applied before precise pixel-to-point projection.

## He Wentian panorama annotation app

The first HK-TreeLabel workflow reproduced on the new VMMS data uses the
`hewentian_pano` trajectory. It reads the original panoramas without modifying
them, combines them with tree inventory points within 50 m of the route, and
generates three square candidate views for each target tree. The reviewer can
draw YOLO-format polygons, drop individual views, save the accepted views, or
reject an ambiguous/occluded tree.

The easiest Windows launcher is to double-click:

```text
启动何文田标注工具.cmd
```

Alternatively, start both services from PowerShell:

```powershell
.\run-hewentian-labeler.ps1
```

Open `http://127.0.0.1:5177/`. The backend health endpoint is
`http://127.0.0.1:8021/api/health`.

Derived data is written below `derived/hewentian_labeler`:

- `view_cache`: reproducible 1024 x 1024 crops and view metadata.
- `annotations`: accepted image copies, YOLO polygon text files, and per-tree
  provenance metadata.
- `runtime/review_state.json`: accepted/rejected progress for resuming later.
- `runtime/checkpoints.json`: per-species drafts used by “保存进度并换树种”.

### Image visibility and preprocessing records

Each candidate view has non-destructive exposure (`-2` to `+2` EV) and
contrast (`0.50` to `1.50`) controls. “自动建议” measures a target-centred ROI
so a bright sky alone does not classify an otherwise visible tree as
overexposed. Hold “按住看原图” to compare the adjusted preview with the source.
Edits are debounce-saved to the current checkpoint after about 0.9 seconds.

The reviewer can record a view as clear, partially visible, or unannotatable,
with an occlusion/exposure/blur/missing-target reason and optional note.
Unannotatable drops only that view; it does not automatically reject the whole
tree. Submitted metadata keeps both the measured quality and the human recipe.
The original image is always preserved. When a recipe changes pixels, a
deterministic enhanced copy and matching YOLO label are also written below the
tree's `enhanced` directory. Keep train/validation/test grouping at `tree_id`
level so original and enhanced variants cannot leak across splits.
Accepted and rejected reviews are also flattened into
`derived/hewentian_labeler/annotations/preprocess_manifest.csv`, which is the
training-ready table for learning or evaluating a future preprocessing policy.

## All-route coordinate-matched panorama app

To inspect the same public-tree-coordinate workflow across **all five indexed
VMMS panorama streams**, double-click `启动全部轨迹树点标注工具.cmd` or run
`run-all-routes-labeler.ps1`. Open `http://127.0.0.1:5178/`; its backend is
`http://127.0.0.1:8022/api/health`. The route selector offers all routes and
each individual stream. This is separate from the original He Wentian app at
port 5177; its reviews remain in `derived/hewentian_labeler` and are not
rewritten. New all-route reviews go to `derived/all_routes_coordinate_labeler`.

The present 50 m public-inventory overlay has 466 candidate tree points:
He Wentian 185, Tsim Sha Tsui stream 1 zero, Tsim Sha Tsui stream 2 35,
Stubbs Road stream 0 eight, and Stubbs Road stream 1 238. The zero route still
appears so the absence of nearby public records is visible rather than being
mistaken for missing VMMS imagery. All five streams contain 20,691 indexed
panorama frames. Candidate crops use frames from the tree's own route only,
and non-He-Wentian tree IDs include a route prefix to prevent collisions.

Inventory coordinates are candidate locations, **not verified visibility or
species ground truth**. A stale point, occluded tree, or wrong panorama heading
still requires human keep/reject and polygon review. The source panoramas stay
read-only under `receive/vmms`.

The current candidate crop calibration assumes that the horizontal midpoint of
each Ladybug equirectangular panorama is the vehicle forward direction and that
x increases clockwise. This was checked on near-route He Wentian samples. Tree
inventory positions can still be stale, occluded, or represent a tree inside a
dense crown group, so the human keep/reject decision remains authoritative.

The point-cloud auxiliary viewer feasibility and implementation stages are
documented in `POINT_CLOUD_AUXILIARY_PLAN.md`.

## He Wentian image-first frame annotation app

When inventory tree coordinates cannot be matched reliably to the panorama,
use the independent image-driven workflow. Double-click:

```text
启动何文田逐帧标注工具.cmd
```

or open `http://127.0.0.1:5177/frame.html` while the normal services are
running. This mode does not use tree coordinates to create tasks. It follows
the `hewentian_pano` frame sequence and supports all 6088 raw frames or
distance-based sampling at 2, 5, or 10 m. The default 5 m sampling produces
about 469 review frames along the 2.60 km route.

For every panorama the reviewer can draw multiple tree polygons, assign a
different species and visibility status to each tree, record an unknown
species, adjust exposure/contrast, or mark the frame as `no_tree` or
`unusable`. Drafts are automatically saved after about 0.9 seconds.

Outputs are isolated below `derived/hewentian_frame_labeler`:

- `preview_cache`: 2048 x 1024 review previews; original 8192 x 4096 frames
  remain read-only under `receive/vmms`.
- `annotations/records`: per-frame JSON with provenance, polygons, species,
  image quality and preprocessing recipe.
- `annotations/labels`: YOLO segmentation labels; `no_tree` frames receive an
  empty negative label file.
- `annotations/classes.json`: stable species-to-class order.
- `annotations/dataset_manifest.csv`: source/label paths and 100 m route block
  IDs for leakage-safe train/validation/test splitting.
- `runtime`: formal review state, auto-saved drafts and the current position.

Adjacent panorama frames are highly correlated. Split the eventual dataset by
`route_block_id`, never by randomly shuffling individual images.
