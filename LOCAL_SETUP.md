# Local LabelPaw Setup

This clone is prepared as a separate manual annotation system for web-crawled images.

## Start

Double-click:

```bat
start-labelpaw-yolo.bat
```

Or run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-labelpaw-yolo.ps1
```

Environment check without opening the GUI:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-labelpaw-yolo.ps1 -CheckOnly
```

The startup script always uses the conda `yolo` environment and stores runtime caches under `.runtime/`.

## Defaults

- Default annotation mode: polygon
- Default save format: YOLO

Polygon labels are written as YOLO segmentation rows:

```txt
class_id x1 y1 x2 y2 x3 y3 ...
```

Coordinates are normalized to `[0, 1]` with six decimal places, matching the previous training-label output.

## Git Backup

The pristine upstream clone is preserved at:

- Branch: `backup/upstream-pristine-e25e5d7`
- Tag: `backup/upstream-pristine-20260608`

Local setup changes are on branch:

- `codex/network-image-labelpaw-setup`
