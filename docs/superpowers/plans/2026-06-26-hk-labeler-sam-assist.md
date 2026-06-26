# HK Labeler SAM Assist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional SAM2/SAM3-assisted crown polygon generation to `hk-tree-labeler` while keeping manual polygon labeling intact.

**Architecture:** The FastAPI backend stays lightweight and launches SAM inference through the existing `YOLO_PYTHON` conda `yolo` environment, matching the current YOLO helper pattern. The frontend exposes a model-assist toggle and model selector; when enabled, a click on blank image space requests a SAM polygon and appends it as a normal editable YOLO segmentation polygon.

**Tech Stack:** FastAPI, Pydantic, Python subprocess helpers, React canvas UI, Node test runner.

---

### Task 1: Backend SAM Model Discovery and Request Wrapper

**Files:**
- Create: `D:\TREE\hk-tree-labeler\backend\app\sam_assistant.py`
- Modify: `D:\TREE\hk-tree-labeler\backend\app\config.py`
- Modify: `D:\TREE\hk-tree-labeler\backend\app\schemas.py`
- Modify: `D:\TREE\hk-tree-labeler\backend\app\main.py`
- Test: `D:\TREE\hk-tree-labeler\backend\tests\test_sam_assistant.py`

- [ ] Write tests that create fake SAM weight files, verify SAM2/SAM3 classification, and verify missing model keys return `model_ready: false`.
- [ ] Run `D:\TREE\hk-tree-labeler\backend\.venv\Scripts\python.exe -m pytest D:\TREE\hk-tree-labeler\backend\tests\test_sam_assistant.py -q` and confirm the tests fail because `sam_assistant` is missing.
- [ ] Implement model scanning, request schema, endpoint wiring, and subprocess command construction.
- [ ] Re-run the backend test and confirm it passes.

### Task 2: SAM Inference Helper

**Files:**
- Create: `D:\TREE\tools\predict_sam_polygon.py`

- [ ] Implement a standalone helper that accepts `--model-type`, `--model`, `--image`, `--x`, `--y`, and optional `--config`.
- [ ] For SAM2, use `build_sam2` and `SAM2ImagePredictor`.
- [ ] For SAM3, use `build_sam3_image_model` and `Sam3Processor`.
- [ ] Convert the best mask contour to normalized polygon, normalized box, confidence, source, and model key JSON.
- [ ] Run `C:\Users\57680\.conda\envs\yolo\python.exe D:\TREE\tools\predict_sam_polygon.py --help` to verify the helper is syntactically usable without loading a model.

### Task 3: Frontend Optional Assist Flow

**Files:**
- Modify: `D:\TREE\hk-tree-labeler\frontend\src\annotationGeometry.js`
- Modify: `D:\TREE\hk-tree-labeler\frontend\src\annotationGeometry.test.mjs`
- Modify: `D:\TREE\hk-tree-labeler\frontend\src\main.jsx`
- Modify: `D:\TREE\hk-tree-labeler\frontend\src\styles.css`

- [ ] Add a pure helper test proving a candidate polygon is appended to existing polygons instead of replacing or truncating them.
- [ ] Run `node --test D:\TREE\hk-tree-labeler\frontend\src\annotationGeometry.test.mjs` and confirm the new test fails.
- [ ] Add the helper and wire the React UI: model list fetch, assist toggle, model selector, per-image click request, status display, and no `slice(0, 1)` truncation.
- [ ] Re-run the frontend geometry test and `npm run build`.

### Task 4: Final Verification and Git Save

**Files:**
- All files modified above.

- [ ] Run backend targeted tests.
- [ ] Run frontend targeted tests.
- [ ] Run frontend build.
- [ ] Run `D:\Git\cmd\git.exe status --short --branch`.
- [ ] Stage only the task files.
- [ ] Commit as `zhenglin <zhenglin9587@outlook.com>`.
- [ ] Push the current branch; if main cannot be pushed safely, report the exact failure and push/offer a task branch as appropriate.
