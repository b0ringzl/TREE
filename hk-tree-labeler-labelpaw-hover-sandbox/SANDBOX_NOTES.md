# LabelPaw Hover SAM Sandbox

This folder is an isolated prototype copied from `D:\TREE\hk-tree-labeler`.
It is for testing LabelPaw-style SAM hover previews without changing the main labeler.

## Ports

- Backend: `http://127.0.0.1:8010`
- Frontend: `http://127.0.0.1:5175`

## Behavior

- Turn on `AI Assist`.
- Move the mouse over a blank crown area.
- The frontend sends debounced hover points to `/api/task/sam-hover`.
- The backend keeps a long-lived SAM worker in the conda `yolo` environment.
- The worker caches the current image features and returns a green preview polygon.
- Click while the preview is visible to confirm it as a normal editable polygon label.

## Isolation

The sandbox does not copy `.venv`, `node_modules`, `dist`, logs, or runtime caches.
Startup scripts first look for sandbox-local dependencies, then reuse the existing main labeler dependencies if sandbox-local ones are absent.
For this prototype, `frontend\node_modules` can be a local junction to the main labeler `node_modules` folder, so no new npm install is required.

SAM weights are still read from:

```text
D:\TREE\LabelPaw-web-images\weights\sam_weights
```

## Start

```powershell
cd D:\TREE\hk-tree-labeler-labelpaw-hover-sandbox
powershell -ExecutionPolicy Bypass -File .\run-session.ps1
```
