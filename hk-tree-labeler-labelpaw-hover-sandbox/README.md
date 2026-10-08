# HK Tree Labeler LabelPaw Hover Sandbox

This is an isolated experiment copied from `D:\TREE\hk-tree-labeler`.
It tests LabelPaw-style SAM hover assistance without changing the main labeler.

Use this sandbox only for trying the hover-preview workflow:

```powershell
cd D:\TREE\hk-tree-labeler-labelpaw-hover-sandbox
powershell -ExecutionPolicy Bypass -File .\run-session.ps1
```

Default ports:

- Backend: `http://127.0.0.1:8010`
- Frontend: `http://127.0.0.1:5175`

SAM weights are read from:

```text
D:\TREE\LabelPaw-web-images\weights\sam_weights
```

When `AI Assist` is enabled, moving the mouse over a blank crown area asks the long-lived SAM worker for a preview polygon. Click while the green preview is visible to add it as a normal editable polygon label.
