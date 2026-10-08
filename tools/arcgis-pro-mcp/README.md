# ArcGIS Pro Vector MCP

This local MCP server exposes selected ArcPy vector geoprocessing operations to
Codex through stdio. The MCP runtime is isolated in `.venv`; ArcPy operations
run with the Python environment bundled with ArcGIS Pro 3.5.2.

## Safety policy

- Inputs must be under `D:\TREE`.
- Outputs must be under `D:\TREE\gis_output`.
- Existing outputs are never overwritten.
- Repair operations copy data first and modify only the copy.
- Every operation is appended to `logs\tool_calls.jsonl`.
- Raw data deletion, in-place field updates, and arbitrary ArcPy code execution
  are intentionally not exposed.

Edit `config.json` to change allowed roots. Restart Codex after changing the MCP
configuration or tool implementation.

## Available tools

- `arcgis_health`
- `describe_dataset`
- `list_workspace`
- `create_file_geodatabase`
- `copy_features`
- `select_by_attribute`
- `buffer_features`
- `clip_features`
- `intersect_features`
- `dissolve_features`
- `merge_features`
- `project_features`
- `spatial_join_features`
- `repair_geometry_copy`

All output tools require a new absolute output path. Use a path under
`D:\TREE\gis_output`, such as
`D:\TREE\gis_output\tree_analysis.gdb\selected_trees`.

## Manual startup

```powershell
.\launch-mcp.cmd
```

The process waits for MCP messages on stdin, so a blank terminal is expected
when it is launched manually. Normally Codex starts it automatically.

