"""Execute one validated ArcPy operation and write a JSON response."""

from __future__ import annotations

import json
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import arcpy

from security import validate_request_paths


def dataset_summary(path: str) -> dict[str, Any]:
    if not arcpy.Exists(path):
        raise FileNotFoundError(path)
    description = arcpy.Describe(path)
    spatial_reference = getattr(description, "spatialReference", None)
    extent = getattr(description, "extent", None)
    fields = []
    try:
        fields = [
            {
                "name": field.name,
                "alias": field.aliasName,
                "type": field.type,
                "length": field.length,
                "nullable": field.isNullable,
            }
            for field in arcpy.ListFields(path)
        ]
    except (OSError, RuntimeError):
        fields = []
    result: dict[str, Any] = {
        "path": path,
        "data_type": getattr(description, "dataType", None),
        "dataset_type": getattr(description, "datasetType", None),
        "shape_type": getattr(description, "shapeType", None),
        "has_z": bool(getattr(description, "hasZ", False)),
        "has_m": bool(getattr(description, "hasM", False)),
        "spatial_reference": {
            "name": getattr(spatial_reference, "name", None),
            "wkid": getattr(spatial_reference, "factoryCode", None),
            "linear_unit": getattr(spatial_reference, "linearUnitName", None),
        },
        "fields": fields,
    }
    if extent:
        result["extent"] = {
            "xmin": extent.XMin,
            "ymin": extent.YMin,
            "xmax": extent.XMax,
            "ymax": extent.YMax,
        }
    try:
        result["feature_count"] = int(arcpy.management.GetCount(path)[0])
    except (OSError, RuntimeError):
        result["feature_count"] = None
    return result


def require_new_output(path: str) -> None:
    if arcpy.Exists(path) or Path(path).exists():
        raise FileExistsError(f"Output already exists; choose a new path: {path}")


def dispatch(operation: str, args: dict[str, Any]) -> dict[str, Any]:
    args = validate_request_paths(operation, args)
    arcpy.env.overwriteOutput = False
    if operation == "health":
        info = arcpy.GetInstallInfo()
        return {
            "status": "ok",
            "product": info.get("ProductName"),
            "version": info.get("Version"),
            "license_level": arcpy.ProductInfo(),
            "python": sys.executable,
        }
    if operation == "describe":
        return dataset_summary(args["input_path"])
    if operation == "list_workspace":
        workspace = args["workspace_path"]
        if not arcpy.Exists(workspace) and not Path(workspace).is_dir():
            raise FileNotFoundError(workspace)
        previous = arcpy.env.workspace
        try:
            arcpy.env.workspace = workspace
            datasets = []
            for feature_dataset in arcpy.ListDatasets(feature_type="feature") or []:
                datasets.append(feature_dataset)
            feature_classes = arcpy.ListFeatureClasses() or []
            for feature_dataset in datasets:
                feature_classes.extend(
                    str(Path(feature_dataset) / name)
                    for name in arcpy.ListFeatureClasses(feature_dataset=feature_dataset)
                    or []
                )
            return {
                "workspace": workspace,
                "feature_datasets": datasets,
                "feature_classes": feature_classes,
                "tables": arcpy.ListTables() or [],
                "rasters": arcpy.ListRasters() or [],
            }
        finally:
            arcpy.env.workspace = previous

    output = args["output_path"]
    require_new_output(output)
    if operation == "create_file_gdb":
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        name = output_path.name
        if not name.lower().endswith(".gdb"):
            name += ".gdb"
        result = arcpy.management.CreateFileGDB(str(output_path.parent), name)
        return {"output_path": str(result[0])}
    if operation == "copy_features":
        arcpy.management.CopyFeatures(args["input_path"], output)
    elif operation == "select_by_attribute":
        layer = f"arcgis_mcp_{uuid.uuid4().hex}"
        try:
            arcpy.management.MakeFeatureLayer(args["input_path"], layer)
            arcpy.management.SelectLayerByAttribute(
                layer, "NEW_SELECTION", args["where_clause"]
            )
            arcpy.management.CopyFeatures(layer, output)
        finally:
            if arcpy.Exists(layer):
                arcpy.management.Delete(layer)
    elif operation == "buffer":
        arcpy.analysis.Buffer(
            args["input_path"],
            output,
            args["distance"],
            dissolve_option=args.get("dissolve_option", "NONE"),
            method=args.get("method", "PLANAR"),
        )
    elif operation == "clip":
        arcpy.analysis.Clip(args["input_path"], args["clip_path"], output)
    elif operation == "intersect":
        arcpy.analysis.Intersect(
            args["input_paths"],
            output,
            join_attributes=args.get("join_attributes", "ALL"),
            output_type=args.get("output_type", "INPUT"),
        )
    elif operation == "dissolve":
        arcpy.management.Dissolve(
            args["input_path"],
            output,
            args.get("dissolve_fields", []),
            multi_part=args.get("multi_part", "MULTI_PART"),
        )
    elif operation == "merge":
        arcpy.management.Merge(args["input_paths"], output)
    elif operation == "project":
        spatial_reference = arcpy.SpatialReference(int(args["output_wkid"]))
        arcpy.management.Project(
            args["input_path"],
            output,
            spatial_reference,
            args.get("transformation", ""),
        )
    elif operation == "spatial_join":
        arcpy.analysis.SpatialJoin(
            args["target_path"],
            args["join_path"],
            output,
            join_operation=args.get("join_operation", "JOIN_ONE_TO_ONE"),
            join_type=args.get("join_type", "KEEP_ALL"),
            match_option=args.get("match_option", "INTERSECT"),
        )
    elif operation == "repair_geometry_copy":
        arcpy.management.CopyFeatures(args["input_path"], output)
        arcpy.management.RepairGeometry(output, "DELETE_NULL")
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    return {"output_path": output, "summary": dataset_summary(output)}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: arcpy_worker.py REQUEST_JSON RESPONSE_JSON")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8-sig"))
        result = dispatch(str(request["operation"]), dict(request.get("args", {})))
        response = {"ok": True, "result": result}
        exit_code = 0
    except Exception as error:  # ArcPy raises several non-standard exceptions.
        response = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "arcpy_messages": arcpy.GetMessages(2),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    response_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

