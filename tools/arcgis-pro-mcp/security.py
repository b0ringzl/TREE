"""Path policy shared by the MCP server and ArcPy worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("ARCGIS_MCP_CONFIG", ROOT / "config.json"))


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "arcpy_launcher",
        "windows_user_profile",
        "allowed_input_roots",
        "allowed_output_roots",
        "worker_timeout_seconds",
        "audit_log",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing ArcGIS MCP settings: {missing}")
    return config


def canonical(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        raise ValueError(f"Path must be absolute: {path}")
    return Path(os.path.abspath(value))


def is_within(path: Path, roots: list[str]) -> bool:
    candidate = os.path.normcase(str(path))
    for root_value in roots:
        root = os.path.normcase(str(canonical(root_value)))
        try:
            if os.path.commonpath((candidate, root)) == root:
                return True
        except ValueError:
            continue
    return False


def validate_input_path(path: str | Path, config: dict[str, Any]) -> str:
    value = canonical(path)
    if not is_within(value, list(config["allowed_input_roots"])):
        raise PermissionError(f"Input path is outside allowed roots: {value}")
    return str(value)


def validate_output_path(path: str | Path, config: dict[str, Any]) -> str:
    value = canonical(path)
    if not is_within(value, list(config["allowed_output_roots"])):
        raise PermissionError(f"Output path is outside allowed roots: {value}")
    return str(value)


def validate_request_paths(operation: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize all paths before ArcPy receives a request."""

    config = load_config()
    normalized = dict(args)
    input_keys = {
        "describe": ("input_path",),
        "list_workspace": ("workspace_path",),
        "copy_features": ("input_path",),
        "select_by_attribute": ("input_path",),
        "buffer": ("input_path",),
        "clip": ("input_path", "clip_path"),
        "intersect": (),
        "dissolve": ("input_path",),
        "merge": (),
        "project": ("input_path",),
        "spatial_join": ("target_path", "join_path"),
        "repair_geometry_copy": ("input_path",),
    }
    output_keys = {
        "create_file_gdb": ("output_path",),
        "copy_features": ("output_path",),
        "select_by_attribute": ("output_path",),
        "buffer": ("output_path",),
        "clip": ("output_path",),
        "intersect": ("output_path",),
        "dissolve": ("output_path",),
        "merge": ("output_path",),
        "project": ("output_path",),
        "spatial_join": ("output_path",),
        "repair_geometry_copy": ("output_path",),
    }
    if (
        operation not in input_keys
        and operation not in output_keys
        and operation not in {"health"}
    ):
        raise ValueError(f"Unsupported operation: {operation}")
    for key in input_keys.get(operation, ()):
        normalized[key] = validate_input_path(normalized[key], config)
    for key in output_keys.get(operation, ()):
        normalized[key] = validate_output_path(normalized[key], config)
    if operation in {"intersect", "merge"}:
        normalized["input_paths"] = [
            validate_input_path(path, config) for path in normalized["input_paths"]
        ]
    return normalized
