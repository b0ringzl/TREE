"""A local, path-restricted ArcPy MCP server for vector operations."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from security import ROOT, load_config, validate_request_paths


CONFIG = load_config()
WORKER = ROOT / "arcpy_worker.py"
WATCHDOG = ROOT / "worker_watchdog.py"
ARCPY_LAUNCHER = Path(CONFIG["arcpy_launcher"])
AUDIT_LOG = Path(CONFIG["audit_log"])

mcp = FastMCP(
    "ArcGIS Pro Vector Tools",
    instructions=(
        "Local ArcPy tools for file-based vector processing. Inputs are limited "
        "to configured roots, outputs are limited to D:\\TREE\\gis_output, and "
        "existing outputs are never overwritten."
    ),
    json_response=True,
)


def append_audit(record: dict[str, Any]) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def worker_environment() -> dict[str, str]:
    """Build the Windows user environment ArcGIS needs, even under a minimal MCP env."""

    profile = Path(CONFIG["windows_user_profile"])
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    local_app_data = profile / "AppData" / "Local"
    roaming_app_data = profile / "AppData" / "Roaming"
    arcgis_root = ARCPY_LAUNCHER.parents[3]
    python_root = arcgis_root / "bin" / "Python" / "envs" / "arcgispro-py3"
    path_parts = [
        system_root / "System32",
        system_root,
        arcgis_root / "bin",
        python_root,
        python_root / "Scripts",
    ]
    inherited_path = os.environ.get("PATH", "")
    if inherited_path:
        path_parts.append(Path(inherited_path))
    environment = dict(os.environ)
    environment.update(
        {
            "SYSTEMROOT": str(system_root),
            "WINDIR": str(system_root),
            "COMSPEC": str(system_root / "System32" / "cmd.exe"),
            "USERPROFILE": str(profile),
            "HOMEDRIVE": profile.drive,
            "HOMEPATH": str(profile)[len(profile.drive) :],
            "LOCALAPPDATA": str(local_app_data),
            "APPDATA": str(roaming_app_data),
            "TEMP": str(local_app_data / "Temp"),
            "TMP": str(local_app_data / "Temp"),
            "PATH": os.pathsep.join(str(path) for path in path_parts),
        }
    )
    return environment


def run_worker(operation: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_request_paths(operation, args)
    started = time.monotonic()
    response: dict[str, Any] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="arcgis-mcp-") as temporary:
            request_path = Path(temporary) / "request.json"
            response_path = Path(temporary) / "response.json"
            request_path.write_text(
                json.dumps(
                    {"operation": operation, "args": normalized},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            environment = worker_environment()
            process = subprocess.Popen(
                [
                    environment["COMSPEC"],
                    "/d",
                    "/c",
                    "call",
                    str(ARCPY_LAUNCHER),
                    str(WORKER),
                    str(request_path),
                    str(response_path),
                ],
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=environment,
            )
            watchdog = subprocess.Popen(
                [
                    sys.executable,
                    str(WATCHDOG),
                    str(os.getpid()),
                    str(process.pid),
                    environment["SYSTEMROOT"],
                ],
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=environment,
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=int(CONFIG["worker_timeout_seconds"])
                )
            except subprocess.TimeoutExpired as error:
                subprocess.run(
                    [
                        str(Path(environment["SYSTEMROOT"]) / "System32" / "taskkill.exe"),
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
                process.communicate(timeout=10)
                raise TimeoutError(
                    f"ArcPy worker exceeded {CONFIG['worker_timeout_seconds']} seconds"
                ) from error
            finally:
                try:
                    watchdog.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    watchdog.terminate()
                    watchdog.wait(timeout=3)
            if not response_path.is_file():
                raise RuntimeError(
                    "ArcPy worker did not write a response. "
                    f"exit={process.returncode}; stderr={stderr[-2000:]}"
                )
            response = json.loads(response_path.read_text(encoding="utf-8-sig"))
            if process.returncode != 0 or not response.get("ok"):
                raise RuntimeError(
                    f"{response.get('error_type', 'ArcPyError')}: "
                    f"{response.get('error', 'ArcPy operation failed')}\n"
                    f"{response.get('arcpy_messages', '')}"
                )
            return dict(response["result"])
    finally:
        append_audit(
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "operation": operation,
                "args": normalized,
                "success": bool(response and response.get("ok")),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )


async def invoke(operation: str, args: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(run_worker, operation, args)


@mcp.tool()
async def arcgis_health() -> dict[str, Any]:
    """Check ArcGIS Pro, ArcPy, license, and worker availability."""

    return await invoke("health", {})


@mcp.tool()
async def describe_dataset(input_path: str) -> dict[str, Any]:
    """Describe a file-based GIS dataset, including CRS, extent, fields, and count."""

    return await invoke("describe", {"input_path": input_path})


@mcp.tool()
async def list_workspace(workspace_path: str) -> dict[str, Any]:
    """List feature datasets, feature classes, tables, and rasters in a workspace."""

    return await invoke("list_workspace", {"workspace_path": workspace_path})


@mcp.tool()
async def create_file_geodatabase(output_path: str) -> dict[str, Any]:
    """Create a new File Geodatabase under the configured output root."""

    return await invoke("create_file_gdb", {"output_path": output_path})


@mcp.tool()
async def copy_features(input_path: str, output_path: str) -> dict[str, Any]:
    """Copy vector features to a new dataset without overwriting existing data."""

    return await invoke(
        "copy_features", {"input_path": input_path, "output_path": output_path}
    )


@mcp.tool()
async def select_by_attribute(
    input_path: str, output_path: str, where_clause: str
) -> dict[str, Any]:
    """Select features with an ArcGIS SQL clause and export them to a new dataset."""

    return await invoke(
        "select_by_attribute",
        {
            "input_path": input_path,
            "output_path": output_path,
            "where_clause": where_clause,
        },
    )


@mcp.tool()
async def buffer_features(
    input_path: str,
    output_path: str,
    distance: str,
    dissolve_option: str = "NONE",
    method: str = "PLANAR",
) -> dict[str, Any]:
    """Buffer vector features; distance examples include '10 Meters' and '50 Feet'."""

    return await invoke(
        "buffer",
        {
            "input_path": input_path,
            "output_path": output_path,
            "distance": distance,
            "dissolve_option": dissolve_option,
            "method": method,
        },
    )


@mcp.tool()
async def clip_features(
    input_path: str, clip_path: str, output_path: str
) -> dict[str, Any]:
    """Clip input features by polygon clip features and write a new dataset."""

    return await invoke(
        "clip",
        {
            "input_path": input_path,
            "clip_path": clip_path,
            "output_path": output_path,
        },
    )


@mcp.tool()
async def intersect_features(
    input_paths: list[str],
    output_path: str,
    join_attributes: str = "ALL",
    output_type: str = "INPUT",
) -> dict[str, Any]:
    """Intersect two or more vector datasets and write a new output dataset."""

    return await invoke(
        "intersect",
        {
            "input_paths": input_paths,
            "output_path": output_path,
            "join_attributes": join_attributes,
            "output_type": output_type,
        },
    )


@mcp.tool()
async def dissolve_features(
    input_path: str,
    output_path: str,
    dissolve_fields: list[str] | None = None,
    multi_part: str = "MULTI_PART",
) -> dict[str, Any]:
    """Dissolve features by zero or more fields into a new dataset."""

    return await invoke(
        "dissolve",
        {
            "input_path": input_path,
            "output_path": output_path,
            "dissolve_fields": dissolve_fields or [],
            "multi_part": multi_part,
        },
    )


@mcp.tool()
async def merge_features(input_paths: list[str], output_path: str) -> dict[str, Any]:
    """Merge two or more vector datasets into a new output dataset."""

    return await invoke(
        "merge", {"input_paths": input_paths, "output_path": output_path}
    )


@mcp.tool()
async def project_features(
    input_path: str,
    output_path: str,
    output_wkid: int,
    transformation: str = "",
) -> dict[str, Any]:
    """Project vector features to an EPSG/WKID coordinate system."""

    return await invoke(
        "project",
        {
            "input_path": input_path,
            "output_path": output_path,
            "output_wkid": output_wkid,
            "transformation": transformation,
        },
    )


@mcp.tool()
async def spatial_join_features(
    target_path: str,
    join_path: str,
    output_path: str,
    join_operation: str = "JOIN_ONE_TO_ONE",
    join_type: str = "KEEP_ALL",
    match_option: str = "INTERSECT",
) -> dict[str, Any]:
    """Spatially join two vector datasets and write a new output dataset."""

    return await invoke(
        "spatial_join",
        {
            "target_path": target_path,
            "join_path": join_path,
            "output_path": output_path,
            "join_operation": join_operation,
            "join_type": join_type,
            "match_option": match_option,
        },
    )


@mcp.tool()
async def repair_geometry_copy(
    input_path: str, output_path: str
) -> dict[str, Any]:
    """Copy a vector dataset and repair geometry only in the new copy."""

    return await invoke(
        "repair_geometry_copy",
        {"input_path": input_path, "output_path": output_path},
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
