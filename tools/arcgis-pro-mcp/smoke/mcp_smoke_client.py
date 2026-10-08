"""Exercise the installed stdio MCP server with real ArcPy vector operations."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]


def structured(result):
    value = getattr(result, "structuredContent", None)
    if value is not None:
        return value
    for item in result.content:
        text = getattr(item, "text", None)
        if text:
            return json.loads(text)
    raise RuntimeError("MCP tool returned no JSON content")


async def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    health_only = "--health-only" in sys.argv[2:]
    points = str(run_dir / "smoke_input.gdb" / "trees")
    selected = str(run_dir / "smoke_input.gdb" / "trees_quality_1")
    buffers = str(run_dir / "smoke_input.gdb" / "trees_buffer_10m")
    projected = str(run_dir / "smoke_input.gdb" / "trees_buffer_wgs84")
    params = StdioServerParameters(
        command=str(ROOT / ".venv" / "Scripts" / "python.exe"),
        args=[str(ROOT / "server.py")],
        env={"ARCGIS_MCP_CONFIG": str(ROOT / "config.json")},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = sorted(tool.name for tool in tools.tools)
            health = structured(await session.call_tool("arcgis_health", {}))
            if health_only:
                print(
                    json.dumps(
                        {
                            "status": "passed",
                            "tool_count": len(tool_names),
                            "arcgis_version": health["version"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return
            description = structured(
                await session.call_tool("describe_dataset", {"input_path": points})
            )
            selected_result = structured(
                await session.call_tool(
                    "select_by_attribute",
                    {
                        "input_path": points,
                        "output_path": selected,
                        "where_clause": "quality = 1",
                    },
                )
            )
            buffer_result = structured(
                await session.call_tool(
                    "buffer_features",
                    {
                        "input_path": selected,
                        "output_path": buffers,
                        "distance": "10 Meters",
                    },
                )
            )
            project_result = structured(
                await session.call_tool(
                    "project_features",
                    {
                        "input_path": buffers,
                        "output_path": projected,
                        "output_wkid": 4326,
                    },
                )
            )
    checks = {
        "tool_count": len(tool_names),
        "tools": tool_names,
        "arcgis_version": health["version"],
        "input_count": description["feature_count"],
        "selected_count": selected_result["summary"]["feature_count"],
        "buffer_count": buffer_result["summary"]["feature_count"],
        "projected_count": project_result["summary"]["feature_count"],
        "projected_wkid": project_result["summary"]["spatial_reference"]["wkid"],
    }
    expected = {
        "input_count": 6,
        "selected_count": 4,
        "buffer_count": 4,
        "projected_count": 4,
        "projected_wkid": 4326,
    }
    for name, value in expected.items():
        if checks[name] != value:
            raise AssertionError(f"{name}: {checks[name]} != {value}")
    (run_dir / "smoke_result.json").write_text(
        json.dumps({"status": "passed", **checks}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", **checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
