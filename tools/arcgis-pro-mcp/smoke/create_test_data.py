"""Create a small projected point feature class for MCP smoke testing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import arcpy


def main() -> None:
    output_dir = Path(sys.argv[1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    gdb = output_dir / "smoke_input.gdb"
    arcpy.management.CreateFileGDB(str(output_dir), gdb.name)
    points = str(gdb / "trees")
    arcpy.management.CreateFeatureclass(
        str(gdb), "trees", "POINT", spatial_reference=arcpy.SpatialReference(2326)
    )
    arcpy.management.AddField(points, "species", "TEXT", field_length=64)
    arcpy.management.AddField(points, "quality", "SHORT")
    rows = [
        (835000.0 + index * 4.0, 820000.0 + index * 3.0, species, quality)
        for index, (species, quality) in enumerate(
            [
                ("Ficus microcarpa", 1),
                ("Ficus microcarpa", 1),
                ("Cinnamomum camphora", 1),
                ("Cinnamomum camphora", 0),
                ("Bauhinia variegata", 1),
                ("Bauhinia variegata", 0),
            ]
        )
    ]
    with arcpy.da.InsertCursor(
        points, ["SHAPE@XY", "species", "quality"]
    ) as cursor:
        for x, y, species, quality in rows:
            cursor.insertRow(((x, y), species, quality))
    print(json.dumps({"gdb": str(gdb), "points": points, "count": len(rows)}))


if __name__ == "__main__":
    main()

