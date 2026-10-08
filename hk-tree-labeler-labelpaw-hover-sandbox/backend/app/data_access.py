from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import SPECIES_DIR, assert_inside_root, safe_name


LAT_ALIASES = ("lat", "latitude", "y", "纬度", "緯度")
LON_ALIASES = ("lon", "lng", "long", "longitude", "x", "经度", "經度")
ID_ALIASES = ("tree_id", "treeid", "id", "Tree_ID", "树木ID", "樹木ID")


@dataclass(frozen=True)
class TreeRecord:
    tree_id: str
    lat: float
    lon: float
    height_m: float


def list_species() -> list[str]:
    if not SPECIES_DIR.exists():
        return []
    return sorted(path.name for path in SPECIES_DIR.iterdir() if path.is_dir())


def species_path(species: str) -> Path:
    path = assert_inside_root(SPECIES_DIR / safe_name(species))
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Species folder not found: {species}")
    return path


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]
    for col in columns:
        lowered = str(col).strip().lower()
        if any(alias.strip().lower() in lowered for alias in aliases):
            return col
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def parse_height_m(species_dir: Path) -> float:
    traits = species_dir / "traits.txt"
    if not traits.exists():
        return 10.0
    raw = traits.read_bytes()
    text = None
    for encoding in ("utf-8", "utf-8-sig", "big5", "gb18030", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return 10.0

    candidates: list[float] = []
    for match in re.finditer(r"(?:height|高度|高)[^\d]{0,24}(\d+(?:\.\d+)?)\s*(?:m|米|metre|meter)?", text, re.I):
        value = float(match.group(1))
        if 1 <= value <= 80:
            candidates.append(value)
    return max(candidates) if candidates else 10.0


def read_traits_text(species: str) -> str:
    folder = species_path(species)
    traits = folder / "traits.txt"
    if not traits.exists():
        return ""
    raw = traits.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "big5", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def load_tree_records(species: str) -> list[TreeRecord]:
    folder = species_path(species)
    tables = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.xlsx")) + sorted(folder.glob("*.xls"))
    if not tables:
        raise FileNotFoundError(f"No coordinate table found in {folder}")

    height_m = parse_height_m(folder)
    records: list[TreeRecord] = []
    seen: dict[str, int] = {}
    for table in tables:
        df = _read_table(table)
        columns = list(df.columns)
        lat_col = _find_column(columns, LAT_ALIASES)
        lon_col = _find_column(columns, LON_ALIASES)
        id_col = _find_column(columns, ID_ALIASES)
        if not lat_col or not lon_col:
            continue
        for row_index, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            raw_id = str(row[id_col]).strip() if id_col and pd.notna(row[id_col]) else f"{table.stem}_{row_index}"
            base_id = safe_name(raw_id.replace("\\", "_").replace("/", "_"))
            seen[base_id] = seen.get(base_id, 0) + 1
            tree_id = base_id if seen[base_id] == 1 else f"{base_id}_{seen[base_id]}"
            records.append(TreeRecord(tree_id=tree_id, lat=lat, lon=lon, height_m=height_m))
    if not records:
        raise ValueError(f"No valid latitude/longitude rows found for {species}")
    return records
