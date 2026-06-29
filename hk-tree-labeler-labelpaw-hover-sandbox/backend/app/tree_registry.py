from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import LABEL_MERGE_MAP, SPECIES_DIR, TREE_REGISTRY_DB, safe_name
from .data_access import ID_ALIASES, LAT_ALIASES, LON_ALIASES, TreeRecord, _find_column, _read_table, parse_height_m

HEIGHT_ALIASES = ("height", "height_m", "tree_height", "楂樺害")


@dataclass(frozen=True)
class RegistryTree:
    registry_id: int
    tree_id: str
    species: str
    source_species: str
    lat: float
    lon: float
    height_m: float
    source_file: str
    source_row: int
    raw_payload: dict[str, Any]

    def to_tree_record(self) -> TreeRecord:
        return TreeRecord(tree_id=self.tree_id, lat=self.lat, lon=self.lon, height_m=self.height_m)

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw_payload", None)
        return data


def normalize_tree_id(value: str) -> str:
    return safe_name(str(value).strip().replace("\\", "_").replace("/", "_"))


def _lookup_key(value: str) -> str:
    return normalize_tree_id(value).casefold()


def _connect(db_path: Path = TREE_REGISTRY_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tree_records (
            registry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tree_id TEXT NOT NULL,
            tree_id_key TEXT NOT NULL,
            species TEXT NOT NULL,
            source_species TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            height_m REAL NOT NULL,
            source_file TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            raw_payload TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_tree_records_tree_id_key ON tree_records(tree_id_key)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_tree_records_species_id ON tree_records(species, tree_id_key)")


def load_label_merge_map(label_map_path: Path = LABEL_MERGE_MAP) -> dict[str, str]:
    if not label_map_path.exists():
        return {}
    df = pd.read_csv(label_map_path)
    if "original_species" not in df.columns or "label_species" not in df.columns:
        return {}
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        original = str(row["original_species"]).strip()
        label = str(row["label_species"]).strip()
        if original and label and original.lower() != "nan" and label.lower() != "nan":
            mapping[original] = label
    return mapping


def canonical_species_name(species: str, label_map: dict[str, str]) -> str:
    return label_map.get(species, species)


def _row_value(row: pd.Series, column: str | None, fallback: Any = None) -> Any:
    if not column:
        return fallback
    value = row.get(column)
    return fallback if pd.isna(value) else value


def iter_registry_rows(species_root: Path = SPECIES_DIR, label_map_path: Path = LABEL_MERGE_MAP) -> list[RegistryTree]:
    label_map = load_label_merge_map(label_map_path)
    rows: list[RegistryTree] = []
    if not species_root.exists():
        return rows

    for species_folder in sorted(path for path in species_root.iterdir() if path.is_dir()):
        source_species = species_folder.name
        species = canonical_species_name(source_species, label_map)
        default_height = parse_height_m(species_folder)
        tables = sorted(species_folder.glob("*.csv")) + sorted(species_folder.glob("*.xlsx")) + sorted(species_folder.glob("*.xls"))
        for table in tables:
            df = _read_table(table)
            columns = list(df.columns)
            lat_col = _find_column(columns, LAT_ALIASES)
            lon_col = _find_column(columns, LON_ALIASES)
            id_col = _find_column(columns, ID_ALIASES)
            height_col = _find_column(columns, HEIGHT_ALIASES)
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

                raw_id = _row_value(row, id_col, f"{table.stem}_{row_index}")
                tree_id = normalize_tree_id(str(raw_id))
                if not tree_id:
                    continue
                raw_height = _row_value(row, height_col, default_height)
                try:
                    height_m = float(raw_height)
                except (TypeError, ValueError):
                    height_m = default_height
                source_row_species = str(_row_value(row, "original_species", source_species)).strip()
                row_species = canonical_species_name(source_row_species, label_map)
                public_species = species if species != source_species else row_species
                raw_payload = {str(key): (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
                rows.append(
                    RegistryTree(
                        registry_id=0,
                        tree_id=tree_id,
                        species=public_species,
                        source_species=source_row_species or source_species,
                        lat=lat,
                        lon=lon,
                        height_m=height_m,
                        source_file=str(table),
                        source_row=int(row_index) + 2,
                        raw_payload=raw_payload,
                    )
                )
    return rows


def rebuild_tree_registry(
    db_path: Path = TREE_REGISTRY_DB,
    species_root: Path = SPECIES_DIR,
    label_map_path: Path = LABEL_MERGE_MAP,
) -> dict[str, int | str]:
    rows = iter_registry_rows(species_root, label_map_path)
    updated_at = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as connection:
        _ensure_schema(connection)
        connection.execute("DELETE FROM tree_records")
        connection.executemany(
            """
            INSERT INTO tree_records
                (tree_id, tree_id_key, species, source_species, lat, lon, height_m, source_file, source_row, raw_payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.tree_id,
                    _lookup_key(row.tree_id),
                    row.species,
                    row.source_species,
                    row.lat,
                    row.lon,
                    row.height_m,
                    row.source_file,
                    row.source_row,
                    json.dumps(row.raw_payload, ensure_ascii=False, default=str),
                    updated_at,
                )
                for row in rows
            ],
        )
    return {"status": "rebuilt", "records": len(rows), "db_path": str(db_path)}


def _row_to_registry_tree(row: sqlite3.Row) -> RegistryTree:
    return RegistryTree(
        registry_id=int(row["registry_id"]),
        tree_id=str(row["tree_id"]),
        species=str(row["species"]),
        source_species=str(row["source_species"]),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        height_m=float(row["height_m"]),
        source_file=str(row["source_file"]),
        source_row=int(row["source_row"]),
        raw_payload=json.loads(row["raw_payload"] or "{}"),
    )


def lookup_tree_id(tree_id: str, db_path: Path = TREE_REGISTRY_DB) -> list[RegistryTree]:
    with _connect(db_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT * FROM tree_records
            WHERE tree_id_key = ?
            ORDER BY species, source_file, source_row
            """,
            (_lookup_key(tree_id),),
        ).fetchall()
    return [_row_to_registry_tree(row) for row in rows]


def get_registry_tree(registry_id: int, db_path: Path = TREE_REGISTRY_DB) -> RegistryTree | None:
    with _connect(db_path) as connection:
        _ensure_schema(connection)
        row = connection.execute("SELECT * FROM tree_records WHERE registry_id = ?", (registry_id,)).fetchone()
    return _row_to_registry_tree(row) if row else None


def ensure_tree_registry(db_path: Path = TREE_REGISTRY_DB) -> dict[str, int | str]:
    if db_path.exists():
        with _connect(db_path) as connection:
            _ensure_schema(connection)
            count = connection.execute("SELECT COUNT(*) AS value FROM tree_records").fetchone()["value"]
        if count:
            return {"status": "ready", "records": int(count), "db_path": str(db_path)}
    return rebuild_tree_registry(db_path)
