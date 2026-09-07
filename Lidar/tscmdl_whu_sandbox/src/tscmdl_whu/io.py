"""Memory-mapped readers for WHU-STree trajectory point clouds and labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class WHUFormatError(ValueError):
    """Raised when a WHU-STree file violates the expected binary layout."""


_PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class PlyHeader:
    path: Path
    format: str
    version: str
    vertex_count: int
    properties: tuple[tuple[str, str], ...]
    dtype: np.dtype
    data_offset: int

    @property
    def record_size(self) -> int:
        return self.dtype.itemsize

    @property
    def expected_file_size(self) -> int:
        return self.data_offset + self.vertex_count * self.record_size


@dataclass(frozen=True)
class PointChunk:
    start: int
    stop: int
    records: np.ndarray
    reference: np.ndarray | None

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def tree(self) -> np.ndarray | None:
        if "tree" in (self.records.dtype.names or ()):
            return self.records["tree"]
        if self.reference is not None:
            return self.reference[:, 0]
        return None

    @property
    def label(self) -> np.ndarray | None:
        if "label" in (self.records.dtype.names or ()):
            return self.records["label"]
        if self.reference is not None:
            return self.reference[:, 1]
        return None

    @property
    def intensity(self) -> np.ndarray:
        return self.records["intensity"]

    def xyz(self, dtype: np.dtype | str | None = None) -> np.ndarray:
        """Return an N x 3 dense coordinate array for downstream tensor code."""
        output_dtype = np.dtype(dtype) if dtype is not None else self.records["x"].dtype
        coordinates = np.empty((len(self), 3), dtype=output_dtype)
        coordinates[:, 0] = self.records["x"]
        coordinates[:, 1] = self.records["y"]
        coordinates[:, 2] = self.records["z"]
        return coordinates


def parse_ply_header(path: str | Path) -> PlyHeader:
    """Parse the header of a scalar-only binary little-endian PLY file."""
    ply_path = Path(path)
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    ply_format: str | None = None
    version: str | None = None
    vertex_count: int | None = None
    current_element: str | None = None
    properties: list[tuple[str, str]] = []
    non_vertex_elements: list[tuple[str, int]] = []

    with ply_path.open("rb") as stream:
        first_line = stream.readline()
        if first_line.rstrip(b"\r\n") != b"ply":
            raise WHUFormatError(f"Not a PLY file: {ply_path}")

        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise WHUFormatError(f"PLY header has no end_header marker: {ply_path}")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise WHUFormatError(f"PLY header is not ASCII: {ply_path}") from exc

            if not line or line.startswith("comment ") or line.startswith("obj_info "):
                continue
            if line == "end_header":
                data_offset = stream.tell()
                break

            parts = line.split()
            if parts[0] == "format" and len(parts) == 3:
                ply_format, version = parts[1], parts[2]
            elif parts[0] == "element" and len(parts) == 3:
                current_element = parts[1]
                try:
                    count = int(parts[2])
                except ValueError as exc:
                    raise WHUFormatError(f"Invalid element count in {ply_path}: {line}") from exc
                if count < 0:
                    raise WHUFormatError(f"Negative element count in {ply_path}: {line}")
                if current_element == "vertex":
                    vertex_count = count
                elif count:
                    non_vertex_elements.append((current_element, count))
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise WHUFormatError(f"Unsupported vertex property in {ply_path}: {line}")
                scalar_type, name = parts[1], parts[2]
                if scalar_type not in _PLY_SCALAR_DTYPES:
                    raise WHUFormatError(
                        f"Unsupported PLY scalar type {scalar_type!r} in {ply_path}"
                    )
                if any(existing_name == name for existing_name, _ in properties):
                    raise WHUFormatError(f"Duplicate vertex property {name!r} in {ply_path}")
                properties.append((name, scalar_type))

    if ply_format != "binary_little_endian" or version != "1.0":
        raise WHUFormatError(
            f"Expected binary_little_endian PLY 1.0, got {ply_format!r} {version!r}: {ply_path}"
        )
    if vertex_count is None:
        raise WHUFormatError(f"PLY has no vertex element: {ply_path}")
    if not properties:
        raise WHUFormatError(f"PLY vertex element has no scalar properties: {ply_path}")
    if non_vertex_elements:
        raise WHUFormatError(
            f"Non-vertex payloads are not supported: {non_vertex_elements!r} in {ply_path}"
        )

    dtype = np.dtype([(name, _PLY_SCALAR_DTYPES[kind]) for name, kind in properties])
    return PlyHeader(
        path=ply_path,
        format=ply_format,
        version=version,
        vertex_count=vertex_count,
        properties=tuple(properties),
        dtype=dtype,
        data_offset=data_offset,
    )


class WHUTrajectoryReader:
    """Read one WHU-STree trajectory without loading its full payload into RAM."""

    REQUIRED_POINT_FIELDS = frozenset({"x", "y", "z", "intensity"})

    def __init__(
        self,
        ply_path: str | Path,
        reference_path: str | Path | None = None,
        *,
        chunk_size: int = 1_000_000,
        strict_file_size: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.header = parse_ply_header(ply_path)
        self.chunk_size = chunk_size
        names = set(self.header.dtype.names or ())
        missing = self.REQUIRED_POINT_FIELDS - names
        if missing:
            raise WHUFormatError(f"Missing required PLY fields: {sorted(missing)}")

        has_tree = "tree" in names
        has_label = "label" in names
        if has_tree != has_label:
            raise WHUFormatError("PLY must contain both tree and label fields, or neither")

        actual_size = self.header.path.stat().st_size
        if actual_size < self.header.expected_file_size:
            raise WHUFormatError(
                f"Truncated PLY: expected {self.header.expected_file_size} bytes, got {actual_size}"
            )
        if strict_file_size and actual_size != self.header.expected_file_size:
            raise WHUFormatError(
                f"Unexpected trailing PLY data: expected {self.header.expected_file_size} bytes, "
                f"got {actual_size}"
            )

        self._records = np.memmap(
            self.header.path,
            mode="r",
            dtype=self.header.dtype,
            offset=self.header.data_offset,
            shape=(self.header.vertex_count,),
        )

        self.reference_path = Path(reference_path) if reference_path is not None else None
        self._reference: np.ndarray | None = None
        if self.reference_path is not None:
            if has_tree:
                raise WHUFormatError("Both embedded PLY labels and an external reference were supplied")
            if not self.reference_path.is_file():
                raise FileNotFoundError(self.reference_path)
            reference = np.load(self.reference_path, mmap_mode="r", allow_pickle=False)
            if reference.dtype != np.dtype(np.int16):
                raise WHUFormatError(
                    f"Reference dtype must be int16, got {reference.dtype}: {self.reference_path}"
                )
            if reference.ndim != 2 or reference.shape[1] != 2:
                raise WHUFormatError(
                    f"Reference shape must be (N, 2), got {reference.shape}: {self.reference_path}"
                )
            if reference.shape[0] != self.header.vertex_count:
                raise WHUFormatError(
                    f"PLY/reference row mismatch: {self.header.vertex_count} != {reference.shape[0]}"
                )
            self._reference = reference

        if has_tree:
            self.annotation_source = "embedded"
        elif self._reference is not None:
            self.annotation_source = "reference"
        else:
            self.annotation_source = "none"

    def __len__(self) -> int:
        return self.header.vertex_count

    @property
    def chunk_count(self) -> int:
        return (len(self) + self.chunk_size - 1) // self.chunk_size

    def read_chunk(self, start: int, stop: int | None = None) -> PointChunk:
        if start < 0 or start >= len(self):
            raise IndexError(f"Chunk start is outside [0, {len(self)}): {start}")
        requested_stop = start + self.chunk_size if stop is None else stop
        if requested_stop <= start:
            raise ValueError("Chunk stop must be greater than start")
        actual_stop = min(requested_stop, len(self))
        reference = None if self._reference is None else self._reference[start:actual_stop]
        return PointChunk(
            start=start,
            stop=actual_stop,
            records=self._records[start:actual_stop],
            reference=reference,
        )

    def iter_chunks(self) -> Iterator[PointChunk]:
        for start in range(0, len(self), self.chunk_size):
            yield self.read_chunk(start)

    def close(self) -> None:
        record_mmap = getattr(self._records, "_mmap", None)
        if record_mmap is not None:
            record_mmap.close()
        reference_mmap = getattr(self._reference, "_mmap", None)
        if reference_mmap is not None:
            reference_mmap.close()

    def __enter__(self) -> "WHUTrajectoryReader":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def discover_reference_path(
    dataset_root: str | Path, road_id: str, trajectory_id: str
) -> Path | None:
    candidate = Path(dataset_root) / "reference_data" / f"{road_id}_{trajectory_id}.npy"
    return candidate if candidate.is_file() else None


def open_whu_trajectory(
    dataset_root: str | Path,
    road_id: str,
    trajectory_id: str,
    *,
    chunk_size: int = 1_000_000,
    use_reference: bool = True,
) -> WHUTrajectoryReader:
    root = Path(dataset_root)
    ply_path = root / road_id / "PCD" / f"{trajectory_id}.ply"
    reference_path = (
        discover_reference_path(root, road_id, trajectory_id) if use_reference else None
    )
    return WHUTrajectoryReader(
        ply_path,
        reference_path,
        chunk_size=chunk_size,
    )
