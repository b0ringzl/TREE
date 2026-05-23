from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import GOOGLE_MAPS_API_KEY, STATIC_IMAGE_SIZE
from .geometry import (
    bearing_deg,
    detail_fov_for_distance,
    fov_for_distance,
    haversine_m,
    offset_point,
    pitch_deg,
    trunk_fov_for_distance,
)


@dataclass(frozen=True)
class StreetViewCandidate:
    pano_id: str
    lat: float
    lon: float
    date: str | None
    distance_m: float
    heading: float
    pitch: float
    fov: int


@dataclass(frozen=True)
class StreetViewShot:
    view_type: str
    filename: str
    pano_id: str
    lat: float
    lon: float
    date: str | None
    distance_m: float
    heading: float
    pitch: float
    fov: int


class StreetViewClient:
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    image_url = "https://maps.googleapis.com/maps/api/streetview"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or GOOGLE_MAPS_API_KEY
        self.http = httpx.AsyncClient(timeout=30)
        self.api_counts: dict[str, int] = {
            "Google Street View Metadata API": 0,
            "Google Street View Static API": 0,
        }

    def _count_api(self, name: str) -> None:
        self.api_counts[name] = self.api_counts.get(name, 0) + 1

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is required for Street View downloads")

    async def close(self) -> None:
        await self.http.aclose()

    async def validate_api_key(self) -> None:
        self._require_key()
        self._count_api("Google Street View Metadata API")
        response = await self.http.get(
            self.metadata_url,
            params={
                "location": "22.2819,114.1589",
                "radius": 50,
                "source": "outdoor",
                "key": self.api_key,
            },
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status in {"OK", "ZERO_RESULTS", "NOT_FOUND"}:
            return
        message = payload.get("error_message") or status or "unknown error"
        raise RuntimeError(f"Google Street View API key validation failed: {message}")

    async def metadata(self, lat: float, lon: float, radius: int = 30) -> dict | None:
        self._require_key()
        self._count_api("Google Street View Metadata API")
        response = await self.http.get(
            self.metadata_url,
            params={
                "location": f"{lat},{lon}",
                "radius": radius,
                "source": "outdoor",
                "key": self.api_key,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "OK":
            return None
        return payload

    async def discover_nearest_three(
        self,
        tree_lat: float,
        tree_lon: float,
        tree_height_m: float,
    ) -> list[StreetViewCandidate]:
        probes = [(tree_lat, tree_lon)]
        for distance in (8, 16, 24):
            for angle in range(0, 360, 45):
                probes.append(offset_point(tree_lat, tree_lon, angle, distance))

        by_pano: dict[str, StreetViewCandidate] = {}
        for probe_lat, probe_lon in probes:
            meta = await self.metadata(probe_lat, probe_lon, radius=35)
            if not meta:
                continue
            pano_id = meta.get("pano_id")
            loc = meta.get("location") or {}
            pano_lat = loc.get("lat")
            pano_lon = loc.get("lng")
            if not pano_id or pano_lat is None or pano_lon is None:
                continue
            distance_m = haversine_m(float(pano_lat), float(pano_lon), tree_lat, tree_lon)
            if not (3 <= distance_m <= 30):
                continue
            candidate = StreetViewCandidate(
                pano_id=str(pano_id),
                lat=float(pano_lat),
                lon=float(pano_lon),
                date=meta.get("date"),
                distance_m=distance_m,
                heading=bearing_deg(float(pano_lat), float(pano_lon), tree_lat, tree_lon),
                pitch=pitch_deg(tree_height_m, distance_m, target_ratio=0.65, max_pitch=42.0),
                fov=fov_for_distance(distance_m),
            )
            current = by_pano.get(candidate.pano_id)
            if current is None or self._candidate_rank(candidate) < self._candidate_rank(current):
                by_pano[candidate.pano_id] = candidate
        return sorted(by_pano.values(), key=self._candidate_rank)[:3]

    def _candidate_rank(self, candidate: StreetViewCandidate) -> tuple[int, float]:
        return (-self._date_value(candidate.date), candidate.distance_m)

    def _date_value(self, date_text: str | None) -> int:
        if not date_text:
            return 0
        parts = date_text.split("-")
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 1
            return year * 100 + month
        except (TypeError, ValueError):
            return 0

    def build_feature_shots(
        self,
        candidates: list[StreetViewCandidate],
        tree_height_m: float,
    ) -> list[StreetViewShot]:
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda item: item.distance_m)
        newest_ordered = sorted(candidates, key=self._candidate_rank)
        closest = ordered[0]
        overview_source = next((item for item in newest_ordered if item.distance_m >= 8), newest_ordered[0])
        detail_source = closest

        return [
            StreetViewShot(
                view_type="feature_overview",
                filename="feature_overview.jpg",
                pano_id=overview_source.pano_id,
                lat=overview_source.lat,
                lon=overview_source.lon,
                date=overview_source.date,
                distance_m=overview_source.distance_m,
                heading=overview_source.heading,
                pitch=pitch_deg(tree_height_m, overview_source.distance_m, target_ratio=0.65, max_pitch=42.0),
                fov=fov_for_distance(overview_source.distance_m),
            ),
            StreetViewShot(
                view_type="trunk_texture",
                filename="trunk_texture.jpg",
                pano_id=closest.pano_id,
                lat=closest.lat,
                lon=closest.lon,
                date=closest.date,
                distance_m=closest.distance_m,
                heading=closest.heading,
                pitch=0.0,
                fov=trunk_fov_for_distance(closest.distance_m),
            ),
            StreetViewShot(
                view_type="detail_closeup",
                filename="detail_closeup.jpg",
                pano_id=detail_source.pano_id,
                lat=detail_source.lat,
                lon=detail_source.lon,
                date=detail_source.date,
                distance_m=detail_source.distance_m,
                heading=detail_source.heading,
                pitch=pitch_deg(tree_height_m, detail_source.distance_m, target_ratio=0.78, max_pitch=55.0),
                fov=detail_fov_for_distance(detail_source.distance_m),
            ),
        ]

    async def download_image(self, shot: StreetViewShot, output_path: Path) -> None:
        self._require_key()
        self._count_api("Google Street View Static API")
        response = await self.http.get(
            self.image_url,
            params={
                "size": STATIC_IMAGE_SIZE,
                "pano": shot.pano_id,
                "heading": f"{shot.heading:.2f}",
                "pitch": f"{shot.pitch:.2f}",
                "fov": str(shot.fov),
                "source": "outdoor",
                "return_error_code": "true",
                "key": self.api_key,
            },
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            raise RuntimeError(f"Street View response was not an image: {content_type}")
        output_path.write_bytes(response.content)
