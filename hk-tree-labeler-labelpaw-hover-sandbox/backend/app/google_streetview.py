from __future__ import annotations

import asyncio
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import httpx

from .config import STATIC_IMAGE_SIZE, load_google_maps_api_key
from .geometry import (
    bearing_deg,
    fov_for_distance,
    haversine_m,
    offset_point,
    pitch_deg,
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
        self.api_key = api_key or load_google_maps_api_key()
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0, read=30.0, write=10.0),
            limits=httpx.Limits(max_connections=24, max_keepalive_connections=0),
            headers={"User-Agent": "hk-tree-labeler/0.1"},
        )
        self.api_counts: dict[str, int] = {
            "Google Street View Metadata API": 0,
            "Google Street View Static API": 0,
        }
        self.api_error_counts: dict[str, int] = {}

    def _count_api(self, name: str) -> None:
        self.api_counts[name] = self.api_counts.get(name, 0) + 1

    def _count_api_error(self, name: str) -> None:
        self.api_error_counts[name] = self.api_error_counts.get(name, 0) + 1

    def _require_key(self) -> None:
        if not self.api_key:
            self.api_key = load_google_maps_api_key()
        if not self.api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is required for Street View downloads")

    async def close(self) -> None:
        await self.http.aclose()

    async def _get_with_retries(
        self,
        api_name: str,
        url: str,
        params: dict,
        retries: int = 3,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            self._count_api(api_name)
            try:
                response = await self.http.get(url, params=params)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"Retryable HTTP status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_error = exc
                self._count_api_error(api_name)
                if attempt >= retries:
                    break
                await asyncio.sleep(0.4 * attempt)
        raise RuntimeError(f"{api_name} request failed after {retries} attempts: {last_error}") from last_error

    async def validate_api_key(self) -> None:
        self._require_key()
        response = await self._get_with_retries(
            "Google Street View Metadata API",
            self.metadata_url,
            params={
                "location": "22.2819,114.1589",
                "radius": 50,
                "source": "outdoor",
                "key": self.api_key,
            },
        )
        payload = response.json()
        status = payload.get("status")
        if status in {"OK", "ZERO_RESULTS", "NOT_FOUND"}:
            return
        message = payload.get("error_message") or status or "unknown error"
        raise RuntimeError(f"Google Street View API key validation failed: {message}")

    async def metadata(self, lat: float, lon: float, radius: int = 30) -> dict | None:
        self._require_key()
        response = await self._get_with_retries(
            "Google Street View Metadata API",
            self.metadata_url,
            params={
                "location": f"{lat},{lon}",
                "radius": radius,
                "source": "outdoor",
                "key": self.api_key,
            },
        )
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
        return await self.discover_candidates(tree_lat, tree_lon, tree_height_m, limit=3)

    async def discover_candidates(
        self,
        tree_lat: float,
        tree_lon: float,
        tree_height_m: float,
        limit: int = 12,
    ) -> list[StreetViewCandidate]:
        probes = [(tree_lat, tree_lon)]
        for distance in (6, 10, 16, 24, 30):
            for angle in range(0, 360, 30):
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
        return sorted(by_pano.values(), key=self._candidate_rank)[:limit]

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

    def angular_gap(self, first: float, second: float) -> float:
        diff = abs((first - second) % 360)
        return min(diff, 360 - diff)

    def heading_spread_score(self, candidates: list[StreetViewCandidate]) -> float:
        if len(candidates) <= 1:
            return 0.0
        return min(self.angular_gap(a.heading, b.heading) for a, b in combinations(candidates, 2))

    def select_angle_fan_candidates(
        self,
        candidates: list[StreetViewCandidate],
        count: int = 3,
        step_degrees: int = 60,
    ) -> list[StreetViewCandidate]:
        if len(candidates) <= count:
            return sorted(candidates, key=self._candidate_rank)

        best_group: list[StreetViewCandidate] | None = None
        best_score: tuple[float, float, int, float] | None = None
        ordered = sorted(candidates, key=self._candidate_rank)
        for anchor in ordered:
            for direction in (1, -1):
                selected: list[StreetViewCandidate] = []
                errors: list[float] = []
                used: set[str] = set()
                for index in range(count):
                    target = (anchor.heading + direction * step_degrees * index) % 360
                    available = [item for item in ordered if item.pano_id not in used]
                    if not available:
                        break
                    picked = min(
                        available,
                        key=lambda item: (
                            self.angular_gap(item.heading, target),
                            self._candidate_rank(item),
                        ),
                    )
                    selected.append(picked)
                    used.add(picked.pano_id)
                    errors.append(self.angular_gap(picked.heading, target))

                if len(selected) != count:
                    continue
                date_score = sum(self._date_value(item.date) for item in selected)
                distance_score = sum(item.distance_m for item in selected)
                score = (-max(errors), -sum(errors), date_score, -distance_score)
                if best_score is None or score > best_score:
                    best_score = score
                    best_group = selected

        return best_group or ordered[:count]

    def build_angle_shots(
        self,
        candidates: list[StreetViewCandidate],
        tree_height_m: float,
    ) -> list[StreetViewShot]:
        if not candidates:
            return []

        fan = self.select_angle_fan_candidates(candidates, count=3, step_degrees=60)
        if len(fan) < 3:
            return []
        labels = ("0", "60", "120")
        shots: list[StreetViewShot] = []
        for label, source in zip(labels, fan):
            shots.append(
                StreetViewShot(
                    view_type=f"view_{label}",
                    filename=f"view_{label}.jpg",
                    pano_id=source.pano_id,
                    lat=source.lat,
                    lon=source.lon,
                    date=source.date,
                    distance_m=source.distance_m,
                    heading=source.heading,
                    pitch=pitch_deg(tree_height_m, source.distance_m, target_ratio=0.65, max_pitch=42.0),
                    fov=fov_for_distance(source.distance_m),
                )
            )
        return shots

    async def download_image(self, shot: StreetViewShot, output_path: Path) -> None:
        self._require_key()
        response = await self._get_with_retries(
            "Google Street View Static API",
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
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            raise RuntimeError(f"Street View response was not an image: {content_type}")
        output_path.write_bytes(response.content)
