from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
    phi1, phi2 = math.radians(from_lat), math.radians(to_lat)
    d_lambda = math.radians(to_lon - from_lon)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def pitch_deg(
    tree_height_m: float,
    distance_m: float,
    camera_height_m: float = 2.5,
    target_ratio: float = 0.65,
    min_pitch: float = -30.0,
    max_pitch: float = 42.0,
) -> float:
    if distance_m <= 0:
        return 0.0
    target_height_m = tree_height_m * target_ratio
    raw_pitch = math.degrees(math.atan((target_height_m - camera_height_m) / distance_m))
    return max(min_pitch, min(max_pitch, raw_pitch))


def fov_for_distance(distance_m: float) -> int:
    if distance_m < 10:
        return 90
    if distance_m < 20:
        return 75
    return 60


def offset_point(lat: float, lon: float, bearing: float, distance_m: float) -> tuple[float, float]:
    angular_distance = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(angular_distance)
        + math.cos(phi1) * math.sin(angular_distance) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(angular_distance) * math.cos(phi1),
        math.cos(angular_distance) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), ((math.degrees(lambda2) + 540) % 360) - 180
