from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageStat


RECIPE_VERSION = "photo_v1"
DEFAULT_EXPOSURE_EV = 0.0
DEFAULT_CONTRAST = 1.0


def _percentile(histogram: list[int], fraction: float) -> int:
    total = sum(histogram)
    if total <= 0:
        return 0
    threshold = max(1, int(math.ceil(total * fraction)))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= threshold:
            return value
    return 255


def _analysis_roi(
    image: Image.Image, roi_fractions: tuple[float, float, float, float]
) -> Image.Image:
    """Use the target-centred region so bright sky does not dominate the score."""
    width, height = image.size
    left_fraction, top_fraction, right_fraction, bottom_fraction = roi_fractions
    left = int(round(width * left_fraction))
    right = int(round(width * right_fraction))
    top = int(round(height * top_fraction))
    bottom = int(round(height * bottom_fraction))
    return image.crop((left, top, right, bottom))


def analyze_image_quality(
    image: Image.Image,
    roi_fractions: tuple[float, float, float, float] = (0.18, 0.12, 0.82, 0.88),
) -> dict[str, float | str]:
    roi = _analysis_roi(image, roi_fractions).convert("RGB")
    roi.thumbnail((512, 512), Image.Resampling.BILINEAR)
    grayscale = roi.convert("L")
    histogram = grayscale.histogram()
    pixel_count = max(1, sum(histogram))
    p05 = _percentile(histogram, 0.05)
    p50 = _percentile(histogram, 0.50)
    p95 = _percentile(histogram, 0.95)
    dark_ratio = sum(histogram[:32]) / pixel_count
    bright_ratio = sum(histogram[240:]) / pixel_count
    statistic = ImageStat.Stat(grayscale)
    mean = float(statistic.mean[0])
    standard_deviation = float(statistic.stddev[0])
    dynamic_range = float(p95 - p05)

    if p50 < 70 or dark_ratio > 0.35:
        status = "underexposed"
    elif p50 > 190 or bright_ratio > 0.35:
        status = "overexposed"
    elif dynamic_range < 70 or standard_deviation < 28:
        status = "low_contrast"
    else:
        status = "normal"

    return {
        "roi_left_fraction": roi_fractions[0],
        "roi_top_fraction": roi_fractions[1],
        "roi_right_fraction": roi_fractions[2],
        "roi_bottom_fraction": roi_fractions[3],
        "mean_luminance": round(mean, 3),
        "luminance_std": round(standard_deviation, 3),
        "p05_luminance": float(p05),
        "p50_luminance": float(p50),
        "p95_luminance": float(p95),
        "dynamic_range": round(dynamic_range, 3),
        "dark_pixel_ratio": round(dark_ratio, 6),
        "bright_pixel_ratio": round(bright_ratio, 6),
        "quality_status": status,
    }


def suggest_recipe(metrics: dict[str, float | str]) -> dict[str, Any]:
    median = max(1.0, float(metrics["p50_luminance"]))
    dynamic_range = max(1.0, float(metrics["dynamic_range"]))
    status = str(metrics["quality_status"])

    exposure_ev = max(-2.0, min(2.0, math.log2(118.0 / median)))
    if status == "normal":
        exposure_ev = 0.0
        contrast = 1.0
    elif status == "low_contrast":
        contrast = max(1.0, min(1.50, 125.0 / dynamic_range))
    else:
        contrast = 1.0

    return {
        "recipe_version": RECIPE_VERSION,
        "exposure_ev": round(exposure_ev, 2),
        "contrast": round(contrast, 2),
        "reason": status,
        "auto_suggested": True,
        "human_accepted": False,
        "quality_before": metrics,
        "quality_after": {},
    }


def apply_photo_recipe(
    image: Image.Image,
    exposure_ev: float = DEFAULT_EXPOSURE_EV,
    contrast: float = DEFAULT_CONTRAST,
) -> Image.Image:
    result = image.convert("RGB")
    exposure_factor = 2.0 ** float(exposure_ev)
    contrast_factor = float(contrast)
    if abs(exposure_factor - 1.0) <= 1e-6 and abs(contrast_factor - 1.0) <= 1e-6:
        return result

    # Keep the saved pixels aligned with the browser's brightness/contrast preview.
    lookup = [
        max(
            0,
            min(
                255,
                int(round(((value * exposure_factor) - 127.5) * contrast_factor + 127.5)),
            ),
        )
        for value in range(256)
    ]
    return result.point(lookup * 3)


def recipe_changes_pixels(exposure_ev: float, contrast: float) -> bool:
    return abs(float(exposure_ev)) >= 0.01 or abs(float(contrast) - 1.0) >= 0.01
