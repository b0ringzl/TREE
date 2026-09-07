from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StartTaskRequest(BaseModel):
    species: str
    target_count: int = Field(ge=1, le=10_000)


class Box(BaseModel):
    class_id: int = 0
    x_center: float = Field(ge=0, le=1)
    y_center: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class Polygon(BaseModel):
    class_id: int = 0
    points: list[tuple[float, float]] = Field(min_length=3)

    @field_validator("points")
    @classmethod
    def validate_points(cls, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in points:
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError("polygon points must be normalized between 0 and 1")
        return points


class PhotoPreprocess(BaseModel):
    recipe_version: str = "photo_v1"
    exposure_ev: float = Field(default=0.0, ge=-2.0, le=2.0)
    contrast: float = Field(default=1.0, ge=0.5, le=1.5)
    reason: Literal["normal", "underexposed", "overexposed", "low_contrast", "mixed"] = "normal"
    auto_suggested: bool = False
    human_accepted: bool = False
    quality_before: dict[str, float | str] = Field(default_factory=dict)
    quality_after: dict[str, float | str] = Field(default_factory=dict)


class VisibilityReview(BaseModel):
    status: Literal["clear", "partial", "unannotatable"] = "clear"
    reason: Literal[
        "none",
        "occluded",
        "exposure_unrecoverable",
        "blur",
        "target_missing",
        "neighboring_tree",
        "other",
    ] = "none"
    note: str = Field(default="", max_length=500)


class ImageAnnotation(BaseModel):
    image: str
    polygons: list[Polygon] = Field(default_factory=list)
    boxes: list[Box] = Field(default_factory=list)
    keep: bool = True
    preprocess: PhotoPreprocess = Field(default_factory=PhotoPreprocess)
    visibility: VisibilityReview = Field(default_factory=VisibilityReview)


class SubmitRequest(BaseModel):
    tree_id: str
    species: str
    annotations: list[ImageAnnotation]


class CheckpointRequest(BaseModel):
    tree_id: str
    species: str
    annotations: list[ImageAnnotation] = Field(default_factory=list)


class RejectRequest(BaseModel):
    tree_id: str
    species: str | None = None
    annotations: list[ImageAnnotation] = Field(default_factory=list)


class ImageAnalyzeRequest(BaseModel):
    image: str


class FrameStartRequest(BaseModel):
    spacing_m: float = Field(default=5.0, ge=0.0, le=50.0)


class FrameTreeLabel(BaseModel):
    label_id: str = Field(min_length=1, max_length=80)
    species: str = Field(min_length=1, max_length=200)
    points: list[tuple[float, float]] = Field(min_length=3)
    visibility: Literal["clear", "partial", "uncertain"] = "clear"
    note: str = Field(default="", max_length=500)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("points")
    @classmethod
    def validate_label_points(
        cls, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        for x, y in points:
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError("frame label points must be normalized between 0 and 1")
        return points


class FrameAnnotationRequest(BaseModel):
    frame_id: str
    frame_status: Literal["annotated", "no_tree", "unusable"] = "annotated"
    labels: list[FrameTreeLabel] = Field(default_factory=list)
    draft_points: list[tuple[float, float]] = Field(default_factory=list)
    preprocess: PhotoPreprocess = Field(default_factory=PhotoPreprocess)
    note: str = Field(default="", max_length=1000)


class FrameAnalyzeRequest(BaseModel):
    frame_id: str


class SamSegmentRequest(BaseModel):
    image: str
    model_key: str
    point: tuple[float, float]
