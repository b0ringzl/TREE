from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class StartTaskRequest(BaseModel):
    species: str
    target_count: int = Field(ge=1, le=10_000)


class ApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1)


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
                raise ValueError("polygon points must be normalized coordinates between 0 and 1")
        return points


class ImageAnnotation(BaseModel):
    image: str
    polygons: list[Polygon] = Field(default_factory=list)
    boxes: list[Box] = Field(default_factory=list)
    keep: bool = True


class SubmitRequest(BaseModel):
    tree_id: str
    species: str
    annotations: list[ImageAnnotation]


class RejectRequest(BaseModel):
    tree_id: str
    species: str | None = None


class SamSegmentRequest(BaseModel):
    image: str
    model_key: str
    point: tuple[float, float]

    @field_validator("point")
    @classmethod
    def validate_point(cls, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("SAM point must be normalized coordinates between 0 and 1")
        return point
