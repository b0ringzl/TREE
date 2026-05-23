from __future__ import annotations

from pydantic import BaseModel, Field


class StartTaskRequest(BaseModel):
    species: str
    target_count: int = Field(ge=1, le=10_000)


class Box(BaseModel):
    class_id: int = 0
    x_center: float = Field(ge=0, le=1)
    y_center: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class ImageAnnotation(BaseModel):
    image: str
    boxes: list[Box]
    keep: bool = True


class SubmitRequest(BaseModel):
    tree_id: str
    species: str
    annotations: list[ImageAnnotation]


class RejectRequest(BaseModel):
    tree_id: str
    species: str | None = None
