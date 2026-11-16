"""Versioned visual observations; model agreement is evidence, not ground truth."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .image_functions import ImageFunctionEvidence


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Coordinate = Annotated[StrictInt, Field(ge=0, le=1000)]
Label = Annotated[str, Field(min_length=1, max_length=64, pattern=r"\S")]
Labels = Annotated[list[Label], Field(max_length=32)]


class VisualObject(StrictModel):
    label: Label
    bbox: tuple[Coordinate, Coordinate, Coordinate, Coordinate]

    @model_validator(mode="after")
    def valid_box(self) -> VisualObject:
        x0, y0, x1, y1 = self.bbox
        if x0 >= x1 or y0 >= y1:
            raise ValueError("bbox must have positive width and height")
        return self


class SceneObjects(StrictModel):
    objects: list[VisualObject] = Field(max_length=12)


class CropObjects(SceneObjects):
    candidate_id: StrictInt = Field(ge=1)


class ReviewObjects(StrictModel):
    results: list[CropObjects] = Field(max_length=12)

    @model_validator(mode="after")
    def unique_ids(self) -> ReviewObjects:
        ids = [row.candidate_id for row in self.results]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate review IDs")
        return self


class FrameReference(StrictModel):
    """Caller-supplied capture metadata, never inferred from processing time."""

    sequence_id: str = Field(min_length=1, max_length=128)
    frame_index: StrictInt = Field(ge=0)
    timestamp_ms: float = Field(ge=0)


ViewPurpose = Literal["primary", "cross_view", "discovery", "confirmation"]


class ViewObject(StrictModel):
    view_id: str
    model: str
    label: str
    bbox_px: tuple[int, int, int, int]
    clipped: bool = False
    evidence_key: str = ""


class RecognitionView(StrictModel):
    view_id: str
    purpose: ViewPurpose
    source_box_px: tuple[int, int, int, int]
    mirrored: bool
    encoded_sha256: str
    input_size: tuple[int, int]
    objects: list[ViewObject]


class ObjectObservation(StrictModel):
    observation_id: str
    candidate_id: int
    label: str
    bbox_px: tuple[int, int, int, int]
    bbox_normalized: tuple[float, float, float, float]
    status: Literal["corroborated", "needs_review"]
    reason: str
    review_label: str | None = None
    review_bbox_px: tuple[int, int, int, int] | None = None
    review_iou: float | None = None
    crop_pixel_sha256: str
    crop_mean_rgba: tuple[float, float, float, float]
    origin: ViewPurpose = "primary"
    supports: list[ViewObject] = Field(default_factory=list)


class ObjectRelation(StrictModel):
    source_id: str
    target_id: str
    center_delta_normalized: tuple[float, float]
    center_distance_px: float
    bbox_iou: float


class ModelCall(StrictModel):
    stage: Literal["detect", "review"]
    model: str
    served_model: str | None = None
    elapsed_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    output_format: Literal["object", "bare_array"] | None = None
    validation_errors: list[str] = Field(default_factory=list)
    cached: bool = False
    view_id: str | None = None
    purpose: ViewPurpose | None = None
    normalizations: list[str] = Field(default_factory=list)
    response_format: Literal["text", "json_object"] = "text"


class RecognitionResult(StrictModel):
    schema_version: Literal["svsg.recognition/1"] = "svsg.recognition/1"
    status: Literal["corroborated", "needs_review", "unavailable"]
    evidence_mode: Literal["multimodal"] = "multimodal"
    image_sha256: str
    pixel_sha256: str
    width: int
    height: int
    coordinate_space: Literal["exif_oriented_rgba8"] = "exif_oriented_rgba8"
    alpha_background: Literal["white"] = "white"
    frame: FrameReference | None = None
    id_scope: Literal["image_observation_not_track"] = "image_observation_not_track"
    label_scope: list[str]
    coverage: Literal["not_exhaustive"] = "not_exhaustive"
    review_mode: Literal["same_model_different_view", "different_backend"]
    observations: list[ObjectObservation]
    corroborated_counts: dict[str, int]
    relations: list[ObjectRelation]
    image_functions: ImageFunctionEvidence
    calls: list[ModelCall]
    warnings: list[str]
    final_answer: str
    pipeline: Literal["crop", "multiview"] = "crop"
    views: list[RecognitionView] = Field(default_factory=list)
