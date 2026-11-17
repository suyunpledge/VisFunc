"""Shared image/time function contract; sampled pixels and inferred motion stay separate."""

from __future__ import annotations

from itertools import pairwise
from math import isclose
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, TypeAdapter, model_validator

from .image_functions import Coordinate, ImageFunctionEvidence, MeanQuery, PixelQuery, QueryId
from .recognition import ModelCall, ObjectObservation, ObjectRelation, StrictModel

Number = StrictFloat | StrictInt
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Unit = Literal["source_px", "ratio", "degrees", "luma_0_255"]


class VideoPixelQuery(PixelQuery):
    frame_index: StrictInt = Field(ge=0)


class VideoMeanQuery(MeanQuery):
    frame_index: StrictInt = Field(ge=0)


class DifferenceQuery(StrictModel):
    id: QueryId
    op: Literal["delta_rgba"]
    before_frame: StrictInt = Field(ge=0)
    after_frame: StrictInt = Field(ge=0)
    x: Coordinate
    y: Coordinate


class MotionQuery(StrictModel):
    id: QueryId
    op: Literal["motion_value"]
    segment_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    t_ms: Number = Field(ge=0)
    mode: Literal["sampled", "fitted"] = "sampled"


VideoQuery = Annotated[
    VideoPixelQuery | VideoMeanQuery | DifferenceQuery | MotionQuery,
    Field(discriminator="op"),
]
VIDEO_QUERY: TypeAdapter[VideoQuery] = TypeAdapter(VideoQuery)


class SampledFunction(StrictModel):
    op: Literal["piecewise_linear"] = "piecewise_linear"
    unit: Unit
    knots: list[tuple[Number, Number]] = Field(min_length=1, max_length=4096)
    provenance: Literal["motion_estimate"] = "motion_estimate"

    @model_validator(mode="after")
    def ordered(self) -> SampledFunction:
        if self.knots[0][0] < 0 or any(a[0] >= b[0] for a, b in pairwise(self.knots)):
            raise ValueError("function times must be nonnegative and strictly increasing")
        return self


class FittedProgress(StrictModel):
    op: Literal["cubic_bezier", "damped_spring"]
    t_start_ms: Number = Field(ge=0)
    t_end_ms: Number = Field(gt=0)
    params: list[Number] = Field(min_length=2, max_length=4)
    rmse: Number = Field(ge=0)
    unit: Literal["ratio"] = "ratio"
    provenance: Literal["fit"] = "fit"

    @model_validator(mode="after")
    def valid(self) -> FittedProgress:
        if self.t_end_ms <= self.t_start_ms:
            raise ValueError("fit must have a positive time interval")
        if self.op == "cubic_bezier":
            if len(self.params) != 4 or any(not 0 <= self.params[i] <= 1 for i in (0, 2)):
                raise ValueError("invalid cubic-bezier control points")
        elif len(self.params) != 2 or not (0 < self.params[0] < 1 and self.params[1] > 0):
            raise ValueError("spring requires damping in (0,1) and positive omega in rad/s")
        return self


class TimeInterval(StrictModel):
    t_start_ms: Number = Field(ge=0)
    t_end_ms: Number = Field(ge=0)
    duration_ms: Number = Field(ge=0)

    @model_validator(mode="after")
    def consistent(self) -> TimeInterval:
        if self.t_end_ms < self.t_start_ms or not isclose(
            self.duration_ms, self.t_end_ms - self.t_start_ms, rel_tol=0, abs_tol=1e-5
        ):
            raise ValueError("duration must equal interval end minus start")
        return self


class DesignTiming(TimeInterval):
    easing: str = Field(min_length=1, max_length=64)
    basis_t_start_ms: Number = Field(ge=0)
    start_shift_ms: Number
    rmse: Number = Field(ge=0)
    rmse_limit: StrictFloat = Field(default=0.05, ge=0.05, le=0.05)
    provenance: Literal["named_curve_alignment"] = "named_curve_alignment"
    is_ground_truth: Literal[False] = False

    @model_validator(mode="after")
    def aligned(self) -> DesignTiming:
        if (
            self.duration_ms <= 0
            or self.rmse > self.rmse_limit
            or not isclose(
                self.t_start_ms,
                self.basis_t_start_ms + self.start_shift_ms,
                rel_tol=0,
                abs_tol=1e-5,
            )
        ):
            raise ValueError("invalid design timing alignment or fit quality")
        return self


class TimingEvidence(StrictModel):
    visible_change: TimeInterval
    estimated_design: DesignTiming | None
    design_unavailable_reason: (
        Literal[
            "truncated_segment",
            "no_named_fit",
            "invalid_alignment",
            "poor_named_fit",
            "estimate_before_timeline",
            "legacy_missing_timing",
        ]
        | None
    )

    @model_validator(mode="after")
    def availability(self) -> TimingEvidence:
        if (self.estimated_design is None) != (self.design_unavailable_reason is not None):
            raise ValueError("missing design timing requires an explicit reason")
        return self


class MotionTiming(TimingEvidence):
    version: Literal["svsg.motion-timing/1"] = "svsg.motion-timing/1"
    unit: Literal["ms"] = "ms"
    clock: Literal["source_timeline"] = "source_timeline"
    primary_channel: str | None = Field(default=None, max_length=64)
    channels: dict[str, TimingEvidence] = Field(default_factory=dict, max_length=16)

    @model_validator(mode="after")
    def primary(self) -> MotionTiming:
        if self.primary_channel is not None:
            if self.primary_channel not in self.channels:
                raise ValueError("unknown primary timing channel")
            primary = self.channels[self.primary_channel]
            if (
                primary.estimated_design != self.estimated_design
                or primary.design_unavailable_reason != self.design_unavailable_reason
            ):
                raise ValueError("primary channel and segment design timing disagree")
        return self


class MotionFunctions(StrictModel):
    segment_id: str = Field(min_length=1, max_length=128)
    kind: Literal["transform", "fade", "instant", "nonrigid"]
    confidence: Literal["low", "medium", "high"]
    t_start_ms: Number = Field(ge=0)
    t_end_ms: Number = Field(ge=0)
    # Legacy t_start_ms/t_end_ms keep their visible-support meaning.
    # Missing timing in older stored bundles must not become a design estimate.
    timing: MotionTiming | None = None
    region: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    truncated: bool = False
    reference: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    # These flags remain visible even where hidden portions have fitted sample values.
    inferred_channels: list[str] = Field(default_factory=list)
    lower_bound_channels: list[str] = Field(default_factory=list)
    samples: dict[str, SampledFunction] = Field(max_length=32)
    fits: dict[str, FittedProgress] = Field(default_factory=dict, max_length=16)
    id_scope: Literal["motion_segment_not_object_track"] = "motion_segment_not_object_track"

    @model_validator(mode="after")
    def domains(self) -> MotionFunctions:
        if self.t_end_ms < self.t_start_ms:
            raise ValueError("invalid motion interval")
        if self.timing is not None:
            visible = self.timing.visible_change
            if (visible.t_start_ms, visible.t_end_ms) != (self.t_start_ms, self.t_end_ms):
                raise ValueError("visible timing disagrees with the motion interval")
            for channel in self.timing.channels.values():
                interval = channel.visible_change
                if not self.t_start_ms <= interval.t_start_ms <= interval.t_end_ms <= self.t_end_ms:
                    raise ValueError("channel visible timing exceeds its segment")
            if self.truncated and (
                self.timing.estimated_design is not None
                or any(
                    channel.estimated_design is not None
                    for channel in self.timing.channels.values()
                )
            ):
                raise ValueError("truncated motion cannot claim a complete design timing")
        for curve in self.samples.values():
            if not self.t_start_ms <= curve.knots[0][0] <= curve.knots[-1][0] <= self.t_end_ms:
                raise ValueError("sample function exceeds its segment")
        for fit in self.fits.values():
            if not self.t_start_ms <= fit.t_start_ms < fit.t_end_ms <= self.t_end_ms:
                raise ValueError("fit exceeds its segment")
        return self


class FrameSemantics(StrictModel):
    status: Literal["corroborated", "needs_review", "unavailable"]
    label_scope: list[str]
    review_mode: Literal["same_model_different_view", "different_backend"]
    observations: list[ObjectObservation]
    relations: list[ObjectRelation]
    calls: list[ModelCall]
    warnings: list[str]
    coverage: Literal["not_exhaustive"] = "not_exhaustive"


class FrameFunctions(StrictModel):
    frame_index: StrictInt = Field(ge=0)
    timestamp_ms: Number = Field(ge=0)
    image_sha256: Digest
    roles: list[str]
    image_functions: ImageFunctionEvidence
    # Content-addressed exact raster lives outside the model-facing summary.
    raster_file: str = Field(pattern=r"^raster-[0-9a-f]{64}\.json$")
    raster_sha256: Digest
    semantics: FrameSemantics | None = None


class FrameDifference(StrictModel):
    op: Literal["subtract_rgba"] = "subtract_rgba"
    before_frame: StrictInt = Field(ge=0)
    after_frame: StrictInt = Field(ge=0)
    elapsed_ms: Number = Field(gt=0)
    changed_pixels: StrictInt = Field(ge=0)
    changed_bbox: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | None
    mean_abs_rgba: tuple[Number, Number, Number, Number]
    meaning: Literal["pixel_change_not_object_motion"] = "pixel_change_not_object_motion"


class VideoFunctions(StrictModel):
    version: Literal["svsg.video-functions/1"] = "svsg.video-functions/1"
    sequence_id: str = Field(min_length=1, max_length=128)
    source_spec_sha256: Digest
    width_px: StrictInt = Field(gt=0)
    height_px: StrictInt = Field(gt=0)
    t_start_ms: Number = Field(ge=0)
    t_end_ms: Number = Field(ge=0)
    clock: Literal["container", "index"]
    time_resolution_ms: Number = Field(gt=0)
    coordinate_space: Literal["decoded_source_rgba8"] = "decoded_source_rgba8"
    coverage: Literal["sampled_frames_only"] = "sampled_frames_only"
    frames: list[FrameFunctions] = Field(min_length=1, max_length=256)
    differences: list[FrameDifference] = Field(max_length=255)
    motions: list[MotionFunctions] = Field(default_factory=list, max_length=256)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self) -> VideoFunctions:
        if self.t_end_ms < self.t_start_ms:
            raise ValueError("invalid source interval")
        for frame in self.frames:
            f = frame.image_functions
            if (f.width_px, f.height_px) != (self.width_px, self.height_px):
                raise ValueError("frame dimensions differ from source")
            if not self.t_start_ms <= frame.timestamp_ms <= self.t_end_ms:
                raise ValueError("frame outside selected interval")
            if frame.raster_file != f"raster-{frame.raster_sha256}.json":
                raise ValueError("raster filename does not match artifact hash")
        for a, b in pairwise(self.frames):
            if a.frame_index >= b.frame_index or a.timestamp_ms >= b.timestamp_ms:
                raise ValueError("frame indices and times must be strictly increasing")
        if len(self.differences) != len(self.frames) - 1:
            raise ValueError("one difference per adjacent evidence pair is required")
        for d, (a, b) in zip(self.differences, pairwise(self.frames), strict=True):
            if (d.before_frame, d.after_frame) != (a.frame_index, b.frame_index):
                raise ValueError("difference frame references are inconsistent")
            if abs(d.elapsed_ms - (b.timestamp_ms - a.timestamp_ms)) > 1e-6:
                raise ValueError("difference uses an incorrect time interval")
            if not 0 <= d.changed_pixels <= self.width_px * self.height_px:
                raise ValueError("invalid changed-pixel count")
            if (d.changed_bbox is None) != (d.changed_pixels == 0):
                raise ValueError("changed bbox inconsistent with count")
            if d.changed_bbox is not None:
                x0, y0, x1, y1 = d.changed_bbox
                if not (0 <= x0 < x1 <= self.width_px and 0 <= y0 < y1 <= self.height_px):
                    raise ValueError("difference bbox exceeds source")
            if any(not 0 <= v <= 255 for v in d.mean_abs_rgba):
                raise ValueError("invalid channel difference")
        if len({m.segment_id for m in self.motions}) != len(self.motions):
            raise ValueError("duplicate motion IDs")
        for m in self.motions:
            x0, y0, x1, y1 = m.region
            if not (0 <= x0 < x1 <= self.width_px and 0 <= y0 < y1 <= self.height_px):
                raise ValueError("motion region exceeds source")
            if not self.t_start_ms <= m.t_start_ms <= m.t_end_ms <= self.t_end_ms:
                raise ValueError("motion outside source interval")
        return self
