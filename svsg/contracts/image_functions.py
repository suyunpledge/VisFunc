"""Versioned parameters and calls for the deterministic raster compiler."""

from __future__ import annotations

import re
from math import isclose
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator

from .contour import Contour

MAX_RASTER_PIXELS = 16 * 1024 * 1024
MAX_REGIONS = 64
MAX_QUERIES = 16
RGBA = Annotated[str, Field(pattern=r"^#[0-9A-F]{8}$")]
QueryId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_]{1,32}$")]
Coordinate = Annotated[StrictInt, Field(ge=0)]
Box = tuple[Coordinate, Coordinate, Coordinate, Coordinate]


class _Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PixelQuery(_Parameters):
    id: QueryId
    op: Literal["pixel_rgba"]
    x: Coordinate
    y: Coordinate


class MeanQuery(_Parameters):
    id: QueryId
    op: Literal["mean_rgba"]
    box: Box

    @model_validator(mode="after")
    def _valid_box(self) -> MeanQuery:
        x0, y0, x1, y1 = self.box
        if x0 >= x1 or y0 >= y1:
            raise ValueError("mean_rgba requires a nonempty half-open box")
        return self


RasterQuery = Annotated[PixelQuery | MeanQuery, Field(discriminator="op")]
RASTER_QUERIES: TypeAdapter[list[RasterQuery]] = TypeAdapter(
    Annotated[list[RasterQuery], Field(max_length=MAX_QUERIES)]
)


class Measurement(_Parameters):
    query: RasterQuery
    value: str = Field(max_length=64)

    @model_validator(mode="after")
    def _value_shape(self) -> Measurement:
        if isinstance(self.query, PixelQuery):
            if re.fullmatch(r"#[0-9A-F]{8}", self.value) is None:
                raise ValueError("pixel_rgba must be a hexadecimal RGBA value")
        elif re.fullmatch(r"\d{1,3}\.\d{2}(?:,\d{1,3}\.\d{2}){3}", self.value) is None or any(
            float(part) > 255 for part in self.value.split(",")
        ):
            raise ValueError("mean_rgba must contain four channel means")
        return self


class ColorCount(_Parameters):
    rgba: RGBA
    area_px: int = Field(gt=0)


class RegionParameters(_Parameters):
    id: StrictInt = Field(gt=0)
    rgba: RGBA
    box: Box
    area_px: StrictInt = Field(gt=0)
    perimeter_px: StrictInt = Field(gt=0)
    centroid_x: float
    centroid_y: float
    contour: Contour | None = None

    @model_validator(mode="after")
    def _geometry(self) -> RegionParameters:
        x0, y0, x1, y1 = self.box
        if self.area_px > (x1 - x0) * (y1 - y0) or x0 >= x1 or y0 >= y1:
            raise ValueError("region area is inconsistent with its bounding box")
        if not (x0 < self.centroid_x < x1 and y0 < self.centroid_y < y1):
            raise ValueError("region centroid is outside its bounding box")
        if self.contour is not None:
            contour = self.contour
            if (
                contour.bounds != self.box
                or contour.area2 != 2 * self.area_px
                or contour.perimeter != self.perimeter_px
                or not isclose(contour.centroid[0], self.centroid_x, abs_tol=1e-9)
                or not isclose(contour.centroid[1], self.centroid_y, abs_tol=1e-9)
            ):
                raise ValueError("contour does not reproduce region measurements")
        return self

    @property
    def fill_ratio(self) -> float:
        x0, y0, x1, y1 = self.box
        return self.area_px / ((x1 - x0) * (y1 - y0))

    @property
    def shape(self) -> str:
        return "filled_rectangle" if self.fill_ratio == 1 else "connected_region"

    @property
    def outline_kind(self) -> str | None:
        if self.contour is None:
            return None
        if self.fill_ratio == 1:
            return "filled_rectangle"
        if self.contour.holes:
            return "holed_region"
        return "convex_region" if self.contour.solidity == 1 else "concave_region"


class QuantizedView(_Parameters):
    """Approximate image segmentation. All geometry lives on this working grid."""

    width: StrictInt = Field(gt=0, le=160)
    height: StrictInt = Field(gt=0, le=160)
    method: Literal["box_median_cut_16", "box_octree_rgba_16"]
    component_count: StrictInt = Field(gt=0)
    regions: list[RegionParameters] = Field(max_length=MAX_REGIONS)

    @model_validator(mode="after")
    def _coverage(self) -> QuantizedView:
        area = self.width * self.height
        if not len(self.regions) <= self.component_count <= area:
            raise ValueError("invalid working-grid component count")
        if len({r.id for r in self.regions}) != len(self.regions):
            raise ValueError("duplicate working-grid IDs")
        if any(r.box[2] > self.width or r.box[3] > self.height for r in self.regions):
            raise ValueError("working-grid region bounds out of range")
        if sum(r.area_px for r in self.regions) > area:
            raise ValueError("working-grid region area out of range")
        return self


class ImageFunctionEvidence(_Parameters):
    version: Literal["svsg.functions/1"] = "svsg.functions/1"
    width_px: StrictInt = Field(gt=0)
    height_px: StrictInt = Field(gt=0)
    pixel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    unique_colors: StrictInt | None = Field(default=None, gt=0)
    top_colors: list[ColorCount] = Field(default_factory=list, max_length=16)
    component_count: StrictInt | None = Field(default=None, gt=0)
    regions: list[RegionParameters] = Field(default_factory=list, max_length=MAX_REGIONS)
    measurements: list[Measurement] = Field(default_factory=list, max_length=MAX_QUERIES)
    quantized: QuantizedView | None = None

    @model_validator(mode="after")
    def _coverage(self) -> ImageFunctionEvidence:
        width, height = self.width_px, self.height_px
        area = width * height
        if area > MAX_RASTER_PIXELS:
            raise ValueError("function evidence exceeds the raster pixel budget")
        if self.quantized is not None and (
            self.quantized.width > width or self.quantized.height > height
        ):
            raise ValueError("working grid cannot exceed source dimensions")
        if len({row.id for row in self.regions}) != len(self.regions):
            raise ValueError("duplicate region IDs")
        if len({row.query.id for row in self.measurements}) != len(self.measurements):
            raise ValueError("duplicate query IDs")
        for row in self.measurements:
            query = row.query
            if isinstance(query, PixelQuery):
                valid = query.x < width and query.y < height
            else:
                valid = query.box[2] <= width and query.box[3] <= height
            if not valid:
                raise ValueError("measurement is outside the raster")
        if self.component_count is None:
            if self.regions:
                raise ValueError("regions require a completed component scan")
        elif not len(self.regions) <= self.component_count <= area:
            raise ValueError("region count is inconsistent with image area")
        if any(r.box[2] > width or r.box[3] > height for r in self.regions):
            raise ValueError("region bounds exceed image dimensions")
        if sum(r.area_px for r in self.regions) > area:
            raise ValueError("region areas exceed image area")
        if self.unique_colors is None:
            if self.top_colors:
                raise ValueError("color counts require a completed palette scan")
        elif not len(self.top_colors) <= self.unique_colors <= area:
            raise ValueError("color count is inconsistent with image area")
        if sum(row.area_px for row in self.top_colors) > area:
            raise ValueError("palette areas exceed image area")
        return self
