"""Image-to-function compilation independent of object detectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from svsg.contracts import IR, SceneType
from svsg.contracts.image_functions import (
    MAX_QUERIES,
    MAX_REGIONS,
    ColorCount,
    ImageFunctionEvidence,
    Measurement,
    PixelQuery,
    QuantizedView,
    RasterQuery,
)
from svsg.l1_compiler.compiler import ImageMeta
from svsg.l1_compiler.detector_adapter import ImageSource

from .raster import RasterFunction, region_mean
from .regions import MAX_SPANS, solve_regions

MAX_PALETTE = 4096
FUNCTION_COMPILER = "raster_function_v1"
GRID_DIM = 160


def rgba_hex(rgba: tuple[int, ...]) -> str:
    return "#" + bytes(rgba).hex().upper()


def _quantized_view(raster: RasterFunction) -> QuantizedView:
    from PIL import Image

    with raster.to_image() as image:
        image.thumbnail((GRID_DIM, GRID_DIM), Image.Resampling.BOX)
        with image.getchannel("A") as alpha:
            opaque = alpha.getextrema() == (255, 255)
        if opaque:
            with image.convert("RGB") as rgb:
                palette = rgb.quantize(
                    colors=16, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
                )
            method = "box_median_cut_16"
        else:
            palette = image.quantize(
                colors=16, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE
            )
            method = "box_octree_rgba_16"
        with palette, palette.convert("RGBA") as rgba:
            grid = RasterFunction(*rgba.size, rgba.tobytes())
    count, regions = solve_regions(grid)
    assert count is not None  # A 160×160 grid is below the 100,000-span budget.
    return QuantizedView.model_validate(
        {
            "width": grid.width,
            "height": grid.height,
            "method": method,
            "component_count": count,
            "regions": regions,
        }
    )


def describe_raster(
    raster: RasterFunction,
    queries: Sequence[RasterQuery] = (),
    *,
    span_limit: int = MAX_SPANS,
) -> ImageFunctionEvidence:
    if len(queries) > MAX_QUERIES or len({q.id for q in queries}) != len(queries):
        raise ValueError("too many function queries or duplicate query IDs")
    with raster.to_image() as image:
        # Pillow returns tuples of channel values for RGBA mode.
        raw_colors = cast(list[tuple[int, tuple[int, ...]]] | None, image.getcolors(MAX_PALETTE))
        colors = sorted(raw_colors or [], key=lambda pair: (-pair[0], pair[1]))
        measurements = []
        for query in queries:
            if isinstance(query, PixelQuery):
                value = rgba_hex(raster.pixel(query.x, query.y))
            else:
                value = ",".join(f"{channel:.2f}" for channel in region_mean(image, query.box))
            measurements.append(Measurement(query=query, value=value))
    component_count, regions = solve_regions(raster, span_limit=span_limit)
    crowded = component_count is None or component_count > MAX_REGIONS * 4
    quantized = _quantized_view(raster) if raw_colors is None or crowded else None
    return ImageFunctionEvidence(
        width_px=raster.width,
        height_px=raster.height,
        pixel_sha256=raster.pixel_sha256,
        unique_colors=len(colors) if raw_colors is not None else None,
        top_colors=[ColorCount(rgba=rgba_hex(color), area_px=area) for area, color in colors[:16]],
        component_count=component_count,
        regions=regions,
        measurements=measurements,
        quantized=quantized,
    )


@dataclass(frozen=True)
class FunctionImageCompiler:
    detector_model = FUNCTION_COMPILER

    def compile(
        self, image: ImageSource, meta: ImageMeta, *, queries: Sequence[RasterQuery] = ()
    ) -> IR:
        evidence = describe_raster(RasterFunction.from_image(image), queries)
        return IR(
            image_id=meta.image_id,
            image_width=meta.width,
            image_height=meta.height,
            timestamp=meta.timestamp,
            scene_type=SceneType.UNKNOWN,
            detector_model=self.detector_model,
            image_functions=evidence,
        )
