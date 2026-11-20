"""Bounded data-only evaluation: exact pixels, signed differences and temporal estimates."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from hashlib import sha256
from operator import itemgetter
from pathlib import Path

from svsg.contracts.video_functions import (
    VIDEO_QUERY,
    DifferenceQuery,
    FittedProgress,
    MotionQuery,
    SampledFunction,
    VideoFunctions,
    VideoMeanQuery,
    VideoPixelQuery,
)
from svsg.image_functions.raster import RasterFunction


def evaluate(function: SampledFunction | FittedProgress, t_ms: float) -> dict:
    if type(t_ms) not in (int, float) or not math.isfinite(t_ms):
        raise ValueError("time must be a finite number")
    if isinstance(function, SampledFunction):
        knots = function.knots
        if not knots[0][0] <= t_ms <= knots[-1][0]:
            raise ValueError("time outside function support; extrapolation is disabled")
        index = bisect_left(knots, t_ms, key=itemgetter(0))
        b, vb = knots[index]
        if b == t_ms:
            value = vb
            support, mode = [b], "sample"
        else:
            a, va = knots[index - 1]
            weight = (t_ms - a) / (b - a)
            value = va * (1 - weight) + vb * weight
            support, mode = [a, b], "interpolated"
        return {
            "value": value,
            "unit": function.unit,
            "evaluation": mode,
            "provenance": function.provenance,
            "support_ms": support,
        }
    if not function.t_start_ms <= t_ms <= function.t_end_ms:
        raise ValueError("time outside fitted interval; extrapolation is disabled")
    if function.op == "cubic_bezier":
        tau = (t_ms - function.t_start_ms) / (function.t_end_ms - function.t_start_ms)
        x1, y1, x2, y2 = function.params
        lo, hi = 0.0, 1.0
        for _ in range(48):
            u = (lo + hi) / 2
            x = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u**2 * x2 + u**3
            if x < tau:
                lo = u
            else:
                hi = u
        u = (lo + hi) / 2
        value = 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u**2 * y2 + u**3
    else:
        damping, omega = function.params
        seconds = (t_ms - function.t_start_ms) / 1000
        wd = omega * math.sqrt(1 - damping**2)
        value = 1 - math.exp(-damping * omega * seconds) * (
            math.cos(wd * seconds) + damping * omega / wd * math.sin(wd * seconds)
        )
    return {
        "value": value,
        "unit": "ratio",
        "evaluation": "fit",
        "provenance": "fit",
        "rmse": function.rmse,
        "support_ms": [function.t_start_ms, function.t_end_ms],
    }


class VideoRuntime:
    def __init__(self, bundle: VideoFunctions, artifact_dir: str | Path):
        self.bundle = bundle
        self.root = Path(artifact_dir)
        self._frames = {f.frame_index: f for f in bundle.frames}
        self._cached: tuple[str, RasterFunction] | None = None

    @classmethod
    def load(cls, manifest: str | Path) -> VideoRuntime:
        from .bridge import read_artifact

        path = Path(manifest)
        bundle = VideoFunctions.model_validate_json(read_artifact(path.parent, path.name))
        return cls(bundle, path.parent)

    def raster(self, frame_index: int) -> RasterFunction:
        from .bridge import read_artifact

        if type(frame_index) is not int or frame_index not in self._frames:
            raise ValueError("frame was not exported; no intermediate image is inferred")
        ref = self._frames[frame_index]
        if self._cached is not None and self._cached[0] == ref.raster_sha256:
            raster = self._cached[1]
        else:
            data = read_artifact(self.root, ref.raster_file)
            if sha256(data).hexdigest() != ref.raster_sha256:
                raise ValueError("raster artifact hash mismatch")
            raster = RasterFunction.from_program(json.loads(data))
        if (raster.width, raster.height) != (
            self.bundle.width_px,
            self.bundle.height_px,
        ) or raster.pixel_sha256 != ref.image_functions.pixel_sha256:
            raise ValueError("raster does not match its frame manifest")
        self._cached = (ref.raster_sha256, raster)
        return raster

    def pixel(self, frame_index: int, x: int, y: int) -> tuple[int, int, int, int]:
        if type(x) is not int or type(y) is not int:
            raise ValueError("pixel coordinates must be integers")
        return self.raster(frame_index).pixel(x, y)

    def call(self, parameters: dict) -> dict:
        """Same pixel_rgba/mean_rgba operations as image queries, plus explicit frame/time."""
        query = VIDEO_QUERY.validate_python(parameters)
        result = {"id": query.id, "op": query.op, "sequence_id": self.bundle.sequence_id}
        if isinstance(query, MotionQuery):
            return {
                **result,
                **self.motion(
                    query.segment_id, query.name, query.t_ms, fitted=query.mode == "fitted"
                ),
            }
        if isinstance(query, DifferenceQuery):
            value = self.delta_pixel(query.before_frame, query.after_frame, query.x, query.y)
            indices = [query.before_frame, query.after_frame]
        elif isinstance(query, VideoPixelQuery):
            value = self.pixel(query.frame_index, query.x, query.y)
            indices = [query.frame_index]
        elif isinstance(query, VideoMeanQuery):
            value = self.raster(query.frame_index).mean(query.box)
            indices = [query.frame_index]
        return {
            **result,
            "value": value,
            "provenance": "decoded_pixels",
            "frames": [
                {
                    "frame_index": i,
                    "timestamp_ms": self._frames[i].timestamp_ms,
                    "pixel_sha256": self._frames[i].image_functions.pixel_sha256,
                }
                for i in indices
            ],
        }

    def delta_pixel(self, before_frame: int, after_frame: int, x: int, y: int) -> tuple:
        if not any(
            d.before_frame == before_frame and d.after_frame == after_frame
            for d in self.bundle.differences
        ):
            raise ValueError("difference is not in the exported evidence chain")
        a, b = self.pixel(before_frame, x, y), self.pixel(after_frame, x, y)
        return tuple(after - before for before, after in zip(a, b, strict=True))

    def motion(self, segment_id: str, name: str, t_ms: float, *, fitted: bool = False) -> dict:
        segment = next((s for s in self.bundle.motions if s.segment_id == segment_id), None)
        if segment is None:
            raise ValueError("unknown motion segment")
        functions = segment.fits if fitted else segment.samples
        if name not in functions:
            raise ValueError("function not available for this segment")
        return {
            **evaluate(functions[name], t_ms),
            "segment_id": segment_id,
            "confidence": segment.confidence,
            "truncated": segment.truncated,
            "inferred_channels": segment.inferred_channels,
            "lower_bound_channels": segment.lower_bound_channels,
            "id_scope": segment.id_scope,
        }
