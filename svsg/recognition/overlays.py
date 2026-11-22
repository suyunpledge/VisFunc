"""Measure screen overlays on a still image or on the frames in a function manifest."""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path

from svsg.image_functions.raster import RasterFunction
from svsg.recognition.geometry import Box
from svsg.recognition.icon_matching import load_icons, match_icons
from svsg.recognition.subtitle_ocr import SubtitleReader, split_caption_band


def toolbar_region(w: int, h: int) -> Box:
    return 0, int(h * 0.85), w, h


def _region(box: Box | None, default: Box, w: int, h: int) -> Box:
    x, y, right, bottom = box if box is not None else default
    if not (0 <= x < right <= w and 0 <= y < bottom <= h):
        raise ValueError("overlay region must be a nonempty box within the source frame")
    return x, y, right, bottom


def _frames(
    source: Path, frame: int | None
) -> Iterator[tuple[str | None, int | None, float | None, RasterFunction]]:
    if source.suffix.lower() == ".json":
        from svsg.video_functions import VideoRuntime

        runtime = VideoRuntime.load(source)
        for ref in runtime.bundle.frames:
            if frame is None or ref.frame_index == frame:
                yield (
                    runtime.bundle.sequence_id,
                    ref.frame_index,
                    ref.timestamp_ms,
                    runtime.raster(ref.frame_index),
                )
        if frame is not None and not any(f.frame_index == frame for f in runtime.bundle.frames):
            raise ValueError("requested frame was not exported")
    else:
        if frame is not None:
            raise ValueError("--frame requires a video function manifest")
        yield None, None, None, RasterFunction.from_image(source)


def measure_overlays(
    source: Path,
    *,
    frame: int | None = None,
    only: str = "both",
    subtitle_region: Box | None = None,
    icon_region: Box | None = None,
    icon_dir: Path | None = None,
    player_toolbar: bool = False,
) -> dict:
    import cv2
    import numpy as np

    if only not in {"both", "subtitles", "icons"}:
        raise ValueError("unknown overlay mode")
    if player_toolbar and icon_region is not None:
        raise ValueError("choose a player toolbar or an explicit icon region")
    reader = SubtitleReader() if only != "icons" else None
    icons = load_icons(icon_dir) if only != "subtitles" else []
    frames = []
    previous = None
    measurements: dict = {}
    for sequence, index, timestamp, raster in _frames(source, frame):
        if raster.pixel_sha256 != previous:
            w, h = raster.width, raster.height
            rgba = np.frombuffer(raster.samples, dtype=np.uint8).reshape(h, w, 4)
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            if rgba[:, :, 3].min() < 255:
                alpha = rgba[:, :, 3:4].astype(np.uint16)
                bgr = ((bgr * alpha + 255 * (255 - alpha) + 127) // 255).astype(np.uint8)
            measurements = {}
            if reader is not None:
                margin = w // 20
                region = _region(subtitle_region, (margin, int(h * 0.52), w - margin, h), w, h)
                measurements["subtitle_region_px"] = region
                lines = reader.read(
                    bgr, region, centered=subtitle_region is None
                )
                if subtitle_region is None:
                    lines, detached = split_caption_band(lines, h)
                    if detached:
                        measurements["detached_text_candidates"] = detached
                measurements["subtitles"] = lines
            if icons:
                default = toolbar_region(w, h) if player_toolbar else (0, 0, w, h)
                region = _region(icon_region, default, w, h)
                measurements["icon_region_px"] = region
                measurements["icons"] = match_icons(bgr, region, icons, review_row=player_toolbar)
            previous = raster.pixel_sha256
        frames.append(
            dict(
                sequence_id=sequence,
                frame_index=index,
                timestamp_ms=timestamp,
                pixel_sha256=previous,
                size_px=[raster.width, raster.height],
                **measurements,
            )
        )
    packet: dict = dict(
        version="svsg.overlays/1",
        coordinate_space="source_pixels_xyxy",
        status="recognition_candidates",
        frames=frames,
    )
    if reader is not None:
        packet["subtitle_engine"] = {
            "package": "rapidocr-onnxruntime",
            "version": version("rapidocr-onnxruntime"),
        }
    if icons:
        packet["icon_templates"] = {
            icon.template_id or icon.label: icon.template_sha256 for icon in icons
        }
    return packet
