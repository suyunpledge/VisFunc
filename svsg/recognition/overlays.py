"""Measure screen overlays on a still image or on the frames in a function manifest."""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path

from svsg.image_functions.raster import RasterFunction
from svsg.recognition.geometry import Box
from svsg.recognition.icon_matching import load_icons, match_icons, supplement_toolbar
from svsg.recognition.subtitle_ocr import SubtitleReader, split_caption_band
from svsg.recognition.subtitle_sequence import CONTEXT_MS, review_caption_sequence, within_window


def toolbar_region(w: int, h: int) -> Box:
    return 0, int(h * 0.85), w, h


def _region(box: Box | None, default: Box, w: int, h: int) -> Box:
    x, y, right, bottom = box if box is not None else default
    if not (0 <= x < right <= w and 0 <= y < bottom <= h):
        raise ValueError("overlay region must be a nonempty box within the source frame")
    return x, y, right, bottom


def _frames(
    source: Path, frame: int | None, *, context_ms: int = 0
) -> Iterator[tuple[str | None, int | None, float | None, RasterFunction]]:
    if source.suffix.lower() == ".json":
        from svsg.video_functions import VideoRuntime

        runtime = VideoRuntime.load(source)
        stamp = None
        if frame is not None:
            target = next((f for f in runtime.bundle.frames if f.frame_index == frame), None)
            if target is None:
                raise ValueError("requested frame was not exported")
            stamp = target.timestamp_ms
        for ref in runtime.bundle.frames:
            if (
                stamp is None
                or ref.frame_index == frame
                or (context_ms and within_window(ref.timestamp_ms, stamp, context_ms))
            ):
                yield (
                    runtime.bundle.sequence_id,
                    ref.frame_index,
                    ref.timestamp_ms,
                    runtime.raster(ref.frame_index),
                )
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
    fallback_icon_dir: Path | None = None,
    player_toolbar: bool = False,
) -> dict:
    import cv2
    import numpy as np

    if only not in {"both", "subtitles", "icons"}:
        raise ValueError("unknown overlay mode")
    if fallback_icon_dir is not None:
        if icon_dir is not None or only == "subtitles":
            raise ValueError("toolbar fallback requires generic icon recognition")
        player_toolbar = True
    if player_toolbar and icon_region is not None:
        raise ValueError("choose a player toolbar or an explicit icon region")
    reader = SubtitleReader() if only != "icons" else None
    icons = load_icons(icon_dir, player_toolbar=player_toolbar) if only != "subtitles" else []
    fallback = load_icons(fallback_icon_dir) if fallback_icon_dir is not None else []
    frames = []
    previous = None
    measurements: dict = {}
    temporal = reader is not None and subtitle_region is None and source.suffix.lower() == ".json"
    for sequence, index, timestamp, raster in _frames(
        source, frame, context_ms=CONTEXT_MS if temporal else 0
    ):
        need_icons = bool(icons) and (frame is None or frame == index)
        key = (raster.pixel_sha256, need_icons)
        if previous != key:
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
                lines = reader.read(bgr, region, centered=subtitle_region is None)
                if subtitle_region is None:
                    lines, detached = split_caption_band(lines, h, player_toolbar=player_toolbar)
                    if detached:
                        measurements["detached_text_candidates"] = detached
                measurements["subtitles"] = lines
            if need_icons:
                default = toolbar_region(w, h) if player_toolbar else (0, 0, w, h)
                region = _region(icon_region, default, w, h)
                measurements["icon_region_px"] = region
                measurements["icons"] = match_icons(bgr, region, icons, review_row=player_toolbar)
                if fallback:
                    measurements["icons"] = supplement_toolbar(
                        measurements["icons"], match_icons(bgr, region, fallback, review_row=True)
                    )
            previous = key
        frames.append(
            dict(
                sequence_id=sequence,
                frame_index=index,
                timestamp_ms=timestamp,
                pixel_sha256=raster.pixel_sha256,
                size_px=[raster.width, raster.height],
                **measurements,
            )
        )
    packet: dict = dict(
        version="svsg.overlays/1",
        coordinate_space="source_pixels_xyxy",
        status="recognition_candidates",
        frames=review_caption_sequence(frames, frame) if temporal else frames,
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
        packet["icon_templates"].update(
            {
                "reference/" + (icon.template_id or icon.label): icon.template_sha256
                for icon in fallback
            }
        )
    return packet
