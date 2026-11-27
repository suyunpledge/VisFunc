"""Scan a source, analyse bounded windows and write the evidence manifest."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from contextlib import closing, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import annotate
from .frames import FrameSource, InvalidTimestamps, write_image
from .motion import analyze_window
from .reflow import measure_reflow
from .segment import LAG, Scan, scan
from .spec import SCHEMA, summarize, validate
from .timing import with_timing

MARGIN = 12
MEMORY_BUDGET = 150_000_000  # Shared budget for cached BGR crops in one task.


@dataclass
class _Window:
    roi: tuple[int, int, int, int]
    start: int
    end: int
    reserved_bytes: int
    truncated: bool
    crops: list[np.ndarray] = field(default_factory=list)
    context: tuple[int, int, int, int] | None = None


def _plan_windows(
    scanned: Scan, pad: int, max_window: int, budget: int, layout_edges: bool = False
) -> list[list[_Window]]:
    batches: list[list[_Window]] = []
    width, height = scanned.size
    for cand in scanned.candidates:
        x1, y1, x2, y2 = cand.bbox
        roi = (
            max(0, x1 - MARGIN),
            max(0, y1 - MARGIN),
            min(width, x2 + MARGIN),
            min(height, y2 + MARGIN),
        )
        stride = (roi[2] - roi[0]) * (roi[3] - roi[1]) * 3
        limit = min(max_window, budget // stride)
        if limit < 2:
            raise ValueError(
                "crop budget cannot fit two frames; reduce max_dim or raise max_crop_bytes"
            )
        start = max(0, cand.first - 1 - pad - LAG)
        desired_start = start
        desired_end = min(len(scanned.times) - 1, cand.last + pad)
        # Always retain the transition that created the candidate, even with a small limit.
        if cand.first >= start + limit:
            start = cand.first - limit + 1
        end = min(desired_end, start + limit - 1)
        window = _Window(
            roi,
            start,
            end,
            (end - start + 1) * stride,
            start > desired_start or end < desired_end,
        )
        if layout_edges and not window.truncated and 5 <= end - start + 1 <= 180:
            # Changed pixels often cover only the moving edge of a panel. Include
            # the stationary edge when the full time window fits the crop budget.
            margin = min(160, max(roi[2] - roi[0], roi[3] - roi[1]))
            context = (
                max(0, roi[0] - margin), max(0, roi[1] - margin),
                min(width, roi[2] + margin), min(height, roi[3] + margin),
            )
            needed = (context[2] - context[0]) * (context[3] - context[1]) * 3 * (end - start + 1)
            if needed <= budget:
                window.context = context
                window.reserved_bytes = needed
        for batch in batches:
            reserved = sum(
                w.reserved_bytes for w in batch if start <= w.end and w.start <= end
            )
            if reserved + window.reserved_bytes <= budget:
                batch.append(window)
                break
        else:
            batches.append([window])
    return batches


def _ambient_share(scanned: Scan, roi: tuple[int, int, int, int]) -> float:
    """Fraction of a region's grid cells that belong to a continuously changing area."""
    cells = scanned.ambient_cells
    if cells is None:
        return 0.0
    gh, gw = cells.shape
    w, h = scanned.size
    x1, y1, x2, y2 = roi
    part = cells[
        y1 * gh // h : max(y1 * gh // h + 1, (y2 * gh + h - 1) // h),
        x1 * gw // w : max(x1 * gw // w + 1, (x2 * gw + w - 1) // w),
    ]
    return float(part.mean()) if part.size else 0.0


def _decode_frames(frames: FrameSource, scanned: Scan, stop: int, needed):
    """Bind later reads to the first pass index and fail on an inconsistent source."""
    with closing(frames.selected_samples(lambda fid: needed(fid - scanned.first_id))) as stream:
        for i, (frame_id, stamp, raw) in enumerate(stream):
            if frame_id != scanned.first_id + i or stamp != scanned.times[i]:
                raise ValueError(
                    "source timestamps or frame indices changed during analysis"
                )
            if needed(i):
                if raw is None:
                    raise ValueError(f"missing pixels at requested frame {frame_id}")
                yield i, raw
            if i == stop:
                return
    raise ValueError("source ended before a requested evidence frame was decoded")


def _analyze_windows(
    frames: FrameSource,
    scanned: Scan,
    batches: list[list[_Window]],
    threshold: int,
    stage: Path | None,
    sheets: bool,
) -> list[tuple[_Window, dict, Path | None]]:
    done = []
    inv = 1.0 / frames.info.scale
    for batch in batches:
        def in_batch(index, windows=batch):
            return any(w.start <= index <= w.end for w in windows)

        for i, raw in _decode_frames(frames, scanned, max(w.end for w in batch), in_batch):
            active = [w for w in batch if w.start <= i <= w.end]
            frame = frames.resized(raw)
            for w in active:
                x1, y1, x2, y2 = w.context or w.roi
                w.crops.append(frame[y1:y2, x1:x2].copy())
                if i != w.end:
                    continue
                times = scanned.times[w.start : w.end + 1]
                results = None
                if w.context:
                    focus = (w.roi[0] - x1, w.roi[1] - y1, w.roi[2] - x1, w.roi[3] - y1)
                    results = measure_reflow(w.crops, times, (x1, y1), inv, focus, threshold)
                    if results is None:
                        a, b, c, d = focus
                        w.crops = [crop[b:d, a:c] for crop in w.crops]
                        x1, y1, x2, y2 = w.roi
                if results is None:
                    results = analyze_window(
                        w.crops,
                        times,
                        origin=(x1, y1),
                        inv_scale=inv,
                        threshold=threshold,
                        allow_layer=not w.truncated and _ambient_share(scanned, w.roi) < 0.5,
                    )
                for result in results:
                    # A window can hold several elements; each keeps only the area it sweeps.
                    sx1, sy1, sx2, sy2 = result.pop("_roi", (0, 0, x2 - x1, y2 - y1))
                    part = replace(
                        w, roi=(x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2), crops=[]
                    )
                    sheet_path = None
                    if stage is not None and sheets:
                        view = result["_view"]
                        duration = times[view["end"]] - times[view["start"]]
                        title = f"{result['kind']} {duration:.0f}ms"
                        easing = result["easing"]
                        if easing:
                            title += "  " + (
                                easing.get("css")
                                or f"spring zeta={easing['damping_ratio']}"
                            )
                        sheet_path = stage / f"window-{len(done)}.png"
                        # Slices keep the crops alive, so do not bind them to a name.
                        write_image(
                            sheet_path,
                            annotate.sheet(
                                [c[sy1:sy2, sx1:sx2] for c in w.crops],
                                times,
                                result,
                                title,
                            ),
                        )
                    done.append((part, result, sheet_path))
                w.crops.clear()
    return sorted(
        done, key=lambda item: (item[0].start + item[1]["_view"]["start"], item[0].roi)
    )


def _check_options(
    max_segments: int,
    min_area: float,
    threshold: int,
    pad: int,
    max_window: int,
    max_frames: int,
    max_crop_bytes: int,
) -> None:
    bounds = {
        "max_segments": (max_segments, 0),
        "threshold": (threshold, 0),
        "pad": (pad, 0),
        "max_window": (max_window, 2),
        "max_frames": (max_frames, 2),
        "max_crop_bytes": (max_crop_bytes, 6),
    }
    for name, (value, minimum) in bounds.items():
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if threshold > 255:
        raise ValueError("threshold must be <= 255")
    if not math.isfinite(min_area) or min_area < 0:
        raise ValueError("min_area must be finite and nonnegative")


def analyze(
    source: str | Path | Sequence[np.ndarray],
    out_dir: str | Path | None = None,
    *,
    fps: float | None = None,
    max_dim: int = 960,
    start_ms: float | None = None,
    end_ms: float | None = None,
    max_segments: int = 12,
    min_area: float = 120.0,
    threshold: int = 12,
    pad: int = 3,
    max_window: int = 300,
    sheets: bool = True,
    max_frames: int = 1_000_000,
    max_crop_bytes: int = MEMORY_BUDGET,
    raw_frames: bool = True,
    ambient_ms: float = 3000.0,
    functions: bool = False,
    sequence_id: str | None = None,
    function_max_frames: int = 64,
    evidence_every_ms: float | None = None,
    layout_edges: bool = False,
    mark_uncertain: bool = False,
) -> dict:
    """Analyse a selected range. Export full-size evidence frames when an output is given."""
    _check_options(
        max_segments, min_area, threshold, pad, max_window, max_frames, max_crop_bytes
    )
    if functions:
        from .functions import require_svsg

        require_svsg()
        if out_dir is None or not raw_frames:
            raise ValueError("function export requires out_dir and raw_frames=True")
        if type(function_max_frames) is not int or not 1 <= function_max_frames <= 256:
            raise ValueError("function_max_frames must be in 1..256")
    if evidence_every_ms is not None and (
        not math.isfinite(evidence_every_ms) or evidence_every_ms <= 0
    ):
        raise ValueError("evidence_every_ms must be finite and positive")
    frames = FrameSource(
        source, fps=fps, max_dim=max_dim, start_ms=start_ms, end_ms=end_ms
    )
    info = frames.info
    warnings = []
    options = {
        "threshold": threshold,
        "min_area": min_area,
        "max_segments": max_segments,
        "max_frames": max_frames,
        "ambient_ms": ambient_ms,
    }
    if not math.isfinite(ambient_ms) or ambient_ms < 0:
        raise ValueError("ambient_ms must be finite and nonnegative")
    try:
        scanned = scan(frames, **options)
    except InvalidTimestamps:
        if info.timestamps != "container":
            raise
        info.timestamps = "index"
        warnings.append(
            "container timestamps were unusable; the selected range was rescanned using source frame indices"
        )
        scanned = scan(frames, **options)
    if frames.fps_assumed:
        warnings.append(
            "the decoder did not report a frame rate; nominal fps is assumed to be 30"
        )
    times = scanned.times
    count = len(times)
    if count < 2:
        raise ValueError("need at least 2 frames in the selected range")
    if scanned.dropped:
        warnings.append(
            f"{scanned.dropped} change region(s) were dropped (min_area / max_segments)"
        )
    batches = _plan_windows(scanned, pad, max_window, max_crop_bytes, layout_edges)
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    stage_context = (
        TemporaryDirectory(dir=out) if out is not None else nullcontext(None)
    )
    with stage_context as directory:
        stage = Path(directory) if directory is not None else None
        done = _analyze_windows(frames, scanned, batches, threshold, stage, sheets)
        segments = []
        frame_roles = {0: {"range-start"}, count - 1: {"range-end"}}
        if evidence_every_ms is not None:
            target = times[0] + evidence_every_ms
            for index, stamp in enumerate(times):
                if stamp >= target:
                    frame_roles.setdefault(index, set()).add("periodic")
                    target = stamp + evidence_every_ms
        for number, (w, result, sheet_path) in enumerate(done, start=1):
            view = result.pop("_view")
            sid = f"s{number}"
            s, e = w.start + view["start"], w.start + view["end"]
            x1, y1, x2, y2 = w.roi
            keys = []
            for k, role in view["keys"]:
                idx = w.start + k
                frame_roles.setdefault(idx, set()).add(f"{sid}:{role}")
                keys.append(
                    {
                        "role": role,
                        "frame": scanned.first_id + idx,
                        "t_ms": round(times[idx], 3),
                    }
                )
            seg = {
                **result,
                "id": sid,
                "t_start_ms": round(times[s], 3),
                "t_end_ms": round(times[e], 3),
                "duration_ms": round(times[e] - times[s], 3),
                "frames": [scanned.first_id + s, scanned.first_id + e],
                "region": [
                    round(x1 * info.width / scanned.size[0]),
                    round(y1 * info.height / scanned.size[1]),
                    round(x2 * info.width / scanned.size[0]),
                    round(y2 * info.height / scanned.size[1]),
                ],
                "keyframes": keys,
                "truncated": w.truncated,
            }
            share = _ambient_share(scanned, w.roi)
            if share >= 0.5 and (seg["confidence"] != "high" or seg["kind"] == "instant"):
                seg["notes"].append(
                    f"{share:.0%} of this region keeps changing on its own; the measurement is contaminated"
                )
                seg["confidence"] = "low"
            if w.truncated:
                seg["notes"].append(
                    "the candidate exceeded the analysis window; only part of its evidence was analysed"
                )
                seg["confidence"] = "low"
            if sheet_path is not None:
                seg["sheet"] = f"{sid}.png"
                sheet_path.rename(stage / seg["sheet"])
            segments.append(seg)
        spec = {
            "schema": SCHEMA,
            "source": {
                "path": info.path,
                "width": info.width,
                "height": info.height,
                "fps_nominal": info.fps,
                "frames_analysed": count,
                "t_start_ms": round(times[0], 3),
                "t_end_ms": round(times[-1], 3),
                "duration_ms": round(times[-1] - times[0], 3),
                "timestamps": info.timestamps,
                "analysis_scale": info.scale,
            },
            "time_resolution_ms": float(np.median(np.diff(times))),
            "units": {
                "position": "source pixels, origin top-left",
                "time": "ms on the source timeline",
            },
            "coverage": {
                "dropped_candidates": scanned.dropped,
                "truncated_segments": sum(w.truncated for w, _, _ in done),
                "ambient_regions": len(scanned.ambient),
            },
            "ambient": [
                {
                    "region": [
                        round(r["bbox"][0] * info.width / scanned.size[0]),
                        round(r["bbox"][1] * info.height / scanned.size[1]),
                        round(r["bbox"][2] * info.width / scanned.size[0]),
                        round(r["bbox"][3] * info.height / scanned.size[1]),
                    ],
                    "t_start_ms": round(r["t_start"], 3),
                    "t_end_ms": round(r["t_end"], 3),
                }
                for r in scanned.ambient
            ],
            "segments": segments,
            "warnings": warnings,
        }
        spec = with_timing(spec)
        if stage is not None:
            if functions and len(frame_roles) > function_max_frames:
                raise ValueError("function evidence frame budget exceeded; increase "
                                 "function_max_frames or select a shorter interval")
            _export_evidence(
                frames, scanned, done, spec, frame_roles, stage, raw_frames, sheets
            )
            if functions:
                from .functions import export_functions

                spec["function_evidence"] = export_functions(
                    spec, stage, stage, sequence_id=sequence_id, max_frames=function_max_frames
                )
        problems = validate(spec)
        if problems:
            raise ValueError("generated invalid motion spec: " + "; ".join(problems))
        if stage is not None:
            (stage / "spec.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            (stage / "summary.md").write_text(summarize(spec, mark_uncertain=mark_uncertain), encoding="utf-8")
            for path in sorted(stage.iterdir(), key=lambda p: p.name == "spec.json"):
                path.replace(out / path.name)
    return spec


def _export_evidence(
    frames: FrameSource,
    scanned: Scan,
    done: list,
    spec: dict,
    frame_roles: dict,
    stage: Path,
    raw_frames: bool,
    sheets: bool,
) -> None:
    if not raw_frames and not (sheets and done):
        return
    refs = []
    for i, raw in _decode_frames(
        frames, scanned, len(scanned.times) - 1 if raw_frames else 0,
        lambda index: (sheets and index == 0) or (raw_frames and index in frame_roles),
    ):
        if i == 0 and sheets and done:
            boxes = [
                (seg["id"], w.roi) for seg, (w, _, _) in zip(spec["segments"], done)
            ]
            write_image(
                stage / "overview.png", annotate.overview(frames.resized(raw), boxes)
            )
            spec["overview"] = "overview.png"
        if raw_frames and i in frame_roles:
            frame_id = scanned.first_id + i
            path = stage / "encoding-frame.png"
            write_image(path, raw)
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                while chunk := stream.read(65536):
                    digest.update(chunk)
            sha256 = digest.hexdigest()
            filename = f"frame-{frame_id:08d}-{sha256}.png"
            path.rename(stage / filename)
            refs.append(
                {
                    "frame": frame_id,
                    "t_ms": round(scanned.times[i], 3),
                    "roles": sorted(frame_roles[i]),
                    "image": filename,
                    "sha256": sha256,
                }
            )
    if raw_frames:
        spec["evidence_frames"] = refs
