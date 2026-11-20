"""Optional adapter to the shared SVSG image/time function implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .spec import validate
from .timing import with_timing


def require_svsg():
    try:
        from svsg.video_functions import export_motion_spec
    except ImportError as exc:
        raise ValueError("function export requires the local SVSG package and Pillow; "
                         "install SVSG with its [functions] extra") from exc
    return export_motion_spec


def export_functions(spec: dict, evidence_dir: str | Path, out_dir: str | Path, *,
                     sequence_id: str | None = None, max_frames: int = 64) -> dict:
    problems = validate(spec)
    if problems:
        raise ValueError("invalid MotionSpec input: " + "; ".join(problems))
    # Artifact references are excluded from the source digest to avoid a hash cycle.
    source = with_timing({k: v for k, v in spec.items() if k != "function_evidence"})
    digest = hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    bundle = require_svsg()(source, evidence_dir, out_dir,
                           sequence_id=sequence_id or f"motion-{digest[:24]}",
                           max_frames=max_frames)
    manifest = Path(out_dir) / "functions.json"
    return {"version": bundle.version, "manifest": manifest.name,
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "sequence_id": bundle.sequence_id, "frames": len(bundle.frames)}


MAX_BATCH = 1000


def _query_examples(segments: list[dict], frames: list[dict]) -> list[dict]:
    examples = []
    for mode in ("sampled", "fitted"):
        for segment in segments:
            functions = segment[mode]
            if functions:
                name, curve = next(iter(functions.items()))
                examples.append({"id": mode, "op": "motion_value", "segment_id": segment["segment_id"],
                                 "name": name, "mode": mode, "t_ms": curve["t_ms"][0]})
                break
    if frames:
        examples.append({"id": "pixel", "op": "pixel_rgba",
                         "frame_index": frames[0]["frame_index"], "x": 0, "y": 0})
    return examples


def catalogue(manifest: str | Path) -> dict:
    """函数清单的目录：给模型先读的那一小页。

    完整的 functions.json 里每一帧都带图像函数描述，一段动效就有上百 KB；
    要发查询，模型需要的只是"有哪些片段、每段能查哪些函数、单位、时间范围、取值范围、哪些帧能查像素"。
    这里从已加载的清单里提取这些，不做任何新的测量。
    """
    require_svsg()
    from svsg.video_functions import VideoRuntime

    bundle = VideoRuntime.load(manifest).bundle
    segments = []
    for motion in bundle.motions:
        sampled = {}
        for name, function in motion.samples.items():
            values = [v for _, v in function.knots]
            sampled[name] = {"unit": function.unit, "t_ms": [function.knots[0][0], function.knots[-1][0]],
                             "range": [min(values), max(values)], "knots": len(values),
                             "provenance": function.provenance}
        fitted = {name: {"op": fit.op, "t_ms": [fit.t_start_ms, fit.t_end_ms], "rmse": fit.rmse}
                  for name, fit in motion.fits.items()}
        timing = motion.timing
        design = timing.estimated_design if timing is not None else None
        segments.append({
            "segment_id": motion.segment_id, "kind": motion.kind, "confidence": motion.confidence,
            "region": motion.region, "visible_t_ms": [motion.t_start_ms, motion.t_end_ms],
            "visible_duration_ms": (timing.visible_change.duration_ms if timing is not None
                                    else motion.t_end_ms - motion.t_start_ms),
            "design": None if design is None else {"easing": design.easing, "duration_ms": design.duration_ms,
                                                    "t_ms": [design.t_start_ms, design.t_end_ms]},
            "design_unavailable_reason": timing.design_unavailable_reason if timing else "legacy_missing_timing",
            "reference": {key: motion.reference[key] for key in ("type", "element_id", "property_name", "motion_group")
                          if key in motion.reference},
            "sampled": sampled, "fitted": fitted,
            "flags": {k: v for k, v in (("truncated", motion.truncated), ("inferred", motion.inferred_channels),
                                        ("lower_bound", motion.lower_bound_channels)) if v},
        })
    frames = [{"frame_index": f.frame_index, "t_ms": f.timestamp_ms, "roles": f.roles} for f in bundle.frames]
    return {
        "version": bundle.version, "sequence_id": bundle.sequence_id, "clock": bundle.clock,
        "coverage": bundle.coverage, "coordinate_space": bundle.coordinate_space,
        "size_px": [bundle.width_px, bundle.height_px], "t_ms": [bundle.t_start_ms, bundle.t_end_ms],
        "frames": frames,
        "segments": segments,
        "query_examples": _query_examples(segments, frames),
    }


def run_queries(manifest: str | Path, calls) -> dict | list:
    """一次调用或一批调用（列表）。批量时每个调用各自成功或报错，互不影响。"""
    require_svsg()
    from svsg.video_functions import VideoRuntime

    runtime = VideoRuntime.load(manifest)
    if isinstance(calls, dict):
        return runtime.call(calls)
    if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_BATCH:
        raise ValueError(f"expected one call object or a list of 1..{MAX_BATCH} calls")
    results = []
    for call in calls:
        try:
            results.append(runtime.call(call))
        except (ValueError, TypeError, OSError) as exc:
            call_id = call.get("id") if isinstance(call, dict) else None
            results.append({"id": call_id if isinstance(call_id, str) else None, "error": str(exc).splitlines()[0]})
    return results
