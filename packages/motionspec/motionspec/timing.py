"""One calculation for JSON and prose timing; no image, model or SVSG dependency."""

from __future__ import annotations

import math

TIMING_SCHEMA = "svsg.motion-timing/1"
MAX_DESIGN_RMSE = 0.05


def _finite(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _interval(start, end) -> dict:
    if not _finite(start) or not _finite(end) or not 0 <= start <= end:
        raise ValueError("invalid visible-change interval")
    return {"t_start_ms": start, "t_end_ms": end, "duration_ms": round(end - start, 6)}


def _design(easing, basis_start, *, truncated) -> tuple[dict | None, str | None]:
    if truncated:
        return None, "truncated_segment"
    if not isinstance(easing, dict) or not easing.get("named"):
        return None, "no_named_fit"
    best = easing["named"][0]
    if not isinstance(best, dict):
        return None, "invalid_alignment"
    rmse, shift, duration = (best.get(k) for k in ("rmse", "start_shift_ms", "duration_ms"))
    if not _finite(rmse) or rmse < 0 or rmse > MAX_DESIGN_RMSE:
        return None, "poor_named_fit"
    if (
        not _finite(shift)
        or not _finite(duration)
        or duration <= 0
        or not isinstance(best.get("name"), str)
        or not best["name"]
    ):
        return None, "invalid_alignment"
    start = round(basis_start + shift, 6)
    end = round(start + duration, 6)
    if not _finite(start) or not _finite(end) or start < 0:
        # Clamping to zero would silently shift the fitted interval and its end.
        return None, "estimate_before_timeline"
    return {
        **_interval(start, end),
        "easing": best["name"],
        "basis_t_start_ms": basis_start,
        "start_shift_ms": shift,
        "rmse": rmse,
        "rmse_limit": MAX_DESIGN_RMSE,
        "provenance": "named_curve_alignment",
        "is_ground_truth": False,
    }, None


def _entry(row, easing, basis_start, truncated):
    visible = _interval(row["t_start_ms"], row["t_end_ms"])
    design, reason = _design(easing, basis_start, truncated=truncated)
    return {
        "visible_change": visible,
        "estimated_design": design,
        "design_unavailable_reason": reason,
    }


def motion_timing(segment: dict) -> dict:
    """Materialize both meanings, retaining the alignment basis for recomputation."""
    easing = segment.get("easing")
    channels = segment.get("channels", {})
    truncated = bool(segment.get("truncated"))
    timed = {
        name: row
        for name, row in channels.items()
        if isinstance(row, dict) and "t_start_ms" in row and "t_end_ms" in row
    }
    primary = next(
        (name for name, row in timed.items() if easing is not None and row.get("easing") == easing),
        None,
    )
    basis = timed[primary]["t_start_ms"] if primary is not None else segment["t_start_ms"]
    return {
        "version": TIMING_SCHEMA,
        "unit": "ms",
        "clock": "source_timeline",
        **_entry(segment, easing, basis, truncated),
        "primary_channel": primary,
        "channels": {
            name: _entry(row, row.get("easing"), row["t_start_ms"], truncated)
            for name, row in timed.items()
        },
    }


def with_timing(spec: dict) -> dict:
    """Upgrade legacy MotionSpec data without mutating its samples or input dict."""
    return {**spec, "segments": [{**seg, "timing": motion_timing(seg)} for seg in spec["segments"]]}
