"""Compile existing MotionSpec measurements; never invent object identities or paths."""

from __future__ import annotations

from svsg.contracts.video_functions import FittedProgress, MotionFunctions, SampledFunction, Unit

UNITS: dict[str, Unit] = {
    "x": "source_px",
    "y": "source_px",
    "feature_x": "source_px",
    "feature_y": "source_px",
    "scale": "ratio",
    "opacity": "ratio",
    "progress": "ratio",
    "rot_deg": "degrees",
    "luma": "luma_0_255",
}


def compile_motion(segment: dict) -> MotionFunctions:
    rows = segment["samples"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
        raise ValueError("invalid motion sample count")
    # Omit incomplete channels rather than silently bridging a missing measurement.
    samples = {}
    for key, unit in UNITS.items():
        if all(key in row for row in rows):
            samples[key] = SampledFunction(unit=unit, knots=[(r["t_ms"], r[key]) for r in rows])
    channels = segment["channels"]
    for name in channels:
        if all(name in row.get("channel_progress", {}) for row in rows):
            samples[f"{name}.progress"] = SampledFunction(
                unit="ratio", knots=[(r["t_ms"], r["channel_progress"][name]) for r in rows]
            )
    fits = {}
    for name, channel in channels.items():
        easing = channel.get("easing")
        if easing is None:
            continue
        params = (
            easing["params"]
            if easing["model"] == "cubic-bezier"
            else [easing["damping_ratio"], easing["angular_frequency"]]
        )
        fits[f"{name}.progress"] = FittedProgress(
            op="cubic_bezier" if easing["model"] == "cubic-bezier" else "damped_spring",
            params=params,
            rmse=easing["rmse"],
            t_start_ms=channel["t_start_ms"],
            t_end_ms=channel["t_end_ms"],
        )
    return MotionFunctions(
        segment_id=segment["id"],
        kind=segment["kind"],
        confidence=segment["confidence"],
        t_start_ms=segment["t_start_ms"],
        t_end_ms=segment["t_end_ms"],
        region=segment["region"],
        truncated=segment.get("truncated", False),
        reference=segment.get("transform_reference", {}),
        notes=segment.get("notes", []),
        inferred_channels=[
            k for k, v in channels.items() if v.get("extrapolated") or v.get("follows_opacity")
        ],
        lower_bound_channels=[k for k, v in channels.items() if v.get("lower_bound")],
        samples=samples,
        fits=fits,
    )
