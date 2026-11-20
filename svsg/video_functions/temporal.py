"""Compile existing MotionSpec measurements; never invent object identities or paths."""

from __future__ import annotations

from svsg.contracts.video_functions import (
    FittedProgress,
    MotionFunctions,
    MotionTiming,
    SampledFunction,
    Unit,
)

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
    "width": "source_px",
    "height": "source_px",
    "color.r": "ratio",
    "color.g": "ratio",
    "color.b": "ratio",
    "color.a": "ratio",
    "property.value": "property_units",
}


def compile_motion(segment: dict) -> MotionFunctions:
    rows = segment["samples"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
        raise ValueError("invalid motion sample count")
    # Omit incomplete channels rather than silently bridging a missing measurement.
    samples = {}
    for key, unit in UNITS.items():
        if all(key in row for row in rows):
            samples[key] = SampledFunction(
                unit=unit,
                knots=[(r["t_ms"], r[key]) for r in rows],
                provenance=(
                    "qt_property_sample"
                    if segment.get("model") == "qt-probe"
                    and (key.startswith("color.") or key in {"property.value", "rot_deg", "scale"})
                    else "motion_estimate"
                ),
            )
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
    # MotionSpec owns alignment. Consume its materialized calculation rather than
    # duplicating the heuristic or introducing a reverse dependency on MotionSpec.
    timing = segment.get("timing")
    if timing is None:
        timing = {
            "visible_change": {
                "t_start_ms": segment["t_start_ms"],
                "t_end_ms": segment["t_end_ms"],
                "duration_ms": segment["t_end_ms"] - segment["t_start_ms"],
            },
            "estimated_design": None,
            "design_unavailable_reason": "legacy_missing_timing",
        }
    reference = dict(segment.get("transform_reference", {}))
    for key in ("element", "element_id", "motion_group"):
        if key in segment:
            reference[key] = segment[key]
    if "property" in channels:
        reference["property_name"] = channels["property"]["name"]
    if "color" in channels:
        reference["color_components"] = "RGBA components in 0..1; not perceptual distance"
    return MotionFunctions(
        segment_id=segment["id"],
        kind=segment["kind"],
        confidence=segment["confidence"],
        t_start_ms=segment["t_start_ms"],
        t_end_ms=segment["t_end_ms"],
        timing=MotionTiming.model_validate(timing),
        region=segment["region"],
        truncated=segment.get("truncated", False),
        reference=reference,
        notes=segment.get("notes", []),
        inferred_channels=[
            k for k, v in channels.items() if v.get("extrapolated") or v.get("follows_opacity")
        ],
        lower_bound_channels=[k for k, v in channels.items() if v.get("lower_bound")],
        samples=samples,
        fits=fits,
    )
