"""Evidence-based field masks for the optional human/model-facing motion summary."""

from .timing import motion_timing

DESIGN = ("start_ms", "duration_ms", "easing")
SHIFT = ("direction", "distance_px")
SIZE = ("width_px", "height_px")
ALPHA = ("opacity_from", "opacity_to")
TIMING_REASONS = {
    "truncated_segment": "only part of the motion was analysed",
    "no_named_fit": "no named curve could be aligned to the samples",
    "poor_named_fit": "the closest named curve exceeds the fit error limit",
    "invalid_alignment": "the fitted curve lacks a valid start or duration",
    "estimate_before_timeline": "the fitted start precedes the clip timeline",
}


def field_uncertainties(seg, timing):
    flags = {}

    def mark(fields, reason):
        for field in fields:
            flags.setdefault(field, reason)

    channels = seg["channels"]
    move = channels.get("translate", {})
    if timing["estimated_design"] is None:
        mark(DESIGN, TIMING_REASONS[timing["design_unavailable_reason"]])
    if seg.get("model") == "qt-probe":
        return flags

    if seg.get("truncated") or seg["confidence"] == "low":
        if seg.get("truncated"):
            reason = "the analysis window ends before all evidence is available"
        elif seg.get("model") == "multibody":
            reason = "too few feature points support this element among independently moving groups"
        elif "appearance" in channels:
            reason = "changing pixels could not be separated into rigid objects"
        else:
            reason = "the visual fit has low confidence for this region"
        mark(DESIGN, reason)
        for name, fields in (
            ("translate", SHIFT),
            ("size", SIZE),
            ("opacity", ALPHA),
            ("scale", ("scale",)),
            ("rotate", ("rotation",)),
        ):
            if name in channels:
                mark(fields, reason)
        if "appearance" in channels:
            mark(SHIFT + SIZE, reason)

    if move.get("lower_bound") or move.get("extrapolated"):
        reason = "part of the path is invisible and the endpoints are not independently observed"
        mark(SHIFT, reason)
        if timing["primary_channel"] == "translate":
            mark(DESIGN, reason)
    if channels.get("scale", {}).get("lower_bound") or channels.get("scale", {}).get(
        "extrapolated"
    ):
        mark(("scale",), "part of the scale change occurs while the element is invisible")
    if move.get("reference") == "tracked_feature_anchor":
        if "rotate" in channels or "scale" in channels:
            mark(SHIFT, "the tracked feature also rotates or scales; the element pivot is unknown")
            mark(SIZE, "feature motion does not identify both element edges")
        elif "luma" in seg and abs(seg["luma"]["to"] - seg["luma"]["from"]) > 6:
            mark(
                SIZE,
                "brightness also changes while features move; "
                "opposite layout edges were not tracked",
            )
            if "opacity" not in channels:
                mark(ALPHA, "brightness and shifting content cannot isolate element opacity")
    if seg.get("model") == "multibody":
        mark(("region",), "point groups locate motion but only approximate the element boundary")
    if "blend" in channels and "opacity" not in channels:
        mark(ALPHA, "a colour change and an opacity change can produce the same pixel blend")
    return flags


def summary_uncertainties(spec):
    """Scene-wide count and per-segment fields have distinct namespaces."""
    segments = {
        seg["id"]: fields
        for seg in spec["segments"]
        if (fields := field_uncertainties(seg, motion_timing(seg)))
    }
    flags = {"segments": segments} if segments else {}
    if spec.get("coverage", {}).get("dropped_candidates"):
        flags["count"] = "candidate windows were dropped before measurement"
    elif not spec["segments"]:
        flags["count"] = "sub-threshold changes cannot be counted here"
    return flags
