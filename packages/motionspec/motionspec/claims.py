"""Check typed model claims against measured motion evidence."""

from __future__ import annotations

from .spec import KINDS, finite, validate

DIRECTIONS = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
SCOPED_FIELDS = {
    "kind",
    "duration_ms",
    "easing",
    "overshoot",
    "distance_px",
    "direction",
    "scale_to",
}
NUMERIC_FIELDS = {"duration_ms", "distance_px", "scale_to"}


def _issue(index: int, claim: dict, reason: str, measured=None) -> dict:
    value = claim.get("value")
    return {
        "claim": index,
        "segment": claim["segment"] if isinstance(claim.get("segment"), str) else None,
        "field": claim["field"] if isinstance(claim.get("field"), str) else None,
        "claimed": value if type(value) in (str, bool) or finite(value) else None,
        "measured": measured,
        "reason": reason,
    }


def _evaluate(spec: dict, by_id: dict, claim: dict) -> tuple[str, str, object]:
    field, value = claim.get("field"), claim.get("value")
    if not isinstance(field, str) or field not in SCOPED_FIELDS | {"count"}:
        return "failed", "unsupported field", None
    extra = set(claim) - {"field", "value", "segment"}
    allowed = {
        "count": {"kind"},
        "duration_ms": {"easing", "channel"},
        "easing": {"channel"},
        "overshoot": {"channel"},
    }
    if extra - allowed.get(field, set()):
        return (
            "failed",
            "unsupported claim option; tolerance is set by the verifier",
            None,
        )
    if field == "count":
        kind = claim.get("kind")
        if (
            "segment" in claim
            or type(value) is not int
            or value < 0
            or (kind is not None and (not isinstance(kind, str) or kind not in KINDS))
        ):
            return (
                "failed",
                "count requires a nonnegative integer and an optional known kind",
                None,
            )
        actual = sum(kind is None or seg["kind"] == kind for seg in by_id.values())
        coverage = spec.get("coverage", {})
        if (
            coverage.get("dropped_candidates")
            or coverage.get("truncated_segments")
            or any("were dropped" in warning for warning in spec.get("warnings", []))
        ):
            return "insufficient_evidence", "candidate coverage is incomplete", actual
        return (
            ("passed", "", actual)
            if actual == value
            else ("failed", "segment count differs", actual)
        )

    sid = claim.get("segment")
    if not isinstance(sid, str) or sid not in by_id:
        return "failed", "no such segment", sorted(by_id)
    if field in NUMERIC_FIELDS and (
        not finite(value) or value < 0 or (field == "scale_to" and value == 0)
    ):
        return "failed", "expected a finite numeric value in range", None
    if field == "overshoot" and type(value) is not bool:
        return "failed", "overshoot requires a boolean", None
    if field in {"kind", "direction", "easing"} and not isinstance(value, str):
        return "failed", "expected a string value", None
    seg = by_id[sid]
    channels = seg["channels"]
    if seg.get("truncated") or any(
        "only its beginning" in note for note in seg.get("notes", [])
    ):
        return "insufficient_evidence", "segment was truncated", None

    measure = seg
    if "channel" in claim:
        name = claim["channel"]
        if not isinstance(name, str) or name not in {
            "translate",
            "scale",
            "rotate",
            "blend",
            "opacity",
        }:
            return "failed", "unknown measurement channel", None
        if name not in channels:
            return "insufficient_evidence", "channel was not measured", None
        measure = channels[name]
    easing = measure.get("easing")
    if field == "kind":
        actual = seg["kind"]
    elif field == "duration_ms":
        actual = measure.get("duration_ms")
        if "easing" in claim:
            name = claim["easing"]
            if not isinstance(name, str):
                return "failed", "easing must be a curve name", None
            row = next(
                (
                    item
                    for item in (easing or {}).get("named", [])
                    if item["name"] == name
                ),
                None,
            )
            if row is None or row["rmse"] > 0.05:
                return (
                    "insufficient_evidence",
                    "no supported duration for the requested easing",
                    None,
                )
            actual = row["duration_ms"]
        if actual is None:
            return "insufficient_evidence", "duration was not measured", None
        tolerance = max(2 * spec["time_resolution_ms"], 0.15 * actual)
    elif field == "easing":
        if not easing or easing["rmse"] > 0.06:
            return "insufficient_evidence", "no reliable easing fit", None
        best = min(easing["named"], key=lambda row: row["rmse"])
        if best["rmse"] > 0.05:
            return "insufficient_evidence", "no named curve fits the samples", None
        ranked = {row["name"]: row["rmse"] for row in easing["named"]}
        if value not in ranked or ranked[value] > min(
            0.05, max(0.04, best["rmse"] * 1.5)
        ):
            return (
                "failed",
                "named easing differs from the measured curve",
                best["name"],
            )
        return "passed", "", best["name"]
    elif field == "overshoot":
        if not easing or easing["rmse"] > 0.06:
            return "insufficient_evidence", "no reliable easing fit", None
        actual = easing["overshoot"] > 0.03
    elif field in {"distance_px", "direction"}:
        move = channels.get("translate")
        if move is None:
            return "insufficient_evidence", "no translation was measured", None
        if field == "direction":
            dx, dy = move["delta"]
            if abs(dx) == abs(dy):
                return (
                    "insufficient_evidence",
                    "no unique cardinal direction",
                    move["delta"],
                )
            actual = max(
                DIRECTIONS,
                key=lambda name: DIRECTIONS[name][0] * dx + DIRECTIONS[name][1] * dy,
            )
        else:
            actual = move["distance_px"]
            tolerance = max(3.0, 0.05 * actual)
    else:
        scale = channels.get("scale")
        if scale is None:
            return "insufficient_evidence", "no scale was measured", None
        actual, tolerance = scale["to"], 0.03

    matches = (
        abs(actual - value) <= tolerance if field in NUMERIC_FIELDS else actual == value
    )
    return (
        ("passed", "", actual)
        if matches
        else ("failed", "claim differs from the measurement", actual)
    )


def check(spec: dict, claims: list[dict]) -> dict:
    invalid_spec = validate(spec)
    if invalid_spec:
        return {
            "passed": False,
            "verdict": "failed",
            "checked": 0,
            "violations": [],
            "insufficient_evidence": [],
            "invalid_spec": invalid_spec,
        }
    if not isinstance(claims, list) or not claims:
        return {
            "passed": False,
            "verdict": "failed",
            "checked": 0,
            "violations": [],
            "insufficient_evidence": [],
            "invalid_claims": "expected a nonempty list of claims",
        }
    by_id = {seg["id"]: seg for seg in spec["segments"]}
    violations, insufficient = [], []
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            violations.append(_issue(i, {}, "claim must be an object"))
            continue
        status, reason, measured = _evaluate(spec, by_id, claim)
        if status == "failed":
            violations.append(_issue(i, claim, reason, measured))
        elif status == "insufficient_evidence":
            insufficient.append(_issue(i, claim, reason, measured))
    verdict = (
        "failed"
        if violations
        else "insufficient_evidence"
        if insufficient
        else "passed"
    )
    return {
        "passed": verdict == "passed",
        "verdict": verdict,
        "checked": len(claims),
        "violations": violations,
        "insufficient_evidence": insufficient,
    }
