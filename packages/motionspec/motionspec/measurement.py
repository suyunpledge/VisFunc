"""Bounded JSON delivery adapter; envelope repair never changes measurement values."""

from __future__ import annotations

import json
import math
import re

NUMBERS = ("start_ms", "duration_ms", "distance_px")
ENUMS = {
    "direction": {"up", "down", "left", "right", "none"},
    "opacity_change": {"fade-in", "fade-out", "none"},
    "easing": {"linear", "ease-in", "ease-out", "ease-in-out"},
}


def _finite_number(value, *, nonnegative):
    try:
        return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)
    except OverflowError:
        return False


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def decode_measurement(text: str, *, numeric_fields=NUMBERS, nonnegative_fields=NUMBERS,
                       enum_fields=None) -> dict:
    """Keep diagnostic rows even on schema failure; accepted gates downstream use."""
    result = {
        "rows": None,
        "strict_format_pass": False,
        "accepted": False,
        "normalizations": [],
        "errors": [],
    }
    if not isinstance(text, str) or len(text) > 200_000:
        result["errors"].append("response exceeds text budget or is not text")
        return result
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.S)
    if fence:
        stripped = fence[1].strip()
        result["normalizations"].append("markdown_fence")
    try:
        data = json.loads(stripped, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        result["errors"].append("invalid JSON")
        return result
    if isinstance(data, list):
        data = {"animations": data}
        result["normalizations"].append("array_envelope")
    rows = data.get("animations") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) > 40 or not all(isinstance(r, dict) for r in rows):
        result["errors"].append("animations must be an array of at most 40 objects")
        return result
    result["rows"] = rows
    for i, row in enumerate(rows):
        if not isinstance(row.get("element"), str) or not row["element"].strip():
            result["errors"].append(f"animations[{i}].element must be nonempty text")
        for key in numeric_fields:
            value = row.get(key)
            if not _finite_number(value, nonnegative=key in nonnegative_fields):
                result["errors"].append(
                    f"animations[{i}].{key} must be a finite number in the declared range"
                )
        for key, choices in (ENUMS if enum_fields is None else enum_fields).items():
            if not isinstance(row.get(key), str) or row[key] not in choices:
                result["errors"].append(f"animations[{i}].{key} is missing or invalid")
    result["accepted"] = not result["errors"]
    result["strict_format_pass"] = result["accepted"] and not result["normalizations"]
    return result


def sum_usage(attempts: list[dict]) -> dict:
    """Sum provider counters recursively, preserving nested cached/reasoning tokens."""

    def add(target, source):
        for key, value in source.items():
            if type(value) is int:
                target[key] = target.get(key, 0) + value
            elif isinstance(value, dict):
                add(target.setdefault(key, {}), value)

    total = {}
    for attempt in attempts:
        add(total, attempt.get("usage") or {})
    return total
