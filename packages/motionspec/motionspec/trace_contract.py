"""Validate and normalize the public, data-only Qt trace before fitting."""

from __future__ import annotations

import math

TRACE_SCHEMA = "motionspec-trace/0.1"
FIELDS = (
    "tick",
    "x",
    "y",
    "width",
    "height",
    "opacity",
    "scale",
    "rotation",
    "visible",
    "scene_x",
    "scene_y",
)
MAX_TICKS = 200_000
MAX_ROWS = 200_000
MAX_ELEMENTS = 20_000


def _number(value, name, *, minimum=None, positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def normalize_trace(trace: dict) -> dict:
    if not isinstance(trace, dict) or trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"expected a {TRACE_SCHEMA} trace")
    ticks = trace.get("ticks_ms")
    if not isinstance(ticks, list) or not 2 <= len(ticks) <= MAX_TICKS:
        raise ValueError("trace ticks must contain 2..200000 samples")
    for i, value in enumerate(ticks):
        _number(value, "timestamp", minimum=0)
        if i and value <= ticks[i - 1]:
            raise ValueError("trace timestamps must be strictly increasing")
    size = trace.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("trace size must contain width and height")
    for value in size:
        _number(value, "window size", positive=True)
    ratio = _number(trace.get("device_pixel_ratio", 1.0), "device pixel ratio", positive=True)
    if max(size) * ratio > 100_000:
        raise ValueError("trace dimensions exceed the coordinate budget")
    elements = trace.get("elements")
    if not isinstance(elements, list) or len(elements) > MAX_ELEMENTS:
        raise ValueError("invalid trace elements or element budget exceeded")
    normalized, identities, total, event_total = [], set(), 0, 0
    for element in elements:
        if not isinstance(element, dict):
            raise ValueError("trace element must be an object")
        identity = element.get("id")
        if type(identity) not in (str, int) or identity in identities:
            raise ValueError("trace element IDs must be unique strings or integers")
        identities.add(identity)
        fields = element.get("fields")
        if (
            not isinstance(fields, list)
            or len(fields) != len(FIELDS)
            or any(type(f) is not str for f in fields)
            or set(fields) != set(FIELDS)
        ):
            raise ValueError("trace fields must name each supported field exactly once")
        order = [fields.index(name) for name in FIELDS]
        rows = element.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("trace element rows must be nonempty")
        total += len(rows)
        if total > MAX_ROWS:
            raise ValueError("trace exceeds the row budget")
        canonical, previous = [], -1
        for row in rows:
            if not isinstance(row, list) or len(row) != len(FIELDS):
                raise ValueError("trace row length does not match fields")
            values = [row[index] for index in order]
            tick = values[0]
            if type(tick) is not int or not previous < tick < len(ticks):
                raise ValueError("trace ticks must be in bounds and strictly increasing")
            previous = tick
            for name, value in zip(FIELDS[1:], values[1:], strict=True):
                _number(value, name)
            if (
                min(values[3:5]) < 0
                or values[6] < 0
                or not 0 <= values[5] <= 1
                or values[8] not in (0, 1)
            ):
                raise ValueError("invalid size, scale, opacity or visibility")
            canonical.append(values)
        events = element.get("events", [])
        if not isinstance(events, list) or len(events) > MAX_TICKS:
            raise ValueError("invalid lifecycle events")
        event_total += len(events)
        if event_total > MAX_ROWS:
            raise ValueError("trace exceeds the lifecycle event budget")
        last_event = -1
        for event in events:
            if not isinstance(event, dict) or event.get("kind") not in ("appeared", "disappeared"):
                raise ValueError("invalid lifecycle event")
            tick = event.get("tick")
            if type(tick) is not int or not last_event < tick < len(ticks):
                raise ValueError("lifecycle ticks must be ordered and in bounds")
            last_event = tick
        normalized.append({**element, "fields": list(FIELDS), "rows": canonical})
    return {**trace, "device_pixel_ratio": ratio, "elements": normalized}
