"""Explicit transforms between model coordinates, crop pixels and source pixels."""

from __future__ import annotations

from math import ceil, floor

Box = tuple[int, int, int, int]


def pixel_box(box: Box, view: Box) -> Box:
    x0, y0, x1, y1 = view
    w, h = x1 - x0, y1 - y0
    return (
        x0 + floor(box[0] * w / 1000),
        y0 + floor(box[1] * h / 1000),
        x0 + ceil(box[2] * w / 1000),
        y0 + ceil(box[3] * h / 1000),
    )


def context_box(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    dx, dy = max(4, ceil((x1 - x0) * 0.15)), max(4, ceil((y1 - y0) * 0.15))
    return max(0, x0 - dx), max(0, y0 - dy), min(width, x1 + dx), min(height, y1 + dy)


def iou(a: Box, b: Box) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    if width <= 0:
        return 0.0
    height = min(a[3], b[3]) - max(a[1], b[1])
    if height <= 0:
        return 0.0
    area = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union else 0.0


def canonical_label(label: str) -> str:
    # Only spelling/standard common-class equivalents; never fuzzy semantic matches.
    value = " ".join(label.lower().replace("_", " ").replace("-", " ").split())
    return {
        "human": "person",
        "man": "person",
        "woman": "person",
        "astronaut": "person",
        "coffee cup": "cup",
        "mug": "cup",
        "automobile": "car",
    }.get(value, value)
