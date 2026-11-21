"""Source-pixel observations and one-to-one matching for video candidates."""

from __future__ import annotations

from dataclasses import dataclass

from svsg.l1_compiler.detector_adapter import RawDetection
from svsg.recognition.geometry import Box, canonical_label, iou


@dataclass(frozen=True)
class _Box:
    bbox: Box
    score: float
    view: Box

    @property
    def area(self) -> int:
        x, y, right, bottom = self.bbox
        return (right - x) * (bottom - y)


def measured_boxes(
    detections: list[RawDetection], view: Box, size: tuple[int, int], label: str, minimum: float
) -> list[_Box]:
    x, y, right, bottom = view
    w, h = right - x, bottom - y
    boxes = []
    for d in detections:
        if canonical_label(d.label) != label or d.detection_score < minimum:
            continue
        left, top, right_edge, bottom_edge = d.bbox
        if not (0 <= left < right_edge <= w and 0 <= top < bottom_edge <= h):
            continue
        if (
            (x > 0 and left <= 2)
            or (y > 0 and top <= 2)
            or (right < size[0] and right_edge >= w - 2)
            or (bottom < size[1] and bottom_edge >= h - 2)
        ):
            continue
        boxes.append(
            _Box((left + x, top + y, right_edge + x, bottom_edge + y), d.detection_score, view)
        )
    return boxes


def _match(current: list[_Box], support: list[_Box], minimum: float) -> dict[int, int]:
    edges = []
    for i, a in enumerate(current):
        for j, b in enumerate(support):
            overlap = iou(a.bbox, b.bbox)
            if overlap >= minimum:
                edges.append((overlap, i, j))
    matches: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in sorted(edges, reverse=True):
        if i not in matches and j not in used:
            matches[i] = j
            used.add(j)
    return matches


def _proposal(box: _Box, prior: _Box | None, detail: _Box | None) -> dict:
    return {
        "bbox_px": list(box.bbox),
        "score": round(box.score, 4),
        "view_px": list(box.view),
        "prior_iou": round(iou(box.bbox, prior.bbox), 4) if prior else None,
        "prior_bbox_px": list(prior.bbox) if prior else None,
        "prior_view_px": list(prior.view) if prior else None,
        "supporting_detail_bbox_px": list(detail.bbox) if detail else None,
        "supporting_detail_view_px": list(detail.view) if detail else None,
    }


def _match_heads(bodies: list[_Box], heads: list[_Box]) -> dict[int, int]:
    """Assign a visible head once, favoring the expected upper-body center."""
    edges = []
    for i, body in enumerate(bodies):
        left, top, right, bottom = body.bbox
        w, h = right - left, bottom - top
        for j, head in enumerate(heads):
            x, y, x1, y1 = head.bbox
            cx, cy = (x + x1) / 2, (y + y1) / 2
            if (
                left <= cx <= right
                and top - 0.08 * h <= cy <= top + 0.42 * h
                and y1 - y <= 0.55 * h
                and x1 - x <= 0.95 * w
            ):
                cost = ((cx - (left + right) / 2) / w) ** 2 + ((cy - top) / h - 0.12) ** 2
                edges.append((cost, i, j))
    matched: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in sorted(edges):
        if i not in matched and j not in used:
            matched[i] = j
            used.add(j)
    return matched
