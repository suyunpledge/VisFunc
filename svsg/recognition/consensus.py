"""Cross-view association without counting repeated outputs as independent objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from svsg.contracts.recognition import ViewObject

from .geometry import Box, iou


@dataclass
class Candidate:
    supports: list[ViewObject] = field(default_factory=list)

    @property
    def first(self) -> ViewObject:
        return self.supports[0]

    def agreement(self, threshold: float, reviewer: str, require_two_models: bool):
        pairs = []
        for a, b in combinations(self.supports, 2):
            if a.clipped or b.clipped or a.view_id == b.view_id or a.label != b.label:
                continue
            if a.model == b.model and a.evidence_key and a.evidence_key == b.evidence_key:
                continue
            if require_two_models and (a.model == b.model or reviewer not in (a.model, b.model)):
                continue
            overlap = iou(a.bbox_px, b.bbox_px)
            if overlap >= threshold:
                pairs.append((overlap, a, b))
        return max(pairs, key=lambda row: row[0]) if pairs else None


def associate(groups: list[Candidate], objects: list[ViewObject], threshold: float) -> None:
    """One detection per group per view; only same-class and overlapping boxes can join."""
    options = []
    for oi, obj in enumerate(objects):
        for gi, group in enumerate(groups):
            first = group.first
            if first.label != obj.label or any(s.view_id == obj.view_id for s in group.supports):
                continue
            overlap = iou(first.bbox_px, obj.bbox_px)
            if overlap >= threshold:
                options.append((overlap, oi, gi))
    used_objects: set[int] = set()
    used_groups: set[int] = set()
    for _, oi, gi in sorted(options, reverse=True):
        if oi not in used_objects and gi not in used_groups:
            groups[gi].supports.append(objects[oi])
            used_objects.add(oi)
            used_groups.add(gi)
    for oi, obj in enumerate(objects):
        if oi not in used_objects:
            # Avoid duplicate boxes within one response inflating the candidate count.
            if any(
                g.first.label == obj.label and iou(g.first.bbox_px, obj.bbox_px) >= 0.8
                for g in groups
            ):
                continue
            groups.append(Candidate([obj]))


def consensus_box(a: Box, b: Box) -> Box:
    # Coordinate median of two corroborating views; not a pixel segmentation estimate.
    x0, y0, x1, y1 = (round((x + y) / 2) for x, y in zip(a, b, strict=True))
    return x0, y0, x1, y1
