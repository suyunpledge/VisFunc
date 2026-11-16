"""Recompute spatial relations from two same-grid region descriptions."""

from __future__ import annotations

from collections import defaultdict
from math import hypot

from svsg.contracts.contour import Contour, ring_edges
from svsg.contracts.image_functions import RegionParameters

RELATIONS = {
    "left_of": "包围框位于左侧",
    "right_of": "包围框位于右侧",
    "above": "包围框位于上方",
    "below": "包围框位于下方",
    "bbox_contains": "包围框包含",
    "bbox_iou": "包围框 IoU",
    "center_distance_px": "重心间距（格长）",
    "shared_edge_px": "共享轮廓边长（格长）",
    "touching": "共享边相接",
}


def _segments(outline: Contour) -> dict[tuple[bool, int], list[tuple[int, int]]]:
    lines: dict[tuple[bool, int], list[tuple[int, int]]] = defaultdict(list)
    for ring in outline.rings:
        for (x0, y0), (x1, y1) in ring_edges(ring):
            if x0 == x1:
                lines[(True, x0)].append((min(y0, y1), max(y0, y1)))
            else:
                lines[(False, y0)].append((min(x0, x1), max(x0, x1)))
    return lines


def shared_edge(a: Contour, b: Contour) -> int:
    aa, bb = _segments(a), _segments(b)
    length = 0
    for line in aa.keys() & bb.keys():
        first, second = sorted(aa[line]), sorted(bb[line])
        i = j = 0
        while i < len(first) and j < len(second):
            lo, hi = first[i]
            other_lo, other_hi = second[j]
            length += max(0, min(hi, other_hi) - max(lo, other_lo))
            if hi <= other_hi:
                i += 1
            else:
                j += 1
    return length


def evaluate_relation(a: RegionParameters, b: RegionParameters, name: str) -> int | float | bool:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    if name == "left_of":
        return ax1 <= bx0
    if name == "right_of":
        return bx1 <= ax0
    if name == "above":
        return ay1 <= by0
    if name == "below":
        return by1 <= ay0
    if name == "bbox_contains":
        return ax0 <= bx0 and ay0 <= by0 and ax1 >= bx1 and ay1 >= by1
    if name == "bbox_iou":
        intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * (
            max(0, min(ay1, by1) - max(ay0, by0))
        )
        union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
        return intersection / union
    if name == "center_distance_px":
        return hypot(a.centroid_x - b.centroid_x, a.centroid_y - b.centroid_y)
    if name in ("shared_edge_px", "touching"):
        if a.contour is None or b.contour is None:
            raise ValueError("边界缺失，证据不足")
        overlap = shared_edge(a.contour, b.contour)
        return overlap if name == "shared_edge_px" else overlap > 0
    raise ValueError("未知空间函数")
