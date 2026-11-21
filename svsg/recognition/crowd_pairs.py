"""Recover a second pedestrian when overlapping bodies have separate visible evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from svsg.image_functions.raster import RasterFunction
from svsg.l1_compiler.detector_adapter import RawDetection
from svsg.recognition.crowd_detector import CrowdHumanDetector
from svsg.recognition.geometry import Box, iou
from svsg.recognition.video_evidence import _Box, _match, _match_heads, _proposal, measured_boxes


@dataclass
class _Pair:
    body: _Box
    head: _Box
    prior: _Box
    neighbors: list[tuple[int, _Box]]


def _contained(a: _Box, b: _Box) -> bool:
    ax, ay, ar, ab = a.bbox
    bx, by, br, bb = b.bbox
    common = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    return common >= 0.65 * min(a.area, b.area)


def _same_observation(a: _Pair, b: _Pair) -> bool:
    heads = iou(a.head.bbox, b.head.bbox)
    overlap = iou(a.body.bbox, b.body.bbox)
    if heads >= 0.35:
        return True
    if overlap < 0.45:
        return False
    # In the same view, the matching stage already required distinct head
    # and prior detections. Across crops those assignments are independent.
    return (
        a.body.view != b.body.view
        or heads >= 0.15
        or overlap >= 0.78
        or iou(a.prior.bbox, b.prior.bbox) >= 0.70
    )


def _review_view(
    existing: list[_Box],
    now: list[RawDetection],
    prior_observations: Callable[[], list[RawDetection]],
    view: Box,
    size: tuple[int, int],
) -> list[_Pair]:
    bodies = measured_boxes(now, view, size, "person", 0.40)
    heads = measured_boxes(now, view, size, "head", 0.35)
    assigned = _match(existing, bodies, 0.40)
    occupied = set(assigned.values())
    head_for = _match_heads(bodies, heads)
    pending = []
    for i, body in enumerate(bodies):
        if i in occupied or i not in head_for or body.score < 0.55:
            continue
        head = heads[head_for[i]]
        if head.score < 0.40:
            continue
        overlaps = [iou(body.bbox, other.bbox) for other in existing]
        peak = max(overlaps, default=0.0)
        if not 0.12 <= peak < 0.78:
            continue
        # Each conflicting incumbent must own a separate body and head in this view.
        neighbors = []
        for k, (overlap, other) in enumerate(zip(overlaps, existing, strict=True)):
            if overlap < 0.35 and not _contained(body, other):
                continue
            other_head = head_for.get(assigned.get(k, -1))
            if other_head is None or iou(head.bbox, heads[other_head].bbox) >= 0.15:
                break
            neighbors.append((k, heads[other_head]))
        else:
            pending.append((i, head, neighbors))
    if not pending:
        return []
    prior = measured_boxes(prior_observations(), view, size, "person", 0.35)
    temporal = _match(bodies, prior, 0.45)
    return [
        _Pair(bodies[i], h, prior[temporal[i]], neighbors)
        for i, h, neighbors in pending
        if i in temporal
    ]


def review_overlaps(
    detector: CrowdHumanDetector,
    current: RasterFunction,
    previous: RasterFunction,
    boxes: list[dict],
    observations: list[RawDetection],
    prior: list[RawDetection],
    *,
    crops: bool,
) -> None:
    size = (current.width, current.height)
    full = (0, 0, *size)
    existing = [_Box(tuple(row["bbox_px"]), row["score"], full) for row in boxes]
    if not existing:
        return
    pairs = _review_view(existing, observations, lambda: prior, full, size)
    if crops and len(existing) >= 2:
        w, h = size
        cw, ch = max(1, round(0.4 * w)), max(1, round(0.4 * h))
        centers = [((b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2) for b in existing]
        with current.to_image() as image, previous.to_image() as earlier:

            def detect_crop(source, view):
                with source.crop(view) as crop:
                    return detector.detect_raster(RasterFunction(cw, ch, crop.tobytes()))

            for y in sorted({0, (h - ch) // 2, h - ch}):
                for x in sorted({0, (w - cw) // 2, w - cw}):
                    view = (x, y, x + cw, y + ch)
                    if sum(x <= cx < x + cw and y <= cy < y + ch for cx, cy in centers) < 2:
                        continue
                    pairs.extend(
                        _review_view(
                            existing,
                            detect_crop(image, view),
                            partial(detect_crop, earlier, view),
                            view,
                            size,
                        )
                    )
    accepted: list[_Pair] = []
    for pair in sorted(pairs, key=lambda p: -p.body.score):
        # Overlapping crops can rediscover the same person with slightly different coordinates.
        if any(_same_observation(pair, other) for other in accepted):
            continue
        entry = _proposal(pair.body, pair.prior, None)
        entry.update(
            source="crowd_overlap",
            head_bbox_px=list(pair.head.bbox),
            head_score=pair.head.score,
            prior_score=pair.prior.score,
            separated_from=[
                {"bbox_px": list(existing[k].bbox), "head_bbox_px": list(h.bbox)}
                for k, h in pair.neighbors
            ],
        )
        boxes.append(entry)
        accepted.append(pair)
