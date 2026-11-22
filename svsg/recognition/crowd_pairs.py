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


def _head_focus_views(
    existing: list[_Box], heads: list[_Box], prior_heads: list[_Box], size: tuple[int, int]
) -> list[Box]:
    """Choose up to two close-ups of persistent heads missing from the incumbent bodies."""
    assigned = set(_match_heads(existing, heads).values())
    w, h = size
    views: list[Box] = []
    for i in sorted(range(len(heads)), key=lambda i: -heads[i].score):
        if i in assigned:
            continue
        head = heads[i].bbox
        if not any(iou(head, other.bbox) >= 0.25 for other in prior_heads):
            continue
        left, top, right, bottom = head
        cx = (left + right) / 2
        # Only spend another detector pass near an existing person's upper body.
        for other in existing:
            x0, y0, x1, y1 = other.bbox
            if x0 - 0.5 * (x1 - x0) < cx < x1 + 0.5 * (x1 - x0) and (
                y0 - 0.35 * (y1 - y0) < top < y0 + 0.55 * (y1 - y0)
            ):
                break
        else:
            continue
        side = min(max(192, 15 * (bottom - top)), w, h)
        x = max(0, min(w - side, round(cx - side / 2)))
        y = max(0, min(h - side, round(top - 0.30 * side)))
        view = (x, y, x + side, y + side)
        if view == (0, 0, w, h) or any(iou(view, v) >= 0.45 for v in views):
            continue
        views.append(view)
        if len(views) == 2:
            break
    return views


def _crop_views(
    existing: list[_Box],
    observations: list[RawDetection],
    prior: list[RawDetection],
    size: tuple[int, int],
    *,
    head_focus: bool,
) -> dict[Box, str]:
    w, h = size
    cw, ch = max(1, round(0.4 * w)), max(1, round(0.4 * h))
    centers = [((b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2) for b in existing]
    views = {
        (x, y, x + cw, y + ch): "grid"
        for y in sorted({0, (h - ch) // 2, h - ch})
        for x in sorted({0, (w - cw) // 2, w - cw})
        if sum(x <= cx < x + cw and y <= cy < y + ch for cx, cy in centers) >= 2
    }
    if not head_focus:
        return views
    full = (0, 0, w, h)
    heads = measured_boxes(observations, full, size, "head", 0.50)
    prior_heads = measured_boxes(prior, full, size, "head", 0.35)
    for view in _head_focus_views(existing, heads, prior_heads, size):
        views.setdefault(view, "head_focus")
    return views


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
        if i in occupied or i not in head_for:
            continue
        head = heads[head_for[i]]
        if head.score < (0.60 if body.score < 0.55 else 0.40):
            continue
        overlaps = [iou(body.bbox, other.bbox) for other in existing]
        peak = max(overlaps, default=0.0)
        if not 0.12 <= peak < 0.78:
            continue
        # Each conflicting incumbent must own a separate body and head in this view.
        neighbors = []
        for k, (overlap, other) in enumerate(zip(overlaps, existing, strict=True)):
            if overlap == 0 or (overlap < 0.35 and not _contained(body, other)):
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
    pairs = []
    for i, head, neighbors in pending:
        if i not in temporal:
            continue
        body, earlier = bodies[i], prior[temporal[i]]
        if body.score < 0.55 and (earlier.score < 0.55 or iou(body.bbox, earlier.bbox) < 0.55):
            continue
        pairs.append(_Pair(body, head, earlier, neighbors))
    return pairs


def review_overlaps(
    detector: CrowdHumanDetector,
    current: RasterFunction,
    previous: RasterFunction,
    boxes: list[dict],
    observations: list[RawDetection],
    prior: list[RawDetection],
    *,
    crops: bool,
    head_focus: bool = True,
) -> None:
    size = (current.width, current.height)
    full = (0, 0, *size)
    existing = [_Box(tuple(row["bbox_px"]), row["score"], full) for row in boxes]
    if not existing:
        return
    pairs = _review_view(existing, observations, lambda: prior, full, size)
    views = _crop_views(existing, observations, prior, size, head_focus=head_focus) if crops else {}
    if views:
        with current.to_image() as image, previous.to_image() as earlier:

            def detect_crop(source, view):
                with source.crop(view) as crop:
                    tile = RasterFunction(crop.width, crop.height, crop.tobytes())
                return detector.detect_raster(tile)

            for view in views:
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
            review_view=views.get(pair.body.view, "full_frame"),
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
