"""Ground video proposals in the current frame, with optional spatial and temporal support."""

from __future__ import annotations

from dataclasses import dataclass

from svsg.image_functions.raster import RasterFunction
from svsg.l1_compiler.detector_adapter import YoloDetector
from svsg.recognition.crowd_detector import CrowdHumanDetector
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


def _measure(
    detector: YoloDetector, frame: RasterFunction, label: str, view: Box, size: tuple[int, int]
) -> list[_Box]:
    x, y, right, bottom = view
    width, height = size
    boxes = []
    for detection in detector.detect_raster(frame):
        if detection.detection_score < 0.25 or canonical_label(detection.label) != label:
            continue
        a, b, c, d = detection.bbox
        if not (0 <= a < c <= frame.width and 0 <= b < d <= frame.height):
            continue
        # An internal crop edge can turn part of a person into a second detection.
        if (
            (x > 0 and a <= 2)
            or (y > 0 and b <= 2)
            or (right < width and c >= frame.width - 2)
            or (bottom < height and d >= frame.height - 2)
        ):
            continue
        boxes.append(_Box((x + a, y + b, x + c, y + d), detection.detection_score, view))
    return boxes


def _detail_boxes(detector: YoloDetector, frame: RasterFunction, label: str) -> list[_Box]:
    w, h = frame.width, frame.height
    cw, ch = round(0.6 * w), round(0.6 * h)
    if cw == w or ch == h:
        return []
    candidates = []
    with frame.to_image() as image:
        for y in (0, h - ch):
            for x in (0, w - cw):
                view = (x, y, x + cw, y + ch)
                with image.crop(view) as crop:
                    tile = RasterFunction(cw, ch, crop.tobytes())
                candidates.extend(_measure(detector, tile, label, view, (w, h)))
    unique: list[_Box] = []
    for box in sorted(candidates, key=lambda b: -b.score):
        if all(iou(box.bbox, other.bbox) < 0.5 for other in unique):
            unique.append(box)
    return unique


def _match(current: list[_Box], support: list[_Box], minimum: float) -> dict[int, int]:
    edges = [
        (iou(a.bbox, b.bbox), i, j) for i, a in enumerate(current) for j, b in enumerate(support)
    ]
    matches: dict[int, int] = {}
    used: set[int] = set()
    for overlap, i, j in sorted(edges, reverse=True):
        if overlap < minimum:
            break
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


def _add_crowd_candidates(
    detector: CrowdHumanDetector,
    current: RasterFunction,
    previous: RasterFunction | None,
    boxes: list[dict],
) -> None:
    full = (0, 0, current.width, current.height)
    observations = detector.detect_raster(current)
    bodies = [
        _Box(d.bbox, d.detection_score, full)
        for d in observations
        if d.label == "person" and d.detection_score >= 0.55
    ]
    heads = [_Box(d.bbox, d.detection_score, full) for d in observations if d.label == "head"]
    available = [
        i
        for i, b in enumerate(bodies)
        if all(iou(b.bbox, other["bbox_px"]) < 0.45 for other in boxes)
    ]
    if not available:
        return
    head_matches = _match_heads(bodies, heads)
    prior = (
        [
            _Box(d.bbox, d.detection_score, full)
            for d in detector.detect_raster(previous)
            if d.label == "person"
        ]
        if previous is not None
        else []
    )
    temporal = _match(bodies, prior, 0.35)
    for i in available:
        if i not in head_matches and i not in temporal:
            continue
        body = bodies[i]
        if any(iou(body.bbox, other["bbox_px"]) >= 0.47 for other in boxes):
            continue
        p = prior[temporal[i]] if i in temporal else None
        head = heads[head_matches[i]] if i in head_matches else None
        entry = _proposal(body, p, None)
        entry.update(
            source="crowd_full_body",
            head_bbox_px=list(head.bbox) if head else None,
            head_score=head.score if head else None,
            prior_score=p.score if p else None,
        )
        boxes.append(entry)


def frame_candidates(
    detector: YoloDetector,
    current: RasterFunction,
    previous: RasterFunction | None,
    label: str,
    *,
    detail: bool | None = None,
    crowd_detector: CrowdHumanDetector | None = None,
) -> dict:
    label = canonical_label(label)
    if crowd_detector is not None and label != "person":
        raise ValueError("crowd detection requires the person label")
    size = (current.width, current.height)
    if previous and (previous.width, previous.height) != size:
        raise ValueError("candidate frames must have matching dimensions")
    full = (0, 0, *size)
    distinct = previous is not None and previous.pixel_sha256 != current.pixel_sha256
    current_boxes = _measure(detector, current, label, full, size)
    prior_boxes = _measure(detector, previous, label, full, size) if distinct and previous else []
    temporal = _match(current_boxes, prior_boxes, 0.35)
    frame_area = current.width * current.height
    if detail is None:
        has_small_box = any(box.area <= 0.0025 * frame_area for box in current_boxes)
        has_support = bool(temporal) or any(box.score >= 0.4 for box in current_boxes)
        use_detail = label == "person" and (has_small_box or not has_support)
    else:
        use_detail = detail
    tiles = _detail_boxes(detector, current, label) if use_detail else []
    spatial = _match(current_boxes, tiles, 0.5)
    boxes = []
    for i, box in enumerate(current_boxes):
        if box.score >= 0.4 or i in temporal or i in spatial:
            boxes.append(
                _proposal(
                    box,
                    prior_boxes[temporal[i]] if i in temporal else None,
                    tiles[spatial[i]] if i in spatial else None,
                )
            )
    # A new tile-only proposal needs a second observation at the same scale.
    # Skip the prior-frame tile pass when there is nothing new to confirm.
    used = set(spatial.values())
    novel = [
        j
        for j, box in enumerate(tiles)
        if j not in used
        and box.score >= 0.45
        and (detail is not None or box.area <= 0.01 * frame_area)
        and all(iou(box.bbox, other["bbox_px"]) < 0.45 for other in boxes)
    ]
    if novel and distinct and previous:
        prior_tiles = _detail_boxes(detector, previous, label)
        tracked = _match(tiles, prior_tiles, 0.35)
        for j in novel:
            box = tiles[j]
            if j in tracked and all(iou(box.bbox, other["bbox_px"]) < 0.45 for other in boxes):
                boxes.append(_proposal(box, prior_tiles[tracked[j]], None))
    if crowd_detector is not None:
        _add_crowd_candidates(crowd_detector, current, previous if distinct else None, boxes)
    result = {
        "label": label,
        "model": detector.model_name,
        "status": "unverified_candidates",
        "detail_views": use_detail,
        "coordinate_space": "source_pixels_xyxy",
        "size_px": list(size),
        "pixel_sha256": current.pixel_sha256,
        "previous_pixel_sha256": previous.pixel_sha256 if previous else None,
        "boxes": boxes,
    }
    if crowd_detector is not None:
        result["crowd_model"] = {
            "name": crowd_detector.model_name,
            "sha256": crowd_detector.weights_sha256,
        }
    return result
