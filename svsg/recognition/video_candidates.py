"""Ground video proposals in the current frame, with optional spatial and temporal support."""

from __future__ import annotations

from svsg.image_functions.raster import RasterFunction
from svsg.l1_compiler.detector_adapter import YoloDetector
from svsg.recognition.crowd_detector import CrowdHumanDetector
from svsg.recognition.crowd_pairs import review_overlaps
from svsg.recognition.geometry import Box, canonical_label, iou
from svsg.recognition.video_evidence import _Box, _match, _match_heads, _proposal, measured_boxes


def _measure(
    detector: YoloDetector, frame: RasterFunction, label: str, view: Box, size: tuple[int, int]
) -> list[_Box]:
    return measured_boxes(detector.detect_raster(frame), view, size, label, 0.25)


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


def _add_crowd_candidates(
    detector: CrowdHumanDetector,
    current: RasterFunction,
    previous: RasterFunction | None,
    boxes: list[dict],
    *,
    crops: bool,
    overlap_review: bool,
    head_focus: bool,
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
    if not available and (not overlap_review or previous is None or not boxes):
        return
    head_matches = _match_heads(bodies, heads)
    prior_observations = detector.detect_raster(previous) if previous is not None else []
    prior = [
        _Box(d.bbox, d.detection_score, full) for d in prior_observations if d.label == "person"
    ]
    temporal = _match(bodies, prior, 0.55)
    for i in available:
        body = bodies[i]
        if i not in temporal and (i not in head_matches or body.score < 0.75):
            continue
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
    if overlap_review and previous is not None:
        review_overlaps(
            detector,
            current,
            previous,
            boxes,
            observations,
            prior_observations,
            crops=crops,
            head_focus=head_focus,
        )


def frame_candidates(
    detector: YoloDetector,
    current: RasterFunction,
    previous: RasterFunction | None,
    label: str,
    *,
    detail: bool | None = None,
    crowd_detector: CrowdHumanDetector | None = None,
    overlap_review: bool = True,
    head_focus: bool = True,
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
        _add_crowd_candidates(
            crowd_detector,
            current,
            previous if distinct else None,
            boxes,
            crops=detail is not False,
            overlap_review=overlap_review,
            head_focus=head_focus,
        )
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
