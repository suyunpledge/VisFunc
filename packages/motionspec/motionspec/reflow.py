"""Measure rectangular layout elements from visible fill boundaries in decoded frames."""

import cv2
import numpy as np

from .motion import _fit_channel, _r


def _boxes(image, tone):
    # A narrow colour band separates dark UI surfaces even when their contrast is low.
    mask = cv2.inRange(image, np.maximum(tone - 8, 0), np.minimum(tone + 8, 255))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    height, width = mask.shape
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue  # a clipped side cannot establish an absolute dimension
        if area >= 20 and area >= 0.40 * w * h and max(w, h) >= 14:
            boxes.append(np.array([x, y, x + w, y + h], dtype=float))
    return boxes


def _tones(image, focus):
    x, y, r, b = focus
    pixels = image[y:b:3, x:r:3].reshape(-1, 3).astype(np.int32)
    bins = pixels // 8
    keys = (bins[:, 0] * 32 + bins[:, 1]) * 32 + bins[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    tones = []
    for idx in np.argsort(-counts)[:8]:
        if counts[idx] < 8:
            break
        tone = np.median(pixels[keys == values[idx]], axis=0).astype(np.int32)
        if all(np.abs(tone - other).max() > 8 for other in tones):
            tones.append(tone)
    return tones


def _track(history, seed):
    """Follow one component backwards. Only an initially collapsed strip may disappear."""
    edges = [seed]
    for choices in reversed(history[:-1]):
        prev = edges[-1]
        distances = [np.abs(box - prev).max() for box in choices]
        if not distances or min(distances) > max(12, 0.4 * max(prev[2:] - prev[:2])):
            break
        order = np.argsort(distances)
        if len(order) > 1 and distances[order[1]] - distances[order[0]] < 2:
            return None
        edges.append(choices[order[0]])
    edges = np.array(edges[::-1])
    if len(edges) < 5:
        return None
    missing = len(history) - len(edges)
    if missing:
        initial = edges[0].copy()
        axis = int(np.argmin(initial[2:] - initial[:2]))
        if (
            initial[axis + 2] - initial[axis] > 4
            or np.ptp(edges[:, axis + 2] - edges[:, axis]) < 10
        ):
            return None
        if np.ptp(edges[:, axis]) <= 1:
            initial[axis + 2] = initial[axis]
        elif np.ptp(edges[:, axis + 2]) <= 1:
            initial[axis] = initial[axis + 2]
        else:
            return None
        # Rounded ends of a newly visible strip understate its transverse span.
        # Only extend the stable sides back into frames too thin to measure them.
        thick = np.flatnonzero(edges[:, axis + 2] - edges[:, axis] >= 9)
        cross = [1 - axis, 3 - axis]
        if len(thick) and np.ptp(edges[thick][:, cross], axis=0).max() <= 4:
            sides = np.median(edges[thick][:, cross], axis=0)
            initial[cross] = sides
            edges[: thick[0], cross] = sides
        edges = np.vstack([np.tile(initial, (missing, 1)), edges])
    return edges


def _geometry(edges):
    sizes = edges[:, 2:] - edges[:, :2]
    change = sizes[-1] - sizes[0]
    axis = int(np.argmax(np.abs(change)))
    shift = edges[-1, :2] - edges[0, :2]
    if abs(change[axis]) >= 5:
        excursion = np.ptp(edges, axis=0)
        if (
            np.ptp(sizes[:, 1 - axis]) > 4
            or excursion[1 - axis] > 4
            or min(excursion[axis], excursion[axis + 2]) > 4
        ):
            return None
        name, values = "size", sizes[:, axis] - sizes[0, axis]
    elif np.ptp(sizes, axis=0).max() <= 4 and np.linalg.norm(shift) >= 5:
        name, values = "translate", (edges[:, :2] - edges[0, :2]) @ (shift / np.linalg.norm(shift))
    else:
        return None
    steps = np.diff(values) * np.sign(values[-1])
    if steps.max() > max(3, 0.45 * abs(values[-1])) or steps[steps < 0].sum() < -max(
        2, 0.08 * abs(values[-1])
    ):
        return None
    return name, values


def _segment(edges, geometry, times, origin, scale):
    name, values = geometry
    first, last, _, progress, curve = _fit_channel(times, values, 0.3 / abs(values[-1]), False)
    if curve is None or curve["rmse"] > 0.065:
        return None
    size = (edges[:, 2:] - edges[:, :2]) * scale
    center = (edges[:, :2] + edges[:, 2:]) / 2
    position = (center + origin) * scale
    channel = {
        "easing": curve,
        "reference": "tracked_box_edges",
        "t_start_ms": _r(times[first], 3),
        "t_end_ms": _r(times[last], 3),
        "duration_ms": _r(times[last] - times[first], 3),
    }
    if name == "translate":
        delta = position[last] - position[first]
        channel.update(
            {
                "reference": "tracked_box_center",
                "from": position[first].round(2).tolist(),
                "to": position[last].round(2).tolist(),
                "delta": delta.round(2).tolist(),
                "distance_px": round(float(np.linalg.norm(delta)), 2),
            }
        )
    else:
        axis = int(np.argmax(np.abs(size[last] - size[first])))
        fixed = min((axis, axis + 2), key=lambda k: np.ptp(edges[:, k]))
        channel.update(
            width_from=float(size[first, 0]),
            width_to=float(size[last, 0]),
            height_from=float(size[first, 1]),
            height_to=float(size[last, 1]),
            anchor=("left", "top", "right", "bottom")[fixed],
        )
    roi = (*edges[:, :2].min(axis=0).astype(int), *edges[:, 2:].max(axis=0).astype(int))
    samples = []
    for k in np.unique(np.linspace(first, last, min(last - first + 1, 40)).round().astype(int)):
        samples.append(
            {
                "t_ms": _r(times[k], 3),
                "progress": _r(progress[k], 4),
                "channel_progress": {name: _r(progress[k], 4)},
                "x": _r(position[k, 0], 2),
                "y": _r(position[k, 1], 2),
                "width": float(size[k, 0]),
                "height": float(size[k, 1]),
            }
        )
    mid = first + int(np.argmin(abs(progress[first : last + 1] - 0.5)))
    keys = {mid: "midpoint", first: "start", last: "end"}
    return {
        "kind": "transform",
        "model": "layout-edges",
        "confidence": "medium",
        "channels": {name: channel},
        "dominant": name,
        "easing": curve,
        "samples": samples,
        "notes": ["Measured four visible fill edges in source pixels."],
        "_roi": tuple(map(int, roi)),
        "_view": {
            "start": first,
            "end": last,
            "pos": center - roi[:2],
            "progress": progress,
            "progress_range": (first, last),
            "keys": sorted(keys.items()),
        },
    }


def measure_reflow(crops, times, origin, scale, focus, threshold):
    """Replace a change window only when a size track explains its changing pixels."""
    if not 5 <= len(crops) <= 180:
        return None
    x, y, r, b = focus
    change = cv2.absdiff(crops[0][y:b, x:r], crops[-1][y:b, x:r]).max(axis=2) > threshold
    if change.sum() < 30:
        return None
    results = []
    for tone in _tones(crops[-1], focus):
        end_boxes = _boxes(crops[-1], tone)
        seeds = [
            box
            for box in end_boxes
            if min(box[2:] - box[:2]) >= 6
            and box[2] > x
            and box[3] > y
            and box[0] < r
            and box[1] < b
        ]
        if not seeds or len(seeds) > 12:
            continue
        history = [_boxes(c, tone) for c in crops[:-1]] + [end_boxes]
        for seed in seeds:
            edges = _track(history, seed)
            geometry = _geometry(edges) if edges is not None else None
            if geometry is None:
                continue
            # A moving neighbour near a window must contribute evidence inside its focus.
            lo = np.maximum(edges[:, :2].min(axis=0).astype(int), [x, y])
            hi = np.minimum(edges[:, 2:].max(axis=0).astype(int), [r, b])
            if np.any(hi <= lo) or change[lo[1] - y : hi[1] - y, lo[0] - x : hi[0] - x].sum() < 20:
                continue
            if any(np.max(np.abs(edges - other)) <= 4 for other, _ in results):
                continue
            results.append((edges, geometry))
    if not any(g[0] == "size" for _, g in results):
        return None
    remainder = change.copy()
    segments = []
    for edges, geometry in results:
        seg = _segment(edges, geometry, np.asarray(times), np.asarray(origin), scale)
        if seg is None:
            return None
        left, top, right, bottom = seg["_roi"]
        remainder[
            max(0, top - y - 3) : max(0, bottom - y + 3),
            max(0, left - x - 3) : max(0, right - x + 3),
        ] = False
        segments.append(seg)
    if remainder.sum() > max(20, 0.06 * change.sum()):
        return None
    return segments
