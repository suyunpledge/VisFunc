"""Reconcile an uncertain caption role with nearby observations of the same lettering."""

from __future__ import annotations

from svsg.recognition.geometry import Box, iou

CONTEXT_MS = 800


def within_window(stamp: float, other: float, window_ms: float) -> bool:
    # Container PTS converted to milliseconds may land a fraction of a nanosecond
    # past an exact sampling boundary (e.g. 2800.0000000000005 - 2000).
    return abs(stamp - other) <= window_ms + 1e-6


def _same_line_role(frame: dict, text: str, box: Box) -> str | None:
    # One OCR view can split a title into two words while another joins it.
    # Compare the union only for fragments on this baseline inside its horizontal span.
    left, top, right, bottom = box
    height = bottom - top
    fragments = []
    for key in ("detached_text_candidates", "subtitles"):
        for row in frame.get(key, []):
            a, b, c, d = row["bbox_px"]
            if (
                a >= left - height / 2
                and c <= right + height / 2
                and abs((b + d - top - bottom) / 2) <= 0.3 * height
            ):
                fragments.append((a, b, c, d, row["text"], key))
    if not fragments:
        return None
    fragments.sort()
    joined = "".join("".join(part[4].split()) for part in fragments)
    extent = (
        min(p[0] for p in fragments),
        min(p[1] for p in fragments),
        max(p[2] for p in fragments),
        max(p[3] for p in fragments),
    )
    if joined != text or iou(extent, box) < 0.65:
        return None
    if all(part[5] == "detached_text_candidates" for part in fragments):
        return "detached"
    return "present"


def _title_support(frames: list[dict], index: int, row: dict) -> list[int]:
    frame = frames[index]
    box = row["bbox_px"]
    _, top, _, bottom = box
    text = "".join(row["text"].split())
    if (
        len(text) < 4
        or (top + bottom) / 2 > frame["size_px"][1] * 0.65
        or bottom - top < frame["size_px"][1] * 0.035
    ):
        return []
    # Preserve the first line of a connected caption block in the target frame.
    if any(
        other is not row and 0 <= other["bbox_px"][1] - bottom <= 1.5 * (bottom - top)
        for other in frame["subtitles"]
    ):
        return []
    support: dict[str, int] = {}
    for direction in (-1, 1):
        stamp = frame["timestamp_ms"]
        j = index + direction
        while 0 <= j < len(frames):
            near = frames[j]
            if (
                not within_window(near["timestamp_ms"], frame["timestamp_ms"], CONTEXT_MS)
                or not within_window(near["timestamp_ms"], stamp, 400)
                or near["sequence_id"] != frame["sequence_id"]
                or near["size_px"] != frame["size_px"]
            ):
                break
            role = _same_line_role(near, text, box)
            if role is None:
                break
            if role == "detached" and near["pixel_sha256"] != frame["pixel_sha256"]:
                support.setdefault(near["pixel_sha256"], near["frame_index"])
                if len(support) == 2:
                    return sorted(support.values())
            stamp = near["timestamp_ms"]
            j += direction
    return []


def review_caption_sequence(frames: list[dict], requested: int | None = None) -> list[dict]:
    """Use original per-frame decisions as witnesses; never propagate a revised decision."""
    output = []
    for i, frame in enumerate(frames):
        if requested is not None and frame["frame_index"] != requested:
            continue
        captions, moved = [], []
        for line in frame["subtitles"]:
            support = _title_support(frames, i, line)
            if support:
                moved.append({**line, "temporal_review": {"support_frame_indices": support}})
            else:
                captions.append(line)
        if moved:
            frame = {
                **frame,
                "subtitles": captions,
                "detached_text_candidates": [*frame.get("detached_text_candidates", []), *moved],
            }
        output.append(frame)
    return output
