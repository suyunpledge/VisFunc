"""Read subtitle candidates from a bounded band and map observations to decoded pixels."""

from __future__ import annotations

import re
from math import ceil, floor, isfinite
from typing import Any

from svsg.recognition.geometry import Box, iou


def _player_readouts(lines: list[dict], height: int) -> set[int]:
    clocks = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*/\s*\d{1,2}:\d{2}(?::\d{2})?)?"
    scales = []
    detached: set[int] = set()
    for i, line in enumerate(lines):
        x, y, right, bottom = line["bbox_px"]
        center_y = (y + bottom) / 2
        if center_y >= 0.82 * height and re.fullmatch(clocks, line["text"]):
            detached.add(i)
        if re.fullmatch(r"[xX×]\s*\d+(?:[.,]\d+)?", line["text"]):
            scales.append((i, (x + right) / 2, center_y, bottom - y))
    # Playback test cards can have several simultaneous speed readouts.
    # A lone cue such as "x2" or a line of dialogue mentioning a time stays eligible.
    for i, x, y, h in scales:
        if not any(abs(y - yy) < 0.45 * h and abs(x - xx) > 2 * h for _, xx, yy, _ in scales):
            continue
        detached.add(i)
        for j, line in enumerate(lines):
            a, b, c, d = line["bbox_px"]
            if (
                re.fullmatch(r"\d+", line["text"])
                and abs((a + c) / 2 - x) < h
                and 0 < y - (b + d) / 2 <= 1.8 * max(h, d - b)
            ):
                detached.add(j)
    return detached


def split_caption_band(
    lines: list[dict], height: int, *, player_toolbar: bool = False
) -> tuple[list[dict], list[dict]]:
    """Keep connected lower captions and separate high, display-sized title lettering."""
    readouts = _player_readouts(lines, height) if player_toolbar else set()
    detached = [line for i, line in enumerate(lines) if i in readouts]
    lines = [line for i, line in enumerate(lines) if i not in readouts]
    lower = []
    for line in lines:
        _, top, _, bottom = line["bbox_px"]
        if 0.70 * height <= (top + bottom) / 2 <= 0.95 * height:
            lower.append(line)
    if not lower:
        captions = []
        for line in lines:
            left, top, right, bottom = line["bbox_px"]
            letters = "".join(line["text"].split())
            size = bottom - top
            display_caps = (
                len(letters) >= 4
                and letters.isupper()
                and (right - left) >= 0.63 * size * len(letters)
                and size >= 0.05 * height
            )
            high = (top + bottom) / 2 <= 0.63 * height
            if high and (display_caps or size >= 0.085 * height):
                detached.append(line)
            else:
                captions.append(line)
        return captions, detached
    _, band_top, _, band_bottom = min(lower, key=lambda row: row["bbox_px"][1])["bbox_px"]
    adjacent_height = band_bottom - band_top
    # Walk the connected block upward so the first line of a tall bilingual or
    # multi-line caption cannot be discarded just because its centre is higher.
    for line in sorted(lines, key=lambda row: -row["bbox_px"][3]):
        _, top, _, bottom = line["bbox_px"]
        if top < band_top and band_top - bottom <= 1.5 * max(bottom - top, adjacent_height):
            band_top, adjacent_height = top, bottom - top
    captions = []
    for line in lines:
        _, top, _, bottom = line["bbox_px"]
        if (top + bottom) / 2 <= 0.65 * height and bottom < band_top:
            detached.append(line)
        else:
            captions.append(line)
    return captions, detached


def _edge_crop(bgr, box: Box, region: Box):
    """Look for a run of similar ink continuing past a detector's line boundary."""
    import cv2
    import numpy as np

    left, top, right, bottom = box
    rx, ry, rr, rb = region
    height = bottom - top
    if height < 6 or right - left < height:
        return None
    x = max(rx, left - 3 * height)
    end = min(rr, right + 3 * height)
    strip = bgr[top:bottom, x:end]
    core = strip[:, left - x : right - x]
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    # Estimate stroke colour from pixels that differ from their local background.
    contrast = cv2.absdiff(gray, cv2.medianBlur(gray, 5))
    strokes = core[contrast >= 35]
    if len(strokes) < 8:
        return None
    bins = (strokes // 32).astype(np.int32)
    keys = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    colour = np.median(strokes[keys == np.bincount(keys).argmax()], axis=0)
    mask = np.max(np.abs(strip.astype(np.int16) - colour), axis=2) <= 38
    density = mask[:, left - x : right - x].mean()
    if not 0.04 <= density <= 0.52:
        return None
    columns = np.flatnonzero(mask.any(axis=0))
    runs = np.split(columns, np.flatnonzero(np.diff(columns) > max(3, round(height * 0.4))) + 1)
    first, last = left, right
    for run in runs:
        if len(run) and x + run[-1] >= left and x + run[0] < right:
            first = min(first, x + int(run[0]))
            last = max(last, x + int(run[-1]) + 1)
    minimum = max(3, round(height * 0.4))
    if max(left - first, last - right) < minimum:
        return None
    # Allow antialiasing at the outside edge, keeping the OCR region as a hard bound.
    first, last = max(rx, first - 1), min(rr, last + 1)
    y0, y1 = max(ry, top - 2), min(rb, bottom + 2)
    patch = bgr[y0:y1, first:last]
    ink = np.max(np.abs(patch.astype(np.int16) - colour), axis=2) <= 38
    return (first, top, last, bottom), patch, (ink.astype(np.uint8) * 255)


def _review_edges(engine, bgr, region: Box, line: dict, others: list[dict]) -> dict:
    expanded = _edge_crop(bgr, line["bbox_px"], region)
    if expanded is None:
        return line
    box, patch, ink = expanded
    if any(other is not line and iou(box, other["bbox_px"]) > 0 for other in others):
        return line
    original = line["text"]
    raw, _ = engine(patch, use_det=False, use_cls=False)
    if not raw or len(raw) != 1:
        return line
    text, score = raw[0]
    if not isfinite(score) or score < 0.85 or text == original or original not in text:
        return line
    # New characters must be on the same side as the measured extension.
    left, _, right, _ = line["bbox_px"]
    before, _, after = text.partition(original)
    if (before and left - box[0] < 3) or (after and box[2] - right < 3):
        return line
    isolated, _ = engine(ink, use_det=False, use_cls=False)
    if not isolated or len(isolated) != 1 or isolated[0][0] != text:
        return line
    ink_score = isolated[0][1]
    if not isfinite(ink_score) or ink_score < 0.85:
        return line
    return dict(
        text=text,
        bbox_px=box,
        score=float(min(score, ink_score)),
        edge_review={"original_text": original, "original_bbox_px": line["bbox_px"]},
    )


def _review_typography(engine, bgr, line: dict) -> dict:
    """Restore only omitted spaces or a terminal mark when two line reads agree."""
    import cv2

    original = line["text"]
    left, top, right, bottom = line["bbox_px"]
    crop = bgr[top:bottom, left:right]
    enlarged = cv2.resize(crop, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    first, _ = engine(enlarged, use_det=False, use_cls=False)
    if not first or len(first) != 1:
        return line
    text, score = first[0]
    if text == original or not isfinite(score) or score < 0.90:
        return line
    body = text
    if original[-1].isalnum() and text.endswith((".", "!", "?", "。", "！", "？")):
        body = text[:-1]
    # Existing letters, punctuation and spaces must survive in their original order.
    if (
        body != body.strip()
        or "  " in body
        or not re.fullmatch(" ?".join(re.escape(char) for char in original), body)
    ):
        return line
    check = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    second, _ = engine(check, use_det=False, use_cls=False)
    if not second or len(second) != 1 or second[0][0] != text:
        return line
    confidence = second[0][1]
    if not isfinite(confidence) or confidence < 0.90:
        return line
    return {
        **line,
        "text": text,
        "score": min(line["score"], score, confidence),
        "typography_review": {"original_text": original},
    }


def _word_cuts(gray, bright: bool) -> list[int]:
    """Propose wide blank columns between glyphs, stable at two ink thresholds."""
    import cv2
    import numpy as np

    ink = gray if bright else 255 - gray
    h, w = ink.shape
    # The outside of an OCR box can include a bright video border around a
    # dark subtitle plate. Estimate contrast from the interior of the line.
    dy, dx = min(h // 8, 3), min(w // 10, 4)
    low, high = np.percentile(ink[dy : h - dy, dx : w - dx], [25, 93])
    if high - low < 65:
        return []
    cuts: list[int] = []
    for fraction in (0.50, 0.68):
        mask = np.asarray(ink >= low + fraction * (high - low), dtype=np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        columns = np.zeros(w, dtype=bool)
        heights = []
        for x, y, width, height, area in stats[1:]:
            if (
                height < max(3, h * 0.28)
                or width > height * 1.5
                or area < height
                or y == 0
                or y + height == h
            ):
                continue
            columns[x : x + width] = True
            heights.append(height)
        if len(heights) < 3:
            return []
        xs = np.flatnonzero(columns)
        gaps = np.diff(xs) - 1
        typical = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 0
        minimum = max(3, 0.30 * np.median(heights), typical + 1.5)
        centers = [int(xs[i] + 1 + gaps[i] // 2) for i in np.flatnonzero(gaps >= minimum)]
        if fraction == 0.50:
            cuts = centers
        else:
            cuts = [x for x in cuts if any(abs(x - other) <= 1 for other in centers)]
    return cuts


def _review_code_spaces(engine, bgr, line: dict) -> dict:
    """Insert a space at a measured gap only when both substrings decode independently."""
    import cv2

    original = line["text"]
    boundaries = {
        match.start() + 1 for match in re.finditer(r"(?=([A-Za-z][0-9]|[0-9][A-Za-z]))", original)
    }
    if not boundaries:
        return line
    boundaries.update(match.start() + 1 for match in re.finditer(r"[a-z][A-Z]", original))
    left, top, right, bottom = line["bbox_px"]
    crop = bgr[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cuts = sorted(set(_word_cuts(gray, True)) | set(_word_cuts(gray, False)))
    spaces: dict[int, int] = {}
    confidence = line["score"]
    for cut in cuts:
        a, _ = engine(crop[:, :cut], use_det=False, use_cls=False)
        if not a or len(a) != 1:
            continue
        prefix, sa = a[0]
        pos = len(prefix)
        if (
            pos not in boundaries
            or not original.startswith(prefix)
            or not isfinite(sa)
            or sa < 0.90
        ):
            continue
        b, _ = engine(crop[:, cut:], use_det=False, use_cls=False)
        if not b or len(b) != 1:
            continue
        suffix, sb = b[0]
        if suffix != original[pos:] or not isfinite(sb):
            continue
        if sb < 0.90 and len(suffix) == 1 and min(crop.shape[0], right - left - cut) >= 9:
            # A digit by itself is sensitive to the edge of its subtitle plate.
            # Keep the word boundary and ask for the same digit inside that edge.
            again, _ = engine(crop[2:-2, cut:-2], use_det=False, use_cls=False)
            if again and len(again) == 1 and again[0][0] == suffix:
                sb = again[0][1]
        if isfinite(sb) and sb >= 0.90:
            spaces[pos] = left + cut
            confidence = min(confidence, float(sa), float(sb))
    if not spaces:
        return line
    return {
        **line,
        "text": "".join((" " if i in spaces else "") + c for i, c in enumerate(original)),
        "score": confidence,
        "spacing_review": {"original_text": original, "gap_x_px": list(spaces.values())},
    }


class SubtitleReader:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR(
            intra_op_num_threads=3,
            inter_op_num_threads=1,
            det_limit_type="max",
            det_limit_side_len=1920,
        )

    def read(self, bgr: Any, region: Box, *, centered: bool = True) -> list[dict]:
        import cv2

        x, y, right, bottom = region
        crop = bgr[y:bottom, x:right]
        height, width = crop.shape[:2]
        lines: list[dict] = []
        # Read at two resolutions; keep raw strings, including punctuation and case.
        for factor in (min(1.0, 960 / width), min(2.0, 1920 / width)):
            cw, ch = max(1, round(width * factor)), max(1, round(height * factor))
            view = cv2.resize(
                crop, (cw, ch), interpolation=(cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
            )
            rows, _ = self.engine(view, use_cls=False)
            for polygon, text, score in rows or []:
                if not isfinite(score) or score < 0.75 or not any(c.isalnum() for c in text):
                    continue
                xs, ys = zip(*polygon, strict=True)
                box = (
                    max(x, x + floor(min(xs) * width / cw)),
                    max(y, y + floor(min(ys) * height / ch)),
                    min(right, x + ceil((max(xs) + 1) * width / cw)),
                    min(bottom, y + ceil((max(ys) + 1) * height / ch)),
                )
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                if centered and abs((box[0] + box[2] - x - right) / 2) > 0.28 * width:
                    continue
                lines.append(dict(text=text, bbox_px=box, score=float(score)))
        kept: list[dict] = []
        for line in sorted(lines, key=lambda row: -row["score"]):
            if all(iou(line["bbox_px"], other["bbox_px"]) < 0.45 for other in kept):
                kept.append(line)
        reviewed = [
            _review_code_spaces(
                self.engine,
                bgr,
                _review_typography(
                    self.engine, bgr, _review_edges(self.engine, bgr, region, line, kept)
                ),
            )
            for line in kept
        ]
        return sorted(reviewed, key=lambda row: (row["bbox_px"][1], row["bbox_px"][0]))
