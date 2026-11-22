"""Locate known UI glyphs by shape or supplied PNG exemplars at multiple pixel sizes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from svsg.recognition.geometry import Box, iou


@dataclass
class Icon:
    label: str
    pixels: Any
    # Transparent monochrome glyphs use the alpha silhouette in either polarity.
    silhouette: bool
    template_sha256: str
    foreground_bgr: Any = None
    template_id: str | None = None
    # The speaker-cross variant shares its left side with a loudspeaker.
    detail_from_x: float | None = None
    check_speaker_margin: bool = False
    outline_speaker: bool = False


def _icon(label, image):
    import numpy as np

    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha >= 128)
    if not xs.size:
        raise ValueError(f"empty icon template: {label}")
    opaque = rgba[ys, xs, :3].astype(float)
    glyph = bool(alpha.min() < 128 and opaque.std(axis=0).max() < 12)
    y, x = slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)
    if glyph:
        pixels = alpha[y, x]
    else:
        pixels = np.asarray(image.convert("RGBA").convert("RGBa"))[y, x].copy()
        if opaque.std(axis=0).max() < 4:
            raise ValueError(f"icon template needs a distinctive shape or pattern: {label}")
    foreground = opaque.mean(axis=0)[::-1]
    digest = sha256(f"RGBA8:{image.width}x{image.height}:".encode("ascii"))
    digest.update(rgba.tobytes())
    return Icon(
        label,
        pixels,
        glyph,
        digest.hexdigest(),
        foreground if glyph and np.ptp(foreground) > 35 else None,
    )


def _transport_variants() -> list[Icon]:
    from PIL import Image, ImageDraw

    variants = []
    for width, height, arm, stroke in ((60, 44, 15, 5), (52, 52, 15, 5)):
        image = Image.new("RGBA", (64, 64))
        draw = ImageDraw.Draw(image)
        for x, dx in ((2, 1), (width + 1, -1)):
            for y, dy in ((2, 1), (height + 1, -1)):
                for a, b in ((arm, stroke), (stroke, arm)):
                    end_x, end_y = x + dx * (a - 1), y + dy * (b - 1)
                    draw.rectangle(
                        (min(x, end_x), min(y, end_y), max(x, end_x), max(y, end_y)), fill="white"
                    )
        icon = _icon("fullscreen", image)
        icon.template_id = f"fullscreen/thin-corners-{width}x{height}"
        variants.append(icon)

    image = Image.new("RGBA", (64, 64))
    draw = ImageDraw.Draw(image)
    for x, y, d in ((4, 59, 1), (59, 4, -1)):
        draw.line([(x, y - 24 * d), (x, y), (x + 24 * d, y)], fill="white", width=6)
        draw.line([(x, y), (x + 23 * d, y - 23 * d)], fill="white", width=8)
    icon = _icon("fullscreen", image)
    icon.template_id = "fullscreen/open-arrows"
    variants.append(icon)

    image = Image.new("RGBA", (64, 64))
    # Some toolbars draw an unfilled speaker with a closed back and no waves.
    ImageDraw.Draw(image).polygon(
        [(4, 19), (25, 19), (44, 4), (44, 58), (25, 43), (4, 43)],
        outline="white",
        width=4,
    )
    icon = _icon("volume", image)
    icon.template_id = "volume/outline-speaker"
    icon.check_speaker_margin = icon.outline_speaker = True
    variants.append(icon)

    for style in ("speaker-only", "adjacent-cross", "diagonal-strike", "outline-cross"):
        image = Image.new("RGBA", (64, 64))
        draw = ImageDraw.Draw(image)
        if style in ("speaker-only", "adjacent-cross"):
            draw.polygon([(4, 22), (16, 22), (35, 5), (35, 58), (16, 41), (4, 41)], fill="white")
        if style == "adjacent-cross":
            for a, b in ((21, 42), (42, 21)):
                draw.line([(38, a), (59, b)], fill="white", width=7)
        elif style == "diagonal-strike":
            # A strike normally extends beyond the speaker and breaks the
            # upper half of the waves. Reserve a clear edge on that side.
            draw.polygon([(6, 21), (17, 21), (33, 5), (33, 53), (17, 39), (6, 39)], fill="white")
            draw.arc((4, 0, 62, 56), -69, 68, fill="white", width=7)
            draw.arc((22, 13, 48, 43), -58, 58, fill="white", width=6)
            draw.line([(7, 0), (63, 56)], fill=(0, 0, 0, 0), width=9)
            draw.line([(2, 3), (60, 61)], fill="white", width=7)
        elif style == "outline-cross":
            draw.line(
                [(43, 15), (43, 5), (22, 22), (4, 22), (4, 43), (22, 43), (43, 59), (43, 49)],
                fill="white",
                width=4,
            )
            for a, b in ((20, 45), (45, 20)):
                draw.line([(36, a), (60, b)], fill="white", width=5)
        icon = _icon("mute", image)
        icon.template_id = f"mute/{style}"
        if style == "adjacent-cross":
            icon.detail_from_x = 0.61
        if style == "outline-cross":
            icon.detail_from_x = 0.59
        icon.check_speaker_margin = style == "speaker-only"
        variants.append(icon)
    return variants


def load_icons(directory: Path | None = None, *, player_toolbar: bool = False) -> list[Icon]:
    from PIL import Image, ImageDraw

    if directory is not None:
        icons = []
        # Keep flat label.png banks compatible. label/style.png supplies another
        # rendering of the same control, without making the style part of its label.
        for path in sorted([*directory.glob("*.png"), *directory.glob("*/*.png")]):
            nested = path.parent != directory
            label = path.parent.name if nested else path.stem
            with Image.open(path) as exemplar:
                icon = _icon(label, exemplar)
            if nested:
                icon.template_id = f"{label}/{path.stem}"
            icons.append(icon)
        if not icons:
            raise ValueError("icon directory must contain PNG templates")
        return icons

    icons = []
    for label in ("play", "pause", "fullscreen", "mute", "volume", "captions", "search", "close"):
        image = Image.new("RGBA", (64, 64))
        draw = ImageDraw.Draw(image)
        if label == "play":
            draw.polygon([(12, 4), (56, 32), (12, 60)], fill="white")
        elif label == "pause":
            for x in (12, 38):
                draw.rectangle((x, 6, x + 13, 57), fill="white")
        elif label == "fullscreen":
            for x, dx in ((6, 1), (58, -1)):
                for y, dy in ((8, 1), (56, -1)):
                    for width, height in ((21, 9), (9, 20)):
                        xe, ye = x + dx * width, y + dy * height
                        draw.rectangle(
                            (min(x, xe), min(y, ye), max(x, xe), max(y, ye)), fill="white"
                        )
        elif label in ("mute", "volume"):
            draw.polygon([(4, 25), (14, 25), (29, 10), (29, 55), (14, 40), (4, 40)], fill="white")
            draw.arc((3, 4, 59, 59), -65, 65, fill="white", width=6)
            draw.arc((20, 17, 46, 46), -53, 53, fill="white", width=5)
            if label == "mute":
                draw.line([(5, 5), (58, 58)], fill=(0, 0, 0, 0), width=13)
                draw.line([(5, 5), (58, 58)], fill="white", width=7)
        elif label == "captions":
            draw.rectangle((3, 10, 61, 54), fill="white")
            for span in ((9, 29, 23, 34), (29, 29, 55, 34), (9, 40, 40, 45), (46, 40, 55, 45)):
                draw.rectangle(span, fill=(0, 0, 0, 0))
        elif label == "search":
            draw.ellipse((4, 4, 42, 42), outline="white", width=7)
            draw.line([(37, 37), (59, 59)], fill="white", width=8)
        else:
            for left, right in ((8, 56), (56, 8)):
                draw.line([(left, 8), (right, 56)], fill="white", width=8)
        icons.append(_icon(label, image))
    if player_toolbar:
        # Common transport-control constructions. Keep these extra hypotheses
        # scoped to the toolbar, where their semantics and size are constrained.
        for label, style in (
            ("fullscreen", "diagonal-arrows"),
            ("fullscreen", "triangular-corners"),
            ("mute", "speaker-cross"),
            ("volume", "wide-speaker"),
            ("pause", "wide-bars"),
        ):
            image = Image.new("RGBA", (64, 64))
            draw = ImageDraw.Draw(image)
            if style == "diagonal-arrows":
                for a, b in ((7, 55), (55, 7)):
                    direction = 1 if a == 7 else -1
                    draw.polygon(
                        [(a, b), (a + 21 * direction, b), (a, b - 21 * direction)], fill="white"
                    )
                    draw.line(
                        [(a, b), (a + 19 * direction, b - 19 * direction)], fill="white", width=6
                    )
            elif style == "triangular-corners":
                for x, dx in ((5, 1), (58, -1)):
                    for y, dy in ((5, 1), (58, -1)):
                        draw.polygon([(x, y), (x + dx * 20, y), (x, y + dy * 20)], fill="white")
            elif style == "wide-bars":
                for x in (9, 37):
                    draw.rectangle((x, 6, x + 16, 57), fill="white")
            else:
                draw.polygon(
                    [(3, 23), (14, 23), (34, 6), (34, 57), (14, 40), (3, 40)], fill="white"
                )
                if style == "speaker-cross":
                    for a, b in ((24, 40), (40, 24)):
                        draw.line([(43, a), (58, b)], fill="white", width=5)
                else:
                    for bounds in ((12, 10, 60, 54), (28, 19, 48, 44)):
                        draw.arc(bounds, -51, 51, fill="white", width=6)
            icon = _icon(label, image)
            icon.template_id = f"{label}/{style}"
            if style == "speaker-cross":
                icon.detail_from_x = 0.62
            icons.append(icon)
        icons.extend(_transport_variants())
    return icons


def _row_support(box: Box, anchors: list[dict]) -> list[Box]:
    left, top, right, bottom = box
    height = bottom - top
    support: list[Box] = []
    for anchor in anchors:
        a, b, c, d = anchor["bbox_px"]
        if not (
            0.6 * height <= d - b <= 1.8 * height
            and abs((b + d - top - bottom) / 2) <= height * 0.35
        ):
            continue
        if all(abs((a + c - x - r) / 2) > max(c - a, r - x) for x, _, r, _ in [box, *support]):
            support.append(anchor["bbox_px"])
            if len(support) == 2:
                break
    return support


def supplement_toolbar(primary: list[dict], references: list[dict]) -> list[dict]:
    """Fill unoccupied positions with strong reference hits supported by another control."""
    strong = [row for row in references if row["score"] >= 0.93]
    anchors = [*primary, *strong]
    kept = list(primary)
    for row in strong:
        box = row["bbox_px"]
        x, y, r, b = box
        # A bare speaker exemplar can lie wholly inside a larger volume glyph.
        # Include containment as well as IoU when deciding whether a spot is occupied.
        occupied = False
        for existing in kept:
            a, t, c, d = existing["bbox_px"]
            overlap = max(0, min(c, r) - max(a, x)) * max(0, min(b, d) - max(y, t))
            if (
                overlap >= 0.5 * min((r - x) * (b - y), (c - a) * (d - t))
                or iou(box, existing["bbox_px"]) >= 0.35
            ):
                occupied = True
                break
        if occupied:
            continue
        support = _row_support(box, [v for v in anchors if v["label"] != row["label"]])
        if support:
            kept.append(
                {
                    **row,
                    "template_id": "reference/" + row.get("template_id", row["label"]),
                    "row_support_px": support,
                }
            )
    return kept


def _gradient_shape(patch, glyph, pad: int) -> float | None:
    """Remove a vertical backdrop ramp only when both glyph margins agree on it."""
    import cv2
    import numpy as np

    a = np.median(patch[:, :pad], axis=1)
    b = np.median(patch[:, -pad:], axis=1)
    background = (a + b) / 2
    if np.ptp(background) < 32 or np.quantile(np.abs(a - b), 0.8) > 15:
        return None
    signal = patch.astype(np.float32) - background.astype(np.float32)[:, None]
    corr = float(cv2.matchTemplate(signal, glyph.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0])
    if abs(corr) < 0.88:
        return None
    blurred = cv2.GaussianBlur(signal, (3, 3), 0.8)
    model = cv2.GaussianBlur(glyph.astype(np.float32), (3, 3), 0.8)
    smooth = float(cv2.matchTemplate(blurred, model, cv2.TM_CCOEFF_NORMED)[0, 0])
    return abs(corr) if corr * np.sign(smooth) > 0 and abs(smooth) >= 0.94 else None


def match_icons(
    bgr: Any,
    region: Box,
    icons: list[Icon],
    *,
    sizes: tuple[int, ...] | None = None,
    review_row: bool = False,
) -> list[dict]:
    import cv2
    import numpy as np

    x0, y0, x1, y1 = region
    crop = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    proposals: list[dict] = []
    secondary: list[dict] = []
    for icon in icons:
        h, w = icon.pixels.shape[:2]
        scales = sizes or sorted(
            {*range(8, 29 if review_row else 21), 24, 28, 32, 40, 48, 64, max(w, h)}
        )
        for size in scales:
            tw, th = max(2, round(size * w / max(w, h))), max(2, round(size * h / max(w, h)))
            pad = max(2, round(size / 6)) if icon.silhouette else 0
            if tw + 2 * pad > x1 - x0 or th + 2 * pad > y1 - y0:
                continue
            template = cv2.resize(icon.pixels, (tw, th), interpolation=cv2.INTER_AREA)
            if icon.silhouette:
                glyph = np.pad(template, pad)
                similarity = np.abs(cv2.matchTemplate(gray, glyph, cv2.TM_CCOEFF_NORMED))
                threshold = 0.93 if icon.check_speaker_margin else 0.86
            else:
                # Use only opaque pixels: transparent margins have no observed RGB.
                mask = (template[:, :, 3] >= 192).astype(np.uint8)
                if not mask.any():
                    continue
                colors = np.ascontiguousarray(
                    np.clip(
                        template[:, :, 2::-1].astype(np.float32)
                        * 255
                        / np.maximum(template[:, :, 3:4], 1),
                        0,
                        255,
                    ).astype(np.uint8)
                )
                error = cv2.matchTemplate(crop, colors, cv2.TM_SQDIFF, mask=mask)
                similarity = 1 - np.sqrt(np.maximum(error, 0) / (3 * mask.sum())) / 255
                spread = colors[mask > 0].std(axis=0)
                tolerance = min(23, 0.65 * float(np.sqrt(np.mean(spread**2))))
                if tolerance < 1:
                    continue
                threshold = 1 - tolerance / 255
            maxima = cv2.dilate(similarity, np.ones((5, 5), np.uint8))
            soften = review_row and icon.silhouette and size <= 32
            review_threshold = 0.88 if icon.check_speaker_margin else 0.73
            ys, xs = np.where(
                (similarity >= (review_threshold if soften else threshold)) & (similarity == maxima)
            )
            soft_glyph = None
            for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
                score = float(similarity[y, x])
                weak = soften and score < threshold
                gradient = None
                if icon.silhouette:
                    patch = gray[y : y + th + 2 * pad, x : x + tw + 2 * pad]
                    if patch.max().item() - patch.min().item() < 32:
                        continue
                    if icon.check_speaker_margin:
                        # Waves or a slash just outside the speaker body would
                        # turn a partial match into a false mute. Require free space.
                        # Thin outline strokes bleed a pixel past the measured edge.
                        start = x + pad + tw + int(icon.outline_speaker)
                        margin = max(3, round(th * 0.70))
                        tail = gray[y + pad : y + pad + th, start : start + margin]
                        if tail.shape[1] != margin:
                            continue
                        bg = float(np.median(patch[glyph <= 24]))
                        ink = glyph >= max(96, 0.8 * int(template.max()))
                        if not ink.any():
                            continue
                        delta = float(np.median(patch[ink])) - bg
                        if (
                            abs(delta) < 64
                            or np.count_nonzero((tail.astype(float) - bg) / delta > 0.4) > 1
                        ):
                            continue
                    if icon.detail_from_x is not None:
                        # The speaker body is shared with volume. Verify the X
                        # separately, in the same polarity as the complete glyph.
                        detail = round(tw * icon.detail_from_x)
                        tail = patch[pad : pad + th, pad + detail : pad + tw]
                        cross = template[:, detail:]
                        agreement = float(
                            cv2.matchTemplate(tail, cross, cv2.TM_CCOEFF_NORMED)[0, 0]
                        )
                        polarity = float(patch[glyph >= 192].mean()) - float(
                            patch[glyph <= 24].mean()
                        )
                        if agreement * np.sign(polarity) < 0.78:
                            continue
                    if weak or (size <= 12 and score < 0.92):
                        # A single bright pixel can give a dark texture a large range.
                        # Compare the interior ink to its surrounding background instead.
                        solid = patch[glyph >= 192]
                        if not solid.size:
                            continue
                        contrast = abs(
                            float(np.median(solid)) - float(np.median(patch[glyph <= 24]))
                        )
                        if contrast < 64:
                            continue
                    if weak:
                        if soft_glyph is None:
                            soft_glyph = cv2.GaussianBlur(glyph, (3, 3), 0.8)
                        soft_patch = cv2.GaussianBlur(patch, (3, 3), 0.8)
                        agreement = abs(
                            float(
                                cv2.matchTemplate(soft_patch, soft_glyph, cv2.TM_CCOEFF_NORMED)[
                                    0, 0
                                ]
                            )
                        )
                        if icon.label in ("volume", "fullscreen") and score >= 0.78:
                            gradient = _gradient_shape(patch, glyph, pad)
                        if agreement < 0.93 and gradient is None:
                            continue
                left, top = x0 + x + pad, y0 + y + pad
                if icon.foreground_bgr is not None:
                    pixels = bgr[top : top + th, left : left + tw][template >= 192]
                    if (
                        not len(pixels)
                        or np.max(np.abs(pixels.mean(axis=0) - icon.foreground_bgr)) > 32
                    ):
                        continue
                row: dict = dict(
                    label=icon.label, bbox_px=(left, top, left + tw, top + th), score=score
                )
                if icon.template_id:
                    row["template_id"] = icon.template_id
                if weak or icon.outline_speaker:
                    if weak:
                        row["smoothed_shape_score"] = agreement
                    if gradient is not None:
                        row["gradient_shape_score"] = gradient
                    secondary.append(row)
                else:
                    proposals.append(row)
    kept: list[dict] = []

    def rank(row: dict) -> tuple[float, int]:
        left, top, right, bottom = row["bbox_px"]
        # Flat-colored logos can match an interior subrectangle equally well.
        return round(row["score"], 5), (right - left) * (bottom - top)

    for row in sorted(proposals, key=rank, reverse=True):
        if all(iou(row["bbox_px"], other["bbox_px"]) < 0.35 for other in kept):
            kept.append(row)
    anchors = kept.copy()
    for row in sorted(secondary, key=rank, reverse=True):
        if any(iou(row["bbox_px"], other["bbox_px"]) >= 0.35 for other in kept):
            continue
        corrected = "gradient_shape_score" in row
        witnesses = [a for a in anchors if a["label"] != row["label"]] if corrected else anchors
        support = _row_support(row["bbox_px"], witnesses)
        required = 1 if corrected else 2
        if len(support) >= required:
            kept.append({**row, "row_support_px": support})
    return kept
