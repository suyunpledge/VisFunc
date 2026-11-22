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


def load_icons(directory: Path | None = None) -> list[Icon]:
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
        scales = sizes or sorted({*range(8, 21), 24, 28, 32, 40, 48, 64, max(w, h)})
        for size in scales:
            tw, th = max(2, round(size * w / max(w, h))), max(2, round(size * h / max(w, h)))
            pad = max(2, round(size / 6)) if icon.silhouette else 0
            if tw + 2 * pad > x1 - x0 or th + 2 * pad > y1 - y0:
                continue
            template = cv2.resize(icon.pixels, (tw, th), interpolation=cv2.INTER_AREA)
            if icon.silhouette:
                glyph = np.pad(template, pad)
                similarity = np.abs(cv2.matchTemplate(gray, glyph, cv2.TM_CCOEFF_NORMED))
                threshold = 0.86
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
            ys, xs = np.where(
                (similarity >= (0.73 if soften else threshold)) & (similarity == maxima)
            )
            soft_glyph = None
            for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
                score = float(similarity[y, x])
                weak = soften and score < threshold
                if icon.silhouette:
                    patch = gray[y : y + th + 2 * pad, x : x + tw + 2 * pad]
                    if patch.max().item() - patch.min().item() < 32:
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
                        if agreement < 0.93:
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
                if weak:
                    row["smoothed_shape_score"] = agreement
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
        support = _row_support(row["bbox_px"], anchors)
        if len(support) == 2:
            kept.append({**row, "row_support_px": support})
    return kept
