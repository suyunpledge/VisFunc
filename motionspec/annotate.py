"""标注图：每个片段一张，关键帧并排 + 轨迹叠加 + 实测/拟合曲线对比。

文字只用 ASCII（OpenCV 自带字体不支持中文）。
"""

from __future__ import annotations

import cv2
import numpy as np

from .fit import evaluate

FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (235, 235, 235)
ACCENT = (80, 200, 255)
FIT = (120, 220, 120)
PANEL = (28, 28, 32)
TILE_H = 240
CAPTION_H = 24


def _text(img, text, org, scale=0.42, color=INK):
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def _tile(crop: np.ndarray, pos, upto: int, start: int, caption: str) -> np.ndarray:
    h, w = crop.shape[:2]
    zoom = min(TILE_H / h, 460 / w)
    size = (max(1, round(w * zoom)), max(1, round(h * zoom)))
    interp = cv2.INTER_NEAREST if zoom > 2 else cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR
    img = cv2.resize(crop, size, interpolation=interp)
    if pos is not None:
        pts = (pos[start : upto + 1] * zoom).round().astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, ACCENT, 2, cv2.LINE_AA)
        cv2.circle(img, (int(pts[0][0]), int(pts[0][1])), 4, ACCENT, 1, cv2.LINE_AA)
        cv2.circle(img, (int(pts[-1][0]), int(pts[-1][1])), 4, ACCENT, -1, cv2.LINE_AA)
    tile = np.full((TILE_H + CAPTION_H, max(size[0], 150), 3), PANEL, np.uint8)
    y0 = (TILE_H - size[1]) // 2
    tile[y0 : y0 + size[1], : size[0]] = img
    _text(tile, caption, (4, TILE_H + 16))
    return tile


def _plot(t_ms: np.ndarray, progress: np.ndarray, easing: dict | None) -> np.ndarray:
    w, h = 300, TILE_H + CAPTION_H
    img = np.full((h, w, 3), PANEL, np.uint8)
    left, right, top, bottom = 34, w - 10, 16, TILE_H - 14
    lo, hi = min(0.0, float(progress.min())) - 0.05, max(1.0, float(progress.max())) + 0.05
    span = max(float(t_ms[-1]), 1e-6)

    def xy(tm, p):
        return (int(left + (right - left) * tm / span), int(bottom - (bottom - top) * (p - lo) / (hi - lo)))

    for level in (0.0, 1.0):
        cv2.line(img, xy(0, level), xy(span, level), (70, 70, 78), 1)
        _text(img, f"{level:.0f}", (8, xy(0, level)[1] + 4), 0.38, (150, 150, 160))
    if easing is not None:
        dense = np.linspace(0, span, 120)
        curve = evaluate(easing, dense)
        pts = np.array([xy(a, b) for a, b in zip(dense, curve)], np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, FIT, 2, cv2.LINE_AA)
    for a, b in zip(t_ms, progress):
        cv2.circle(img, xy(a, b), 3, ACCENT, -1, cv2.LINE_AA)
    _text(img, "dots=measured  line=fit", (left, h - 30), 0.36, (150, 150, 160))
    _text(img, f"0 .. {span:.0f} ms", (left, h - 12), 0.38)
    return img


def sheet(crops: list[np.ndarray], times: list[float], result: dict, title: str) -> np.ndarray:
    view = result["_view"]
    start, end = view["start"], view["end"]
    t = np.asarray(times, dtype=float)
    progress = view.get("progress")
    tiles = []
    for k, role in view["keys"]:
        caption = f"{role} +{t[k] - t[start]:.0f}ms"
        if progress is not None:
            caption += f" p={progress[k]:.2f}"
        tiles.append(_tile(crops[k], view.get("pos"), k, start, caption))
    a, b = view.get("progress_range", (start, end))
    if progress is not None and b - a >= 2:
        seg = np.asarray(progress[a : b + 1], dtype=float)
        tiles.append(_plot(t[a : b + 1] - t[a], seg, result.get("easing")))
    gap = 6
    width = sum(x.shape[1] for x in tiles) + gap * (len(tiles) + 1)
    width = max(width, 520)
    canvas = np.full((30 + TILE_H + CAPTION_H + gap, width, 3), (16, 16, 18), np.uint8)
    _text(canvas, title[: max(40, width // 8)], (gap, 20), 0.5)
    x = gap
    for tile in tiles:
        canvas[30 : 30 + tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1] + gap
    return canvas


def overview(frame: np.ndarray, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> np.ndarray:
    """第一帧上画出各片段所在区域（分析尺度坐标）。"""
    img = frame.copy()
    for label, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), ACCENT, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, max(0, y1 - 16)), (x1 + 10 * len(label) + 6, max(16, y1)), ACCENT, -1)
        _text(img, label, (x1 + 3, max(12, y1 - 4)), 0.45, (20, 20, 20))
    return img
