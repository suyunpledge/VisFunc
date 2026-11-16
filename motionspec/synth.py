"""合成动效片段：已知真值的测试素材，也用作演示。

每个片段返回 (帧列表, 真值字典)。精灵用仿射变换做亚像素合成，和真实渲染一样有抗锯齿。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .fit import bezier_ease, spring_response

W, H = 640, 400
FONT = cv2.FONT_HERSHEY_SIMPLEX


def background() -> np.ndarray:
    ys = np.linspace(26, 44, H, dtype=np.float32)[:, None]
    bg = np.repeat(np.repeat(ys, W, axis=1)[:, :, None], 3, axis=2)
    bg[..., 0] += 10
    bg = bg.astype(np.uint8)
    cv2.rectangle(bg, (0, 0), (W, 36), (52, 46, 44), -1)
    cv2.putText(bg, "Workspace  /  Demo", (14, 24), FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    for i in range(4):
        cv2.putText(bg, f"static line {i + 1}", (20, 360 - i * 18), FONT, 0.4, (110, 110, 118), 1, cv2.LINE_AA)
    return bg


def card(w: int = 200, h: int = 110, hue=(210, 140, 60)) -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.float32)
    cv2.rectangle(mask, (4, 4), (w - 5, h - 5), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    img[:] = hue
    cv2.circle(img, (30, 32), 14, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.putText(img, "Quick setup", (54, 38), FONT, 0.55, (250, 250, 250), 1, cv2.LINE_AA)
    cv2.putText(img, "Connect a model", (16, 70), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(img, (16, 82), (96, 98), (250, 250, 250), 1, cv2.LINE_AA)
    return img, mask


def compose(bg: np.ndarray, sprite, cx: float, cy: float, scale: float = 1.0, alpha: float = 1.0, rot_deg: float = 0.0):
    img, mask = sprite
    h, w = mask.shape
    c, s = math.cos(math.radians(rot_deg)) * scale, math.sin(math.radians(rot_deg)) * scale
    m = np.array([[c, -s, cx - c * w / 2 + s * h / 2], [s, c, cy - s * w / 2 - c * h / 2]], np.float32)
    size = (bg.shape[1], bg.shape[0])
    layer = cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR)
    cover = cv2.warpAffine(mask, m, size, flags=cv2.INTER_LINEAR)[..., None] * alpha
    return (bg * (1 - cover) + layer * cover).round().astype(np.uint8)


def _timeline(fps: float, duration_ms: float, lead: int, tail: int) -> tuple[np.ndarray, int]:
    moving = int(round(duration_ms * fps / 1000.0))
    tau = np.concatenate([np.zeros(lead), np.arange(moving + 1) / moving, np.ones(tail)])
    return tau, moving


def slide(fps: float = 60.0, duration_ms: float = 400.0, distance: float = 300.0, curve=(0.33, 1.0, 0.68, 1.0), lead=10, tail=12):
    bg, sprite = background(), card()
    tau, _ = _timeline(fps, duration_ms, lead, tail)
    p = bezier_ease(*curve, tau)
    frames = [compose(bg, sprite, 150 + distance * v, 190) for v in p]
    truth = {"kind": "transform", "duration_ms": duration_ms, "delta": [distance, 0.0], "curve": list(curve), "start_ms": lead * 1000 / fps}
    return frames, truth


def fade(fps: float = 60.0, duration_ms: float = 250.0, lead=10, tail=12):
    bg, sprite = background(), card()
    tau, _ = _timeline(fps, duration_ms, lead, tail)
    frames = [compose(bg, sprite, 320, 190, alpha=float(v)) for v in tau]
    return frames, {"kind": "fade", "duration_ms": duration_ms, "curve": [0.0, 0.0, 1.0, 1.0]}


def pop(fps: float = 60.0, zeta: float = 0.35, omega: float = 28.0, start=0.6, lead=10, total_ms=700.0):
    bg, sprite = background(), card()
    steps = int(round(total_ms * fps / 1000.0))
    t = np.concatenate([np.zeros(lead), np.arange(steps + 1) / fps])
    p = spring_response(zeta, omega, t)
    frames = [compose(bg, sprite, 320, 190, scale=start + (1 - start) * float(v)) for v in p]
    return frames, {"kind": "transform", "zeta": zeta, "omega": omega, "scale_to": 1 / start}


def stagger(fps: float = 60.0, lead=8, gap_ms: float = 500.0):
    """两个元素先后各动一次：上方卡片右移，随后下方卡片淡入。"""
    bg = background()
    a, b = card(), card(180, 90, (90, 170, 80))
    tau_a, moving_a = _timeline(fps, 300.0, lead, 0)
    wait = int(round(gap_ms * fps / 1000.0))
    tau_b, _ = _timeline(fps, 200.0, 0, 12)
    frames = []
    for v in bezier_ease(0.42, 0.0, 0.58, 1.0, tau_a):
        frames.append(compose(bg, a, 140 + 200 * v, 110))
    held = compose(bg, a, 340, 110)
    frames += [held.copy() for _ in range(wait)]
    for v in tau_b:
        frames.append(compose(held, b, 200, 270, alpha=float(v)))
    return frames, {"segments": 2}


def enter(fps: float = 60.0, duration_ms: float = 300.0, rise: float = 24.0, curve=(0.0, 0.0, 0.2, 1.0),
          opacity_curve=None, lead=10, tail=12):
    """进入：上移 rise 像素同时淡入。opacity_curve 不给时和位移共用一条曲线。"""
    bg, sprite = background(), card()
    tau, _ = _timeline(fps, duration_ms, lead, tail)
    p = bezier_ease(*curve, tau)
    a = p if opacity_curve is None else bezier_ease(*opacity_curve, tau)
    frames = [compose(bg, sprite, 320, 190 + rise * (1 - v), alpha=float(o)) for v, o in zip(p, a)]
    return frames, {"delta": [0.0, -rise], "duration_ms": duration_ms, "start_ms": lead * 1000 / fps}


def leave(fps: float = 60.0, duration_ms: float = 250.0, drop: float = 24.0, curve=(0.42, 0.0, 1.0, 1.0), lead=10, tail=12):
    """退出：下移 drop 像素同时淡出。"""
    bg, sprite = background(), card()
    tau, _ = _timeline(fps, duration_ms, lead, tail)
    p = bezier_ease(*curve, tau)
    frames = [compose(bg, sprite, 320, 190 + drop * v, alpha=float(1 - v)) for v in p]
    return frames, {"delta": [0.0, drop], "duration_ms": duration_ms}


def popover(fps: float = 60.0, duration_ms: float = 300.0, start=0.9, curve=(0.0, 0.0, 0.2, 1.0), lead=10, tail=12):
    """弹出层：从 start 倍放大到原大小，同时淡入。"""
    bg, sprite = background(), card()
    tau, _ = _timeline(fps, duration_ms, lead, tail)
    p = bezier_ease(*curve, tau)
    frames = [compose(bg, sprite, 320, 190, scale=start + (1 - start) * v, alpha=float(v)) for v in p]
    return frames, {"scale_from": start, "duration_ms": duration_ms}


def listing(fps: float = 60.0, items: int = 3, gap_ms: float = 66.667, duration_ms: float = 300.0, rise: float = 20.0,
            curve=(0.0, 0.0, 0.2, 1.0), lead=10, tail=12):
    """列表项依次进入：外观相同的几行，各自上移并淡入，起点相隔 gap_ms。"""
    bg, row = background(), card(180, 60)
    total = int(round((duration_ms + gap_ms * (items - 1)) * fps / 1000.0))
    frames = []
    for i in range(lead + total + tail + 1):
        t_ms = (i - lead) * 1000.0 / fps
        frame = bg
        for j in range(items):
            v = float(bezier_ease(*curve, np.clip((t_ms - gap_ms * j) / duration_ms, 0, 1)))
            frame = compose(frame, row, 320, 110 + 70 * j + rise * (1 - v), alpha=v)
        frames.append(frame)
    starts = [lead * 1000 / fps + gap_ms * j for j in range(items)]
    return frames, {"starts_ms": starts, "delta": [0.0, -rise], "duration_ms": duration_ms}


CASES = {"slide": slide, "fade": fade, "pop": pop, "stagger": stagger,
         "enter": enter, "leave": leave, "popover": popover, "listing": listing}


def write_video(path, frames, fps: float = 60.0) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise OSError(f"cannot open video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
