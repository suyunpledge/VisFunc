"""第二遍：对单个候选窗口（已裁到区域）做跟踪、混合度估计、分类和缓动拟合。

四种结果：
- instant：一帧内完成的突变
- fade：没有东西在动，画面在首尾两帧之间线性混合 → 透明度 / 颜色过渡
- transform：有特征点在动 → 平移 / 缩放 / 旋转随时间的曲线
- nonrigid：以上都解释不了（辉光、模糊、粒子）→ 只给统计量
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .fit import bezier_ease, fit_easing, fit_open
from .layer import fit_layers
from .segment import changed_mask, to_gray

LK = {"winSize": (15, 15), "maxLevel": 3,
      "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
MAX_POINTS = 300
COHERENT = 0.6  # 低于这个一致性就不当作单个刚体
LAYER_MAX_FRAMES = 180
LAYER_MAX_MS = 3000.0


def _track(grays: list[np.ndarray], threshold: int) -> dict:
    """逐步估计运动。

    新找的特征点必须确实在动才算数，避免被静止背景拉回零；
    已经确认属于运动物体的点会一直带着走，这样缓出尾段每帧零点几像素的位移也能量到。
    """
    n = len(grays)
    low = max(4, threshold // 2)
    steps: list[dict | None] = []
    active = []
    carried = np.zeros((0, 2), np.float32)
    kernel = np.ones((7, 7), np.uint8)
    for k in range(n - 1):
        g0, g1 = grays[k], grays[k + 1]
        mask = (cv2.absdiff(g0, g1) > low).astype(np.uint8)
        is_active = int(mask.sum()) >= 4
        active.append(is_active)
        if not is_active:
            steps.append(None)  # 画面没变：位置不动，带着的点原样保留
            continue
        # 每一步都重新找点：新点既补充跟踪，也用来检验"这块区域是不是一个整体在动"
        pts, known = carried, len(carried)
        fresh = cv2.goodFeaturesToTrack(g0, 150, 0.01, 3, mask=cv2.dilate(mask, kernel) * 255)
        if fresh is not None:
            pts = np.vstack([pts, fresh.reshape(-1, 2)])
        if len(pts) < 4:
            steps.append(None)
            carried = np.zeros((0, 2), np.float32)
            continue
        p0 = pts.astype(np.float32).reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK)
        p0f, p1f = p0.reshape(-1, 2), p1.reshape(-1, 2)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1)
        ok &= np.linalg.norm(p0f - back.reshape(-1, 2), axis=1) < 0.7
        is_known = np.arange(len(pts)) < known
        fresh_moving = ok & ~is_known & (np.linalg.norm(p1f - p0f, axis=1) > 0.2)
        moving = ok & (is_known | fresh_moving)
        if int(moving.sum()) < 4:
            steps.append(None)
            carried = np.zeros((0, 2), np.float32)
            continue
        a, b = p0f[moving], p1f[moving]
        median = np.median(b - a, axis=0)
        matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        keep = np.ones(len(a), bool)
        if matrix is None:
            matrix = np.array([[1, 0, median[0]], [0, 1, median[1]]], dtype=float)
        elif int(inliers.sum()) >= 4:
            keep = inliers.ravel().astype(bool)
        # 一致性只在新点上量：带着走的点本来就是按模型筛出来的，拿它们算会虚高
        agree = None
        if int(fresh_moving.sum()) >= 6:
            fa, fb = p0f[fresh_moving], p1f[fresh_moving]
            predicted = fa @ matrix[:, :2].T + matrix[:, 2]
            agree = float((np.linalg.norm(predicted - fb, axis=1) < 1.5).mean())
        steps.append({"median": median, "matrix": matrix, "from": a[keep], "to": b[keep], "agree": agree})
        carried = b[keep][:MAX_POINTS]

    valid = [s is not None for s in steps]
    # 一致性：在动的点里有多少服从同一个相似变换。刚体的界面元素接近 1，
    # 3D 场景、粒子、多个物体各动各的会明显偏低
    agree = [s["agree"] for s in steps if s is not None and s["agree"] is not None]
    out = {"valid": valid, "active": active, "ok": False, "coherence": float(np.median(agree)) if agree else None}
    first = next((s for s in steps if s is not None), None)
    if first is None:
        return out

    anchor = first["from"].mean(axis=0)
    spread = first["from"].max(axis=0) - first["from"].min(axis=0)
    half = float(max(8.0, max(spread) / 2))

    cums = [np.eye(3)]
    for s in steps:
        cums.append(np.vstack([s["matrix"], [0, 0, 1]]) @ cums[-1] if s is not None else cums[-1])
    scale = np.array([math.hypot(c[0, 0], c[1, 0]) for c in cums])
    rot = np.degrees(np.unwrap([math.atan2(c[1, 0], c[0, 0]) for c in cums]))

    # 缩放和旋转都可忽略时按纯平移处理：中位数光流比逐步拟合的相似变换更稳
    rigid = float(np.abs(scale - 1).max()) < 0.03 and float(np.abs(rot).max()) < 1.5
    pivot = None
    pivot_error = None
    if rigid:
        pos = [anchor.copy()]
        for s in steps:
            pos.append(pos[-1] + (s["median"] if s is not None else 0.0))
        scale, rot = np.ones(n), np.zeros(n)
    else:
        # A fixed point is a possible parameterization only if it explains the trajectory.
        matrices = np.asarray(cums)
        lhs = np.eye(2) - matrices[:, :2, :2]
        shift = matrices[:, :2, 2]
        h, w = grays[0].shape
        try:
            fixed, _, rank, _ = np.linalg.lstsq(lhs.reshape(-1, 2), shift.ravel(), rcond=None)
            pivot_error = float(np.linalg.norm(lhs @ fixed - shift, axis=1).max())
            if (rank == 2 and pivot_error <= 1.2
                    and -0.5 * w <= fixed[0] <= 1.5 * w and -0.5 * h <= fixed[1] <= 1.5 * h):
                pivot = fixed
        except np.linalg.LinAlgError:
            pass
        base = pivot if pivot is not None else anchor
        pos = [(c @ np.array([base[0], base[1], 1.0]))[:2] for c in cums]
    anchor_path = np.array([(c @ np.array([*anchor, 1.0]))[:2] for c in cums])
    if rigid:
        anchor_path = np.asarray(pos)
    out.update(ok=True, pos=np.asarray(pos), anchor_path=anchor_path, scale=scale, rot=rot, half=half,
               rigid=rigid, pivot=pivot, pivot_error=pivot_error)
    return out


def _blend(crops: list[np.ndarray]) -> dict:
    """把每一帧投影到"首帧→末帧"的连线上：a=0 是首帧，a=1 是末帧。

    真正的淡入淡出、颜色过渡，残差接近 0；位移、形变的中间帧不是首尾的线性混合，残差大。
    """
    first = crops[0].astype(np.float32)
    delta = crops[-1].astype(np.float32) - first
    energy = float((delta * delta).sum())
    if energy / delta.size < 0.5:
        return {"ok": False}
    alpha, residual = [], []
    for crop in crops:
        d = crop.astype(np.float32) - first
        a = float((d * delta).sum()) / energy
        r = d - a * delta
        alpha.append(a)
        residual.append(math.sqrt(float((r * r).sum()) / energy))
    return {"ok": True, "alpha": np.array(alpha), "residual": np.array(residual), "strength": math.sqrt(energy / delta.size)}


def _trim(p: np.ndarray, eps: float) -> tuple[int, int]:
    """起点：最后一个仍在 0 附近的帧。终点：第一个进入并保持在 1 附近的帧。"""
    n = len(p)
    moved = np.nonzero(np.abs(p) >= eps)[0]
    start = max(0, int(moved[0]) - 1) if len(moved) else 0
    unsettled = np.nonzero(np.abs(p - 1) >= eps)[0]
    end = min(n - 1, int(unsettled[-1]) + 1) if len(unsettled) else n - 1
    if end <= start:
        start, end = 0, n - 1
    return start, end


def _alpha_eps(p: np.ndarray) -> float:
    """透明度进度的起止门限。窗口首尾几帧是静止的，它们的抖动就是噪声水平。

    固定用 2% 会把缓入缓出的头尾截掉（in-out-quad 的第一帧只走了 0.6%）。
    无损截图噪声接近 0，门限降到 0.4%；有损视频噪声大，自动回到 2%。
    """
    noise = max(float(np.std(p[:3])), float(np.std(p[-3:])))
    return float(np.clip(0.004 + 5 * noise, 0.004, 0.02))


def _period_ms(values: np.ndarray, times: np.ndarray) -> float | None:
    if len(values) < 16:
        return None
    step = float(np.median(np.diff(times)))
    index = np.arange(len(values))
    v = values - np.polyval(np.polyfit(index, values, 1), index)
    power = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2
    if len(power) < 4 or power[1:].sum() <= 0:
        return None
    k = int(np.argmax(power[1:])) + 1
    if k < 2 or power[k] / power[1:].sum() < 0.45:
        return None
    return round(len(v) * step / k, 1)


def _r(value, digits: int = 2):
    return round(float(value), digits)


def _fit_channel(t: np.ndarray, value: np.ndarray, eps: float, returns: bool) -> tuple:
    peak = int(np.argmax(np.abs(value))) if returns else None
    if peak is not None:
        progress = value / value[peak]
        moving = np.flatnonzero(np.abs(progress) >= max(0.02, eps))
        start, end = max(0, int(moving[0]) - 1), min(len(value) - 1, int(moving[-1]) + 1)
        return start, end, peak, progress, None
    progress = value / value[-1]
    start, end = _trim(progress, eps)
    span = progress[end] - progress[start]
    if abs(span) <= 1e-9:
        return start, end, None, progress, None
    progress = (progress - progress[start]) / span
    return start, end, None, progress, fit_easing(t, progress, start, end)


def _analyze_single(
    crops: list[np.ndarray],
    times: list[float],
    *,
    origin: tuple[int, int],
    inv_scale: float,
    threshold: int = 12,
) -> dict:
    """crops：裁到区域的分析尺度 BGR 帧；origin：区域左上角（分析尺度）；inv_scale：分析→源像素。

    返回的坐标、距离都是源视频像素。"_view" 里是画标注图用的内部数据，写 JSON 前要去掉。
    """
    n = len(crops)
    t = np.asarray(times, dtype=float)
    grays = [to_gray(c) for c in crops]
    track = _track(grays, threshold)
    blend = _blend(crops)
    luma = np.array([float(g.mean()) for g in grays])
    active = track["active"]
    n_active = int(sum(active))
    notes: list[str] = []

    def to_source(p) -> list[float]:
        return [_r((p[0] + origin[0]) * inv_scale, 1), _r((p[1] + origin[1]) * inv_scale, 1)]

    result: dict = {"luma": {"from": _r(luma[0], 1), "to": _r(luma[-1], 1)}, "notes": notes}
    view: dict = {"pos": None, "progress": None, "start": 0, "end": n - 1}
    result["_view"] = view

    # ---- 分类 ----
    has_motion = False
    channel = None
    effects: dict = {}
    if track["ok"]:
        pos, scale, rot, half = track["pos"], track["scale"], track["rot"], track["half"]
        offset = np.linalg.norm(pos - pos[0], axis=1)
        effects = {
            "translate": (float(offset.max()), float(offset[-1])),
            "scale": (float(np.abs(scale - 1).max()) * half, abs(float(scale[-1]) - 1) * half),
            "rotate": (float(np.radians(np.abs(rot)).max()) * half, abs(math.radians(float(rot[-1]))) * half),
        }
        channel = max(effects, key=lambda name: effects[name][0])
        enough = sum(track["valid"]) >= max(2, 0.4 * n_active)
        coherence = track["coherence"]
        has_motion = enough and effects[channel][0] >= 2.0 and (coherence is None or coherence >= COHERENT)
        if enough and effects[channel][0] >= 2.0 and not has_motion:
            notes.append(f"points in this region move independently (coherence {track['coherence']:.2f}); it is not one rigid element")

    jump = None
    blend_mid = blend_max = None
    if blend["ok"]:
        steps = np.diff(blend["alpha"])
        if steps.max() > 0.8:
            jump = int(np.argmax(steps))
        mid = (blend["alpha"] > 0.1) & (blend["alpha"] < 0.9)
        blend_max = float(blend["residual"].max())
        blend_mid = float(np.median(blend["residual"][mid])) if mid.any() else blend_max

    if jump is not None and blend_max < 0.35:
        kind = "instant"
    elif blend["ok"] and blend_mid < 0.15 and blend_max < 0.35:
        kind = "fade"  # 先于光流判定：渐显的纹理会让光流给出假的缩放
    elif has_motion:
        kind = "transform"
    elif blend["ok"] and blend_mid < 0.25 and blend_max < 0.5:
        kind = "fade"
    else:
        kind = "nonrigid"
    result["kind"] = kind

    channels: dict = {}
    result["channels"] = channels
    progress = None
    easing = None
    channel_progress: dict[str, np.ndarray] = {}

    if kind == "instant":
        view.update(start=jump, end=jump + 1)
        result["confidence"] = "high"
        channels["change"] = {"mean_abs_delta": _r(blend["strength"], 1)}
        notes.append("change completes within one frame interval")

    elif kind == "transform":
        pos, scale, rot = track["pos"], track["scale"], track["rot"]
        view["pos"] = pos
        channel_ranges = []
        for name, (peak_effect, net_effect) in effects.items():
            returns = net_effect < 0.3 * peak_effect
            if name == "translate":
                if name != channel and peak_effect < max(2.0 / inv_scale, 1.0):
                    continue
                offset = pos - pos[0]
                target = offset[int(np.argmax(np.linalg.norm(offset, axis=1)))] if returns else offset[-1]
                value = offset @ (target / np.linalg.norm(target))
                eps = float(np.clip(0.3 / peak_effect, 0.002, 0.05))
            elif name == "scale":
                value, eps = scale - scale[0], 0.015
                if name != channel and np.abs(value).max() < 0.02:
                    continue
            else:
                value, eps = rot - rot[0], 0.015
                if name != channel and np.abs(value).max() < 1.0:
                    continue
            first, last, peak, p, curve = _fit_channel(t, value, eps, returns)
            channel_ranges.append((first, last))
            channel_progress[name] = p
            target_frame = last if peak is None else peak
            data = {"t_start_ms": _r(t[first], 3), "t_end_ms": _r(t[last], 3),
                    "duration_ms": _r(t[last] - t[first], 3), "easing": curve}
            if peak is not None:
                data.update(pattern="out-and-back", peak_ms=_r(t[peak] - t[first], 1))
            if name == "translate":
                delta = (pos[target_frame] - pos[first]) * inv_scale
                data.update({"from": to_source(pos[first]), "to": to_source(pos[target_frame]),
                             "delta": [_r(x, 1) for x in delta], "distance_px": _r(np.linalg.norm(delta), 1),
                             "reference": "tracked_feature_anchor"})
            elif name == "scale":
                data.update({"from": 1.0, "to": _r(scale[target_frame] / scale[first], 3),
                             "peak": _r(scale[first:last + 1].max() / scale[first], 3)})
            else:
                data.update(from_deg=0.0, to_deg=_r(rot[target_frame] - rot[first], 1))
            channels[name] = data
            if name == channel:
                progress, easing = p, curve
                view["progress_range"] = (first, last)
                if peak is not None:
                    view["peak"] = peak
                    result.update(pattern=data["pattern"], peak_ms=data["peak_ms"])

        view.update(start=min(a for a, _ in channel_ranges), end=max(b for _, b in channel_ranges))
        if track["pivot"] is not None:
            result["origin"] = to_source(track["pivot"])
            notes.append("a fitted fixed point explains the transform; the element center and design pivot are unknown")
        elif not track["rigid"]:
            notes.append("translation follows the tracked feature anchor; the element center and design pivot are unknown")
        result["transform_reference"] = {
            "type": "fitted_fixed_point" if track["pivot"] is not None else "tracked_feature_anchor",
            "feature_anchor": to_source(track["anchor_path"][0]),
            "feature_displacement_px": [_r(v * inv_scale, 1) for v in
                                         track["anchor_path"][view["end"]] - track["anchor_path"][view["start"]]],
            "fixed_point_max_error_px": None if track["pivot_error"] is None else _r(track["pivot_error"] * inv_scale, 2),
        }
        result["dominant"] = channel
        coverage = sum(track["valid"]) / max(1, n_active)
        evidence = track["coherence"]
        result["confidence"] = "high" if coverage >= 0.8 and evidence is not None and evidence >= 0.8 else "medium" if coverage >= 0.55 else "low"
        if not track["rigid"] and result["confidence"] == "high" and len(channels) > 1:
            result["confidence"] = "medium"
        if coverage < 0.8:
            notes.append(f"tracking held on {coverage:.0%} of the moving frames")
        if track["rigid"] and abs(luma[-1] - luma[0]) > 6:
            notes.append("region brightness also changes; an opacity change may be combined with the motion")

    elif kind == "fade":
        progress = blend["alpha"]
        start, end = _trim(progress, _alpha_eps(progress))
        view.update(start=start, end=end)
        progress = (progress - progress[start]) / max(progress[end] - progress[start], 1e-9)
        easing = fit_easing(t, progress, start, end)
        channels["blend"] = {"mean_abs_delta": _r(blend["strength"], 1), "residual": _r(blend_mid, 3),
                             "t_start_ms": _r(t[start], 3), "t_end_ms": _r(t[end], 3),
                             "duration_ms": _r(t[end] - t[start], 3), "easing": easing}
        result["confidence"] = "high" if blend_mid < 0.12 else "medium"
        notes.append("frames closely match a mix of the first and last frame (opacity or colour transition)")

    else:  # nonrigid
        result["confidence"] = "low"
        stats = {"luma_min": _r(luma.min(), 1), "luma_max": _r(luma.max(), 1)}
        period = _period_ms(luma, t)
        if period:
            stats["luma_period_ms"] = period
        if blend_mid is not None:
            stats["blend_residual"] = _r(blend_mid, 3)
        if track["coherence"] is not None:
            stats["motion_coherence"] = _r(track["coherence"], 2)
        channels["appearance"] = stats
        moving = [k for k, a in enumerate(active) if a or changed_mask(crops[k], crops[k + 1], threshold).sum() >= 4]
        if moving:
            view.update(start=max(0, moving[0]), end=min(n - 1, moving[-1] + 1))
        notes.append("neither rigid motion nor a simple cross-fade explains this region; inspect the key frames")

    # ---- 缓动 ----
    s, e = view["start"], view["end"]
    curves = [ch["easing"] for ch in channels.values() if ch.get("easing")]
    if easing and easing["rmse"] > 0.15:
        notes.append("the fitted curve does not describe the samples; this is probably not one eased motion")
        result["confidence"] = "low"
    elif any(curve["rmse"] > 0.06 for curve in curves):
        notes.append("at least one easing fit is poor; treat it as approximate")
        if result["confidence"] == "high":
            result["confidence"] = "medium"
    if any(curve["points"] < 8 for curve in curves):
        notes.append("few samples in at least one channel; several easing curves may fit")
    result["easing"] = easing
    view["progress"] = progress

    # ---- 关键帧与采样 ----
    keys: list[tuple[int, str]] = [(s, "start")]
    a, b = view.get("progress_range", (s, e))
    if progress is not None and b - a >= 2:
        speed = np.abs(np.diff(progress[a : b + 1])) / np.maximum(np.diff(t[a : b + 1]), 1e-6)
        keys.append((a + int(np.argmax(speed)) + 1, "peak-velocity"))
        if "peak" in view:
            keys.append((view["peak"], "peak"))
        elif easing and easing["overshoot"] > 0.03:
            keys.append((a + int(np.argmax(progress[a : b + 1])), "overshoot"))
        else:
            keys.append((a + int(np.argmin(np.abs(progress[a : b + 1] - 0.5))), "midpoint"))
    elif e - s >= 2:
        keys.append(((s + e) // 2, "midpoint"))
    keys.append((e, "end"))
    by_frame = {k: role for k, role in reversed(keys)}
    by_frame[e] = "end"
    view["keys"] = sorted(by_frame.items())

    picks = np.unique(np.linspace(s, e, min(e - s + 1, 40)).round().astype(int))
    samples = []
    for k in picks:
        row: dict = {"t_ms": _r(t[k], 3)}
        if progress is not None:
            row["progress"] = _r(progress[k], 3)
        if kind == "transform":
            row["x"], row["y"] = to_source(track["pos"][k])
            row["feature_x"], row["feature_y"] = to_source(track["anchor_path"][k])
            row["channel_progress"] = {name: _r(p[k], 3) for name, p in channel_progress.items()}
            if not track["rigid"]:
                row["scale"] = _r(track["scale"][k] / track["scale"][s], 3)
                row["rot_deg"] = _r(track["rot"][k] - track["rot"][s], 1)
        else:
            row["luma"] = _r(luma[k], 1)
        samples.append(row)
    result["samples"] = samples
    return result


def _hidden_end(t: np.ndarray, value: np.ndarray, valid: np.ndarray, hidden: int | None, eps: float) -> dict | None:
    """一条"起点一侧看不见"的曲线：value 在完整显示的那一端为 0，越往起点越大，但最前面几帧量不到。

    hidden 是已知"动效从这里开始"的帧（透明度曲线的起点）。看不见的部分不超过全程的三成时，
    用 fit_open 把起点幅度外推出来；否则只报看得见的部分，并标成下界。
    """
    idx = np.flatnonzero(valid)
    if len(idx) < 3:
        return None
    moving = idx[np.abs(value[idx]) >= eps]
    if not len(moving):
        return None
    first = int(idx[0])
    later = idx[idx > moving[-1]]
    end = int(later[0]) if len(later) else int(moving[-1])
    amplitude, start, model = float(value[first]), first, None
    if hidden is not None and hidden < first < end:
        span = t[end] - t[hidden]
        use = idx[(idx >= first) & (idx <= end)]
        if (t[first] - t[hidden]) / span <= 0.3 and len(use) >= 4:
            fit = fit_open((t[use] - t[hidden]) / span, value[use], True)
            # 外推只允许比看得见的部分多三成；再多就是曲线拟合在自由发挥
            if (fit and fit["rmse"] < 0.05 and fit["amplitude"] * amplitude > 0
                    and 0.98 * abs(amplitude) <= abs(fit["amplitude"]) <= 1.3 * abs(amplitude)):
                amplitude, start, model = fit["amplitude"], hidden, fit
    if abs(amplitude) < eps or end <= start:
        return None
    if start < first:
        known_t, known_p = np.r_[t[start], t[idx]], np.r_[0.0, 1 - value[idx] / amplitude]
    else:
        known_t, known_p = t[idx], 1 - value[idx] / amplitude
    progress = np.interp(t, known_t, known_p)
    progress[:start] = 0.0
    if model is not None:
        gap = np.arange(start, first)
        progress[gap] = bezier_ease(*model["params"], (t[gap] - t[start]) / (t[end] - t[start]))
    return {"start": start, "end": end, "progress": progress, "amplitude": amplitude,
            "extrapolated": model is not None,
            "lower_bound": model is None and (first > 0 if hidden is None else hidden < first)}


def _coupled(value: np.ndarray, alpha: np.ndarray, valid: np.ndarray, eps: float) -> float | None:
    """位移（或缩放）是否和透明度走同一条曲线。是的话返回透明度为 0 那一端的幅度。

    两者共用缓动时 value = A·(1 − alpha)，过原点的直线，A 就是看不见那一端的值。
    这比外推贝塞尔稳得多：不依赖曲线形状，对整数像素的量化噪声也不敏感。
    """
    u = 1 - alpha[valid]
    v = value[valid]
    if int(((u > 0.05) & (u < 0.95)).sum()) < 4 or float(u.max()) < 0.5:
        return None
    slope = float((v * u).sum() / (u * u).sum())
    tolerance = max(eps, 0.04 * abs(slope))
    if abs(slope) < eps or float(np.abs(v - slope * u).max()) > 2.5 * tolerance:
        return None
    if float(np.sqrt(np.mean((v - slope * u) ** 2))) > tolerance:
        return None
    return slope


def _layer_channel(t, value, valid, hidden, eps, enter) -> dict | None:
    """把 _hidden_end 用到正向时间上。退出动效是进入动效的时间反演。"""
    n = len(t)
    if enter:
        return _hidden_end(t, value, valid, hidden, eps)
    fit = _hidden_end(-t[::-1], value[::-1], valid[::-1], None if hidden is None else n - 1 - hidden, eps)
    if fit is None:
        return None
    return {**fit, "start": n - 1 - fit["end"], "end": n - 1 - fit["start"], "progress": 1 - fit["progress"][::-1]}


def _layer_results(fit: dict, crops: list[np.ndarray], t: np.ndarray, origin, inv_scale: float) -> list[dict]:
    """把图层模型的每个元素整理成一个片段结果。"""
    enter = fit["enter"]
    height, width = crops[0].shape[:2]
    several = len(fit["elements"]) > 1
    results = []

    def to_source(p) -> list[float]:
        return [_r((p[0] + origin[0]) * inv_scale, 1), _r((p[1] + origin[1]) * inv_scale, 1)]

    for el in fit["elements"]:
        valid = el.valid
        idx = np.flatnonzero(valid)
        notes: list[str] = []
        channels: dict = {}
        ranges: dict[str, tuple[int, int]] = {}
        progresses: dict[str, np.ndarray] = {}

        def add(name, start, end, progress, extra, channels=channels, ranges=ranges, progresses=progresses):
            span = progress[end] - progress[start]
            if abs(span) > 1e-9:
                progress = (progress - progress[start]) / span
            channels[name] = {"t_start_ms": _r(t[start], 3), "t_end_ms": _r(t[end], 3),
                              "duration_ms": _r(t[end] - t[start], 3),
                              "easing": fit_easing(t, progress, start, end), **extra}
            ranges[name] = (start, end)
            progresses[name] = progress

        # 透明度：相对"完整显示那一帧"的比例
        alpha = np.clip(el.alpha, -0.05, 1.05)
        hidden = None
        if float(alpha[idx].min()) < 0.9:
            p = alpha if enter else 1 - alpha
            start, end = _trim(p, _alpha_eps(p))
            add("opacity", start, end, p, {"from": 0.0 if enter else 1.0, "to": 1.0 if enter else 0.0,
                                           "reference": "fully_shown_frame"})
            hidden = start if enter else end

        # 平移：沿最大位移方向投影
        shift = np.where(valid[:, None], el.shift, 0.0)
        far = int(idx[np.argmax(np.linalg.norm(shift[idx], axis=1))])
        reach = float(np.linalg.norm(shift[far]))
        unit = shift[far] / reach if reach > 0 else np.array([1.0, 0.0])
        offset = np.zeros(2)  # 看不见那一端相对完整显示位置的位移

        def channel(value, eps, channels=channels, ranges=ranges, progresses=progresses, alpha=alpha,
                    valid=valid, hidden=hidden):
            if "opacity" in channels:
                slope = _coupled(value, np.clip(alpha, 0, 1), valid, eps)
                if slope is not None:
                    start, end = ranges["opacity"]
                    return {"start": start, "end": end, "progress": progresses["opacity"], "amplitude": slope,
                            "extrapolated": True, "lower_bound": False, "coupled": True}
            fit = _layer_channel(t, value, valid, hidden, eps, enter)
            return None if fit is None else {**fit, "coupled": False}

        move = channel(shift @ unit, 0.3) if reach >= 1.0 else None
        centre = np.array(el.center)
        if move is not None:
            offset = move["amplitude"] * unit
            delta = (-offset if enter else offset) * inv_scale
            a, b = (centre + offset, centre) if enter else (centre, centre + offset)
            add("translate", move["start"], move["end"], move["progress"], {
                "from": to_source(a), "to": to_source(b),
                "delta": [_r(delta[0], 1), _r(delta[1], 1)], "distance_px": _r(np.linalg.norm(delta), 1),
                "reference": "layer_center", "extrapolated": move["extrapolated"],
                "lower_bound": move["lower_bound"], "visible_distance_px": _r(reach * inv_scale, 1),
                "follows_opacity": move["coupled"]})
            if move["coupled"]:
                notes.append("translation is proportional to (1 - opacity), so both share one curve; the invisible end follows from that ratio")
            elif move["extrapolated"]:
                notes.append("the element is invisible at one end; that end of the translation is extrapolated from the fitted curve")
            elif move["lower_bound"]:
                notes.append("part of the translation happens while the element is invisible; the distance is a lower bound")

        # 缩放：相对完整显示时的大小
        grow = None
        if float(np.abs(el.scale[idx] - 1).max()) >= 0.012:
            grow = channel(el.scale - 1, 0.004)
        if grow is not None:
            far_scale = 1 + grow["amplitude"]
            add("scale", grow["start"], grow["end"], grow["progress"], {
                "from": _r(far_scale if enter else 1.0, 3), "to": _r(1.0 if enter else far_scale, 3),
                "peak": _r(max(float(el.scale[idx].max()), far_scale, 1.0), 3),
                "reference": "fully_shown_frame", "extrapolated": grow["extrapolated"],
                "lower_bound": grow["lower_bound"], "follows_opacity": grow["coupled"]})
            if grow["coupled"]:
                notes.append("scale is proportional to (1 - opacity), so both share one curve; the invisible end follows from that ratio")
            elif grow["extrapolated"]:
                notes.append("the element is invisible at one end; that end of the scale is extrapolated from the fitted curve")
        if move is None and grow is None:
            continue

        half = max(el.box[2] - el.box[0], el.box[3] - el.box[1]) / 2
        effect = {"translate": abs(move["amplitude"]) if move else 0.0,
                  "scale": abs(grow["amplitude"]) * half if grow else 0.0}
        dominant = max(effect, key=effect.get)
        start, end = min(a for a, _ in ranges.values()), max(b for _, b in ranges.values())

        # 中心点轨迹：量得到的帧用实测，量不到的用拟合进度补
        pos = centre + shift
        if move is not None:
            along = (1 - progresses["translate"]) if enter else progresses["translate"]
            pos[~valid] = centre + along[~valid, None] * offset

        # 多个元素共用一个窗口时，各自只保留自己扫过的范围
        roi = (0, 0, width, height)
        if several:
            x0, y0, x1, y1 = el.box
            dx, dy = pos[:, 0] - centre[0], pos[:, 1] - centre[1]
            spare = half * max(0.0, float(el.scale.max()) - 1) + 10
            roi = (max(0, int(x0 + dx.min() - spare)), max(0, int(y0 + dy.min() - spare)),
                   min(width, int(math.ceil(x1 + dx.max() + spare))), min(height, int(math.ceil(y1 + dy.max() + spare))))
        ends = [to_gray(c[roi[1]:roi[3], roi[0]:roi[2]]) for c in (crops[0], crops[-1])]

        quality = float(np.median(el.corr[idx]))
        confidence = "high" if fit["residual"] < 0.05 and quality >= 0.95 and len(idx) >= 6 else "medium"
        if any(ch.get("lower_bound") for ch in channels.values()):
            confidence = "medium"
        progress = progresses[dominant]
        a, b = ranges[dominant]
        keys = {start: "start", end: "end"}
        if b - a >= 2:
            speed = np.abs(np.diff(progress[a : b + 1])) / np.maximum(np.diff(t[a : b + 1]), 1e-6)
            keys.setdefault(a + int(np.argmin(np.abs(progress[a : b + 1] - 0.5))), "midpoint")
            keys.setdefault(a + int(np.argmax(speed)) + 1, "peak-velocity")
        samples = []
        for k in np.unique(np.linspace(start, end, min(end - start + 1, 40)).round().astype(int)):
            row = {"t_ms": _r(t[k], 3), "progress": _r(progress[k], 3)}
            row["x"], row["y"] = to_source(pos[k])
            row["opacity"] = _r(float(np.clip(alpha[k], 0, 1)), 3)
            row["channel_progress"] = {name: _r(p[k], 3) for name, p in progresses.items()}
            if grow is not None:
                row["scale"] = _r(el.scale[k], 3)
            samples.append(row)
        notes.insert(0, "layer model: each frame is the background plus the element at some position, scale and opacity")
        results.append({
            "luma": {"from": _r(ends[0].mean(), 1), "to": _r(ends[1].mean(), 1)},
            "notes": notes,
            "kind": "transform",
            "model": "layer",
            "layer": {"direction": "enter" if enter else "exit", "residual": _r(fit["residual"], 4),
                      "plain_fade_residual": _r(fit["plain"], 4), "correlation": _r(quality, 3),
                      "frames_measured": int(len(idx))},
            "channels": channels,
            "transform_reference": {"type": "layer_center", "center": to_source(centre)},
            "dominant": dominant,
            "confidence": confidence,
            "easing": channels[dominant]["easing"],
            "samples": samples,
            "_roi": roi,
            "_view": {"pos": pos - np.array([roi[0], roi[1]]), "progress": progress, "start": start, "end": end,
                      "progress_range": (a, b), "keys": sorted(keys.items())},
        })
    return results


def analyze_window(
    crops: list[np.ndarray],
    times: list[float],
    *,
    origin: tuple[int, int],
    inv_scale: float,
    threshold: int = 12,
    allow_layer: bool = True,
) -> list[dict]:
    """分析一个窗口，返回其中的片段（通常一个；多个元素依次进入时会拆成多个）。

    先试图层模型（进入 / 退出类动效），解释不了再走跟踪和混合度那条路。
    每个结果可带 "_roi"：该片段在窗口内的子区域（分析尺度，相对窗口左上角）。
    """
    # 进入 / 退出都是短动效；又长又大的窗口多半是持续变化的内容，不值得试
    if allow_layer and len(crops) <= LAYER_MAX_FRAMES and times[-1] - times[0] <= LAYER_MAX_MS:
        fit = fit_layers(crops, threshold)
        if fit is not None:
            results = _layer_results(fit, crops, np.asarray(times, dtype=float), origin, inv_scale)
            if results:
                return results
    return [_analyze_single(crops, times, origin=origin, inv_scale=inv_scale, threshold=threshold)]
