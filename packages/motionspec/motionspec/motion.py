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
SPLIT_AREA = 60_000  # 遮罩分类时的目标像素数
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
    tolerance = max(eps, 0.045 * abs(slope))
    err = np.abs(v - slope * u)
    check_err = np.sort(err)[:-1] if len(err) >= 6 else err
    if abs(slope) < eps or float(check_err.max()) > 2.5 * tolerance:
        return None
    if float(np.sqrt(np.mean(check_err**2))) > tolerance:
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
        boxed = el.box[0] > 1 and el.box[1] > 1 and el.box[2] < width - 1 and el.box[3] < height - 1
        for k in np.unique(np.linspace(start, end, min(end - start + 1, 40)).round().astype(int)):
            row = {"t_ms": _r(t[k], 3), "progress": _r(progress[k], 3)}
            row["x"], row["y"] = to_source(pos[k])
            row["opacity"] = _r(float(np.clip(alpha[k], 0, 1)), 3)
            row["channel_progress"] = {name: _r(p[k], 3) for name, p in progresses.items()}
            # 元素的外接框（完整显示那一帧里和背景不同的范围）乘当前缩放。框碰到窗口边时，
            # 里面混进了窗口边上的残差（例如遮罩补偿留下的一圈），不是元素本身的尺寸，就不给
            if boxed:
                row["width"] = _r(el.size[0] * el.scale[k] * inv_scale, 1)
                row["height"] = _r(el.size[1] * el.scale[k] * inv_scale, 1)
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


def _split_backdrop(crops: list[np.ndarray], threshold: int) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """窗口里是否有一大片区域按同一条曲线在首尾两帧之间混合（遮罩变暗、整体变色），
    同时还有一块区域在做别的事（对话框弹出）。是的话返回 (混合区域掩码, 其余部分的外接框)。

    模态框是最常见的例子：遮罩把整个窗口调暗，对话框在上面滑入淡入。两者混在一个窗口里时，
    图层模型以为整片画面都是"新出现的元素"，跟踪又被变暗的背景干扰，两边都测不准。
    """
    # 这里只是分类（哪里是遮罩、哪里是别的元素），在缩小的图上做就够了，掩码最后放大回原尺寸
    full_h, full_w = crops[0].shape[:2]
    factor = min(1.0, math.sqrt(SPLIT_AREA / (full_h * full_w)))
    if factor < 1.0:
        size = (max(8, round(full_w * factor)), max(8, round(full_h * factor)))
        crops = [crops[0], *crops[1:-1:max(1, (len(crops) - 2) // 24)], crops[-1]]
        crops = [cv2.resize(c, size, interpolation=cv2.INTER_AREA) for c in crops]
    f0, f1 = crops[0].astype(np.float32), crops[-1].astype(np.float32)
    delta = f1 - f0
    power = (delta * delta).sum(axis=2)
    changed = power > (3 * threshold) ** 2
    touched = power > threshold ** 2  # 暗背景变暗只有十几个灰度，算不上 changed，但仍是遮罩的证据
    if changed.mean() < 0.05:
        return None
    sampled = crops[1:-1:max(1, (len(crops) - 2) // 24)]
    # 形态学核按像素算，缩小后要跟着缩，否则细的运动带会被开运算整条抹掉
    kernel = np.ones((5, 5) if factor >= 0.75 else (3, 3), np.uint8)
    grow = 2 if factor >= 0.75 else max(1, round(4 * factor))

    def evaluate(basis):
        """以 basis 里的像素估一条共同的混合曲线，返回 (曲线, 每个像素的最大偏离比例, 最大变化幅度)。"""
        curve, worst, swing = [], np.zeros(changed.shape, np.float32), np.zeros(changed.shape, np.float32)
        for crop in sampled:
            moved = crop.astype(np.float32) - f0
            b = float(np.median((moved * delta).sum(axis=2)[basis] / power[basis]))
            curve.append(b)
            worst = np.maximum(worst, np.sqrt(((moved - b * delta) ** 2).sum(axis=2)) / np.maximum(np.sqrt(power), 2.0 * threshold))
            swing = np.maximum(swing, np.sqrt((moved * moved).sum(axis=2)))
        return curve, worst, swing

    # 第一遍：用变化大的像素估曲线，找出变化幅度大又对不上的那块（对话框）
    curve, worst, swing = evaluate(changed)
    other = (worst >= 0.3) & (swing > 3 * threshold)
    other = cv2.dilate(cv2.morphologyEx(other.astype(np.uint8), cv2.MORPH_OPEN, kernel), kernel, iterations=grow) > 0
    # 第二遍：只用那块之外的像素估曲线。对话框占了变化像素的大头时，第一遍的曲线其实是对话框的
    basis = touched & ~other
    if basis.sum() < 0.05 * basis.size:
        return None
    curve, worst, swing = evaluate(basis)
    consistent = (worst < 0.3) & touched & ~other
    if not any(0.15 < b < 0.85 for b in curve):
        return None
    other = (worst >= 0.3) & (swing > 3 * threshold) & ~consistent
    other = cv2.dilate(cv2.morphologyEx(other.astype(np.uint8), cv2.MORPH_OPEN, kernel), kernel, iterations=grow)
    if other.sum() < 0.01 * other.size or other.sum() > 0.6 * other.size:
        return None
    if consistent.sum() < 0.4 * (touched & ~(other > 0)).sum():
        return None
    ys, xs = np.nonzero(other)
    h, w = other.shape
    pad = 6
    box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))
    # 混合区域得在那块"别的东西"外面：一张边移动边淡入的卡片，内部也按同一条曲线混合，
    # 但它被移动的边缘包着，那是同一个元素，不是遮罩
    outside = int(consistent.sum() - consistent[box[1]:box[3], box[0]:box[2]].sum())
    if outside < 0.15 * consistent.sum() or outside < 0.02 * consistent.size:
        return None
    if factor < 1.0:
        consistent = cv2.resize(consistent.astype(np.uint8), (full_w, full_h), interpolation=cv2.INTER_NEAREST) > 0
        sx, sy = full_w / w, full_h / h
        box = (max(0, int(box[0] * sx) - 2), max(0, int(box[1] * sy) - 2),
               min(full_w, int(math.ceil(box[2] * sx)) + 2), min(full_h, int(math.ceil(box[3] * sy)) + 2))
    return consistent, box


def analyze_window(
    crops: list[np.ndarray],
    times: list[float],
    *,
    origin: tuple[int, int],
    inv_scale: float,
    threshold: int = 12,
    allow_layer: bool = True,
    allow_split: bool = True,
) -> list[dict]:
    split = _split_backdrop(crops, threshold) if allow_split and len(crops) >= 5 else None
    if split is not None:
        blend, (x1, y1, x2, y2) = split
        first, last = crops[0], crops[-1]
        # 遮罩部分：把其余区域冻结在首帧，剩下的就是一次纯混合
        frozen = []
        for crop in crops:
            copy = crop.copy()
            copy[y1:y2, x1:x2] = first[y1:y2, x1:x2]
            frozen.append(copy)
        backdrop = _analyze_single(frozen, times, origin=origin, inv_scale=inv_scale, threshold=threshold)
        backdrop["notes"].append("a large area blends towards its final state on one shared curve (dimming overlay or "
                                 "colour change); the element drawn on top of it is reported separately")
        # 其余区域：把遮罩的变化从每个像素里减掉（包括对话框底下和它扫过的地方），
        # 背景就停在末帧的状态，元素落在一张静止的背景上
        f0 = first.astype(np.float32)
        mapped = np.empty_like(f0)
        for channel in range(f0.shape[2]):  # 遮罩是逐通道的仿射：末帧 = 增益 × 首帧 + 偏移
            x = f0[..., channel][blend]
            y = last[..., channel].astype(np.float32)[blend]
            gain, offset = np.polyfit(x, y, 1) if np.ptp(x) > 1 else (1.0, float(np.mean(y - x)))
            mapped[..., channel] = gain * f0[..., channel] + offset
        shift = (mapped - f0)[y1:y2, x1:x2]
        delta = (last.astype(np.float32) - f0)[blend]
        parts = []
        for crop in crops:
            b = float(np.median(((crop.astype(np.float32) - f0)[blend] * delta).sum(axis=1)
                                / np.maximum((delta * delta).sum(axis=1), 1e-6)))
            fixed = crop[y1:y2, x1:x2].astype(np.float32) + (1.0 - b) * shift
            parts.append(np.clip(fixed, 0, 255).round().astype(np.uint8))
        others = analyze_window(parts, times, origin=(origin[0] + x1, origin[1] + y1), inv_scale=inv_scale,
                                threshold=threshold, allow_layer=allow_layer, allow_split=False)
        for result in others:
            sx1, sy1, sx2, sy2 = result.get("_roi", (0, 0, x2 - x1, y2 - y1))
            result["_roi"] = (x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2)
            result["notes"].append("measured after removing the dimming or colour change behind it")
        return [backdrop, *others]
    return _analyze_window_plain(crops, times, origin=origin, inv_scale=inv_scale, threshold=threshold,
                                 allow_layer=allow_layer)


def _analyze_window_plain(
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
    box = _solid_box(crops, times, origin, inv_scale, threshold) if len(crops) >= 5 else None
    if box is not None:
        return [box]
    # 进入 / 退出都是短动效；又长又大的窗口多半是持续变化的内容，不值得试
    if allow_layer and len(crops) <= LAYER_MAX_FRAMES and times[-1] - times[0] <= LAYER_MAX_MS:
        fit = fit_layers(crops, threshold)
        if fit is not None:
            results = _layer_results(fit, crops, np.asarray(times, dtype=float), origin, inv_scale)
            if results:
                return results
    single = _analyze_single(crops, times, origin=origin, inv_scale=inv_scale, threshold=threshold)
    if single["kind"] == "nonrigid":
        bodies = _multibody(crops, times, origin, inv_scale, threshold)
        if bodies:
            return bodies
    return [single]


def _multibody(crops: list[np.ndarray], times, origin, inv_scale: float, threshold: int) -> list[dict] | None:
    """一块区域里有几个已存在的元素各自平移（列表重排、两项对调）。

    单个相似变换解释不了这种情况，原先只能报 nonrigid。这里把特征点从首帧一路跟到末帧，
    按整条位移轨迹聚类：轨迹逐帧一致的点属于同一个元素。每一类单独报一个平移片段。
    只处理平移；任何一类点数太少、或几类加起来解释不了大部分运动点，就放弃。
    """
    n = len(crops)
    t = np.asarray(times, dtype=float)
    grays = [to_gray(c) for c in crops]
    moving = np.zeros(grays[0].shape, np.uint8)
    for k in range(n - 1):
        moving |= (cv2.absdiff(grays[k], grays[k + 1]) > max(4, threshold // 2)).astype(np.uint8)
    moving = cv2.dilate(moving, np.ones((9, 9), np.uint8))
    start = cv2.goodFeaturesToTrack(grays[0], 400, 0.01, 3, mask=moving * 255)
    if start is None or len(start) < 16:
        return None
    pts = start.reshape(-1, 2).astype(np.float32)
    path = np.full((n, len(pts), 2), np.nan, np.float32)
    path[0] = pts
    alive = np.ones(len(pts), bool)
    lost_at = np.full(len(pts), n, int)
    for k in range(n - 1):
        if not alive.any():
            break
        p0 = path[k][alive].reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(grays[k], grays[k + 1], p0, None, **LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(grays[k + 1], grays[k], p1, None, **LK)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0.reshape(-1, 2) - back.reshape(-1, 2), axis=1) < 0.7)
        index = np.flatnonzero(alive)
        path[k + 1, index[ok]] = p1.reshape(-1, 2)[ok]
        alive[index[~ok]] = False
        lost_at[index[~ok]] = k + 1
    # 互相交错的元素会彼此遮挡，被挡住的那个上面的点在交错时会跟丢。
    # 只用交错之前的那段轨迹来分组：取一半运动点还活着的最后一帧
    reach_all = np.nanmax(np.linalg.norm(path - path[0], axis=2), axis=0)
    active = reach_all >= 2.0
    if active.sum() < 16:
        return None
    horizon = int(np.sort(lost_at[active])[len(lost_at[active]) // 2])
    keep = active & (lost_at >= horizon)
    if horizon < 4 or keep.sum() < 16:
        return None
    track = path[:horizon, keep] - path[0, keep]
    reach = np.linalg.norm(track, axis=2).max(axis=0)
    movers = np.flatnonzero(reach >= 1.0)
    if len(movers) < 16:
        return None
    left = list(movers)
    groups = []
    while left:
        seed = max(left, key=lambda i: reach[i])
        gap = np.linalg.norm(track[:, left] - track[:, [seed]], axis=2).max(axis=0)
        members = [i for i, g in zip(left, gap) if g <= 1.5 + 0.05 * reach[seed]]
        left = [i for i in left if i not in set(members)]
        if len(members) >= 8:
            groups.append(members)
    if len(groups) < 2 or sum(len(g) for g in groups) < 0.6 * len(movers):
        return None
    starts = path[0, keep]
    # 每一组用首帧里它的外接框做模板，逐帧在预测位置附近匹配，穿过遮挡
    height, width = grays[0].shape
    tracks = []
    for members in groups:
        x0, y0 = np.floor(starts[members].min(axis=0) - 4).astype(int)
        x1, y1 = np.ceil(starts[members].max(axis=0) + 4).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        template = grays[0][y0:y1, x0:x1].astype(np.float32)
        if template.size < 64 or float(template.std()) < 4:
            return None
        box = (x0, y0, x1, y1)
        forward, f_score = _follow(grays, template, box, np.zeros(2), range(1, n))
        # 末帧里在整个窗口找一次，再倒着跟回来：被遮挡的元素在交错之后能重新对上
        whole = cv2.matchTemplate(grays[-1].astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
        _, best_end, _, (ex, ey) = cv2.minMaxLoc(np.nan_to_num(whole, nan=-1.0))
        if best_end >= 0.9:
            backward, b_score = _follow(grays, template, box, np.array([ex - x0, ey - y0], float), range(n - 2, -1, -1))
            backward[-1], b_score[-1] = np.array([ex - x0, ey - y0], float), best_end
        else:
            backward, b_score = forward, f_score
        use_back = b_score > f_score
        shift = np.where(use_back[:, None], backward, forward)
        score = np.maximum(f_score, b_score)
        good = score >= 0.9
        good[0] = True
        if good.sum() < 0.6 * n:
            return None
        for axis in range(2):  # 两个方向都对不上的帧（真正被挡住的那几帧）按时间插值
            shift[~good, axis] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), shift[good, axis])
        tracks.append((members, (x0, y0, x1, y1), shift))
    results = []
    for members, (bx0, by0, bx1, by1), shift in sorted(tracks, key=lambda item: -len(item[0])):
        centre = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2])
        box0 = np.array([[bx0, by0], [bx1, by1]], dtype=float)
        pos = centre + shift
        offset = shift[-1]
        far = shift[int(np.argmax(np.linalg.norm(shift, axis=1)))]
        reach_body = float(np.linalg.norm(far))
        if reach_body < 2.0:
            continue
        target = offset if np.linalg.norm(offset) >= 0.3 * reach_body else far
        value = shift @ (target / np.linalg.norm(target))
        eps = float(np.clip(0.3 / reach_body, 0.002, 0.05))
        first, last, peak, progress, curve = _fit_channel(t, value, eps, abs(value[-1]) < 0.3 * reach_body)
        k_to = last if peak is None else peak
        delta = (pos[k_to] - pos[first]) * inv_scale
        channel = {"t_start_ms": _r(t[first], 3), "t_end_ms": _r(t[last], 3), "duration_ms": _r(t[last] - t[first], 3),
                   "easing": curve, "from": [_r((pos[first][0] + origin[0]) * inv_scale, 1), _r((pos[first][1] + origin[1]) * inv_scale, 1)],
                   "to": [_r((pos[k_to][0] + origin[0]) * inv_scale, 1), _r((pos[k_to][1] + origin[1]) * inv_scale, 1)],
                   "delta": [_r(delta[0], 1), _r(delta[1], 1)], "distance_px": _r(np.linalg.norm(delta), 1),
                   "reference": "tracked_point_group"}
        if peak is not None:
            channel.update(pattern="out-and-back", peak_ms=_r(t[peak] - t[first], 1))
        pad = 8
        lo = (box0.min(axis=0) + shift.min(axis=0)) - pad
        hi = (box0.max(axis=0) + shift.max(axis=0)) + pad
        roi = (max(0, int(lo[0])), max(0, int(lo[1])), min(width, int(math.ceil(hi[0]))), min(height, int(math.ceil(hi[1]))))
        keys = {first: "start", last: "end"}
        if last - first >= 2:
            keys.setdefault(first + int(np.argmin(np.abs(progress[first : last + 1] - 0.5))), "midpoint")
        samples = []
        for k in np.unique(np.linspace(first, last, min(last - first + 1, 40)).round().astype(int)):
            samples.append({"t_ms": _r(t[k], 3), "progress": _r(progress[k], 3),
                            "x": _r((pos[k][0] + origin[0]) * inv_scale, 1), "y": _r((pos[k][1] + origin[1]) * inv_scale, 1),
                            "channel_progress": {"translate": _r(progress[k], 3)}})
        results.append({
            "luma": {"from": _r(grays[0][roi[1]:roi[3], roi[0]:roi[2]].mean(), 1), "to": _r(grays[-1][roi[1]:roi[3], roi[0]:roi[2]].mean(), 1)},
            "notes": [f"several elements in this region move differently; {len(groups)} groups of tracked points were separated by their trajectories",
                      "translation follows the tracked point group; the element boundary is approximate"],
            "kind": "transform",
            "model": "multibody",
            "channels": {"translate": channel},
            "dominant": "translate",
            "confidence": "medium" if len(members) >= 20 else "low",
            "easing": curve,
            "samples": samples,
            "_roi": roi,
            "_view": {"pos": pos - np.array([roi[0], roi[1]]), "progress": progress, "start": first, "end": last,
                      "progress_range": (first, last), "keys": sorted(keys.items())},
        })
        if peak is not None:
            results[-1].update(pattern="out-and-back", peak_ms=channel["peak_ms"])
    return results if len(results) >= 2 else None


def _follow(grays, template, box, start, order):
    """模板逐帧跟踪：在预测位置附近匹配，带亚像素插值。返回每帧的位移和匹配分数。"""
    n = len(grays)
    height, width = grays[0].shape
    x0, y0, x1, y1 = box
    shift = np.zeros((n, 2))
    score = np.full(n, -1.0)
    order = list(order)
    previous = order[0] - 1 if order[0] > 0 and (len(order) < 2 or order[1] > order[0]) else order[0] + 1
    step = 1 if previous < order[0] else -1
    shift[previous] = start
    score[previous] = 1.0
    velocity = np.zeros(2)
    for k in order:
        guess = shift[k - step] + velocity
        radius = int(max(8, 2 * np.linalg.norm(velocity) + 6))
        gx, gy = int(round(x0 + guess[0])), int(round(y0 + guess[1]))
        sx0, sy0 = max(0, gx - radius), max(0, gy - radius)
        sx1, sy1 = min(width, gx + (x1 - x0) + radius), min(height, gy + (y1 - y0) + radius)
        if sx1 - sx0 < x1 - x0 or sy1 - sy0 < y1 - y0:
            shift[k] = guess
            continue
        result = np.nan_to_num(cv2.matchTemplate(grays[k][sy0:sy1, sx0:sx1].astype(np.float32), template,
                                                 cv2.TM_CCOEFF_NORMED), nan=-1.0)
        _, best, _, (mx, my) = cv2.minMaxLoc(result)
        fx = fy = 0.0
        if 0 < mx < result.shape[1] - 1:
            a, b, c = result[my, mx - 1], result[my, mx], result[my, mx + 1]
            fx = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else 0.0
        if 0 < my < result.shape[0] - 1:
            a, b, c = result[my - 1, mx], result[my, mx], result[my + 1, mx]
            fy = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else 0.0
        score[k] = best
        if best < 0.9:  # 可能被挡住：按预测走，速度不更新
            shift[k] = guess
            continue
        shift[k] = np.array([sx0 + mx + fx - x0, sy0 + my + fy - y0])
        velocity = shift[k] - shift[k - step]
    return shift, score


def _solid_box(crops: list[np.ndarray], times, origin, inv_scale: float, threshold: int) -> dict | None:
    """纯色块的尺寸变化：进度条变长、标签指示条边移边变宽、面板展开。

    相似变换只有等比缩放，表达不了"只变宽"；图层模型和跟踪都假设元素形状不变。
    这里直接逐帧找出那块颜色的外接框，看它四条边怎么动：一条边不动、对边在动就是从那条边长出来；
    两条边都动但距离在变，就是边移边变宽。只接受尺寸确实变了、而且框的变化平滑连续的情况。
    """
    n = len(crops)
    t = np.asarray(times, dtype=float)
    f0, f1 = crops[0].astype(np.int16), crops[-1].astype(np.int16)
    support = np.abs(f1 - f0).max(axis=2) > threshold
    if support.sum() < 30:
        return None
    height, width = support.shape
    near = cv2.dilate(support.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    best = None
    ring = np.concatenate([f0[:2].reshape(-1, 3), f0[-2:].reshape(-1, 3), f0[:, :2].reshape(-1, 3), f0[:, -2:].reshape(-1, 3)])
    backdrop = np.median(ring, axis=0)  # 窗口边上的颜色当作背景，元素的颜色要和它不同
    for which, frame in (("end", f1), ("start", f0)):  # 元素的颜色：变化区域里最常见的那个，先看末帧（长出来的），再看首帧（缩回去的）
        pixels = frame[support]
        pixels = pixels[np.abs(pixels - backdrop).max(axis=1) > 18]
        if len(pixels) < 0.15 * support.sum():
            continue
        colours = (pixels // 16).astype(np.int32)
        keys = colours[:, 0] * 256 + colours[:, 1] * 16 + colours[:, 2]
        values, counts = np.unique(keys, return_counts=True)
        if counts.max() < 0.3 * len(pixels):
            continue
        tone = np.median(pixels[keys == values[np.argmax(counts)]], axis=0)
        boxes = []
        for crop in crops:
            mask = (np.abs(crop.astype(np.int16) - tone).max(axis=2) <= 18).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            box = None
            if count > 1:
                overlap = [int(near[labels == i].sum()) if stats[i, cv2.CC_STAT_AREA] >= 6 else 0 for i in range(1, count)]
                if max(overlap) > 0:
                    i = 1 + int(np.argmax(overlap))
                    x, y, w, h, area = stats[i]
                    if area >= 0.8 * w * h or (min(w, h) <= 12 and area >= 0.6 * w * h):  # 实心的块（小块的圆角占比大）
                        box = (x, y, x + w, y + h)
            boxes.append(box)
        if sum(b is not None for b in boxes) < 0.6 * n or boxes[-1 if which == "end" else 0] is None:
            continue
        edges = np.array([b if b is not None else (np.nan,) * 4 for b in boxes], dtype=float)
        empty = np.isnan(edges[:, 0])
        size = np.stack([edges[:, 2] - edges[:, 0], edges[:, 3] - edges[:, 1]], axis=1)
        size[empty] = 0.0  # 框不存在的帧：长度还是 0
        change = np.abs(size - size[0]).max(axis=0)
        span = float(change.max())
        if span < 3 or span < 0.05 * max(1.0, float(size.max())):
            continue
        axis = int(np.argmax(change))
        if float(change[1 - axis]) > max(2.0, 0.1 * span):  # 两个方向一起变是缩放，交给跟踪
            continue
        if float(np.abs(np.diff(size[:, axis])).max()) > 0.35 * span:  # 一帧跳变太大：是淡入或突变，不是长出来
            continue
        known = edges[~empty]
        limit = (width, height)[axis]
        if (known[:, axis] <= 1).any() or (known[:, axis + 2] >= limit - 1).any():  # 被窗口边裁掉的"变短"不算
            continue
        best = (edges, size, axis, empty)
        break
    if best is None:
        return None
    edges, size, axis, empty = best
    seen = np.flatnonzero(~empty)
    for column in range(4):  # 不存在的帧用看得见的帧补全
        values = edges[:, column]
        values[empty] = np.interp(np.flatnonzero(empty), seen, values[seen])
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    moved = np.abs(edges - edges[seen[0]]).max(axis=0)
    fixed = lo if moved[lo] <= moved[hi] else hi
    size[empty, 1 - axis] = np.interp(np.flatnonzero(empty), seen, size[seen, 1 - axis])
    centre = np.stack([(edges[:, 0] + edges[:, 2]) / 2, (edges[:, 1] + edges[:, 3]) / 2], axis=1)
    centre[empty, axis] = edges[empty, fixed]  # 0 尺寸的帧：中心落在固定边上
    value = size[:, axis] - size[0, axis]
    eps = float(np.clip(0.3 / max(abs(float(value[-1])), 1e-6), 0.002, 0.05))
    first, last, _, progress, curve = _fit_channel(t, value, eps, False)
    if curve is None:
        return None
    s = inv_scale
    anchored = min(moved[lo], moved[hi]) <= 1.0
    channels = {"size": {"t_start_ms": _r(t[first], 3), "t_end_ms": _r(t[last], 3), "duration_ms": _r(t[last] - t[first], 3),
                         "easing": curve, "width_from": _r(size[first, 0] * s, 1), "width_to": _r(size[last, 0] * s, 1),
                         "height_from": _r(size[first, 1] * s, 1), "height_to": _r(size[last, 1] * s, 1),
                         "anchor": ["left", "top", "right", "bottom"][fixed] if anchored else "none",
                         "reference": "solid_colour_box"}}
    shift = centre[last] - centre[first]
    if not anchored and float(np.linalg.norm(shift)) >= 2:  # 两条边都在动：同时有位移
        direction = shift / np.linalg.norm(shift)
        travel = (centre - centre[first]) @ direction
        a, b, _, move_progress, c2 = _fit_channel(t, travel, float(np.clip(0.3 / max(abs(float(travel[-1])), 1e-6), 0.002, 0.05)), False)
        channels["translate"] = {"t_start_ms": _r(t[a], 3), "t_end_ms": _r(t[b], 3), "duration_ms": _r(t[b] - t[a], 3),
                                 "easing": c2, "from": [_r((centre[first][0] + origin[0]) * s, 1), _r((centre[first][1] + origin[1]) * s, 1)],
                                 "to": [_r((centre[last][0] + origin[0]) * s, 1), _r((centre[last][1] + origin[1]) * s, 1)],
                                 "delta": [_r(shift[0] * s, 1), _r(shift[1] * s, 1)], "distance_px": _r(np.linalg.norm(shift) * s, 1),
                                 "reference": "solid_colour_box_center"}
    pad = 4
    x0 = max(0, int(np.floor(edges[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(edges[:, 1].min())) - pad)
    x1 = min(width, int(np.ceil(edges[:, 2].max())) + pad)
    y1 = min(height, int(np.ceil(edges[:, 3].max())) + pad)
    keys = {first: "start", last: "end"}
    if last - first >= 2:
        keys.setdefault(first + int(np.argmin(np.abs(progress[first : last + 1] - 0.5))), "midpoint")
    samples = []
    for k in np.unique(np.linspace(first, last, min(last - first + 1, 40)).round().astype(int)):
        row = {"t_ms": _r(t[k], 3), "progress": _r(progress[k], 3), "x": _r((centre[k][0] + origin[0]) * s, 1),
               "y": _r((centre[k][1] + origin[1]) * s, 1), "width": _r(size[k, 0] * s, 1), "height": _r(size[k, 1] * s, 1),
               "channel_progress": {"size": _r(progress[k], 3)}}
        if "translate" in channels:
            row["channel_progress"]["translate"] = _r(move_progress[k], 3)
        samples.append(row)
    grays = (to_gray(crops[0]), to_gray(crops[-1]))
    return {
        "luma": {"from": _r(grays[0][y0:y1, x0:x1].mean(), 1), "to": _r(grays[1][y0:y1, x0:x1].mean(), 1)},
        "notes": ["a solid-coloured box changes size; edges are measured on the box of that colour, frame by frame"],
        "kind": "transform",
        "model": "solid-box",
        "channels": channels,
        "dominant": "size",
        "confidence": "high" if len(seen) >= 0.8 * n and curve["rmse"] < 0.03 else "medium",
        "easing": curve,
        "samples": samples,
        "_roi": (x0, y0, x1, y1),
        "_view": {"pos": centre - np.array([x0, y0]), "progress": progress, "start": first, "end": last,
                  "progress_range": (first, last), "keys": sorted(keys.items())},
    }
