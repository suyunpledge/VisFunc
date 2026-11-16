"""把"进度随时间"的实测曲线拟合成缓动函数：cubic-bezier、命名曲线、弹簧。

只用 numpy。固定 x1/x2 后 y1/y2 是线性最小二乘，所以只需在 (x1, x2) 上搜索。
"""

from __future__ import annotations

import numpy as np

# CSS 关键字、Material 常用曲线、easings.net 的贝塞尔近似（Qt 的 OutCubic 等同名曲线同形）
NAMED: dict[str, tuple[float, float, float, float]] = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease-in": (0.42, 0.0, 1.0, 1.0),
    "ease-out": (0.0, 0.0, 0.58, 1.0),
    "ease-in-out": (0.42, 0.0, 0.58, 1.0),
    "material-standard": (0.4, 0.0, 0.2, 1.0),
    "material-decelerate": (0.0, 0.0, 0.2, 1.0),
    "material-accelerate": (0.4, 0.0, 1.0, 1.0),
    "in-quad": (0.11, 0.0, 0.5, 0.0),
    "out-quad": (0.5, 1.0, 0.89, 1.0),
    "in-out-quad": (0.45, 0.0, 0.55, 1.0),
    "in-cubic": (0.32, 0.0, 0.67, 0.0),
    "out-cubic": (0.33, 1.0, 0.68, 1.0),
    "in-out-cubic": (0.65, 0.0, 0.35, 1.0),
    "out-quart": (0.25, 1.0, 0.5, 1.0),
    "out-expo": (0.16, 1.0, 0.3, 1.0),
    "in-out-sine": (0.37, 0.0, 0.63, 1.0),
    "out-back": (0.34, 1.56, 0.64, 1.0),
}

_S = np.linspace(0.0, 1.0, 513)
_B1 = 3 * (1 - _S) ** 2 * _S
_B2 = 3 * (1 - _S) * _S**2
_B3 = _S**3


def _param_at(x1: float, x2: float, tau: np.ndarray) -> np.ndarray:
    return np.interp(tau, _B1 * x1 + _B2 * x2 + _B3, _S)


def bezier_ease(x1: float, y1: float, x2: float, y2: float, tau) -> np.ndarray:
    """CSS cubic-bezier(x1, y1, x2, y2) 在归一化时间 tau 处的进度。"""
    s = _param_at(x1, x2, np.clip(np.asarray(tau, dtype=float), 0.0, 1.0))
    return 3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s**2 * y2 + s**3


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _best_y(x1: float, x2: float, tau: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    s = _param_at(x1, x2, tau)
    b1 = 3 * (1 - s) ** 2 * s
    b2 = 3 * (1 - s) * s**2
    rest = p - s**3
    a11, a12, a22 = float(b1 @ b1) + 1e-9, float(b1 @ b2), float(b2 @ b2) + 1e-9
    r1, r2 = float(b1 @ rest), float(b2 @ rest)
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        y1 = y2 = 0.5
    else:
        y1 = (r1 * a22 - r2 * a12) / det
        y2 = (r2 * a11 - r1 * a12) / det
    y1 = float(np.clip(y1, -1.0, 2.0))
    y2 = float(np.clip(y2, -1.0, 2.0))
    return y1, y2, _rmse(b1 * y1 + b2 * y2 + s**3, p)


def fit_bezier(tau: np.ndarray, p: np.ndarray) -> dict:
    best = (1e9, 0.0, 0.0, 1.0, 1.0)
    center, span, step = (0.5, 0.5), 0.5, 0.05
    for _ in range(3):
        xs1 = np.clip(np.arange(center[0] - span, center[0] + span + 1e-9, step), 0.0, 1.0)
        xs2 = np.clip(np.arange(center[1] - span, center[1] + span + 1e-9, step), 0.0, 1.0)
        for x1 in np.unique(xs1):
            for x2 in np.unique(xs2):
                y1, y2, err = _best_y(float(x1), float(x2), tau, p)
                if err < best[0]:
                    best = (err, float(x1), y1, float(x2), y2)
        center, span, step = (best[1], best[3]), step, step / 5
    err, x1, y1, x2, y2 = best
    params = [round(v, 3) for v in (x1, y1, x2, y2)]
    return {
        "model": "cubic-bezier",
        "params": params,
        "css": "cubic-bezier({}, {}, {}, {})".format(*params),
        "rmse": round(err, 4),
    }


def fit_open(tau: np.ndarray, value: np.ndarray, hidden_start: bool) -> dict | None:
    """一端看不见时的贝塞尔拟合：已知端的值为 0，另一端的幅度未知，一并解出来。

    用在"边移动边淡入"上：透明度为 0 的那几帧位置量不到，但动效的起点恰恰在那里。
    hidden_start=True 表示 value = A·(1 − ease)，起点幅度 A 未知；
    否则 value = A·ease，终点幅度 A 未知。固定 (x1, x2) 后对 (A, A·y1, A·y2) 是线性的。
    返回 {"amplitude", "params", "rmse"}，rmse 已按幅度归一化。
    """
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    value = np.asarray(value, dtype=float)
    if len(value) < 4:
        return None
    limit = 3.0 * float(np.abs(value).max())
    best = None
    center, span, step = (0.5, 0.5), 0.5, 0.05
    for _ in range(2):
        xs1 = np.unique(np.clip(np.arange(center[0] - span, center[0] + span + 1e-9, step), 0.0, 1.0))
        xs2 = np.unique(np.clip(np.arange(center[1] - span, center[1] + span + 1e-9, step), 0.0, 1.0))
        for x1 in xs1:
            for x2 in xs2:
                s = _param_at(float(x1), float(x2), tau)
                b1, b2, b3 = 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s**2, s**3
                cols = np.stack([1 - b3, -b1, -b2] if hidden_start else [b3, b1, b2], axis=1)
                sol, *_ = np.linalg.lstsq(cols, value, rcond=None)
                amp = float(sol[0])
                if abs(amp) < 1e-9 or abs(amp) > limit:
                    continue
                y1, y2 = float(sol[1] / amp), float(sol[2] / amp)
                if not (-0.3 <= y1 <= 1.6 and -0.3 <= y2 <= 1.6):
                    continue
                err = _rmse(cols @ sol, value) / abs(amp)
                if best is None or err < best[0]:
                    best = (err, amp, float(x1), y1, float(x2), y2)
        if best is None:
            return None
        center, span, step = (best[2], best[4]), step, step / 4
    err, amp, x1, y1, x2, y2 = best
    return {"amplitude": amp, "params": [x1, y1, x2, y2], "rmse": err}


def rank_named(t_ms: np.ndarray, p: np.ndarray, start: int, end: int) -> list[dict]:
    """每条命名曲线都允许起点前移、终点后延再比较，并报告对齐后的时长。

    原因：缓入的头、缓出的尾每帧位移远小于 1 像素，实测的起止点会比设计值短一截。
    "out-cubic 用 400ms"和"实测 350ms 的某条贝塞尔"描述的是同一段可见运动，
    而开发者要的是前一种说法。
    """
    n = len(p)
    slack = max(1, int(np.ceil(0.5 * (end - start))))
    lo, hi = max(0, start - slack), min(n - 1, end + slack)
    window_t, window_p = t_ms[lo : hi + 1], p[lo : hi + 1]
    rows = []
    for name, pts in NAMED.items():
        best = None
        for s in range(lo, start + 1):
            for e in range(end, hi + 1):
                span = t_ms[e] - t_ms[s]
                if span <= 0:
                    continue
                err = _rmse(bezier_ease(*pts, (window_t - t_ms[s]) / span), window_p)
                cost = err + 2e-4 * ((start - s) + (e - end))  # 同样贴合时取更短的对齐
                if best is None or cost < best[0]:
                    best = (cost, err, s, e)
        _, err, s, e = best
        rows.append(
            {
                "name": name,
                "rmse": round(err, 4),
                "duration_ms": round(float(t_ms[e] - t_ms[s]), 1),
                "start_shift_ms": round(float(t_ms[s] - t_ms[start]), 1),
            }
        )
    return sorted(rows, key=lambda r: r["rmse"])


def spring_response(zeta, omega, t):
    """欠阻尼弹簧的阶跃响应：从 0 出发，收敛到 1。"""
    wd = omega * np.sqrt(1 - zeta**2)
    return 1 - np.exp(-zeta * omega * t) * (np.cos(wd * t) + (zeta * omega / wd) * np.sin(wd * t))


def fit_spring(t_s: np.ndarray, p: np.ndarray) -> dict:
    zetas = np.linspace(0.05, 0.95, 37)
    omegas = np.geomspace(3.0, 200.0, 90)
    best = (1e9, 0.5, 20.0)
    for _ in range(2):
        z = zetas[:, None, None]
        w = omegas[None, :, None]
        err = np.sqrt(np.mean((spring_response(z, w, t_s[None, None, :]) - p[None, None, :]) ** 2, axis=2))
        i, j = np.unravel_index(int(np.argmin(err)), err.shape)
        if err[i, j] < best[0]:
            best = (float(err[i, j]), float(zetas[i]), float(omegas[j]))
        zetas = np.clip(np.linspace(best[1] - 0.03, best[1] + 0.03, 25), 0.02, 0.98)
        omegas = np.linspace(best[2] * 0.93, best[2] * 1.07, 41)
    err, zeta, omega = best
    return {
        "model": "spring",
        "damping_ratio": round(zeta, 3),
        "angular_frequency": round(omega, 2),
        "stiffness": round(omega**2, 1),  # 质量取 1
        "damping": round(2 * zeta * omega, 2),
        "rmse": round(err, 4),
    }


def fit_easing(t_ms: np.ndarray, p: np.ndarray, start: int, end: int) -> dict | None:
    """t_ms、p：整个取帧窗口的时间和归一化进度（p[start]≈0，p[end]≈1）。

    自由贝塞尔和弹簧只在 [start, end] 内拟合；命名曲线的比较会用到窗口两端的静止帧。
    """
    t_ms = np.asarray(t_ms, dtype=float)
    p = np.asarray(p, dtype=float)
    seg_t = t_ms[start : end + 1] - t_ms[start]
    seg_p = p[start : end + 1]
    if len(seg_p) < 4 or seg_t[-1] <= 0:
        return None
    bezier = fit_bezier(seg_t / seg_t[-1], seg_p)
    overshoot = float(max(0.0, p.max() - 1.0))
    result = dict(bezier)
    if overshoot > 0.03 and len(seg_p) >= 8:
        spring = fit_spring(seg_t / 1000.0, seg_p)
        if spring["rmse"] < bezier["rmse"] * 0.8:
            result = dict(spring)
            result["bezier_alternative"] = bezier
    result["named"] = rank_named(t_ms, p, start, end)
    result["overshoot"] = round(overshoot, 3)
    result["points"] = int(len(seg_p))
    return result


def evaluate(easing: dict, t_ms: np.ndarray) -> np.ndarray:
    """用拟合结果在给定时间点上重算进度，画图和校验用。"""
    t_ms = np.asarray(t_ms, dtype=float)
    if easing["model"] == "spring":
        return spring_response(easing["damping_ratio"], easing["angular_frequency"], t_ms / 1000.0)
    return bezier_ease(*easing["params"], t_ms / max(t_ms[-1], 1e-9))
