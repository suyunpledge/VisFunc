"""图层模型：处理"进入 / 退出"类动效，即元素一边移动或缩放、一边改变透明度。

光流在这类动效上不可靠：元素还很淡的时候特征点跟不上，前半段位移会丢。
这里换一个思路。取首尾两帧，一帧里没有这个元素（背景 B），另一帧里元素完整显示（E）。
那么中间每一帧都应当满足

    帧_k = B + a_k · 变换_k(E − B)

a_k 是透明度，变换_k 是平移加等比缩放。对每一帧求出让这个等式最贴合的 (平移, 缩放, a)，
就同时得到三条曲线。归一化相关对整体增益不敏感，所以元素很淡时位置照样量得准。

同一窗口里可以有多个元素（依次进入的列表项）：先按 E − B 的连通域拆开分别求解，
再把运动一致的并回去，所以拆分依据是"动得一不一样"，不是"形状连不连着"。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SEARCH = 64  # 单帧内允许的最大位移（分析尺度像素）
MAX_ELEMENTS = 6
WORK_AREA = 80_000  # 求解时的目标像素数
MIN_ALPHA = 0.04  # 低于这个透明度就认为看不见，位置不可测
MIN_CORR = 0.5


def _principal(diff: np.ndarray, support: np.ndarray) -> np.ndarray:
    """E − B 的主颜色方向。投影到它上面得到带符号的标量图，等亮度的颜色变化也不会丢。"""
    px = diff[support].astype(np.float64)
    _, vectors = np.linalg.eigh(px.T @ px)
    return vectors[:, -1].astype(np.float32)


def _warp_matrix(center, origin, shift, scale, window_origin=(0.0, 0.0)) -> np.ndarray:
    """把模板（左上角在 origin，绕 center 缩放）放到平移 shift 之后的位置，输出到 window 坐标。"""
    tx = center[0] + shift[0] - window_origin[0] - scale * (center[0] - origin[0])
    ty = center[1] + shift[1] - window_origin[1] - scale * (center[1] - origin[1])
    return np.array([[scale, 0, tx], [0, scale, ty]], np.float32)


class _Element:
    def __init__(self, mask: np.ndarray, scalar: np.ndarray, color: np.ndarray, n: int):
        ys, xs = np.nonzero(mask)
        self.mask = mask
        self.box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        x0, y0, x1, y1 = self.box
        self.origin = (x0, y0)
        self.center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        keep = mask[y0:y1, x0:x1, None].astype(np.float32)
        self.scalar = scalar[y0:y1, x0:x1] * keep[..., 0]
        self.color = color[y0:y1, x0:x1] * keep
        self.energy = float((self.scalar**2).sum())
        self.shift = np.full((n, 2), np.nan)
        self.scale = np.ones(n)
        self.alpha = np.zeros(n)
        self.corr = np.zeros(n)
        self.valid = np.zeros(n, bool)

    def render(self, k: int, size: tuple[int, int], color: bool) -> np.ndarray:
        shift = self.shift[k] if self.valid[k] else self._nearest_shift(k)
        m = _warp_matrix(self.center, self.origin, shift, self.scale[k])
        return cv2.warpAffine(self.color if color else self.scalar, m, size, flags=cv2.INTER_LINEAR)

    def _nearest_shift(self, k: int) -> np.ndarray:
        known = np.flatnonzero(self.valid)
        if not len(known):
            return np.zeros(2)
        return self.shift[known[np.argmin(np.abs(known - k))]]


def _fit_element(el: _Element, frames: list[np.ndarray], order: list[int]) -> None:
    """沿 order（从元素完整的一端走向元素消失的一端）逐帧求平移和缩放。"""
    x0, y0, x1, y1 = el.box
    th, tw = el.scalar.shape
    pad = SEARCH + max(th, tw)
    margin = 4 + int(math.ceil(0.08 * max(th, tw)))
    prev = np.zeros(2)
    prev_scale = 1.0
    last_step = 0.0
    tried = 0
    el.valid[:] = False
    for k in order:
        if tried >= 6 and int(el.valid.sum()) < 3:
            return  # 有信号的前几帧都对不上：这不是进入/退出的图层，不再往下算
        padded = cv2.copyMakeBorder(frames[k], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        # 粗定位：在上一帧位置附近做归一化互相关。搜索半径跟着上一步的速度走，
        # 否则列表里外观相同的相邻项会被当成"这个元素跳过去了"
        radius = int(min(SEARCH, max(24.0, 2.5 * last_step + 8.0)))
        ax, ay = x0 + pad + int(round(prev[0])), y0 + pad + int(round(prev[1]))
        rx0, ry0 = max(0, ax - radius), max(0, ay - radius)
        rx1, ry1 = min(padded.shape[1], ax + tw + radius), min(padded.shape[0], ay + th + radius)
        if rx1 - rx0 < tw or ry1 - ry0 < th:
            continue
        region = padded[ry0:ry1, rx0:rx1]
        if float((region**2).sum()) < (MIN_ALPHA**2) * el.energy:
            continue
        score = cv2.matchTemplate(region, el.scalar, cv2.TM_CCORR_NORMED)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        gy, gx = np.mgrid[0 : score.shape[0], 0 : score.shape[1]]
        away = np.hypot(gx - (ax - rx0), gy - (ay - ry0)) / radius
        _, _, _, where = cv2.minMaxLoc(score - 0.05 * away.astype(np.float32))  # 同样贴合时取离上一帧近的
        peak = float(score[where[1], where[0]])
        if peak < MIN_CORR * 0.6:
            continue
        coarse = np.array([rx0 + where[0] - x0 - pad, ry0 + where[1] - y0 - pad], dtype=float)

        # 细化：在 (dx, dy, 缩放) 上做坐标下降，直接最大化归一化相关
        wx0 = int(math.floor(x0 + coarse[0] - margin))
        wy0 = int(math.floor(y0 + coarse[1] - margin))
        ww, wh = tw + 2 * margin, th + 2 * margin
        window = padded[wy0 + pad : wy0 + pad + wh, wx0 + pad : wx0 + pad + ww]
        if window.shape != (wh, ww):
            continue
        norm = float((window**2).sum())
        if norm < (MIN_ALPHA**2) * el.energy:
            continue
        tried += 1

        def corr(p, window=window, norm=norm, wx0=wx0, wy0=wy0, ww=ww, wh=wh):
            m = _warp_matrix(el.center, el.origin, (p[0], p[1]), p[2], (wx0, wy0))
            warped = cv2.warpAffine(el.scalar, m, (ww, wh), flags=cv2.INTER_LINEAR)
            denom = math.sqrt(float((warped**2).sum()) * norm)
            return float((warped * window).sum()) / denom if denom > 0 else 0.0

        best_p = [coarse[0], coarse[1], prev_scale]
        best = corr(best_p)
        steps = [0.5, 0.5, 0.01]
        while steps[0] > 0.02:
            improved = False
            for i in range(3):
                for sign in (1.0, -1.0):
                    trial = list(best_p)
                    trial[i] += sign * steps[i]
                    if not 0.3 <= trial[2] <= 3.0:
                        continue
                    value = corr(trial)
                    if value > best + 2e-5:
                        best_p, best, improved = trial, value, True
                        break
            if not improved:
                steps = [s / 2 for s in steps]
        if best < MIN_CORR:
            continue
        el.shift[k] = best_p[:2]
        el.scale[k] = best_p[2]
        el.corr[k] = best
        el.valid[k] = True
        last_step = float(np.hypot(best_p[0] - prev[0], best_p[1] - prev[1]))
        prev, prev_scale = np.array(best_p[:2]), best_p[2]


def _solve_alpha(elements: list[_Element], colors: list[np.ndarray], size) -> np.ndarray:
    """每帧对所有元素的透明度做联合最小二乘，返回每帧的残差占该帧变化量的比例。"""
    n = len(colors)
    residual = np.zeros(n)
    for k in range(n):
        layers = [el.render(k, size, True) for el in elements]
        basis = np.stack([layer.ravel() for layer in layers], axis=1).astype(np.float64)
        target = colors[k].ravel().astype(np.float64)
        gram = basis.T @ basis + 1e-6 * np.eye(len(elements))
        alphas = np.linalg.solve(gram, basis.T @ target)
        for el, a in zip(elements, alphas):
            el.alpha[k] = float(a)
        power = float((target**2).sum())
        residual[k] = float(((basis @ alphas - target) ** 2).sum()) / power if power > 0 else 0.0
    return residual


def _same_motion(a: _Element, b: _Element) -> bool:
    both = a.valid & b.valid
    if both.sum() < 3 or abs(int(a.valid.sum()) - int(b.valid.sum())) > 2:
        return False
    return (
        float(np.abs(a.shift[both] - b.shift[both]).max()) < 1.0
        and float(np.abs(a.scale[both] - b.scale[both]).max()) < 0.015
        and float(np.abs(a.alpha - b.alpha).max()) < 0.12
    )


def _fit_direction(first: np.ndarray, last: np.ndarray, crops: list[np.ndarray], enter: bool, threshold: int):
    n = len(crops)
    back, full = (first, last) if enter else (last, first)
    diff = full - back
    support = np.abs(diff).max(axis=2) > threshold
    if int(support.sum()) < 30:
        return None
    direction = _principal(diff, support)
    scalar = diff @ direction
    frames_color = [c - back for c in crops]
    frames = [f @ direction for f in frames_color]
    size = (diff.shape[1], diff.shape[0])
    order = list(range(n - 1, -1, -1)) if enter else list(range(n))
    # 不考虑运动的基准：每帧只是 E − B 的整体缩放（纯淡入淡出）
    energy = float((diff**2).sum())
    plain = np.zeros(n)
    for k, f in enumerate(frames_color):
        power = float((f**2).sum())
        if power > 0:
            plain[k] = float(((f - (float((f * diff).sum()) / energy) * diff) ** 2).sum()) / power

    grown = cv2.dilate(support.astype(np.uint8), np.ones((7, 7), np.uint8))
    count, labels = cv2.connectedComponents(grown, connectivity=8)
    parts = []
    for label in range(1, count):
        mask = (labels == label) & cv2.dilate(support.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if int((mask & support).sum()) >= 30:
            parts.append(mask)
    if not parts:
        return None
    parts.sort(key=lambda m: float((scalar[m] ** 2).sum()), reverse=True)
    elements = [_Element(mask, scalar, diff, n) for mask in parts[:MAX_ELEMENTS]]
    for el in elements:
        _fit_element(el, frames, order)
    _solve_alpha(elements, frames_color, size)

    # 运动一致的并成一个元素（没有底板的多行文字会被连通域拆散）
    merged = True
    while merged and len(elements) > 1:
        merged = False
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if _same_motion(elements[i], elements[j]):
                    union = _Element(elements[i].mask | elements[j].mask, scalar, diff, n)
                    _fit_element(union, frames, order)
                    elements = [e for t, e in enumerate(elements) if t not in (i, j)] + [union]
                    _solve_alpha(elements, frames_color, size)
                    merged = True
                    break
            if merged:
                break

    # 多个元素时再做一轮：减掉其他元素的重建之后重新定位，消除相互干扰
    if len(elements) > 1:
        for el in elements:
            others = [o for o in elements if o is not el]
            cleaned = [
                (frames[k] - sum(o.alpha[k] * o.render(k, size, False) for o in others)).astype(np.float32)
                for k in range(n)
            ]
            _fit_element(el, cleaned, order)
    residual = _solve_alpha(elements, frames_color, size)
    elements = [el for el in elements if el.valid.sum() >= 3]
    if not elements:
        return None
    # 只在看得见元素的帧上比较两种解释
    seen = np.any([el.valid for el in elements], axis=0)
    # 运动带来的改善集中在少数几帧（位移最大时元素往往最淡），所以看改善最大的几帧而不是平均
    gain = float(np.sort((plain - residual)[seen])[-3:].mean())
    return {"enter": enter, "elements": elements, "gain": gain,
            "residual": float(residual[seen].mean()), "plain": float(plain[seen].mean())}


def _preexists(el: _Element, back: np.ndarray, full: np.ndarray, support: np.ndarray) -> bool:
    """元素是不是本来就在背景帧里，只是挪了位置或换了大小。

    图层模型只对"背景帧里没有这个元素"成立。一张不透明卡片从 A 滑到 B，也能被硬解成
    "B 处进入一个元素、A 处有个反向元素"，数值上贴合但没有意义。
    判据：在背景帧里按多个缩放比例找这个元素；找到了，并且那个位置后来确实变了，就是挪动。
    只找到一个没变过的相同元素不算（列表里外观相同的静止项）。
    """
    x0, y0, x1, y1 = el.box
    template = full[y0:y1, x0:x1]
    if float(template.std()) < 6:
        return False
    th, tw = template.shape
    for scale in np.arange(0.5, 1.51, 0.05):
        size = (int(round(tw * scale)), int(round(th * scale)))
        if min(size) < 8 or size[0] > back.shape[1] or size[1] > back.shape[0]:
            continue
        resized = cv2.resize(template, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        score = np.nan_to_num(cv2.matchTemplate(back, resized, cv2.TM_CCOEFF_NORMED), nan=0.0, posinf=0.0, neginf=0.0)
        _, peak, _, where = cv2.minMaxLoc(score)
        if peak >= 0.8 and float(support[where[1] : where[1] + size[1], where[0] : where[0] + size[0]].mean()) >= 0.15:
            return True
    return False


def fit_layers(crops: list[np.ndarray], threshold: int) -> dict | None:
    """尝试用图层模型解释整个窗口。解释不了（或不需要）时返回 None。

    返回 {"enter": bool, "residual": 模型残差占比, "plain": 纯淡入淡出的残差占比, "elements": [...]}，
    元素的 shift 是相对"完整显示那一帧"的位移，单位为传入帧的像素。
    """
    if len(crops) < 5:
        return None
    h, w = crops[0].shape[:2]
    # 开销随面积线性增长，大区域先降采样；位置精度相应变粗，但大元素本来也不需要亚像素
    factor = float(np.clip(math.sqrt(WORK_AREA / (h * w)), 0.25, 1.0))
    if factor != 1.0:
        size = (max(8, round(w * factor)), max(8, round(h * factor)))
        crops = [cv2.resize(c, size, interpolation=cv2.INTER_AREA) for c in crops]
    floats = [c.astype(np.float32) for c in crops]
    best = None
    for enter in (True, False):
        fit = _fit_direction(floats[0], floats[-1], floats, enter, threshold)
        if fit is not None and (best is None or fit["residual"] - fit["gain"] < best["residual"] - best["gain"]):
            best = fit
    if best is None:
        return None

    moved = False
    for el in best["elements"]:
        ok = el.valid
        if np.abs(el.scale[ok] - 1).max() < 0.012:
            el.scale[:] = 1.0
        if np.abs(el.shift[ok]).max() >= 1.0 or np.abs(el.scale[ok] - 1).max() >= 0.012:
            moved = True
    quality = float(np.median(np.concatenate([el.corr[el.valid] for el in best["elements"]])))
    best["correlation"] = quality
    # 接受条件：模型贴合、确实有运动、并且比"纯淡入淡出"解释得明显更好
    if not moved or best["residual"] > 0.2 or quality < 0.85 or best["gain"] < 0.06:
        return None
    back, full = (floats[0], floats[-1]) if best["enter"] else (floats[-1], floats[0])
    support = np.abs(full - back).max(axis=2) > threshold
    grays = [cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) for image in (back, full)]
    if any(_preexists(el, grays[0], grays[1], support) for el in best["elements"]):
        return None
    if factor != 1.0:
        for el in best["elements"]:
            el.shift /= factor
            el.center = (el.center[0] / factor, el.center[1] / factor)
            el.box = tuple(int(round(v / factor)) for v in el.box)
    return best
