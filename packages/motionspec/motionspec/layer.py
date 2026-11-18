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

    def render_box(self, k: int, size: tuple[int, int], color: bool) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """和 render 一样，但只渲染元素落点的外接框（框外本来就是 0）。返回 (框, 图块)。"""
        shift = self.shift[k] if self.valid[k] else self._nearest_shift(k)
        scale = self.scale[k]
        x0, y0, x1, y1 = self.box
        cx, cy = self.center
        left = cx + shift[0] + scale * (x0 - cx)
        top = cy + shift[1] + scale * (y0 - cy)
        right = cx + shift[0] + scale * (x1 - cx)
        bottom = cy + shift[1] + scale * (y1 - cy)
        bx0, by0 = max(0, int(math.floor(left)) - 1), max(0, int(math.floor(top)) - 1)
        bx1, by1 = min(size[0], int(math.ceil(right)) + 2), min(size[1], int(math.ceil(bottom)) + 2)
        if bx1 <= bx0 or by1 <= by0:
            return None
        m = _warp_matrix(self.center, self.origin, shift, scale, (bx0, by0))
        patch = cv2.warpAffine(self.color if color else self.scalar, m, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR)
        return (bx0, by0, bx1, by1), patch

    def _nearest_shift(self, k: int) -> np.ndarray:
        known = np.flatnonzero(self.valid)
        if not len(known):
            return np.zeros(2)
        return self.shift[known[np.argmin(np.abs(known - k))]]


def _edge(profile: np.ndarray) -> tuple[float, float]:
    """一维轮廓上元素两条边的亚像素位置：轮廓从两端向内第一次越过平台值一半的地方（线性插值）。

    整数外接框会丢掉抗锯齿的边缘像素（它们和背景差得少，过不了阈值），每边少半个到一个多像素，
    缩小分析时放大回来就是好几个像素。半高位置对边缘覆盖率是无偏的。
    """
    inside = profile[profile > 0]
    if not len(inside):
        return 0.0, float(len(profile))
    half = 0.5 * float(np.median(inside))
    above = np.flatnonzero(profile >= half)
    a, b = int(above[0]), int(above[-1])
    left = float(a)
    if a > 0:
        lo, hi = profile[a - 1], profile[a]
        left = a - (hi - half) / (hi - lo) if hi > lo else float(a)
    right = float(b + 1)
    if b + 1 < len(profile):
        hi, lo = profile[b], profile[b + 1]
        right = b + 1 - (half - lo) / (hi - lo) if hi > lo else float(b + 1)
    return left, right


def _extent(template: np.ndarray) -> tuple[float, float]:
    """元素模板（E − B 在主颜色方向上的投影）的宽和高，亚像素。"""
    magnitude = np.abs(template)
    left, right = _edge(magnitude.sum(axis=0))
    top, bottom = _edge(magnitude.sum(axis=1))
    return right - left, bottom - top


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
    last_k = None
    for k in order:
        if tried >= 6 and int(el.valid.sum()) < 3:
            return  # 有信号的前几帧都对不上：这不是进入/退出的图层，不再往下算
        # 和上一帧完全一样（动效前后的静止帧）：结果也一样，不用再搜
        if last_k is not None and el.valid[last_k] and np.array_equal(frames[k], frames[last_k]):
            el.shift[k], el.scale[k], el.corr[k], el.valid[k] = el.shift[last_k], el.scale[last_k], el.corr[last_k], True
            last_step = 0.0
            last_k = k
            continue
        last_k = k
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

        def descend(best_p, best, axes):
            steps = [0.5, 0.5, 0.01]
            while steps[0] > 0.02:
                improved = False
                for i in axes:
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
            return best_p, best

        # 先只搜位移；缩放差一点就能明显更贴合时才把缩放也放进来（大多数进入/退出不缩放）
        best_p, best = descend(best_p, best, (0, 1))
        if prev_scale != 1.0 or max(corr([best_p[0], best_p[1], best_p[2] + d]) for d in (0.01, -0.01)) > best + 2e-5:
            best_p, best = descend(best_p, best, (0, 1, 2))
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
    # 和整窗最小二乘完全等价：每个元素只在自己的落点框里非零，所以内积只需要在框（和框的交集）里算，
    # 残差 = |t|² - 2a·b + aᵀGa，不必把整窗的基底拼出来
    n, m = len(colors), len(elements)
    residual = np.zeros(n)
    for k in range(n):
        target = colors[k]
        flat = target.reshape(-1)
        power = float(np.dot(flat, flat))
        pieces = [el.render_box(k, size, True) for el in elements]
        gram = 1e-6 * np.eye(m)
        rhs = np.zeros(m)
        for i, piece in enumerate(pieces):
            if piece is None:
                continue
            (x0, y0, x1, y1), patch = piece
            a = patch.reshape(-1)
            gram[i, i] += float(np.dot(a, a))
            rhs[i] = float(np.dot(a, target[y0:y1, x0:x1].reshape(-1)))
            for j in range(i + 1, m):
                other = pieces[j]
                if other is None:
                    continue
                (u0, v0, u1, v1), patch_j = other
                ix0, iy0, ix1, iy1 = max(x0, u0), max(y0, v0), min(x1, u1), min(y1, v1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                cross = float(np.dot(patch[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0].reshape(-1),
                                     patch_j[iy0 - v0 : iy1 - v0, ix0 - u0 : ix1 - u0].reshape(-1)))
                gram[i, j] += cross
                gram[j, i] += cross
        alphas = np.linalg.solve(gram, rhs)
        for el, a in zip(elements, alphas):
            el.alpha[k] = float(a)
        if power > 0:
            left = power - 2 * float(alphas @ rhs) + float(alphas @ gram @ alphas)
            residual[k] = max(0.0, left) / power
    return residual


def _contains(outer: _Element, inner: _Element) -> bool:
    return outer.box[0] <= inner.box[0] and outer.box[1] <= inner.box[1] and outer.box[2] >= inner.box[2] and outer.box[3] >= inner.box[3]


def _same_motion(a: _Element, b: _Element) -> bool:
    both = a.valid & b.valid
    # 一个元素的框完全包住另一个：多半是同一个元素被拆成了外框和内容。外框那块在元素还很淡时
    # 常常跟丢、透明度也受背景影响，所以只看两者都测到的帧上位移是否逐帧一致
    nested = _contains(a, b) or _contains(b, a)
    if both.sum() < 3 or (not nested and abs(int(a.valid.sum()) - int(b.valid.sum())) > 2):
        return False
    if not (float(np.abs(a.shift[both] - b.shift[both]).max()) < 1.0
            and float(np.abs(a.scale[both] - b.scale[both]).max()) < 0.015):
        return False
    moving = both & (np.abs(a.shift).max(axis=1) >= 1.0)
    if nested and int(both.sum()) >= 6 and int(moving.sum()) >= 3:
        return True
    return float(np.abs(a.alpha - b.alpha).max()) < 0.12


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
    if float(np.sort(plain)[-3:].mean()) < 0.06:
        return None

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
    elements = [el for el in elements if el.valid.sum() >= 3]
    if not elements:
        return None
    first_res = _solve_alpha(elements, frames_color, size)
    first_seen = np.any([el.valid for el in elements], axis=0)
    first_corr = float(np.median(np.concatenate([el.corr[el.valid] for el in elements])))
    if first_corr < 0.7 or float(first_res[first_seen].mean()) > 0.45:
        return None
    if not any(np.abs(el.shift[el.valid]).max() >= 1.0 or np.abs(el.scale[el.valid] - 1).max() >= 0.012 for el in elements):
        return None
    grays = [cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) for image in (back, full)]
    if any(_preexists(el, grays[0], grays[1], support) for el in elements) or _displaced(elements, grays[0], grays[1], support):
        return None

    def merge(elements):
        """运动一致的并成一个元素（没有底板的多行文字会被连通域拆散）。"""
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
        return elements

    elements = merge(elements)
    # 多个元素时再做一轮：减掉其他元素的重建之后重新定位，消除相互干扰
    if len(elements) > 1:
        for el in elements:
            others = [o for o in elements if o is not el]
            cleaned = [
                (frames[k] - sum(o.alpha[k] * o.render(k, size, False) for o in others)).astype(np.float32)
                for k in range(n)
            ]
            _fit_element(el, cleaned, order)
        _solve_alpha(elements, frames_color, size)
        # 重新定位后，原先在元素很淡时跟丢的外框可能已经测到了位移，再并一次
        elements = merge(elements)
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


def _displaced(elements: list[_Element], back: np.ndarray, full: np.ndarray, support: np.ndarray) -> bool:
    """整片变化区域能否由背景帧里挪过来的同一块画面解释。

    _preexists 靠纹理匹配，纯色卡片（低纹理）会漏掉：它整体上移时，变化区域被拆成上下两条，
    各自看起来像"一块新出现的色块"。这里把所有元素的掩码合在一起，在背景帧里按位移搜索：
    如果某个位移处的背景帧画面与"完整显示"帧在掩码内几乎一致，而完整显示帧在同一位移处已经不是这样，
    说明元素本来就在背景帧的那个位置，后来挪走了。外观相同的静止项在两帧里都在原处，不会被算进来。
    """
    mask = np.zeros(back.shape, bool)
    reach = 8.0
    for el in elements:
        mask |= el.mask
        ok = el.valid
        if ok.any():
            reach = max(reach, float(np.abs(el.shift[ok]).max()))
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return False
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    weight = mask[y0:y1, x0:x1].astype(np.float32)
    template = full[y0:y1, x0:x1].astype(np.float32)
    energy = float((((template - back[y0:y1, x0:x1].astype(np.float32)) ** 2) * weight).sum())
    if energy <= 0:
        return False
    radius = int(min(1.6 * reach + 16, 0.6 * max(back.shape)))

    def errors(image):
        padded = cv2.copyMakeBorder(image.astype(np.float32), radius, radius, radius, radius, cv2.BORDER_REPLICATE)
        return cv2.matchTemplate(padded[y0 : y1 + 2 * radius, x0 : x1 + 2 * radius], template, cv2.TM_SQDIFF, mask=weight) / energy

    in_back, in_full = errors(back), errors(full)
    in_back[radius - 2 : radius + 3, radius - 2 : radius + 3] = np.inf  # 不位移的解释就是"淡入"本身
    for index in np.argsort(in_back, axis=None)[:5]:
        dy, dx = divmod(int(index), in_back.shape[1])
        if in_back[dy, dx] >= 0.15:
            break
        if in_full[dy, dx] >= 0.3:
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
    if _displaced(best["elements"], grays[0], grays[1], support):
        return None
    for el in best["elements"]:
        width, height = _extent(el.scalar)
        el.size = (width / factor, height / factor)
    if factor != 1.0:
        for el in best["elements"]:
            el.shift /= factor
            el.center = (el.center[0] / factor, el.center[1] / factor)
            el.box = tuple(int(round(v / factor)) for v in el.box)
    return best
