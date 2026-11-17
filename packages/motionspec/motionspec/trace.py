"""Qt 探针（离线的那一半）：把 qt_probe 录下的轨迹转成和视觉路径相同的规格。

轨迹里是控件属性的真实取值，不经过像素反推，所以位移、透明度没有测量误差，
时间精度就是取样间隔。缓动曲线仍然是拟合出来的（探针看不到动画对象本身），
但拟合用的是精确的进度值。

    from motionspec.trace import spec_from_trace
    spec = spec_from_trace(json.load(open("trace.json", encoding="utf-8")), "out/")
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .motion import _fit_channel, _r
from .spec import SCHEMA, summarize, validate
from .trace_contract import normalize_trace
from .timing import with_timing

MERGE_TICKS = 3  # Convert nominal frame intervals to milliseconds before merging.
X, Y, W, H, OPACITY, SCALE, ROTATION, VISIBLE, SCENE_X, SCENE_Y = range(10)


def _runs(rows: list[list[float]], ticks: np.ndarray) -> list[np.ndarray]:
    """把稀疏的行切成一段段连续变化，并按帧补齐（缺的帧沿用上一行的值）。"""
    last_tick = len(ticks) - 1
    merge_ms = MERGE_TICKS * float(np.median(np.diff(ticks)))
    out, current = [], [rows[0]]
    for row in rows[1:]:
        elapsed = ticks[int(row[0])] - ticks[int(current[-1][0])]
        if elapsed <= merge_ms + 1e-6:
            current.append(row)
        else:
            out.append(current)
            current = [row]
    out.append(current)
    dense = []
    for i, run in enumerate(out):
        if len(run) < 2:
            continue
        # 两端各补一段静止帧。整数像素的位置在缓入、缓出的头尾几帧不会变，
        # 留出余量，命名曲线匹配时才能把起点前移、终点后延，还原设计时长
        pad = max(6, (int(run[-1][0]) - int(run[0][0])) // 2)
        floor = int(out[i - 1][-1][0]) + 1 if i else 0
        ceiling = int(out[i + 1][0][0]) - 1 if i + 1 < len(out) else last_tick
        first, last = max(floor, int(run[0][0]) - pad), min(ceiling, int(run[-1][0]) + pad)
        table = np.zeros((last - first + 1, 11))
        cursor = 0
        for tick in range(first, last + 1):
            while cursor + 1 < len(run) and run[cursor + 1][0] <= tick:
                cursor += 1
            table[tick - first] = [tick, *run[cursor][1:]]
        dense.append(table)
    return dense


def _channel(t: np.ndarray, value: np.ndarray, eps: float) -> dict:
    peak = float(np.abs(value).max())
    returns = abs(float(value[-1])) < 0.3 * peak
    start, end, top, progress, easing = _fit_channel(t, value, eps, returns)
    data = {"t_start_ms": _r(t[start], 3), "t_end_ms": _r(t[end], 3), "easing": easing}
    data["duration_ms"] = _r(data["t_end_ms"] - data["t_start_ms"], 3)
    if top is not None:
        data.update(pattern="out-and-back", peak_ms=_r(t[top] - t[start], 1))
    # 属性值是精确的：起点取片段第一帧（变化前的静止值），终点取最后一帧（稳定后的值）
    return {"data": data, "start": start, "end": end, "peak": top, "progress": progress,
            "origin": 0, "target": len(value) - 1 if top is None else top}


def _segment(element: dict, table: np.ndarray, ticks: np.ndarray, ratio: float, size: tuple[int, int]) -> dict | None:
    t = ticks[table[:, 0].astype(int)]
    v = table[:, 1:]
    width, height = v[:, W] * v[:, SCALE] * ratio, v[:, H] * v[:, SCALE] * ratio
    centre = np.stack([v[:, SCENE_X] * ratio + width / 2, v[:, SCENE_Y] * ratio + height / 2], axis=1)
    alpha = v[:, OPACITY] * v[:, VISIBLE]
    channels: dict = {}
    fitted: dict[str, dict] = {}
    effect: dict[str, float] = {}

    offset = centre - centre[0]
    reach = float(np.linalg.norm(offset, axis=1).max())
    # Classify translation from the item's position properties. A changing size
    # or scale can move its centre without an independent translation. Keep the
    # real scaled centre above for samples, endpoints and region geometry.
    position = v[:, [X, Y]] * ratio
    position_reach = float(np.linalg.norm(position - position[0], axis=1).max())
    if reach >= 0.75 and position_reach >= 0.75:
        far = offset[int(np.argmax(np.linalg.norm(offset, axis=1)))]
        net = offset[-1]
        target = net if np.linalg.norm(net) >= 0.3 * reach else far
        value = offset @ (target / np.linalg.norm(target))
        fit = _channel(t, value, float(np.clip(0.25 / reach, 0.001, 0.05)))
        a, b = centre[fit["origin"]], centre[fit["target"]]
        fit["data"].update({"from": [_r(a[0], 1), _r(a[1], 1)], "to": [_r(b[0], 1), _r(b[1], 1)],
                            "delta": [_r(b[0] - a[0], 1), _r(b[1] - a[1], 1)],
                            "distance_px": _r(np.linalg.norm(b - a), 1), "reference": "element_center"})
        channels["translate"], fitted["translate"], effect["translate"] = fit["data"], fit, reach

    own_w, own_h = v[:, W] * ratio, v[:, H] * ratio
    grow_w, grow_h = float(np.abs(own_w - own_w[0]).max()), float(np.abs(own_h - own_h[0]).max())
    if max(grow_w, grow_h) >= 0.75 and own_w[0] > 0 and own_h[0] > 0:
        rw, rh = own_w / own_w[0], own_h / own_h[0]
        if float(np.abs(rw - rh).max()) < 0.02:  # 等比例变化就是缩放
            fit = _channel(t, rw - 1, 0.004)
            fit["data"].update({"from": 1.0, "to": _r(rw[fit["target"]], 3), "peak": _r(rw.max(), 3), "reference": "size"})
            channels["scale"], fitted["scale"] = fit["data"], fit
        else:
            lead = own_w if grow_w >= grow_h else own_h
            fit = _channel(t, lead - lead[0], float(np.clip(0.25 / max(grow_w, grow_h), 0.001, 0.05)))
            k = fit["target"]
            fit["data"].update({"width_from": _r(own_w[fit["origin"]], 1), "width_to": _r(own_w[k], 1),
                                "height_from": _r(own_h[fit["origin"]], 1), "height_to": _r(own_h[k], 1)})
            channels["size"], fitted["size"] = fit["data"], fit
        effect["scale" if "scale" in channels else "size"] = max(grow_w, grow_h)

    scale = v[:, SCALE]
    if "scale" not in channels and float(np.abs(scale - scale[0]).max()) >= 0.005 and float(scale.max()) > 0:
        fit = _channel(t, scale - scale[0], 0.004)
        fit["data"].update({"from": _r(scale[fit["origin"]], 3), "to": _r(scale[fit["target"]], 3),
                            "peak": _r(scale.max(), 3), "reference": "scale_property"})
        channels["scale"], fitted["scale"] = fit["data"], fit
        effect["scale"] = float(np.abs(scale - scale[0]).max()) * float(max(own_w[0], own_h[0])) / 2

    rotation = v[:, ROTATION]
    if float(np.abs(rotation - rotation[0]).max()) >= 0.5:
        fit = _channel(t, rotation - rotation[0], 0.004)
        fit["data"].update(from_deg=_r(rotation[fit["origin"]], 1), to_deg=_r(rotation[fit["target"]], 1))
        channels["rotate"], fitted["rotate"] = fit["data"], fit
        effect["rotate"] = math.radians(float(np.abs(rotation - rotation[0]).max())) * float(max(own_w[0], own_h[0])) / 2

    if float(np.abs(alpha - alpha[0]).max()) >= 0.02:
        fit = _channel(t, alpha - alpha[0], 0.004)
        fit["data"].update({"from": _r(alpha[fit["origin"]], 3), "to": _r(alpha[fit["target"]], 3), "reference": "opacity_property"})
        channels["opacity"], fitted["opacity"] = fit["data"], fit
        effect.setdefault("opacity", 0.0)

    if not channels:
        return None
    dominant = max(effect, key=effect.get)
    lead = fitted[dominant]
    start, end = min(f["start"] for f in fitted.values()), max(f["end"] for f in fitted.values())
    # Preserve the recorded origin directly: subtracting the half-size back
    # from the centre can round 100 to 99.99999999999999 before floor().
    left, top = v[:, SCENE_X] * ratio, v[:, SCENE_Y] * ratio
    region = [max(0, int(math.floor(left.min()))), max(0, int(math.floor(top.min()))),
              min(size[0], int(math.ceil((left + width).max()))), min(size[1], int(math.ceil((top + height).max())))]
    if region[2] <= region[0] or region[3] <= region[1]:
        return None  # 整个在窗口外
    samples = []
    for k in np.unique(np.linspace(start, end, min(end - start + 1, 40)).round().astype(int)):
        samples.append({"t_ms": _r(t[k], 3), "progress": _r(lead["progress"][k], 3), "x": _r(centre[k][0], 1),
                        "y": _r(centre[k][1], 1), "opacity": _r(alpha[k], 3), "scale": _r(scale[k], 3),
                        "channel_progress": {name: _r(f["progress"][k], 3) for name, f in fitted.items()}})
    t_start, t_end = _r(t[start], 3), _r(t[end], 3)
    seg = {
        "kind": "fade" if set(channels) == {"opacity"} else "transform",
        "model": "qt-probe",
        "confidence": "high",
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "duration_ms": _r(t_end - t_start, 3),
        "frames": [int(table[start, 0]), int(table[end, 0])],
        "region": region,
        "element": {k: element[k] for k in ("class", "name", "text", "path") if element.get(k)},
        "dominant": dominant,
        "channels": channels,
        "easing": lead["data"]["easing"],
        "samples": samples,
        "truncated": False,
        "notes": ["read from live Qt properties, not from pixels; values are exact, the easing curve is fitted"],
        "_effect": max(effect.values()) if effect else 0.0,
    }
    if lead["peak"] is not None:
        seg.update(pattern="out-and-back", peak_ms=lead["data"]["peak_ms"])
    return seg


def spec_from_trace(trace: dict, out_dir: str | Path | None = None, *, max_segments: int = 40) -> dict:
    if type(max_segments) is not int or not 0 <= max_segments <= 4096:
        raise ValueError("max_segments must be an integer between 0 and 4096")
    trace = normalize_trace(trace)
    ticks = np.asarray(trace["ticks_ms"], dtype=float)
    if len(ticks) < 2:
        raise ValueError("trace has fewer than two samples")
    ratio = float(trace.get("device_pixel_ratio") or 1.0)
    size = (max(1, round(trace["size"][0] * ratio)), max(1, round(trace["size"][1] * ratio)))
    found = []
    for element in trace["elements"]:
        for table in _runs(element["rows"], ticks):
            seg = _segment(element, table, ticks, ratio, size)
            if seg is not None:
                found.append(seg)
    found.sort(key=lambda s: -s["_effect"])
    dropped = max(0, len(found) - max_segments)
    kept = sorted(found[:max_segments], key=lambda s: (s["t_start_ms"], s["region"][1], s["region"][0]))
    for number, seg in enumerate(kept, start=1):
        seg.pop("_effect")
        kept[number - 1] = {"id": f"s{number}", **seg}
    warnings = []
    if dropped:
        warnings.append(f"{dropped} smaller property change(s) were dropped (max_segments)")
    if trace.get("truncated"):
        warnings.append("the recorder hit its element limit; part of the interface was not observed")
    step = float(np.median(np.diff(ticks)))
    spec = {
        "schema": SCHEMA,
        "source": {
            "path": "qt-probe:" + (trace.get("title") or ""),
            "width": size[0],
            "height": size[1],
            "fps_nominal": round(1000.0 / step, 3) if step > 0 else 60.0,
            "frames_analysed": int(len(ticks)),
            "t_start_ms": round(float(ticks[0]), 3),
            "t_end_ms": round(float(ticks[-1]), 3),
            "duration_ms": round(float(ticks[-1]) - float(ticks[0]), 3),
            "timestamps": "probe",
            "analysis_scale": 1.0,
            "device_pixel_ratio": ratio,
        },
        "time_resolution_ms": step,
        "units": {"position": "device pixels in the window, origin top-left", "time": "ms on the probe clock"},
        "coverage": {"dropped_candidates": dropped, "truncated_segments": 0, "ambient_regions": 0},
        "ambient": [],
        "segments": kept,
        "lifecycle": [
            {"element_id": element["id"], "kind": event["kind"],
             "timestamp_ms": float(ticks[event["tick"]])}
            for element in trace["elements"] for event in element.get("events", [])
        ],
        "warnings": warnings,
    }
    spec = with_timing(spec)
    problems = validate(spec)
    if problems:
        raise ValueError("generated invalid motion spec: " + "; ".join(problems))
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (out / "summary.md").write_text(summarize(spec), encoding="utf-8")
    return spec
