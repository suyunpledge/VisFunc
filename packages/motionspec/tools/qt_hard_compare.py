"""给 qt_hard_scenes.py 的场景自动判分：视觉路径和探针，各自对真值逐项比较。

    python tools/qt_hard_compare.py out/qt-hard [--only 场景,场景] [--no-probe]

判分规则：真值片段按区域重叠和时间重叠找最匹配的实测片段；
起点 ±34ms、时长 ±15%（至少 17ms）、位移 ±10%（至少 3px）、尺寸 ±10%（至少 3px）、缓动族一致。
起点和时长只比较估计设计时序，缺少估计计为未命中。多出来的实测片段记为误报。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motionspec import analyze  # noqa: E402
from motionspec.trace import spec_from_trace  # noqa: E402

FAMILY = {
    "linear": "linear", "ease": "ease-in-out", "ease-in": "ease-in", "ease-out": "ease-out", "ease-in-out": "ease-in-out",
    "material-standard": "ease-in-out", "material-decelerate": "ease-out", "material-accelerate": "ease-in",
    "in-quad": "ease-in", "out-quad": "ease-out", "in-out-quad": "ease-in-out", "in-cubic": "ease-in",
    "out-cubic": "ease-out", "in-out-cubic": "ease-in-out", "out-quart": "ease-out", "out-expo": "ease-out",
    "in-out-sine": "ease-in-out", "out-back": "overshoot",
}


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])  # noqa: E731
    return inter / max(1e-9, min(area(a), area(b)))  # 相对较小那个的覆盖率：大区域包着小元素也算对上


def timing(seg) -> tuple[float, float] | None:
    design = (seg.get("timing") or {}).get("estimated_design")
    if design:
        return design["t_start_ms"], design["duration_ms"]
    return None


def family(seg) -> str | None:
    easing = seg.get("easing")
    if not easing:
        return None
    if easing.get("model") == "spring" or easing.get("overshoot", 0) > 0.03:
        return "overshoot"
    best = easing["named"][0]
    return FAMILY.get(best["name"]) if best["rmse"] <= 0.05 else None


def judge(want: dict, seg: dict | None, ratio: float) -> dict:
    marks = {"found": seg is not None}
    if seg is None:
        return marks
    ch = seg["channels"]
    design = timing(seg)
    marks["start"] = design is not None and abs(design[0] - want["start_ms"]) <= 34
    marks["duration"] = design is not None and abs(design[1] - want["duration_ms"]) <= max(17, 0.15 * want["duration_ms"])
    if "delta" in want:
        dx, dy = (v * ratio for v in want["delta"])
        got = ch.get("translate", {}).get("delta")
        tol = max(3.0, 0.10 * (dx * dx + dy * dy) ** 0.5)
        marks["move"] = got is not None and abs(got[0] - dx) <= tol and abs(got[1] - dy) <= tol
    if "size" in want:
        size = ch.get("size")
        ok = size is not None
        for axis, (a, b) in want["size"].items():
            if not ok:
                break
            got = (size.get(f"{axis}_from"), size.get(f"{axis}_to"))
            ok = None not in got and all(abs(g - v * ratio) <= max(3.0, 0.10 * abs(b - a) * ratio) for g, v in zip(got, (a, b)))
        marks["size"] = ok
    if "opacity" in want:
        fade = ch.get("opacity")
        marks["opacity"] = fade is not None and abs(fade["from"] - want["opacity"][0]) <= 0.1 and abs(fade["to"] - want["opacity"][1]) <= 0.1
    marks["easing"] = family(seg) == FAMILY[want["easing"]]
    return marks


def describe(seg) -> str:
    ch = seg["channels"]
    design = timing(seg)
    label = f"design {design[0]:.0f}+{design[1]:.0f}ms" if design else "design unavailable"
    parts = [seg["kind"], label, f"region {seg['region']}"]
    for name in ("translate", "size", "scale", "opacity", "blend", "appearance", "change"):
        if name in ch:
            c = ch[name]
            if name == "translate":
                parts.append(f"move {c['delta']}")
            elif name == "size":
                parts.append(f"size {c.get('width_from')}x{c.get('height_from')}->{c.get('width_to')}x{c.get('height_to')}")
            elif name == "scale":
                parts.append(f"scale {c['from']}->{c['to']}")
            elif name == "opacity":
                parts.append(f"opacity {c['from']}->{c['to']}")
            else:
                parts.append(name)
    if seg.get("easing"):
        parts.append(f"[{seg['easing']['named'][0]['name']}]")
    parts.append(seg["confidence"])
    return " ".join(parts)


def score(scene: dict, spec: dict, ratio: float, label: str) -> tuple[int, int, int]:
    segs = list(spec["segments"])
    got_total = want_total = 0
    for want in scene["segments"]:
        rect = [v * ratio for v in want["rect"]]
        t0, t1 = want["start_ms"], want["start_ms"] + want["duration_ms"]

        def fit(seg, rect=rect, t0=t0, t1=t1, want=want):
            overlap = min(t1, seg["t_end_ms"]) - max(t0, seg["t_start_ms"])
            value = iou(rect, seg["region"]) if overlap > -34 else 0.0
            move = seg["channels"].get("translate", {}).get("delta")
            if value and "delta" in want and move:  # 区域相同的几个元素（对调的两行）按位移方向区分
                value += 0.01 * (want["delta"][0] * move[0] + want["delta"][1] * move[1] > 0)
            return value

        best = max(segs, key=fit) if segs else None
        if best is not None and fit(best) < 0.3:
            best = None
        if best is not None:
            segs.remove(best)
        marks = judge(want, best, ratio)
        got, total = sum(marks.values()), len(marks) if marks["found"] else len(judge(want, {"channels": {}, "t_start_ms": 0, "duration_ms": 0, "kind": ""}, ratio))
        got_total += got
        want_total += total
        bad = [k for k, v in marks.items() if not v]
        print(f"   {label:<6} {want['element']:<11} {got}/{total} " + ("ok" if not bad else "miss " + ",".join(bad))
              + ("" if best is None else f"   <- {describe(best)}"))
    for seg in segs:
        print(f"   {label:<6} {'(extra)':<11} {describe(seg)}")
    return got_total, want_total, len(segs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="out/qt-hard")
    parser.add_argument("--only")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    totals = {"vision": [0, 0, 0], "probe": [0, 0, 0]}
    for name, scene in truth["scenes"].items():
        if args.only and name not in args.only.split(","):
            continue
        began = time.time()
        vision = None if args.no_vision else analyze(root / name / "frames", root / name / "result",
                                                     fps=truth["fps"], raw_frames=False)
        print(f"\n## {name}  ({time.time() - began:.1f}s)")
        trace = json.loads((root / name / "trace.json").read_text(encoding="utf-8"))
        ratio = float(trace.get("device_pixel_ratio") or 1.0)
        probe = None if args.no_probe else spec_from_trace(trace, root / name / "probe")
        for label, spec in (("vision", vision), ("probe", probe)):
            if spec is None:
                continue
            got, total, extra = score(scene, spec, ratio, label)
            totals[label][0] += got
            totals[label][1] += total
            totals[label][2] += extra
    print()
    for label, (got, total, extra) in totals.items():
        if total:
            print(f"{label}: {got}/{total} = {100 * got / total:.0f}%, extra segments {extra}")


if __name__ == "__main__":
    main()
