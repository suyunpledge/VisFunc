"""函数层的精度与开销：把视觉分析导出成 SVSG 函数，逐帧求值，和 Qt 探针记录的真实属性比较。

    python tools/qt_hard_scenes.py out/qt-hard
    python tools/function_accuracy.py out/qt-hard [--video] [--only 场景]

每个真实元素（探针里有变化的控件）找区域重叠最大的视觉片段，在两者都覆盖的每个时间点上
比较中心位置（x, y）、宽高（width, height）和透明度（opacity）。只比较函数里实际存在的量；
缺失的量记为"不可用"，单独计数。时间：分析、导出、加载、每次查询分别计时。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motionspec import analyze  # noqa: E402
from motionspec.trace import spec_from_trace  # noqa: E402

KEYS = ("x", "y", "width", "height", "opacity")


def overlap(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: max(1, (r[2] - r[0]) * (r[3] - r[1]))  # noqa: E731
    return inter / min(area(a), area(b))


def truth_curves(trace: dict) -> list[dict]:
    """探针轨迹里每个有几何或透明度变化的元素：逐帧的中心、宽高、透明度（设备像素）。"""
    ratio = float(trace.get("device_pixel_ratio") or 1.0)
    ticks = trace["ticks_ms"]
    out = []
    for element in trace["elements"]:
        rows = element["rows"]
        if len(rows) < 2:
            continue
        table = {}
        cursor = 0
        for tick in range(rows[0][0], rows[-1][0] + 1):
            while cursor + 1 < len(rows) and rows[cursor + 1][0] <= tick:
                cursor += 1
            r = rows[cursor]
            w, h = r[3] * r[6] * ratio, r[4] * r[6] * ratio
            visible = r[8] if w * h > 0 else 1.0
            table[ticks[tick]] = {"x": r[9] * ratio + w / 2, "y": r[10] * ratio + h / 2, "width": w, "height": h,
                                  "opacity": r[5] * visible}
        values = list(table.values())
        if all(abs(v[k] - values[0][k]) < 0.5 for v in values for k in KEYS):
            continue
        xs = [v["x"] - v["width"] / 2 for v in values] + [v["x"] + v["width"] / 2 for v in values]
        ys = [v["y"] - v["height"] / 2 for v in values] + [v["y"] + v["height"] / 2 for v in values]
        out.append({"name": element.get("name") or element.get("class"), "region": [min(xs), min(ys), max(xs), max(ys)],
                    "table": table})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="out/qt-hard")
    parser.add_argument("--only")
    parser.add_argument("--video", action="store_true", help="analyse the H.264 clip.mp4 of each scene instead of PNG frames")
    args = parser.parse_args()
    from svsg.video_functions import VideoRuntime

    root = Path(args.root)
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    errors = {k: [] for k in KEYS}
    missing = {k: 0 for k in KEYS}
    timings = {"analyze": [], "export": [], "load": [], "query": []}
    for name in truth["scenes"]:
        if args.only and name not in args.only.split(","):
            continue
        source = root / name / ("clip.mp4" if args.video else "frames")
        out = root / name / ("fx-video" if args.video else "fx")
        began = time.perf_counter()
        spec = analyze(source, out, fps=None if args.video else truth["fps"], functions=True,
                       sequence_id=f"{name}-{'video' if args.video else 'png'}")
        took = time.perf_counter() - began
        export_s = spec.get("function_evidence", {}).get("seconds")
        timings["analyze"].append(took)
        began = time.perf_counter()
        runtime = VideoRuntime.load(out / "functions" / "functions.json") if (out / "functions" / "functions.json").exists() \
            else VideoRuntime.load(out / "functions.json")
        timings["load"].append(time.perf_counter() - began)
        motions = {m.segment_id: m for m in runtime.bundle.motions}
        print(f"\n## {name}  analyze+export {took:.1f}s" + (f" (export {export_s:.1f}s)" if export_s else ""))
        trace = json.loads((root / name / "trace.json").read_text(encoding="utf-8"))
        for element in truth_curves(trace):
            # 区域重叠的候选里，选位置曲线和真值最贴合的那个（区域相同的几个元素靠曲线区分）
            candidates = [s for s in spec["segments"] if overlap(element["region"], s["region"]) >= 0.3]

            def fit(s, element=element):
                fn = motions[s["id"]]
                if "x" not in fn.samples or "y" not in fn.samples:
                    # 没有位置曲线的片段（淡入、变色）等于说"没在动"：代价是真值自己的位移量
                    rows = list(element["table"].values())
                    return sum(abs(r["x"] - rows[0]["x"]) + abs(r["y"] - rows[0]["y"]) for r in rows) / len(rows) + 1e-3
                knots = {k: fn.samples[k].knots for k in ("x", "y")}
                lo, hi = knots["x"][0][0], knots["x"][-1][0]
                times = [t for t in element["table"] if lo <= t <= hi]
                if not times:
                    return 1e6
                cost, base = 0.0, None
                for t in times:
                    v = [runtime.call({"id": "q", "op": "motion_value", "segment_id": s["id"], "name": k, "t_ms": t})["value"] for k in ("x", "y")]
                    r = [element["table"][t][k] for k in ("x", "y")]
                    base = base or (v, r)
                    cost += abs((v[0] - base[0][0]) - (r[0] - base[1][0])) + abs((v[1] - base[0][1]) - (r[1] - base[1][1]))
                return cost / len(times)

            seg = min(candidates, key=fit) if candidates else None
            if seg is None:
                print(f"   {element['name']:<12} no matching segment")
                for k in KEYS:
                    missing[k] += 1
                continue
            fn = motions[seg["id"]]
            line = []
            for key in KEYS:
                if key not in fn.samples:
                    missing[key] += 1
                    continue
                knots = fn.samples[key].knots
                lo, hi = knots[0][0], knots[-1][0]
                diffs = []
                base_truth = base_fn = None
                for t, row in element["table"].items():
                    if not lo <= t <= hi:
                        continue
                    began = time.perf_counter()
                    value = runtime.call({"id": "q", "op": "motion_value", "segment_id": seg["id"], "name": key, "t_ms": t})["value"]
                    timings["query"].append(time.perf_counter() - began)
                    # 位置的参照点可能不同（特征锚点 vs 元素中心），比较相对首个共同时刻的变化量
                    if key in ("x", "y"):
                        if base_truth is None:
                            base_truth, base_fn = row[key], value
                        diffs.append(abs((value - base_fn) - (row[key] - base_truth)))
                    else:
                        diffs.append(abs(value - row[key]))
                if diffs:
                    errors[key].append(max(diffs))
                    line.append(f"{key} max {max(diffs):.1f}")
            print(f"   {element['name']:<12} <- {seg['id']} {seg['kind']}/{seg.get('model', 'track')}: " + (", ".join(line) or "no comparable function"))
    print("\nworst error per element (median / max), and elements without the function:")
    for key in KEYS:
        values = errors[key]
        text = f"{statistics.median(values):.1f} / {max(values):.1f}" if values else "-"
        print(f"  {key:<8} {text:<14} compared {len(values)}, unavailable {missing[key]}")
    for stage, values in timings.items():
        if values:
            unit = "ms" if stage in ("query", "load") else "s"
            scale = 1000 if unit == "ms" else 1
            print(f"  time {stage:<8} median {statistics.median(values) * scale:.2f}{unit}  total {sum(values) * scale:.1f}{unit}")


if __name__ == "__main__":
    main()
