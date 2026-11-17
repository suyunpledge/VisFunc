"""把 motionspec 的两条路径（看像素的视觉分析、读属性的 Qt 探针）和真值并排打印。

    python tools/qt_scenes.py out/qt      （需要 PySide6，生成帧、探针轨迹和真值）
    python tools/qt_compare.py out/qt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motionspec import analyze  # noqa: E402
from motionspec.trace import spec_from_trace  # noqa: E402


def best_named(easing: dict | None) -> str:
    if not easing:
        return "-"
    row = easing["named"][0]
    return f"{row['name']} {row['duration_ms']:.0f}ms"


def line(label: str, seg: dict) -> str:
    ch = seg["channels"]
    text = f"   {label:<8} {seg['kind']:<9} start {seg['t_start_ms']:.0f}  {seg['duration_ms']:.0f}ms"
    if "translate" in ch:
        text += f"  move {ch['translate']['delta']} [{best_named(ch['translate']['easing'])}]"
    if "scale" in ch:
        text += f"  scale {ch['scale']['from']}->{ch['scale']['to']}"
    if "opacity" in ch:
        text += f"  opacity {ch['opacity']['from']:g}->{ch['opacity']['to']:g} [{best_named(ch['opacity']['easing'])}]"
    if "blend" in ch:
        text += f"  fade [{best_named(ch['blend']['easing'])}]"
    who = seg.get("element", {})
    if who:
        text += f"  <{who.get('name') or who.get('class')}>"
    return text


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "out/qt")
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    for name, scene in truth["scenes"].items():
        began = time.time()
        vision = analyze(root / name / "frames", root / name / "result", fps=truth["fps"], raw_frames=False)
        took = time.time() - began
        probe = None
        if (root / name / "trace.json").exists():
            trace = json.loads((root / name / "trace.json").read_text(encoding="utf-8"))
            probe = spec_from_trace(trace, root / name / "probe")
        print(f"\n## {name}  (expected {len(scene['segments'])}; vision {len(vision['segments'])} in {took:.1f}s"
              + (f"; probe {len(probe['segments'])}" if probe else "") + ")")
        ratio = vision["source"]["width"] / truth["size"][0]  # 屏幕缩放：真值是逻辑像素，截图是设备像素
        for want in scene["segments"]:
            text = f"   truth    {want['kind']:<9} start {want['start_ms']:.0f}  {want['duration_ms']}ms {want['easing']}"
            if "delta" in want:
                text += f"  move {[round(v * ratio, 1) for v in want['delta']]}"
            if "opacity" in want:
                text += f"  opacity {want['opacity']}"
            print(text)
        for seg in vision["segments"]:
            print(line("vision", seg))
        for seg in (probe or {}).get("segments", []):
            print(line("probe", seg))


if __name__ == "__main__":
    main()
