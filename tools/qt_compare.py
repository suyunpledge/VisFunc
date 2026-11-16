"""把 motionspec 对 Qt 场景的分析结果和 tools/qt_scenes.py 写下的真值并排打印。

    python tools/qt_scenes.py out/qt      （需要 PySide6）
    python tools/qt_compare.py out/qt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motionspec import analyze  # noqa: E402


def best_named(easing: dict | None) -> str:
    if not easing:
        return "-"
    row = easing["named"][0]
    return f"{row['name']} {row['duration_ms']:.0f}ms (rmse {row['rmse']})"


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "out/qt")
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    for name, scene in truth["scenes"].items():
        began = time.time()
        spec = analyze(root / name / "frames", root / name / "result", fps=truth["fps"], raw_frames=False)
        print(f"\n## {name}  ({time.time() - began:.1f}s, {len(spec['segments'])} segment(s), expected {len(scene['segments'])})")
        ratio = spec["source"]["width"] / truth["size"][0]  # 屏幕缩放：真值是逻辑像素，截图是设备像素
        for want in scene["segments"]:
            text = f"   truth    {want['kind']:<9} start {want['start_ms']:.0f}  {want['duration_ms']}ms {want['easing']}"
            if "delta" in want:
                text += f"  move {[round(v * ratio, 1) for v in want['delta']]}"
            if "opacity" in want:
                text += f"  opacity {want['opacity']}"
            print(text)
        for seg in spec["segments"]:
            ch = seg["channels"]
            text = f"   measured {seg['kind']:<9} start {seg['t_start_ms']:.0f}  {seg['duration_ms']:.0f}ms"
            if "translate" in ch:
                text += f"  move {ch['translate']['delta']} [{best_named(ch['translate']['easing'])}]"
            if "scale" in ch:
                text += f"  scale {ch['scale']['from']}->{ch['scale']['to']}"
            if "opacity" in ch:
                text += f"  opacity {ch['opacity']['from']:g}->{ch['opacity']['to']:g} [{best_named(ch['opacity']['easing'])}]"
            if "blend" in ch:
                text += f"  fade [{best_named(ch['blend']['easing'])}]"
            print(text + f"  ({seg['confidence']}{', ' + seg['model'] if seg.get('model') else ''})")
        for warning in spec["warnings"]:
            print("   warning:", warning)


if __name__ == "__main__":
    main()
