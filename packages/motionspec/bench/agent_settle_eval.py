"""端到端：Agent 点击后等界面稳定，再只读变化区域的文字。1080p、60fps，按帧时间模拟。

场景：
  A  带遮罩的确认框 250ms 淡入（100ms 开始，350ms 结束），输入框里光标每 530ms 闪一次
  B  同 A，右下有视频一直在播，每次操作新开观察器（默认 ambient 3000ms）
  C  同 B，但会话里一直用同一个观察器：先看 4 秒，rearm()，Agent 操作 300ms 不送帧，再看
  D  不带遮罩的小弹窗（只贴对话框本身），同样有光标闪烁

    python bench/agent_settle_eval.py screens > agent.json   （需要 svsg[overlays]）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from motionspec.settle import SettleWatcher

from svsg.recognition.screen_text import ScreenTextReader

STEP = 1000 / 60
DIALOG = (700, 420, 1220, 660)
VIDEO = (1500, 600, 1800, 780)
WANT = [
    "Remove 3 devices?",
    "Laptop-101, Laptop-102 and Laptop-104 will be signed out.",
    "Keep devices",
    "Remove",
]


def ease(p: float) -> float:
    return 1 - (1 - p) ** 3


def frame_at(before, after, t_rel, *, video_i=None, caret=True):
    p = min(1.0, max(0.0, (t_rel - 100) / 250))
    frame = before.copy() if p == 0 else cv2.addWeighted(before, 1 - ease(p), after, ease(p), 0)
    if caret and int(t_rel // 530) % 2 == 0:
        frame[121:139, 206:208] = 0  # blinking caret after "Morgan Lee"
    if video_i is not None:
        rng = np.random.default_rng(video_i)
        base = rng.integers(40, 200, (18, 30, 3), dtype=np.uint8)
        frame[VIDEO[1] : VIDEO[3], VIDEO[0] : VIDEO[2]] = cv2.resize(
            base, (300, 180), interpolation=cv2.INTER_LINEAR
        )
    return frame


def run(watcher, frames):
    costs = []
    for t, frame in frames:
        start = time.perf_counter()
        done = watcher.push(frame, t)
        costs.append((time.perf_counter() - start) * 1000)
        if done:
            break
    return watcher.result(), watcher.last_frame, costs


def norm(s):
    return "".join(s.split()).lower()


def read(reader, frame, result):
    regions = [r["bbox_px"] for r in result["changed_regions"]]
    got = reader.read(frame, regions)
    texts = [norm(line["text"]) for line in got["lines"]]
    return {
        "regions_read": got["regions_px"],
        "read_ms": got["elapsed_ms"],
        "found_exact": sum(norm(w) in texts for w in WANT),
        "of": len(WANT),
        "lines": len(got["lines"]),
    }


def main():
    folder = Path(sys.argv[1])
    before = cv2.imread(str(folder / "settings-light.png"))
    after = cv2.imread(str(folder / "dialog-after.png"))
    popup = before.copy()
    left, top, right, bottom = DIALOG
    popup[top:bottom, left:right] = after[top:bottom, left:right]
    reader = ScreenTextReader()
    reader.read(before)  # warm up once, as a resident service would be
    full_ms = reader.read(after)["elapsed_ms"]
    out = {"full_frame_read_ms_after_dialog": full_ms, "scenarios": {}}

    def report(name, result, last, costs, offset=0.0):
        end = 350.0 + offset
        out["scenarios"][name] = {
            "state": result["state"],
            "last_change_ms": result["last_change_ms"],
            "true_end_ms": end,
            "decided_ms": result["elapsed_ms"],
            "wait_after_true_end_ms": None
            if result["elapsed_ms"] is None
            else round(result["elapsed_ms"] - end, 1),
            "per_frame_ms_mean": round(float(np.mean(costs)), 2),
            "per_frame_ms_max": round(float(np.max(costs)), 2),
            "changed_regions": [x["bbox_px"] for x in result["changed_regions"]][:4],
            "ambient_regions": [x["bbox_px"] for x in result["ambient_regions"]][:2],
            "text": read(reader, last, result) if result["changed_regions"] else None,
        }

    n = 300
    res, last, costs = run(
        SettleWatcher(), ((i * STEP, frame_at(before, after, i * STEP)) for i in range(n))
    )
    report("A_modal", res, last, costs)

    res, last, costs = run(
        SettleWatcher(),
        ((i * STEP, frame_at(before, after, i * STEP, video_i=i)) for i in range(n)),
    )
    report("B_modal_video_fresh_watcher", res, last, costs)

    watcher = SettleWatcher()
    idle = 240  # four seconds of looking at the screen before acting
    for i in range(idle):
        watcher.push(frame_at(before, after, -1000, video_i=i), i * STEP)
    watcher.rearm()
    t_arm = (idle - 1) * STEP
    gap = 300.0  # the agent clicks; no frames are pushed meanwhile
    frames = (
        (t_arm + gap + j * STEP, frame_at(before, after, j * STEP, video_i=idle + j))
        for j in range(n)
    )
    res, last, costs = run(watcher, frames)
    report("C_modal_video_rearmed_session", res, last, costs, offset=gap)

    res, last, costs = run(
        SettleWatcher(), ((i * STEP, frame_at(before, popup, i * STEP)) for i in range(n))
    )
    report("D_popup_no_dim", res, last, costs)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
