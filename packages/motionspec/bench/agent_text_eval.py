"""桌面截图整屏读字的三种方式对比：字幕读取（overlays）、直接调 RapidOCR、screen-text。

    python bench/agent_screens.py screens        （需要 PySide6）
    python bench/agent_text_eval.py screens > text.json

指标（每个真值控件）：
- found：某条读出的行与控件框相交，且行文字包含控件文字（忽略空白和大小写）
- own_box：某条行文字与控件文字完全相同，且行中心在控件框内（Agent 可以直接点它）
- conflicts：相互重叠（IoU > 0.3）但文字不同的行对数；Agent 无法判断该信哪一条
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import cv2
from rapidocr_onnxruntime import RapidOCR

from svsg.recognition.overlays import measure_overlays
from svsg.recognition.screen_text import ScreenTextReader


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def iou(a, b) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter or 1)


def intersects(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def truth_items(truth: dict) -> list[dict]:
    items = []
    s = truth["scale"]
    for t in truth["texts"]:
        box = list(t["bbox_px"])
        if t["kind"] == "QGroupBox":  # 分组框只有标题是文字
            box[3] = box[1] + round(24 * s)
        items.append({"text": t["text"], "bbox_px": box})
    return items


def score(lines: list[dict], truth: dict) -> dict:
    items = truth_items(truth)
    found = own = 0
    misses = []
    for t in items:
        want = norm(t["text"])
        near = [ln for ln in lines if intersects(ln["bbox_px"], t["bbox_px"])]
        if any(want in norm(ln["text"]) for ln in near):
            found += 1
        else:
            misses.append({"truth": t["text"], "near": [ln["text"] for ln in near][:3]})
        cx = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
        if any(
            norm(ln["text"]) == want
            and t["bbox_px"][0] <= cx(ln["bbox_px"])[0] <= t["bbox_px"][2]
            and t["bbox_px"][1] <= cx(ln["bbox_px"])[1] <= t["bbox_px"][3]
            for ln in near
        ):
            own += 1
    conflicts = [
        (a["text"], b["text"])
        for i, a in enumerate(lines)
        for b in lines[i + 1 :]
        if iou(a["bbox_px"], b["bbox_px"]) > 0.3 and norm(a["text"]) != norm(b["text"])
    ]
    return {
        "items": len(items),
        "found": found,
        "own_box": own,
        "lines": len(lines),
        "conflicts": len(conflicts),
        "conflict_examples": conflicts[:4],
        "misses": misses[:6],
    }


def plain_ocr(engine: RapidOCR, png: Path) -> list[dict]:
    result, _ = engine(cv2.imread(str(png)))
    lines = []
    for quad, text, conf in result or []:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        lines.append({"text": text, "bbox_px": [min(xs), min(ys), max(xs), max(ys)], "score": conf})
    return lines


def clicks(result: dict, truth: dict) -> dict:
    """按文字找点击位置：reachable 为某个精确匹配落在控件框内；unique 只看屏幕上文字唯一的控件。"""
    from collections import Counter

    from svsg.recognition.screen_text import find_text

    items = truth_items(truth)
    counts = Counter(t["text"] for t in items)
    reach = unique = unique_ok = 0
    for t in items:
        b = t["bbox_px"]
        hits = find_text(result, t["text"], exact=True)
        ins = [
            h
            for h in hits
            if b[0] <= h["center_px"][0] <= b[2] and b[1] <= h["center_px"][1] <= b[3]
        ]
        reach += bool(ins)
        if counts[t["text"]] == 1:
            unique += 1
            unique_ok += bool(hits) and hits[0] in ins
    return {"reachable": reach, "unique": unique, "unique_first_hit_correct": unique_ok}


def main() -> None:
    folder = Path(sys.argv[1])
    engine = RapidOCR(
        intra_op_num_threads=3,
        inter_op_num_threads=1,
        det_limit_type="max",
        det_limit_side_len=1920,
    )
    pngs = sorted(folder.glob("*.png"))
    reader = ScreenTextReader()
    plain_ocr(engine, pngs[0])
    reader.read(pngs[0])
    measure_overlays(pngs[0], only="subtitles")  # 三种读法都先预热
    out = {}
    for png in pngs:
        truth = json.loads(
            png.with_suffix("").with_suffix(".truth.json").read_text(encoding="utf-8")
        )
        w, h = truth["size_px"]
        t0 = time.perf_counter()
        project = measure_overlays(png, only="subtitles", subtitle_region=(0, 0, w, h))["frames"][
            0
        ]["subtitles"]
        t1 = time.perf_counter()
        plain = plain_ocr(engine, png)
        t2 = time.perf_counter()
        screen_result = reader.read(png)
        screen = screen_result["lines"]
        t3 = time.perf_counter()
        out[png.stem] = {
            "overlays_subtitles": {"seconds": round(t1 - t0, 2), **score(project, truth)},
            "plain_rapidocr": {"seconds": round(t2 - t1, 2), **score(plain, truth)},
            "screen_text": {
                "seconds": round(t3 - t2, 2),
                **score(screen, truth),
                **clicks(screen_result, truth),
            },
        }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
