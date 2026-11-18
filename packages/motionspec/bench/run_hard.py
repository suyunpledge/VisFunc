"""第二批场景（tools/qt_hard_scenes.py）上的模型实测：位移、尺寸变化、颜色变化、回弹都要问到。

    python tools/qt_hard_scenes.py out/qt-hard
    python bench/run_hard.py out/qt-hard bench/results/hard1 [--old-code 某个旧版本的 motionspec 包目录]

四种条件：
    video              只给视频（GPT 两个模型拿不到视频，改传带时间戳的静帧）
    video+vision-old   视频 + 用旧版代码（--old-code）对同一段视频做的视觉分析
    video+vision       视频 + 当前代码的视觉分析
    video+probe        视频 + Qt 探针摘要

问题与 bench/run.py 不同，结果不能和第一批直接比较。判分规则在 TRUTH 下方的 judge() 里，先于调用定下。
密钥从本机 Forge 配置读取，只在进程内使用；结果文件不含密钥。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import uuid
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import client  # noqa: E402
from providers import find  # noqa: E402

from motionspec import analyze  # noqa: E402
from motionspec.frames import read_image, write_image  # noqa: E402
from motionspec.trace import spec_from_trace  # noqa: E402
from motionspec.measurement import decode_measurement, sum_usage  # noqa: E402

MODELS = [
    ("mimo-v2.6-flash", "mimo-v2.6-flash", r"mimo-v2\.6-flash$", 10),
    ("mimo-v2.6-pro", "mimo-v2.6-pro", r"mimo-v2\.6-pro$", 10),
    ("seed-2.1-turbo", "doubao-seed-2-1-turbo-260628", r"volces", 5),
    ("seed-2.1-pro", "doubao-seed-2-1-pro-260915", r"volces", 5),
    ("gpt-6.1-sol", "gpt-6.1-sol", r"openai\.com", None),
    ("gpt-6.1-terra", "gpt-6.1-terra", r"openai\.com", None),
]
CONDITIONS = ("video", "video+vision-old", "video+vision", "video+probe")
FRAME_FPS = 10
START = 166.7

# 真值：设备像素（逻辑像素 × 1.5）。move 是 [下限, 上限]：两条边移得不一样多时（边移边变宽），
# 任何一条边或中心的移动量都算对。只从一条固定边长出来/缩回去的元素，move 为 0。
TRUTH = {
    "modal-open": [
        {"element": "dim overlay", "start": START, "duration": 200, "dx": [0, 0], "dy": [0, 0], "dw": 0, "dh": 0,
         "opacity": "fade-in", "color": "no", "easing": "linear"},
        {"element": "dialog", "start": START, "duration": 250, "dx": [0, 0], "dy": [-24, -24], "dw": 0, "dh": 0,
         "opacity": "fade-in", "color": "no", "easing": "ease-out"},
    ],
    "accordion": [
        {"element": "expanding panel", "start": START, "duration": 300, "dx": [0, 0], "dy": [0, 0], "dw": 0, "dh": 270,
         "opacity": "none", "color": "no", "easing": "ease-out"},
        {"element": "rows below", "start": START, "duration": 300, "dx": [0, 0], "dy": [270, 270], "dw": 0, "dh": 0,
         "opacity": "none", "color": "no", "easing": "ease-out"},
    ],
    "sidebar-collapse": [
        {"element": "sidebar", "start": START, "duration": 300, "dx": [0, 0], "dy": [0, 0], "dw": -234, "dh": 0,
         "opacity": "none", "color": "no", "easing": "ease-in-out"},
        {"element": "content", "start": START, "duration": 300, "dx": [-234, -234], "dy": [0, 0], "dw": 0, "dh": 0,
         "opacity": "none", "color": "no", "easing": "ease-in-out"},
    ],
    "scroll": [
        {"element": "list", "start": START, "duration": 250, "dx": [0, 0], "dy": [-180, -180], "dw": 0, "dh": 0,
         "opacity": "none", "color": "no", "easing": "ease-out"},
    ],
    "tab-indicator": [
        {"element": "indicator", "start": START, "duration": 250, "dx": [180, 225], "dy": [0, 0], "dw": 45, "dh": 0,
         "opacity": "none", "color": "no", "easing": "ease-out"},
    ],
    "reorder-swap": [
        {"element": "row moving down", "start": START, "duration": 300, "dx": [0, 0], "dy": [75, 75], "dw": 0, "dh": 0,
         "opacity": "none", "color": "no", "easing": "ease-in-out"},
        {"element": "row moving up", "start": START, "duration": 300, "dx": [0, 0], "dy": [-75, -75], "dw": 0, "dh": 0,
         "opacity": "none", "color": "no", "easing": "ease-in-out"},
    ],
    "overshoot": [
        {"element": "card", "start": START, "duration": 400, "dx": [0, 0], "dy": [-90, -90], "dw": 0, "dh": 0,
         "opacity": "none", "color": "no", "easing": "overshoot"},
    ],
    "button-color": [
        {"element": "button", "start": START, "duration": 200, "dx": [0, 0], "dy": [0, 0], "dw": 0, "dh": 0,
         "opacity": "none", "color": "yes", "easing": "linear"},
    ],
    "progress": [
        {"element": "progress fill", "start": START, "duration": 600, "dx": [0, 0], "dy": [0, 0], "dw": 600, "dh": 0,
         "opacity": "none", "color": "no", "easing": "linear"},
    ],
}
FIELDS = ("start", "duration", "move", "size", "opacity", "color", "easing")

QUESTION = """This is a screen recording of a desktop application: {width}x{height} pixels, {duration} ms long, 60 frames per second.
List every UI animation in it, one entry per animated element (elements that only move because their parent moves are not separate entries).

Reply with JSON only, in exactly this shape:
{{"animations": [{{"element": "<short description>", "start_ms": <number>, "duration_ms": <number>,
  "dx_px": <number>, "dy_px": <number>, "dw_px": <number>, "dh_px": <number>,
  "opacity_change": "fade-in|fade-out|none", "color_change": "yes|no",
  "easing": "linear|ease-in|ease-out|ease-in-out|overshoot"}}]}}

Definitions:
- start_ms from the start of the recording; duration_ms is the designed duration of the animation.
- dx_px, dy_px: how far the element moves (right and down are positive). An element that only grows or shrinks from a fixed edge does not move: use 0.
- dw_px, dh_px: change of the element's width and height (positive = grows). 0 if its size does not change.
- opacity_change: a translucent overlay that darkens the window counts as fade-in.
- color_change: yes if the element changes colour without fading in or out.
- easing: overshoot = goes past its final position and settles back.
All distances are in pixels of this recording.
{extra}"""

EXTRA = {
    "video": "",
    "video+vision-old": "- A motion analysis of this same recording follows, made by a tool that tracks pixels frame by frame. Use it together with the recording.\n",
    "video+vision": "- A motion analysis of this same recording follows, made by a tool that tracks pixels frame by frame. Use it together with the recording.\n",
    "video+probe": "- A trace of this same recording follows, read directly from the application's widget properties while it was recorded. Use it together with the recording.\n",
}


def number(value):
    try:
        return value if type(value) in (int, float) and math.isfinite(value) else None
    except OverflowError:
        return None


def judge(want: dict, got: dict | None) -> dict:
    marks = {name: False for name in FIELDS}
    if got is None:
        return marks
    start, duration = number(got.get("start_ms")), number(got.get("duration_ms"))
    marks["start"] = start is not None and abs(start - want["start"]) <= 34
    marks["duration"] = duration is not None and abs(duration - want["duration"]) <= max(17, 0.15 * want["duration"])

    def near(value, lo, hi):
        tol = max(4.0, 0.12 * max(abs(lo), abs(hi)))
        return value is not None and lo - tol <= value <= hi + tol

    marks["move"] = near(number(got.get("dx_px")), *want["dx"]) and near(number(got.get("dy_px")), *want["dy"])
    marks["size"] = near(number(got.get("dw_px")), want["dw"], want["dw"]) and near(number(got.get("dh_px")), want["dh"], want["dh"])
    marks["opacity"] = str(got.get("opacity_change", "")).strip().lower() == want["opacity"]
    marks["color"] = str(got.get("color_change", "")).strip().lower() == want["color"]
    marks["easing"] = str(got.get("easing", "")).strip().lower() == want["easing"]
    return marks


def score(truth: list[dict], answer: list[dict] | None) -> dict:
    """真值和回答按"起点 + 位移/尺寸方向"贪心配对，再逐项比较。多出来的回答单独记数。"""
    free = list(answer or [])
    marks = []
    for want in truth:
        def cost(row, want=want):
            start = number(row.get("start_ms"))
            c = abs(start - want["start"]) if start is not None else 1e6
            for key, target in (("dx_px", sum(want["dx"]) / 2), ("dy_px", sum(want["dy"]) / 2), ("dw_px", want["dw"]), ("dh_px", want["dh"])):
                value = number(row.get(key)) or 0.0
                c += abs(value - target) / 2
            return c
        got = min(free, key=cost) if free else None
        if got is not None:
            free.remove(got)
        marks.append(judge(want, got))
    return {"fields": {name: [sum(m[name] for m in marks), len(marks)] for name in FIELDS},
            "count_ok": answer is not None and len(answer) == len(truth), "extra": len(free), "parsed": answer is not None}


def prepare(root: Path, work: Path, old_code: Path | None) -> dict:
    meta = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    scenes = {}
    for name in TRUTH:
        folder = work / name
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / "clip.mp4"
        if not video.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(meta["fps"]), "-i",
                            str(root / name / "frames" / "%04d.png"), "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", str(video)], check=True)
        vision = analyze(video, folder / "vision", raw_frames=False)
        spec_from_trace(json.loads((root / name / "trace.json").read_text(encoding="utf-8")), folder / "probe")
        if old_code is not None and not (folder / "vision-old" / "summary.md").exists():
            # 旧版代码在它自己的目录里运行，导入的就是那一版
            subprocess.run([sys.executable, "-m", "motionspec", "analyze", str(video.resolve()), "-o",
                            str((folder / "vision-old").resolve()), "--no-raw-frames"], cwd=old_code, check=True,
                           stdout=subprocess.DEVNULL)
        frames = sorted((root / name / "frames").glob("*.png"))
        step = max(1, round(meta["fps"] / FRAME_FPS))
        stills = []
        for index in range(0, len(frames), step):
            small = folder / f"still-{index:04d}.png"
            if not small.exists():
                write_image(small, read_image(frames[index]))
            stills.append((round(index * 1000 / meta["fps"]), small))

        def sheets(sub):
            spec = json.loads((folder / sub / "spec.json").read_text(encoding="utf-8"))
            return [folder / sub / seg["sheet"] for seg in spec["segments"] if seg.get("sheet")]

        scenes[name] = {
            "video": video, "stills": stills, "width": vision["source"]["width"], "height": vision["source"]["height"],
            "duration": round(vision["source"]["duration_ms"]),
            "vision": ((folder / "vision" / "summary.md").read_text(encoding="utf-8"), sheets("vision")),
            "vision-old": ((folder / "vision-old" / "summary.md").read_text(encoding="utf-8"), sheets("vision-old"))
            if (folder / "vision-old" / "summary.md").exists() else None,
            "probe": ((folder / "probe" / "summary.md").read_text(encoding="utf-8"), []),
        }
    return scenes


def build(scene: dict, condition: str, vendor: str, fps: int | None) -> list[dict]:
    parts = []
    media = ""
    if fps is not None:
        parts.append(client.video(scene["video"], vendor, fps=fps))
    else:
        for at, path in scene["stills"]:
            parts.append(client.text(f"Frame at {at} ms:"))
            parts.append(client.image(path))
        media = f"- The recording is given as still frames sampled every {round(1000 / FRAME_FPS)} ms, each labelled with its time.\n"
    parts.append(client.text(QUESTION.format(width=scene["width"], height=scene["height"], duration=scene["duration"],
                                             extra=media + EXTRA[condition])))
    source = {"video+vision": "vision", "video+vision-old": "vision-old", "video+probe": "probe"}.get(condition)
    if source:
        summary, images = scene[source]
        label = "Widget property trace summary" if source == "probe" else "Motion analysis summary"
        parts.append(client.text(f"{label}:\n\n{summary}"))
        for sheet in images:
            parts.append(client.text(f"Annotated key frames for segment {sheet.stem}:"))
            parts.append(client.image(sheet))
    return parts


def decode(text: str) -> dict:
    return decode_measurement(
        text, numeric_fields=("start_ms", "duration_ms", "dx_px", "dy_px", "dw_px", "dh_px"),
        nonnegative_fields=("start_ms", "duration_ms"),
        enum_fields={"opacity_change": {"fade-in", "fade-out", "none"},
                     "color_change": {"yes", "no"},
                     "easing": {"linear", "ease-in", "ease-out", "ease-in-out", "overshoot"}},
    )


def parse(text: str) -> list[dict] | None:
    delivery = decode(text)
    return delivery["rows"] if delivery["accepted"] else None


def ask(model_row, name: str, scene: dict, condition: str, max_tokens: int, slots: dict,
        *, repeat: int = 0, run_id: str | None = None, max_attempts: int = 3) -> dict:
    if not 1 <= max_attempts <= 4 or not 1 <= max_tokens <= 16000:
        raise ValueError("invalid attempt or output token budget")
    label, model, pattern, fps = model_row
    provider = find(slots.get(label, pattern))
    parts = build(scene, condition, client.vendor_of(provider.base_url), fps)
    attempts = []
    began = time.perf_counter()
    for attempt in range(max_attempts):
        reply = client.chat(provider.base_url, provider.key, model, parts, max_tokens=max_tokens)
        attempts.append({"attempt": attempt + 1, "max_tokens": max_tokens, **reply})
        retryable = reply["status"] in (0, 408, 429) or reply["status"] >= 500
        retryable |= reply["ok"] and not reply["text"].strip()
        if not retryable or attempt + 1 == max_attempts:
            break
        time.sleep(2 * (attempt + 1))
    delivery = decode(reply["text"] if reply["ok"] else "")
    answer = delivery["rows"]
    scores = score(TRUTH[name], answer)
    wrong = not scores["count_ok"] or any(a != b for a, b in scores["fields"].values())
    return {"model": label, "scene": name, "condition": condition, "ok": reply["ok"], "status": reply["status"],
            "run_id": run_id, "repeat": repeat, "task_id": f"{run_id}:{repeat}:{label}:{name}:{condition}",
            "input_sha256": hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "provider_slot": getattr(provider, "slot", None), "requested_model": model,
            "attempts": attempts, "seconds": round(time.perf_counter() - began, 3),
            "request_seconds": sum(a["seconds"] for a in attempts), "usage": sum_usage(attempts),
            "usage_unknown_attempts": sum(type((attempt.get("usage") or {}).get("total_tokens")) is not int
                                          for attempt in attempts),
            "text": reply["text"], "error": None if reply["ok"] else str(reply["error"])[:400],
            "delivery": delivery, "transport_fail": not reply["ok"],
            "format_fail": not delivery["strict_format_pass"] if reply["ok"] else None,
            "answer_wrong": wrong if answer is not None else None,
            "parsed": answer if delivery["accepted"] else None, "score": scores,
            "end_to_end_score": scores if delivery["accepted"] else score(TRUTH[name], None)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes")
    parser.add_argument("out")
    parser.add_argument("--old-code", help="directory of an older motionspec package (contains the motionspec/ folder)")
    parser.add_argument("--models")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--providers", help="JSON file mapping model labels to exact Forge slot regexes")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.jsonl").exists() or (out / "protocol.json").exists():
        raise ValueError("use a new output directory; existing experiment evidence is immutable")
    if args.repeat < 1 or not 1 <= args.workers <= 16 or not 1 <= args.max_tokens <= 16000:
        raise ValueError("invalid repeat, worker or token budget")
    run_id = uuid.uuid4().hex
    scenes = prepare(Path(args.scenes), out / "data", Path(args.old_code) if args.old_code else None)
    models = [m for m in MODELS if not args.models or m[0] in args.models.split(",")]
    slots = json.loads(Path(args.providers).read_text(encoding="utf-8")) if args.providers else {}
    for model in models:
        find(slots.get(model[0], model[2]))  # 有歧义的提供方在付费调用之前就报错
    conditions = [c for c in args.conditions.split(",") if c in CONDITIONS and (c != "video+vision-old" or args.old_code)]
    jobs = [(m, name, scene, c, repeat) for repeat in range(args.repeat) for name, scene in scenes.items() for c in conditions for m in models]
    print(f"{len(jobs)} calls: {len(models)} models x {len(scenes)} scenes x {len(conditions)} conditions x {args.repeat}", flush=True)
    (out / "protocol.json").write_text(json.dumps({"run_id": run_id, "repeat": args.repeat, "truth": TRUTH, "question": QUESTION, "extra": EXTRA, "conditions": conditions,
                                                   "models": [m[:2] for m in models], "slots": slots, "fields": FIELDS},
                                                  indent=2), encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, (out / "raw.jsonl").open("a", encoding="utf-8") as log:
        for row in pool.map(lambda job: ask(*job[:4], args.max_tokens, slots, repeat=job[4], run_id=run_id), jobs):
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            got = sum(v[0] for v in row["score"]["fields"].values())
            total = sum(v[1] for v in row["score"]["fields"].values())
            print(f"  {row['model']:<16} {row['scene']:<17} {row['condition']:<17} {got}/{total}  {row['seconds']}s"
                  + ("" if row["ok"] else f"  HTTP {row['status']}") + ("" if row["parsed"] is not None or not row["ok"] else "  (unparseable)"),
                  flush=True)


if __name__ == "__main__":
    main()
