"""模型实测：同一段界面动效视频，接入 motionspec 前后，模型测得准不准。

    python tools/qt_scenes.py out/qt          （先生成素材，需要 PySide6）
    python bench/run.py out/qt bench/results/run1

三种条件：
    video            只给视频（各家用其最高抽帧率）
    video+vision     视频 + motionspec 视觉分析的摘要和标注图（只需要一段录屏）
    video+probe      视频 + Qt 探针的摘要（需要能在应用里挂探针）

密钥从本机 Forge 配置读取，只在进程内使用；原始回答存到结果目录，不含密钥。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
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
    # (显示名, 模型名, 提供方匹配, 视频抽帧率；None 表示这个模型拿不到视频、改传抽帧图片)
    ("mimo-v2.6-flash", "mimo-v2.6-flash", r"mimo-v2\.6-flash$", 10),
    ("mimo-v2.6-pro", "mimo-v2.6-pro", r"mimo-v2\.6-pro$", 10),
    ("seed-2.1-turbo", "doubao-seed-2-1-turbo-260628", r"volces", 5),
    ("seed-2.1-pro", "doubao-seed-2-1-pro-260915", r"volces", 5),
    ("gpt-6.1-sol", "gpt-6.1-sol", r"openai\.com", None),
    ("gpt-6.1-terra", "gpt-6.1-terra", r"openai\.com", None),
]
CONDITIONS = ("video", "video+vision", "video+probe")
FAMILY = {"linear": "linear", "out-cubic": "ease-out", "in-cubic": "ease-in", "in-out-quad": "ease-in-out", "in-out-cubic": "ease-in-out"}
FRAME_FPS = 10  # 给拿不到视频的模型抽帧用

QUESTION = """This is a screen recording of a desktop application: {width}x{height} pixels, {duration} ms long, recorded at 60 frames per second.
Measure every UI animation in it. One entry per animated element; ignore elements that only move because their parent moves.

Reply with JSON only, no prose, in exactly this shape:
{{"animations": [{{"element": "<short description>", "start_ms": <number>, "duration_ms": <number>,
  "direction": "up|down|left|right|none", "distance_px": <number>,
  "opacity_change": "fade-in|fade-out|none", "easing": "linear|ease-in|ease-out|ease-in-out"}}]}}

Rules:
- start_ms is measured from the start of the recording; duration_ms is the designed duration of the animation.
- distance_px is in pixels of this recording; use direction "none" and distance 0 for an element that does not move.
- easing: linear = constant speed; ease-in = starts slow, ends fast; ease-out = starts fast, ends slow; ease-in-out = slow at both ends.
{extra}"""

EXTRA = {
    "video": "",
    "video+vision": (
        "- A motion analysis of this same recording follows, made by a tool that tracks pixels frame by frame "
        "(a summary table, then one annotated image per segment). Use it together with the recording.\n"
    ),
    "video+probe": (
        "- A motion trace of this same recording follows, read directly from the application's widget properties "
        "while it was recorded. Use it together with the recording.\n"
    ),
}


def prepare(root: Path, work: Path) -> dict:
    """为每个场景准备：mp4、抽帧图片、视觉分析摘要与标注图、探针摘要。"""
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    scenes = {}
    for name, scene in truth["scenes"].items():
        folder = work / name
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / "clip.mp4"
        if not video.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(truth["fps"]), "-i",
                            str(root / name / "frames" / "%04d.png"), "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", str(video)], check=True)
        vision = analyze(video, folder / "vision", raw_frames=False)
        spec_from_trace(json.loads((root / name / "trace.json").read_text(encoding="utf-8")), folder / "probe")
        stills = []
        frames = sorted((root / name / "frames").glob("*.png"))
        step = max(1, round(truth["fps"] / FRAME_FPS))
        for index in range(0, len(frames), step):
            small = folder / f"still-{index:04d}.png"
            if not small.exists():
                write_image(small, read_image(frames[index]))
            stills.append((round(index * 1000 / truth["fps"]), small))
        ratio = vision["source"]["width"] / truth["size"][0]
        wanted = []
        for seg in scene["segments"]:
            delta = [v * ratio for v in seg.get("delta", [0, 0])]
            axis = 0 if abs(delta[0]) >= abs(delta[1]) else 1
            direction = "none" if max(map(abs, delta)) == 0 else (("right", "left") if axis == 0 else ("down", "up"))[delta[axis] < 0]
            fade = "none" if "opacity" not in seg and seg["kind"] != "fade" else (
                "fade-out" if seg.get("opacity") == [1, 0] else "fade-in")
            wanted.append({"start_ms": seg["start_ms"], "duration_ms": seg["duration_ms"], "direction": direction,
                           "distance_px": round(max(map(abs, delta)), 1), "opacity_change": fade,
                           "easing": FAMILY[seg["easing"]]})
        scenes[name] = {
            "video": video, "stills": stills, "truth": wanted,
            "width": vision["source"]["width"], "height": vision["source"]["height"],
            "duration": round(vision["source"]["duration_ms"]),
            "vision_summary": (folder / "vision" / "summary.md").read_text(encoding="utf-8"),
            "vision_sheets": [folder / "vision" / seg["sheet"] for seg in vision["segments"] if seg.get("sheet")],
            "probe_summary": (folder / "probe" / "summary.md").read_text(encoding="utf-8"),
        }
    return scenes


def build(scene: dict, condition: str, vendor: str, fps: int | None) -> list[dict]:
    parts = []
    if fps is not None:
        parts.append(client.video(scene["video"], vendor, fps=fps))
        media = ""
    else:  # 拿不到视频的模型：按时间戳给抽帧图片
        for at, path in scene["stills"]:
            parts.append(client.text(f"Frame at {at} ms:"))
            parts.append(client.image(path))
        media = f"- The recording is given as still frames sampled every {round(1000 / FRAME_FPS)} ms, each labelled with its time.\n"
    question = QUESTION.format(width=scene["width"], height=scene["height"], duration=scene["duration"], extra=media + EXTRA[condition])
    parts.append(client.text(question))
    if condition == "video+vision":
        parts.append(client.text("Motion analysis summary:\n\n" + scene["vision_summary"]))
        for sheet in scene["vision_sheets"]:
            parts.append(client.text(f"Annotated key frames for segment {sheet.stem}:"))
            parts.append(client.image(sheet))
    elif condition == "video+probe":
        parts.append(client.text("Widget property trace summary:\n\n" + scene["probe_summary"]))
    return parts


def parse(text: str) -> list[dict] | None:
    delivery = decode_measurement(text)
    return delivery["rows"] if delivery["accepted"] else None


def number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def score(truth: list[dict], answer: list[dict] | None) -> dict:
    """逐项打分。先把回答里的动效和真值按起点时间一一配对，再比各字段。"""
    fields = ("start_ms", "duration_ms", "direction", "distance_px", "opacity_change", "easing")
    result = {name: [0, 0] for name in (*fields, "count")}
    result["count"] = [int(answer is not None and len(answer) == len(truth)), 1]
    free = list(answer or [])
    for want in truth:
        got = None
        if free:
            def cost(row, want=want):
                start = number(row.get("start_ms"))
                mismatch = 0 if (str(row.get("direction")) == want["direction"]) else 150
                return (abs(start - want["start_ms"]) if start is not None else 1e6) + mismatch
            got = min(free, key=cost)
            free.remove(got)
        for name in fields:
            result[name][1] += 1
            if got is None:
                continue
            if name == "start_ms":
                value = number(got.get(name))
                ok = value is not None and abs(value - want[name]) <= 34  # 两帧
            elif name == "duration_ms":
                value = number(got.get(name))
                ok = value is not None and abs(value - want[name]) <= max(17, 0.15 * want[name])
            elif name == "distance_px":
                value = number(got.get(name))
                ok = value is not None and abs(value - want[name]) <= max(3, 0.10 * want[name])
            else:
                ok = str(got.get(name, "")).strip().lower() == want[name]
            result[name][0] += int(ok)
    return result


def ask(model_row, scene_name: str, scene: dict, condition: str, *, max_attempts=2, max_tokens=2000,
        provider_pattern=None) -> dict:
    if not 1 <= max_attempts <= 4 or not 1 <= max_tokens <= 8000:
        raise ValueError("invalid attempt or output token budget")
    label, model, pattern, fps = model_row
    provider = find(provider_pattern or pattern)
    vendor = client.vendor_of(provider.base_url)
    parts = build(scene, condition, vendor, fps)
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
    delivery = decode_measurement(reply["text"] if reply["ok"] else "")
    answer = delivery["rows"]
    scores = score(scene["truth"], answer)
    wrong = any(v[0] != v[1] for v in scores.values())
    return {"model": label, "scene": scene_name, "condition": condition, "ok": reply["ok"], "status": reply["status"],
            "seconds": round(time.perf_counter() - began, 3), "usage": sum_usage(attempts),
            "provider_slot": getattr(provider, "slot", None), "base_url": provider.base_url,
            "requested_model": model,
            "request_seconds": sum(a["seconds"] for a in attempts), "attempts": attempts,
            "usage_unknown_attempts": sum(not bool(a.get("usage")) for a in attempts),
            "media": f"video@{fps}fps" if fps is not None else f"{len(scene['stills'])} stills@{FRAME_FPS}fps",
            "text": reply["text"], "error": None if reply["ok"] else str(reply["error"])[:400],
            "delivery": delivery, "transport_fail": not reply["ok"],
            "format_fail": not delivery["strict_format_pass"] if reply["ok"] else None,
            "answer_wrong": wrong if answer is not None else None,
            "parsed": answer, "score": scores,
            "end_to_end_score": scores if delivery["accepted"] else score(scene["truth"], None)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", help="folder written by tools/qt_scenes.py")
    parser.add_argument("out")
    parser.add_argument("--models", help="comma-separated display names; default all")
    parser.add_argument("--scenes-only", help="comma-separated scene names; default all")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--providers", help="JSON file mapping model labels to exact Forge slot regexes")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = prepare(Path(args.scenes), out / "data")
    if args.scenes_only:
        scenes = {k: v for k, v in scenes.items() if k in args.scenes_only.split(",")}
    models = [m for m in MODELS if not args.models or m[0] in args.models.split(",")]
    providers = json.loads(Path(args.providers).read_text(encoding="utf-8")) if args.providers else {}
    for model in models:
        find(providers.get(model[0], model[2]))  # Fail ambiguous selection before any paid call.
    conditions = [c for c in args.conditions.split(",") if c in CONDITIONS]
    # 相邻的调用轮流打到不同模型上，避免同时对一家并发太多
    jobs = [(m, name, scene, c) for _ in range(args.repeat) for name, scene in scenes.items() for c in conditions for m in models]
    print(f"{len(jobs)} calls: {len(models)} models x {len(scenes)} scenes x {len(conditions)} conditions x {args.repeat}")
    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, (out / "raw.jsonl").open("a", encoding="utf-8") as log:
        for row in pool.map(lambda job: ask(*job, max_attempts=args.max_attempts,
                                          max_tokens=args.max_tokens,
                                          provider_pattern=providers.get(job[0][0])), jobs):
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            got, total = (sum(v[i] for v in row["score"].values()) for i in (0, 1))
            print(f"  {row['model']:<16} {row['scene']:<13} {row['condition']:<13} {got}/{total}  {row['seconds']}s"
                  + ("" if row["ok"] else f"  HTTP {row['status']}") + ("" if row["parsed"] is not None or not row["ok"] else "  (unparseable)"))
    (out / "truth.json").write_text(json.dumps({k: v["truth"] for k, v in scenes.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
