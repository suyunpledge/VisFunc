"""Record isolated Chromium pages and their CDP/rAF ground truth in one run."""

from __future__ import annotations

import argparse
import base64
import json
import random
import subprocess
import tempfile
from pathlib import Path

from motion_corpus import BENCH, cases, pack, save
from playwright.sync_api import sync_playwright

EASINGS = ("linear", "ease-in", "ease-out", "ease-in-out")


def encode(frames, folder):
    """Preserve compositor timestamps with millisecond precision and verify encoded PTS."""
    pts = [round((t - frames[0][0]) * 1000) for t, _ in frames]
    if sorted(set(pts)) != pts or len(pts) < 5:
        raise ValueError("insufficient or non-increasing screencast timestamps")
    with tempfile.TemporaryDirectory() as work:
        temp = Path(work)
        lines = ["ffconcat version 1.0"]
        for i, (_, image) in enumerate(frames):
            name = f"{i:04}.jpg"
            (temp / name).write_bytes(base64.b64decode(image))
            lines += [f"file {name}", "option framerate 1000"]
            if i + 1 < len(pts):
                lines.append(f"duration {(pts[i + 1] - pts[i]) / 1000:.3f}")
        (temp / "frames.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
        video = (folder / "video.mp4").resolve()
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-safe",
                "0",
                "-f",
                "concat",
                "-i",
                "frames.txt",
                "-fps_mode",
                "vfr",
                "-enc_time_base",
                "1:1000",
                "-video_track_timescale",
                "1000",
                "-c:v",
                "libx264",
                "-bf",
                "0",
                "-crf",
                "23",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            cwd=temp,
            check=True,
        )
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ]
    )
    actual = [
        round(float(f["best_effort_timestamp_time"]) * 1000) for f in json.loads(probe)["frames"]
    ]
    if actual != pts:
        raise ValueError("encoded PTS differ from screencast")
    return pts


def record(browser, case, folder):
    rng = random.Random(case["seed"])
    config = {
        **case,
        "x": rng.randint(36, 100),
        "y": rng.randint(54, 89),
        "span": rng.randint(40, 82),
        "duration": rng.choice((240, 300, 360, 420)),
        "css": rng.choice(EASINGS),
    }
    page = browser.new_page(viewport={"width": 480, "height": 280}, device_scale_factor=1)
    session = page.context.new_cdp_session(page)
    started, frames = [], []

    def receive(event):
        frames.append((event["metadata"]["timestamp"], event["data"]))
        session.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    try:
        page.goto((BENCH / "browser_scene.html").as_uri())
        page.evaluate("c => window.prepare(c)", config)
        session.send("Animation.enable")
        session.on("Animation.animationStarted", lambda event: started.append(event["animation"]))
        session.on("Page.screencastFrame", receive)
        session.send("Page.startScreencast", {"format": "jpeg", "quality": 88, "everyNthFrame": 1})
        page.wait_for_timeout(220)
        page.evaluate("window.trigger()")
        page.wait_for_timeout(config["duration"] + 280)
        session.send("Page.stopScreencast")
        trace = page.evaluate("window.finish()")
    finally:
        page.context.close()

    if len(trace["animations"]) != 1 or not started:
        raise ValueError(f"{case['id']} has no unique recorded animation")
    motion = trace["animations"][0]
    event = started[0]
    if abs(event["source"]["duration"] - motion["duration"]) > 1:
        raise ValueError("CDP and document timing disagree")
    family = case["family"]
    easing = motion["timing"]["easing"]
    if easing == "linear" and motion["keyframes"][0]["easing"] != "linear":
        easing = motion["keyframes"][0]["easing"]
    if easing not in EASINGS or easing != config["css"]:
        raise ValueError(f"unexpected effective browser easing {easing!r} in {family}")

    folder.mkdir(parents=True, exist_ok=True)
    pts = encode(frames, folder)
    epoch0 = frames[0][0] * 1000
    start = trace["origin"] + motion["start"] - epoch0
    if not 0 < start < start + motion["duration"] < pts[-1]:
        raise ValueError("capture does not contain the entire animation")
    first, last = trace["states"][0]["entries"], trace["states"][-1]["entries"]
    rows = []
    for element, final in last.items():
        if element in first:
            initial = first[element]["rect"]
        else:  # list insertion starts at the pre-existing row's top with zero height
            x, y, r, _ = first["follower"]["rect"]
            initial = [x, y, r, y]
        x, y, r, b = initial
        xx, yy, rr, bb = final["rect"]
        rows.append(
            {
                "element": element,
                "start_ms": round(start, 3),
                "duration_ms": motion["duration"],
                "easing": easing,
                "delta": [xx - x, yy - y],
                "resize": [(rr - xx) - (r - x), (bb - yy) - (b - y)],
                "rect": [min(x, xx), min(y, yy), max(r, rr), max(b, bb)],
            }
        )
    trace.update(
        cdp_animations=started,
        chrome_version=browser.version,
        config=config,
        capture_epoch_s=[t for t, _ in frames],
        encoded_pts_ms=pts,
    )
    save(folder / "trace.json", trace)
    save(folder / "truth.json", {"segments": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--chrome")
    parser.add_argument("--m1-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with sync_playwright() as client:
        browser = client.chromium.launch(
            channel=None if args.chrome else "chrome", executable_path=args.chrome, headless=True
        )
        splits = ("m1-test",) if args.m1_test else ("dev","test")
        for split in splits:
            for case in cases("browser", split):
                record(browser, case, args.work / case["id"])
                print(case["id"], flush=True)
        browser.close()
    name = "browser-m1-test.zip" if args.m1_test else "browser.zip"
    pack("browser", args.work, args.output or BENCH / "corpus" / name, splits)


if __name__ == "__main__":
    main()
