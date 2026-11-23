"""Replay a moving video alongside a known 625 ms notice; read it back from the screen cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from motionspec.frames import FrameSource
from motionspec.screen_history import ScreenHistory, WindowHistory

from svsg.recognition.screen_text import ScreenTextReader


def replay(video: Path, start_s: float, code: str, reader: ScreenTextReader) -> dict:
    start = start_s * 1000
    source = FrameSource(video, start_ms=start, end_ms=start + 8000, max_dim=420)
    stream = iter(source)
    previous = next(stream, None)
    if previous is None:
        raise ValueError(f"empty video interval in {video}")
    upcoming = next(stream, None)
    pool = ScreenHistory()
    watch = None
    for index in range(64):
        elapsed = index * 125
        target = start + elapsed
        while upcoming is not None and abs(upcoming[1] - target) < abs(previous[1] - target):
            previous, upcoming = upcoming, next(stream, None)
        if upcoming is None and target - previous[1] > 125:
            raise ValueError(f"{video} needs eight seconds of video after {start_s:g} s")
        frame = previous[2]
        h, w = frame.shape[:2]
        canvas = np.full((h + 84, w, 3), 250, np.uint8)
        canvas[:h] = frame
        label = f"PULSE {code}" if 2000 <= elapsed < 2625 else "Waiting"
        cv2.putText(
            canvas, label, (25, h + 51), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 2, cv2.LINE_AA
        )
        if watch is None:
            watch = WindowHistory("video replay", (0, 0, w, h + 84), canvas, 8, 8)
            pool.watches[watch.id] = watch
        else:
            watch.sample(canvas, elapsed)
    cursor = 0
    hits = []
    while True:
        page = pool.read(watch.id, cursor, None, None, False, region=[0, h + 12, w, h + 78])
        for row in page["events"]:
            if 2000 <= row["at_ms"] < 2625:
                event = pool.read(watch.id, 0, row["event_id"], reader.read, False)["events"][0]
                hits += event.get("text", [])
        if not page["has_more"]:
            break
        cursor = page["events"][-1]["event_id"]
    return {
        "video": video.name,
        "expected": code,
        "sampled_frames": page["samples"],
        "events": page["latest_event_id"],
        "repeated_motion_samples": page["repeated_motion_samples"],
        "evicted_crops": page["evicted_crops"],
        "text": hits,
        "recovered": code in " ".join(hits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--start", type=float, default=8)
    args = parser.parse_args()
    reader = ScreenTextReader()
    results = [
        replay(path, args.start, str(62841 + i), reader) for i, path in enumerate(args.videos)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(row["recovered"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
