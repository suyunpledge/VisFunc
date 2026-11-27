"""Freeze media for V0–V5, without using truth to crop, select frames or summarize."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from motion_corpus import BENCH, TOOLS, cases, save

sys.path.insert(0, str(TOOLS.parent))
from motionspec import analyze
from motionspec.spec import summarize
from motionspec.trace import spec_from_trace


def select(source, split="test", per_family=2):
    counts = Counter()
    chosen = []
    for case in cases(source, split):
        family = case.get("family")
        if family and counts[family] < per_family:
            counts[family] += 1
            chosen.append(case["id"])
    return chosen


def frame_data(video):
    cap = cv2.VideoCapture(str(video))
    frames, times = [], []
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            frames.append(img)
    finally:
        cap.release()
    if len(frames) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("video lacks distinct presentation timestamps")
    return frames, np.array(times)


def make_dense(frames, times, folder):
    # Motion-only crops use pixel differences, independent of the analyzer and ground truth.
    mask = np.zeros(frames[0].shape[:2], np.uint8)
    for img in frames[1:]:
        mask |= (cv2.absdiff(img, frames[0]).max(axis=2) > 18).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    parts = [s for s in stats[1:] if s[4] >= 36]
    if parts:
        x = max(0, min(s[0] for s in parts) - 8)
        y = max(0, min(s[1] for s in parts) - 8)
        r = min(w, max(s[0] + s[2] for s in parts) + 8)
        b = min(h, max(s[1] + s[3] for s in parts) + 8)
    else:
        x, y, r, b = 0, 0, w, h
    variants = []
    # The smallest contact sheet already covers twelve instants across the complete clip.
    for count in (1, 2, 3, 4, 6, 8):
        ids = np.unique(
            np.round(np.linspace(0, len(frames) - 1, min(len(frames), 12 * count))).astype(int)
        )
        names = []
        tw = 144
        th = max(32, min(118, round(tw * (b - y) / (r - x))))
        for n, group in enumerate(np.array_split(ids, math.ceil(len(ids) / 12))):
            sheet = np.full((3 * (th + 22), 4 * tw, 3), 28, np.uint8)
            for j, idx in enumerate(group):
                tile = cv2.resize(frames[idx][y:b, x:r], (tw, th), interpolation=cv2.INTER_AREA)
                left, top = (j % 4) * tw, (j // 4) * (th + 22)
                sheet[top + 22 : top + 22 + th, left : left + tw] = tile
                cv2.putText(
                    sheet,
                    f"{times[idx]:.0f} ms",
                    (left + 4, top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            name = f"dense-{count}-{n}.png"
            ok, png = cv2.imencode(".png", sheet)
            if not ok:
                raise ValueError("PNG encoder failed")
            (folder / name).write_bytes(png.tobytes())
            names.append(name)
        variants.append(
            {"images": names, "frames": len(ids), "timestamps_ms": times[ids].round(3).tolist()}
        )
    return {"crop_xyxy": list(map(int, (x, y, r, b))), "tile_px": [tw, th], "variants": variants}


def prepare(work, source, scene, archive):
    folder = work / scene
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "clip.mp4"
    video.write_bytes(archive.read(scene + "/video.mp4"))
    spec = analyze(video, folder / "evidence", functions=True, sheets=False)
    (folder / "summary.txt").write_text(summarize(spec), encoding="utf8")
    if source == "qt":
        trace = json.loads(archive.read(scene + "/trace.json"))
        (folder / "probe.txt").write_text(summarize(spec_from_trace(trace)), encoding="utf8")
    frames, times = frame_data(video)
    sparse = []
    for i in range(8):
        t = i * 100
        if t > times[-1]:
            break
        idx = int(np.argmin(np.abs(times - t)))
        name = f"frame-{i}.png"
        (folder / name).write_bytes(cv2.imencode(".png", frames[idx])[1].tobytes())
        sparse.append([name, round(float(times[idx]), 3)])
    save(
        folder / "media.json",
        {
            "size": [frames[0].shape[1], frames[0].shape[0]],
            "sparse": sparse,
            "duration_ms": round(float(times[-1]), 3),
            "dense": make_dense(frames, times, folder),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    for source in ("qt", "browser"):
        with ZipFile(BENCH / "corpus" / (source + ".zip")) as archive:
            for scene in [*select(source), select(source, "dev", 1)[0]]:
                prepare(args.work, source, scene, archive)
