"""Prepare short, frame-aligned MOT17 clips. Ground truth is only read by the scorer."""
import argparse
import configparser
import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import sleep
from urllib.request import urlopen

import cv2
import numpy as np

from motionspec import analyze
from svsg.image_functions.compiler import describe_raster
from svsg.image_functions.projection import function_context
from svsg.image_functions.raster import RasterFunction

REV = "f93908467986c77765667925a526ad07be8ad630"
BASE = f"https://huggingface.co/datasets/Lekim89/MOT17/resolve/{REV}/train"
SPLITS = {"dev": ("02", "04", "05"), "holdout": ("09", "10", "13")}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_bytes())


def download(url, path):
    if not path.exists():
        for attempt in range(3):
            try:
                with urlopen(url, timeout=40) as response:
                    data = response.read()
                path.write_bytes(data)
                break
            except OSError:
                if attempt == 2:
                    raise
                sleep(attempt + 1)
    return path.read_bytes()


def geometry(clip, last_frame):
    spec = analyze(clip, sheets=False, max_dim=480)
    evidence = describe_raster(RasterFunction.from_image(last_frame))
    image = function_context(evidence)
    grid = image.get("quantized") or image
    regions = [{k: r[k] for k in ("box", "area_px", "rgba", "shape", "centroid_x", "centroid_y")}
               for r in grid["regions"]]
    return {
        "frame_size_px": [evidence.width_px, evidence.height_px],
        "last_frame_color_grid_size": [grid.get("width", evidence.width_px), grid.get("height", evidence.height_px)],
        "color_regions_in_grid_coordinates": regions,
        "motion_segments_in_source_pixels": [
            {k: s[k] for k in ("region", "t_start_ms", "t_end_ms", "kind", "confidence")}
            for s in spec["segments"]
        ],
        "meaning": "Color regions and motion segments are geometric measurements without object labels.",
    }


def prepare(root, seq):
    folder = root / seq
    folder.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/MOT17-{seq}-FRCNN"
    ini = configparser.ConfigParser()
    ini.read_string(download(f"{url}/seqinfo.ini", folder/"seqinfo.ini").decode("utf-8-sig"))
    fps = ini.getint("Sequence", "frameRate")
    indices = [1 + round((2 + i/5)*fps) for i in range(11)]
    def fetch(i):
        raw = download(f"{url}/img1/{i:06d}.jpg", folder/f"{i:06d}.jpg")
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    with ThreadPoolExecutor(max_workers=5) as pool:
        frames = list(pool.map(fetch, indices))
    h,w = frames[0].shape[:2]
    clip = folder / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w,h))
    if not writer.isOpened():
        raise RuntimeError("cannot encode evaluation video")
    try:
        for frame in [*frames, frames[-1], frames[-1]]:
            writer.write(frame)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(clip))
    decoded = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded.append(frame)
    cap.release()
    if len(decoded) != 13:
        raise ValueError("unexpected encoded frame count")
    last = folder/"last.png"
    if not cv2.imwrite(str(last), decoded[-1]):
        raise OSError("cannot write evaluation frame")
    save(folder/"geometry.json", geometry(clip, last))
    gt = download(f"{url}/gt/gt.txt", folder/"gt.txt")
    save(folder/"source.json", {"revision": REV, "sequence": seq, "source_frames": indices,
         "last_frame": indices[-1], "size": [w,h], "clip_sha256": sha256(clip.read_bytes()).hexdigest(),
         "last_png_sha256": sha256(last.read_bytes()).hexdigest(), "gt_sha256": sha256(gt).hexdigest()})
    print("prepared", seq, indices[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for split in SPLITS.values():
        for sequence in split:
            prepare(args.root, sequence)
