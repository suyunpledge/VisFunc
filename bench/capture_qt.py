"""Render the Qt split, saving the same-run property trace and H.264 clip."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

from motion_corpus import BENCH, TOOLS, cases, pack, save

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_SCALE_FACTOR"] = "1"
sys.path.insert(0, str(TOOLS))
from PySide6.QtCore import QEasingCurve, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QWidget
from qt_hard_scenes import faded, frame
from qt_scenes import FPS, STYLE, ease, load_probe

CODEC = ["-an", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-threads", "1"]
CURVES = {
    "linear": QEasingCurve.Linear,
    "out-cubic": QEasingCurve.OutCubic,
    "in-cubic": QEasingCurve.InCubic,
    "in-out-quad": QEasingCurve.InOutQuad,
}


def render(case, folder, app):
    rng = random.Random(case["seed"])
    family = case["family"]
    root = QWidget()
    root.setObjectName("root")
    root.setStyleSheet(STYLE)
    root.setFixedSize(480, 280)
    root.setAttribute(Qt.WA_DontShowOnScreen, True)
    x, y = rng.randint(100, 165), rng.randint(35, 65)
    start = rng.randint(8, 16) * 1000 / FPS
    duration = rng.randint(13, 28) * 1000 / FPS
    curve = rng.choice(sorted(CURVES))
    span = rng.randint(32, 90)
    elements = []
    truth = []

    def add(name, box, delta=(0, 0), resize=(0, 0), fade=False, delay=0):
        bx, by, w, h = box
        dx, dy = delta
        dw, dh = resize
        widget = frame(root, "row", *box, name)
        alpha = faded(widget, 0) if fade else None
        truth.append(
            {
                "element": name,
                "start_ms": start + delay,
                "duration_ms": duration,
                "easing": curve,
                "delta": list(delta),
                "resize": list(resize),
                "rect": [
                    min(bx, bx + dx),
                    min(by, by + dy),
                    max(bx + w, bx + dx + w + dw),
                    max(by + h, by + dy + h + dh),
                ],
                **({"opacity": [0, 1]} if fade else {}),
            }
        )
        elements.append((widget, box, delta, resize, alpha, delay))

    if family == "panel":
        add("Details", (x, y, 136, 20), resize=(0, span))
        add("Next", (x, y + 29, 136, 36), delta=(0, span))
    elif family == "sidebar":
        add("Menu", (20, y, 140, 150), resize=(-span, 0))
        add("Content", (170, y + 30, 120, 48), delta=(-span, 0))
    elif family == "overlap":
        add("Sync", (x, y, 92, 40), delta=(span, 0))
        add(
            "Queued",
            (x + 50, y + 100, 110, 42),
            delta=(-span, 0),
            delay=rng.randint(2, 5) * 1000 / FPS,
        )
    elif family == "fade":
        add("Saved", (x, y, 120, 50), fade=True)
    elif family == "vertical":
        add("Tools", (x, y + 75, 124, 42), delta=(0, rng.choice((-span, span))))
    else:
        add("Account", (x, y, 112, 50), delta=(rng.choice((-span, span)), 0))

    root.show()
    probe = load_probe().Recorder([root])
    folder.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "480x280",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        *CODEC,
        str(folder / "video.mp4"),
    ]
    encoder = subprocess.Popen(args, stdin=subprocess.PIPE)
    try:
        for i in range(math.ceil((start + duration + 280) * FPS / 1000) + 1):
            stamp = i * 1000 / FPS
            for widget, (bx, by, w, h), (dx, dy), (dw, dh), alpha, delay in elements:
                p = ease(CURVES[curve], stamp, start + delay, duration)
                widget.setGeometry(
                    bx + round(dx * p), by + round(dy * p), w + round(dw * p), h + round(dh * p)
                )
                if alpha:
                    alpha.setOpacity(p)
            app.processEvents()
            image = root.grab().toImage().convertToFormat(QImage.Format_RGB888)
            assert (image.width(), image.height(), image.bytesPerLine()) == (480, 280, 1440)
            encoder.stdin.write(image.constBits())
            probe.sample(stamp)
    finally:
        encoder.stdin.close()
        status = encoder.wait()
        root.close()
    if status:
        raise RuntimeError(f"ffmpeg exited {status}")
    save(folder / "truth.json", {"segments": truth})
    probe.dump(folder / "trace.json")


def legacy(work, group):
    path = work / ("legacy-" + group)
    script = "qt_scenes.py" if group == "basic" else "qt_hard_scenes.py"
    subprocess.run([sys.executable, str(TOOLS / script), str(path)], check=True)
    data = json.loads((path / "truth.json").read_text(encoding="utf8"))
    rects = {
        "card-slide": [[210, 110, 710, 260]],
        "toast-enter": [[600, 480, 860, 548]],
        "toast-exit": [[600, 464, 860, 524]],
        "toast-fade": [[600, 480, 860, 524]],
        "list-stagger": [[560, 110 + i * 50, 860, 166 + i * 50] for i in range(3)],
        "two-at-once": [[210, 110, 510, 320], [600, 480, 860, 524]],
    }
    for name, scene in data["scenes"].items():
        folder = work / name
        folder.mkdir(exist_ok=True)
        for i, row in enumerate(scene["segments"]):
            if group == "basic":
                row["rect"] = rects[name][i]
            row.setdefault("element", f"item-{i + 1}")
            row.setdefault("delta", [0, 0])
            size = row.get("size", {})
            row["resize"] = [
                size[k][1] - size[k][0] if k in size else 0 for k in ("width", "height")
            ]
        save(folder / "truth.json", scene)
        (folder / "trace.json").write_bytes((path / name / "trace.json").read_bytes())
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(path / name / "frames/%04d.png"),
                *CODEC,
                str(folder / "video.mp4"),
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="empty staging directory")
    parser.add_argument("--output", type=Path, default=BENCH / "corpus/qt.zip")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    app = QApplication([])
    for group in ("basic", "hard"):
        legacy(args.work, group)
    for split in ("dev", "test"):
        for case in cases("qt", split):
            if "legacy" not in case:
                render(case, args.work / case["id"], app)
    pack("qt", args.work, args.output)
