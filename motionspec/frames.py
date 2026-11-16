"""帧来源：视频文件、图片目录或内存帧序列，统一成 (序号, 时间戳ms, 分析尺度BGR帧)。

录屏通常是可变帧率，所以时间戳优先取容器里的真实值，取不到才退回 序号/帧率。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class InvalidTimestamps(ValueError):
    """The current decode pass cannot be used as a time index."""


def _image_order(path: Path) -> tuple:
    parts = re.split(r"([0-9]+)", path.name.casefold())
    return (tuple(int(part) if i % 2 else part for i, part in enumerate(parts)), path.name)


@dataclass
class SourceInfo:
    path: str
    width: int
    height: int
    fps: float
    scale: float  # 分析尺度 / 源尺度
    timestamps: str = "container"  # container | index


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread 在 Windows 上读不了非 ASCII 路径，走字节解码。"""
    data = np.frombuffer(path.read_bytes(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"cannot encode image for {path}")
    path.write_bytes(buf.tobytes())


class FrameSource:
    def __init__(
        self,
        source: str | Path | Sequence[np.ndarray],
        *,
        fps: float | None = None,
        max_dim: int = 960,
        start_ms: float | None = None,
        end_ms: float | None = None,
    ) -> None:
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps must be finite and positive")
        if type(max_dim) is not int or max_dim <= 0:
            raise ValueError("max_dim must be a positive integer")
        for name, value in (("start_ms", start_ms), ("end_ms", end_ms)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        self._frames: Sequence[np.ndarray] | None = None
        self._images: list[Path] | None = None
        self._video: str | None = None
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.max_dim = max_dim
        self.fps_assumed = False

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                if fps is None:
                    raise ValueError("image directories require an explicit fps")
                self._images = sorted(
                    (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT),
                    key=_image_order,
                )
                if not self._images:
                    raise ValueError(f"no images in {path}")
                first = read_image(self._images[0])
                if first is None:
                    raise ValueError(f"cannot read {self._images[0]}")
                h, w = first.shape[:2]
                src_fps = fps
                stamps = "index"
            elif path.is_file():
                self._video = str(path)
                cap = cv2.VideoCapture(self._video)
                try:
                    if not cap.isOpened():
                        raise ValueError(f"cannot open video {path}")
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    reported = float(cap.get(cv2.CAP_PROP_FPS))
                finally:
                    cap.release()
                self.fps_assumed = fps is None and (not math.isfinite(reported) or reported <= 0)
                src_fps = fps if fps is not None else 30.0 if self.fps_assumed else reported
                stamps = "index" if fps is not None else "container"
            else:
                raise FileNotFoundError(str(path))
            name = str(path)
        else:
            if len(source) < 2:
                raise ValueError("need at least 2 frames")
            self._frames = source
            h, w = self._bgr(source[0]).shape[:2]
            src_fps = fps or 60.0
            stamps = "index"
            name = "<frames>"

        if w <= 0 or h <= 0:
            raise ValueError("source has no frames")
        scale = min(1.0, max_dim / max(w, h))
        self.info = SourceInfo(name, w, h, float(src_fps), scale, stamps)
        self._size = (max(1, round(w * scale)), max(1, round(h * scale)))

    @staticmethod
    def _bgr(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or not frame.size:
            raise ValueError("frames must be nonempty uint8 arrays")
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frames must be grayscale, BGR or BGRA")
        return frame

    def resized(self, frame: np.ndarray) -> np.ndarray:
        frame = self._bgr(frame)
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        return frame

    def _raw(self) -> Iterator[tuple[int, float, np.ndarray]]:
        step = 1000.0 / self.info.fps
        if self._frames is not None:
            for i, frame in enumerate(self._frames):
                yield i, i * step, frame
        elif self._images is not None:
            for i, path in enumerate(self._images):
                frame = read_image(path)
                if frame is None:
                    raise ValueError(f"cannot read {path}")
                yield i, i * step, frame
        else:
            cap = cv2.VideoCapture(self._video)
            try:
                if not cap.isOpened():
                    raise ValueError(f"cannot open video {self._video}")
                i = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # 解码之后再取：POS_MSEC 返回的是刚读出那一帧的时间戳（实测，取早了会错一帧）
                    stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                    yield i, stamp, frame
                    i += 1
            finally:
                cap.release()

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        for i, t, frame in self.selected_raw():
            yield i, t, self.resized(frame)

    def selected_raw(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Use one clock for the pass and validate it before applying the interval."""
        previous = -1.0
        step = 1000.0 / self.info.fps
        for i, stamp, frame in self._raw():
            t = i * step if self.info.timestamps == "index" else stamp
            if not math.isfinite(t) or t < 0 or t <= previous:
                raise InvalidTimestamps(f"invalid timestamp at source frame {i}")
            previous = t
            if self.start_ms is not None and t < self.start_ms:
                continue
            if self.end_ms is not None and t > self.end_ms:
                break
            frame = self._bgr(frame)
            if frame.shape[:2] != (self.info.height, self.info.width):
                raise ValueError(f"source frame {i} has different dimensions")
            yield i, t, frame
