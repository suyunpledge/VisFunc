"""A bounded, window-scoped record of sampled pixel changes between agent calls."""

from __future__ import annotations

import base64
import math
import threading
import time
from collections import deque
from uuid import uuid4

import cv2
import numpy as np

from .segment import changed_mask

MAX_CROP_BYTES = 4 * 1024 * 1024
PAGE_SIZE = 12


def _preview(frame):
    h, w = frame.shape[:2]
    scale = min(1, 512 / max(w, h))
    small = (
        cv2.resize(
            frame,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1
        else frame
    )
    return cv2.GaussianBlur(small, (3, 3), 0)


def _evidence(frame, mask, area):
    if cv2.countNonZero(mask) < 6:
        return None
    # Connect nearby glyphs so that short-lived text can be read as a line.
    joined = cv2.dilate(mask, np.ones((3, 9), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    sx, sy = frame.shape[1] / mask.shape[1], frame.shape[0] / mask.shape[0]
    boxes = []
    for x, y, w, h, _ in sorted(stats[1:], key=lambda row: -int(row[4])):
        if cv2.countNonZero(mask[y : y + h, x : x + w]) < 6:
            continue
        boxes.append([int(x * sx), int(y * sy), math.ceil((x + w) * sx), math.ceil((y + h) * sy)])
        if len(boxes) == 6:
            break
    if not boxes:
        return None
    left = max(0, min(b[0] for b in boxes) - 12)
    top = max(0, min(b[1] for b in boxes) - 10)
    right = min(frame.shape[1], max(b[2] for b in boxes) + 12)
    bottom = min(frame.shape[0], max(b[3] for b in boxes) + 10)
    crop = frame[top:bottom, left:right]
    scale = min(1, 800 / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 78])
    dx, dy = area[:2]
    meta = {
        "changed_regions_px": [[dx + x, dy + y, dx + r, dy + b] for x, y, r, b in boxes],
        "crop_bbox_px": [dx + left, dy + top, dx + right, dy + bottom],
    }
    return meta, jpeg.tobytes() if ok and jpeg.nbytes <= 64 * 1024 else None


class WindowHistory:
    """One worker owns its capture backend. Readers take snapshots and never consume events."""

    def __init__(self, title, area, baseline, duration, fps):
        self.id = uuid4().hex
        self.title, self.area = title, area
        self.duration, self.fps = duration, fps
        self.previous = _preview(baseline)
        h, w = self.previous.shape[:2]
        self.motion_counts = np.zeros((math.ceil(h / 12), math.ceil(w / 12)), np.int16)
        self.repeated_samples = 0
        self.repeated_box = None
        self.started = time.monotonic()
        self.samples = 1
        self.sampled_until_ms = 0
        self.serial = 0
        self.events = deque(maxlen=min(481, math.ceil(duration * fps) + 1))
        self.crops = {}
        self.crop_bytes = 0
        self.evicted_crops = 0
        self.state = "recording"
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def sample(self, frame, elapsed_ms):
        current = _preview(frame)
        mask = changed_mask(self.previous, current, 14)
        self.previous = current
        # Keep an animated video pane from evicting a toast in a neighbouring static pane.
        # A blinking caret has time to decay; one-off changes are kept at full sensitivity.
        grid = cv2.resize(
            mask.astype(np.float32), self.motion_counts.shape[::-1], interpolation=cv2.INTER_AREA
        )
        active = grid > 0.035
        repeated = (self.motion_counts >= 6) & active
        self.motion_counts = np.clip(self.motion_counts + np.where(active, 2, -1), 0, 8).astype(
            np.int16
        )
        repeated_box = None
        if np.any(repeated):
            moving = cv2.resize(
                repeated.astype(np.uint8), mask.shape[::-1], interpolation=cv2.INTER_NEAREST
            )
            x, y, w, h = cv2.boundingRect(moving)
            dx, dy, right, bottom = self.area
            sx, sy = (right - dx) / mask.shape[1], (bottom - dy) / mask.shape[0]
            repeated_box = [
                dx + int(x * sx),
                dy + int(y * sy),
                dx + math.ceil((x + w) * sx),
                dy + math.ceil((y + h) * sy),
            ]
            mask[moving != 0] = 0
        result = _evidence(frame, mask, self.area)
        with self.lock:
            if self.stop.is_set():
                return
            self.samples += 1
            self.sampled_until_ms = round(elapsed_ms)
            if repeated_box is not None:
                self.repeated_samples += 1
                self.repeated_box = repeated_box
            if result is not None:
                meta, jpeg = result
                self.serial += 1
                if len(self.events) == self.events.maxlen:
                    expired = self.crops.pop(self.events[0]["event_id"], b"")
                    self.crop_bytes -= len(expired)
                self.events.append(dict(meta, event_id=self.serial, at_ms=round(elapsed_ms)))
                if jpeg:
                    while self.crops and self.crop_bytes + len(jpeg) > MAX_CROP_BYTES:
                        self.crop_bytes -= len(self.crops.pop(next(iter(self.crops))))
                        self.evicted_crops += 1
                    self.crops[self.serial] = jpeg
                    self.crop_bytes += len(jpeg)

    def run(self, capture, locate):
        stream = None
        state = "completed"
        try:
            stream = capture(self.area, max_fps=self.fps)
            while not self.stop.is_set():
                now = time.monotonic()
                if now - self.started >= self.duration:
                    break
                if tuple(locate(self.title)) != self.area:
                    state = "window_changed"
                    break
                _, frame = next(stream)
                elapsed = time.monotonic() - self.started
                if elapsed > self.duration:
                    break
                if frame.shape[:2] != (self.area[3] - self.area[1], self.area[2] - self.area[0]):
                    state = "window_changed"
                    break
                self.sample(frame, elapsed * 1000)
                self.stop.wait(max(0, 1 / self.fps - (time.monotonic() - now)))
        except Exception:
            state = "capture_unavailable"
        finally:
            try:
                if stream is not None and hasattr(stream, "close"):
                    stream.close()
            finally:
                with self.lock:
                    self.state = "stopped" if self.stop.is_set() else state

    def snapshot(self, after, event_id, region=None):
        with self.lock:
            rows = [
                (meta.copy(), self.crops.get(meta["event_id"]))
                for meta in self.events
                if meta["event_id"] > after
                and event_id in (None, meta["event_id"])
                and (
                    region is None
                    or any(
                        region[0] < r and x < region[2] and region[1] < b and y < region[3]
                        for x, y, r, b in meta["changed_regions_px"]
                    )
                )
            ]
            first = self.events[0]["event_id"] if self.events else self.serial + 1
            info = {
                "history_id": self.id,
                "window_title": self.title,
                "area_px": list(self.area),
                "state": self.state,
                "samples": self.samples,
                "sampled_until_ms": self.sampled_until_ms,
                "repeated_motion_samples": self.repeated_samples,
                "last_repeated_motion_bbox_px": self.repeated_box,
                "latest_event_id": self.serial,
                "missed_events": max(0, first - 1 - after),
                "evicted_crops": self.evicted_crops,
                "has_more": len(rows) > PAGE_SIZE,
            }
        return info, rows[:PAGE_SIZE]

    def finish(self):
        with self.lock:
            self.stop.set()
            if self.state == "recording":
                self.state = "stopped"


class ScreenHistory:
    def __init__(self):
        self.watches = {}

    def start(self, title, area, frame, duration, fps, capture, locate):
        watch = WindowHistory(title, area, frame, duration, fps)
        if len(self.watches) >= 4:
            oldest = self.watches.pop(next(iter(self.watches)))
            oldest.finish()
        self.watches[watch.id] = watch
        threading.Thread(
            target=watch.run, args=(capture, locate), daemon=True, name="motionspec-window-history"
        ).start()
        return {
            "history_id": watch.id,
            "sample_hz": fps,
            "duration_s": duration,
            "tool": "read_screen_history",
        }

    def finish_window(self, title, area):
        for watch in self.watches.values():
            if watch.title.casefold() == title.casefold() or watch.area == area:
                watch.finish()

    def close(self):
        for watch in self.watches.values():
            watch.finish()
        self.watches.clear()

    def read(self, history_id, after, event_id, reader, stop, include_image=False, region=None):
        watch = self.watches.get(history_id)
        if watch is None:
            raise ValueError("unknown or evicted history_id")
        if (
            type(after) is not int
            or after < 0
            or (event_id is not None and (type(event_id) is not int or event_id < 1))
        ):
            raise ValueError("invalid history cursor or event_id")
        if include_image and event_id is None:
            raise ValueError("include_image requires a single event_id")
        if region is not None and (
            len(region) != 4
            or not all(type(x) is int for x in region)
            or region[0] >= region[2]
            or region[1] >= region[3]
        ):
            raise ValueError("region must be a positive screen-pixel rectangle")
        if stop:
            watch.finish()
        info, rows = watch.snapshot(after, event_id, region)
        for meta, crop in rows:
            meta["crop_available"] = crop is not None
            if include_image and crop:
                info["event_jpeg"] = base64.b64encode(crop).decode("ascii")
        if reader is not None:
            # At most two OCR calls per request. Older events remain individually queryable.
            for meta, crop in rows[-2:]:
                if crop:
                    image = cv2.imdecode(np.frombuffer(crop, np.uint8), cv2.IMREAD_COLOR)
                    try:
                        meta["text"] = [line["text"] for line in reader(image)["lines"][:16]]
                    except Exception as exc:
                        meta["ocr_error"] = str(exc)
        return {**info, "events": [meta for meta, _ in rows]}
