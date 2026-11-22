"""Tell an agent when the screen has stopped changing after an action, and what changed.

An agent that clicks and then screenshots too early reads a half-open dialog; one that sleeps a
fixed second wastes time on every step. `SettleWatcher` takes frames as they arrive and answers
as soon as the screen has been quiet for `quiet_ms` after the last change:

    watcher = SettleWatcher()
    for t_ms, frame in capture_screen():
        if watcher.push(frame, t_ms):
            break
    result = watcher.result()   # state, timing, changed regions in source pixels

States:
- ``settled``: something changed and has been still for ``quiet_ms``;
- ``unchanged``: nothing changed within ``expect_change_ms`` (the action may have had no effect);
- ``timeout``: still changing at ``timeout_ms``;
- ``ended``: the frames ran out first (recorded clips).

"Quiet" means no frame differs from its predecessor or from the frame ``quiet_ms`` earlier, so the
slow tail of an ease-out keeps the wait going even when each step is below the threshold.
``last_change_ms`` is the last frame that broke that, so it can trail the true end by a few frames.

Regions that keep changing for ``ambient_ms`` (video, an animated backdrop, a blinking caret) stop
counting as change and are reported separately. A loading spinner is indistinguishable from such a
region, so callers that must wait for loading to finish should set ``ambient_ms=0`` or watch that
region on its own. Fixed areas such as a clock can be excluded with ``ignore``. An agent that keeps
one watcher for a session and calls ``rearm()`` before each action learns such areas only once.

Comparison happens on a copy scaled so the longest side is at most ``max_dim``; reported boxes are
in the pixels of the frames that were pushed.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence

import cv2
import numpy as np

from .segment import ACTIVE, changed_mask

Box = tuple[int, int, int, int]
STATES = ("settled", "unchanged", "timeout", "ended")
MAX_REGIONS = 32
GAP_MS = 250.0  # A longer pause between frames is time the caller spent acting, not watching.


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


class SettleWatcher:
    def __init__(
        self,
        *,
        quiet_ms: float = 200.0,
        expect_change_ms: float = 1000.0,
        timeout_ms: float = 10_000.0,
        ambient_ms: float = 3000.0,
        threshold: int = 12,
        cell: int = 16,
        max_dim: int = 960,
        ignore: Sequence[Box] = (),
        min_region_px: int = 16,
    ) -> None:
        self.quiet_ms = _finite("quiet_ms", quiet_ms, positive=True)
        self.expect_change_ms = _finite("expect_change_ms", expect_change_ms)
        self.timeout_ms = _finite("timeout_ms", timeout_ms, positive=True)
        self.ambient_ms = _finite("ambient_ms", ambient_ms)
        if type(threshold) is not int or not 0 <= threshold < 255:
            raise ValueError("threshold must be an integer in [0, 255)")
        if type(cell) is not int or cell < 2:
            raise ValueError("cell must be an integer of at least 2")
        if type(max_dim) is not int or max_dim < 16:
            raise ValueError("max_dim must be an integer of at least 16")
        self.threshold, self.cell, self.max_dim = threshold, cell, max_dim
        self.ignore = [tuple(int(v) for v in box) for box in ignore]
        for box in self.ignore:
            if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(
                    "ignore boxes must be [left, top, right, bottom] with positive size"
                )
        self.min_region_px = min_region_px
        # A run of changes survives short pauses: a caret blinks, a video stalls for a frame.
        self.pause_ms = min(600.0, self.ambient_ms / 4) if self.ambient_ms else 0.0
        self.frames = 0
        self._state: str | None = None
        self._result: dict | None = None

    # -- frame intake -------------------------------------------------------------------------
    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            raise ValueError("frames must be uint8 arrays")
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        h, w = frame.shape[:2]
        if self.frames == 0:
            self.size = (w, h)
            self.scale = min(1.0, self.max_dim / max(w, h))
            self.small = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
            self.grid = (
                (self.small[0] + self.cell - 1) // self.cell,
                (self.small[1] + self.cell - 1) // self.cell,
            )
            self.ignored = np.zeros(self.small[::-1], bool)
            for left, top, right, bottom in self.ignore:
                self.ignored[
                    self._to_small(top, h) : self._to_small(bottom, h, up=True),
                    self._to_small(left, w) : self._to_small(right, w, up=True),
                ] = True
        elif (w, h) != self.size:
            raise ValueError("all frames must have the size of the first frame")
        if (w, h) != self.small:
            frame = cv2.resize(frame, self.small, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(frame, (3, 3), 0)

    def _to_small(self, v: int, limit: int, up: bool = False) -> int:
        v = min(max(v, 0), limit) * self.scale
        return math.ceil(v) if up else math.floor(v)

    def push(self, frame: np.ndarray, t_ms: float) -> bool:
        """Add a BGR, BGRA or grayscale frame of constant size. Returns True once decided.

        Frames pushed after the decision still teach the watcher about restless areas and become
        the baseline for ``rearm()``; the decided result and ``last_frame`` stay as they were.
        """
        if not math.isfinite(t_ms):
            raise ValueError("timestamps must be finite")
        small = self._prepare(frame)
        self._latest = frame
        if self.frames == 0:
            self.last_frame = frame
            self.t0 = self.t = float(t_ms)
            self.baseline = self.prev = small
            shape = self.grid[::-1]
            self.run_start = np.full(shape, np.nan)
            self.last_active = np.full(shape, -np.inf)
            self.ambient_seen = np.zeros(shape, bool)
            self.first_change = self.last_change = self.last_step = None
            self.clock = self.clock0 = 0.0
            self.step = 1000.0 / 30
            self.history = deque([(0.0, small)])
            self.frames = 1
            self.arm_offset = 0
            return self._decide()
        if t_ms <= self.t:
            raise ValueError("timestamps must increase")
        # Waiting windows and restless-area runs advance by observed time only: while the caller
        # is busy acting and pushes no frames, runs neither end nor grow and no wait expires.
        # A change made during such a gap still shows as the difference across it.
        dt = t_ms - self.t
        if dt > GAP_MS:
            dt = self.step
        else:
            self.step = dt
        self.clock += dt
        self.t = float(t_ms)
        self.frames += 1
        mask = changed_mask(self.prev, small, self.threshold)
        mask[self.ignored] = False
        self.prev = small
        active = (
            cv2.resize(mask.astype(np.float32), self.grid, interpolation=cv2.INTER_AREA) > ACTIVE
        )
        # The tail of an ease-out moves less than the threshold per frame but still moves, so
        # also compare with the frame from `quiet_ms` ago: "quiet" means no drift over the window.
        self.history.append((self.clock, small))
        while len(self.history) > 1 and self.history[1][0] <= self.clock - self.quiet_ms:
            self.history.popleft()
        drift = changed_mask(self.history[0][1], small, self.threshold)
        drift[self.ignored] = False
        drifting = (
            cv2.resize(drift.astype(np.float32), self.grid, interpolation=cv2.INTER_AREA) > ACTIVE
        )
        ambient = np.zeros_like(active)
        if self.ambient_ms:
            now = self.clock
            ended = ~np.isnan(self.run_start) & (now - self.last_active > self.pause_ms)
            self.run_start[ended] = np.nan
            self.run_start[active & np.isnan(self.run_start)] = now
            self.last_active[active] = now
            ambient = ~np.isnan(self.run_start) & (now - self.run_start >= self.ambient_ms)
            self.ambient_seen |= ambient
            # Cells touching a restless area flicker with it (anti-aliased edges, motion blur).
            ambient = cv2.dilate(ambient.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if self._state is not None:
            return True
        self.last_frame = frame
        stepped = (active & ~ambient).any()
        if stepped:
            self.last_step = self.t
        # Drift right after a step only echoes that step; it matters once steps have stopped.
        drifted = (drifting & ~ambient).any() and (
            self.last_step is None or self.t - self.last_step >= self.quiet_ms
        )
        if stepped or drifted:
            if self.first_change is None:
                self.first_change = self.t
            self.last_change = self.t
        return self._decide()

    def _decide(self) -> bool:
        elapsed = self.clock - self.clock0
        if self.last_change is not None and self.t - self.last_change >= self.quiet_ms:
            self._state = "settled"
        elif self.last_change is None and elapsed >= self.expect_change_ms:
            self._state = "unchanged"
        elif elapsed >= self.timeout_ms:
            self._state = "timeout"
        if self._state is None:
            return False
        self._result = self._build_result()
        return True

    def update(
        self,
        *,
        quiet_ms: float | None = None,
        expect_change_ms: float | None = None,
        timeout_ms: float | None = None,
    ) -> None:
        """Change how long to wait without forgetting the baseline or learned restless areas."""
        if quiet_ms is not None:
            self.quiet_ms = _finite("quiet_ms", quiet_ms, positive=True)
        if expect_change_ms is not None:
            self.expect_change_ms = _finite("expect_change_ms", expect_change_ms)
        if timeout_ms is not None:
            self.timeout_ms = _finite("timeout_ms", timeout_ms, positive=True)

    def rearm(self) -> None:
        """Wait for the next action from the latest frame, keeping what was learned.

        Restless areas found so far stay ambient, so a playing video is learned once per session
        instead of delaying every step by ``ambient_ms``. Call it just before acting.
        """
        if self.frames == 0:
            return
        self._state = self._result = None
        self.t0 = self.t
        self.clock0 = self.clock
        self.baseline = self.prev
        self.last_frame = self._latest
        self.first_change = self.last_change = self.last_step = None
        self.arm_offset = self.frames - 1

    @property
    def state(self) -> str | None:
        return self._state

    # -- result -------------------------------------------------------------------------------
    def _boxes(self, mask: np.ndarray, *, join: int) -> list[dict]:
        if join:
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((join, join), np.uint8)) > 0
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        w, h = self.size
        boxes = []
        for x, y, bw, bh, _ in stats[1:]:
            box = (
                max(0, math.floor(x / self.scale)),
                max(0, math.floor(y / self.scale)),
                min(w, math.ceil((x + bw) / self.scale)),
                min(h, math.ceil((y + bh) / self.scale)),
            )
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area >= self.min_region_px:
                boxes.append({"bbox_px": list(box), "area_px": area})
        boxes.sort(key=lambda r: -r["area_px"])
        return boxes

    def result(self) -> dict:
        """The decided result, or the current state as ``ended`` if no decision was reached."""
        if self.frames == 0:
            raise ValueError("no frames were pushed")
        return self._result if self._result is not None else self._build_result()

    def _build_result(self) -> dict:
        state = self._state or "ended"
        rel = lambda t: None if t is None else round(t - self.t0, 3)  # noqa: E731
        out = {
            "version": "motionspec.settle/1",
            "state": state,
            "frames": self.frames - self.arm_offset,
            "size_px": list(self.size),
            "elapsed_ms": rel(self.t),
            "first_change_ms": rel(self.first_change),
            "last_change_ms": rel(self.last_change),
            "quiet_for_ms": None
            if self.last_change is None
            else round(self.t - self.last_change, 3),
            "settings": {
                "quiet_ms": self.quiet_ms,
                "expect_change_ms": self.expect_change_ms,
                "timeout_ms": self.timeout_ms,
                "ambient_ms": self.ambient_ms,
                "threshold": self.threshold,
                "ignore": [list(b) for b in self.ignore],
            },
        }
        # Net change from the first frame to the last; changes that reverted are not listed.
        net = changed_mask(self.baseline, self.prev, self.threshold) > 0
        net &= ~self.ignored
        restless = np.zeros(self.small[::-1], bool)
        if self.ambient_seen.any():
            cells = cv2.dilate(self.ambient_seen.astype(np.uint8), np.ones((3, 3), np.uint8))
            restless = cv2.resize(cells, self.small, interpolation=cv2.INTER_NEAREST) > 0
            net &= ~restless
        regions = self._boxes(net, join=5)
        out["changed_regions"] = regions[:MAX_REGIONS]
        out["changed_regions_truncated"] = len(regions) > MAX_REGIONS
        out["ambient_regions"] = self._boxes(restless, join=0)[:MAX_REGIONS]
        return out


def watch(frames: Iterable[tuple[float, np.ndarray]], **settings) -> tuple[dict, np.ndarray | None]:
    """Consume (t_ms, frame) pairs until decided; return the result and the last frame."""
    watcher = SettleWatcher(**settings)
    for t_ms, frame in frames:
        if watcher.push(frame, t_ms):
            break
    if watcher.frames == 0:
        raise ValueError("the source produced no frames")
    return watcher.result(), watcher.last_frame


def _dpi_aware() -> None:
    import sys

    if sys.platform == "win32":
        try:  # Physical pixels, so boxes match what the agent clicks on scaled displays.
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


def _grabber(region: Box | None, backend: str):
    """Return (name, grab) where grab() gives one BGR frame of `region` or the primary screen."""
    if backend not in ("auto", "mss", "pil"):
        raise ValueError("capture backend must be auto, mss or pil")
    if region is not None:
        left, top, right, bottom = (int(v) for v in region)
        if left >= right or top >= bottom:
            raise ValueError("capture region must have positive width and height")
    if backend in ("auto", "mss"):
        try:
            import mss
        except ImportError:
            if backend == "mss":
                raise
        else:
            shooter = (getattr(mss, "MSS", None) or mss.mss)()
            area = (
                shooter.monitors[1]
                if region is None
                else {"left": left, "top": top, "width": right - left, "height": bottom - top}
            )
            return "mss", lambda: np.ascontiguousarray(np.asarray(shooter.grab(area))[:, :, :3])
    from PIL import ImageGrab

    def grab() -> np.ndarray:
        image = ImageGrab.grab(bbox=region, all_screens=region is not None)
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    return "pil", grab


def capture_screen(
    region: Box | None = None, *, max_fps: float = 30.0, backend: str = "auto"
) -> Iterator[tuple[float, np.ndarray]]:
    """Grab the screen (or `region`, in physical pixels) as BGR frames with ms timestamps.

    Uses mss when installed (``pip install mss``; about 30 fps for a 2560x1600 screen and 90 fps
    for a 1280x800 window here), otherwise Pillow (about 15 and 20 fps).
    """
    interval = 1.0 / _finite("max_fps", max_fps, positive=True)
    _dpi_aware()
    _, grab = _grabber(region, backend)
    while True:
        start = time.perf_counter()
        frame = grab()
        yield start * 1000.0, frame
        spare = interval - (time.perf_counter() - start)
        if spare > 0:
            time.sleep(spare)
