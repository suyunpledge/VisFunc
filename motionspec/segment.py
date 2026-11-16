"""Find and rank motion regions while retaining only recent grid state."""

from __future__ import annotations

import heapq
from array import array
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .frames import FrameSource

LAG = 6
QUIET = 4
GAP = 3
ACTIVE = 0.02  # A grid cell counts as changed above this fraction of changed pixels.


@dataclass
class Candidate:
    first: int  # Local index of the first changed frame, compared with its predecessor.
    last: int
    bbox: tuple[int, int, int, int]
    mass: float
    peak_area: float


@dataclass
class Scan:
    times: array
    first_id: int
    size: tuple[int, int]
    candidates: list[Candidate]
    dropped: int
    ambient: list[dict] = field(default_factory=list)
    ambient_cells: np.ndarray | None = None  # Grid cells that were ambient at some point.


class _Ambient:
    """Hold grids back for `window_ms` so cells in long continuous runs never reach the components.

    A region that keeps changing (an animated backdrop, a video, a spinner) is not one motion event.
    Left in, it joins every change that touches it in space and time into a single clip-long candidate,
    which then gets truncated and takes the real UI motion with it. Looking ahead by the window length
    tells us, before a grid is released, whether each changed cell belongs to such a run.
    """

    def __init__(self, shape: tuple[int, int], window_ms: float, emit):
        self.window_ms, self.emit = window_ms, emit
        # A run survives short pauses: the fringe of a moving backdrop and a blinking caret both flicker.
        self.pause_ms = min(600.0, window_ms / 4)
        self.kernel = np.ones((3, 3), np.uint8)
        self.run_start = np.full(shape, -1, np.int64)
        self.run_start_t = np.zeros(shape)
        self.last = np.full(shape, -GAP - 2, np.int64)
        self.last_t = np.zeros(shape)
        self.buffer: deque[tuple[int, float, np.ndarray, np.ndarray]] = deque()
        self.frames = np.zeros(shape, np.int64)  # Frames each cell spent in an ambient run.
        self.first_t = np.full(shape, np.inf)
        self.last_seen_t = np.full(shape, -np.inf)

    def push(self, index: int, t: float, grid: np.ndarray) -> None:
        active = grid > ACTIVE
        ended = (self.run_start >= 0) & (t - self.last_t > self.pause_ms)
        if ended.any():
            long = ended & (self.last_t - self.run_start_t >= self.window_ms)
            if long.any():
                for at, _, _, mask in self.buffer:
                    mask |= long & (self.run_start <= at) & (at <= self.last)
            self.run_start[ended] = -1
        fresh = active & (self.run_start < 0)
        self.run_start[fresh] = index
        self.run_start_t[fresh] = t
        self.last[active] = index
        self.last_t[active] = t
        self.buffer.append((index, t, grid, np.zeros(grid.shape, bool)))
        while self.buffer and t - self.buffer[0][1] > self.window_ms:
            self._release()

    def _release(self) -> None:
        index, t, grid, mask = self.buffer.popleft()
        alive = (
            (self.run_start >= 0)
            & (self.run_start <= index)
            & (index <= self.last)
            & (self.last_t - self.run_start_t >= self.window_ms)
        )
        mask = (mask | alive) & (grid > ACTIVE)
        if mask.any():
            # Changed cells touching an ambient cell belong to the same restless area.
            mask = (cv2.dilate(mask.astype(np.uint8), self.kernel) > 0) & (grid > ACTIVE)
            self.frames[mask] += 1
            self.first_t[mask] = np.minimum(self.first_t[mask], t)
            self.last_seen_t[mask] = np.maximum(self.last_seen_t[mask], t)
            grid = grid.copy()
            grid[mask] = 0.0
        self.emit(index, grid)

    def finish(self, size: tuple[int, int]) -> list[dict]:
        while self.buffer:
            self._release()
        seen = (self.frames > 0).astype(np.uint8)
        if not seen.any():
            return []
        gh, gw = seen.shape
        w, h = size
        count, labels = cv2.connectedComponents(seen, connectivity=8)
        regions = []
        for label in range(1, count):
            cells = labels == label
            ys, xs = np.nonzero(cells)
            regions.append(
                {
                    "bbox": (
                        int(xs.min()) * w // gw,
                        int(ys.min()) * h // gh,
                        ((int(xs.max()) + 1) * w + gw - 1) // gw,
                        ((int(ys.max()) + 1) * h + gh - 1) // gh,
                    ),
                    "t_start": float(self.first_t[cells].min()),
                    "t_end": float(self.last_seen_t[cells].max()),
                    "cells": int(cells.sum()),
                }
            )
        return sorted(regions, key=lambda r: -r["cells"])


def changed_mask(prev: np.ndarray, cur: np.ndarray, threshold: int) -> np.ndarray:
    delta = cv2.absdiff(prev, cur)
    if delta.ndim == 3:
        delta = np.maximum(np.maximum(delta[:, :, 0], delta[:, :, 1]), delta[:, :, 2])
    return (delta > threshold).astype(np.uint8)


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)


class _Components:
    def __init__(
        self,
        size: tuple[int, int],
        grid_size: tuple[int, int],
        min_area: float,
        limit: int,
    ):
        self.w, self.h = size
        self.gw, self.gh = grid_size
        self.min_area, self.limit = min_area, limit
        self.last_id = np.full((self.gh, self.gw), -1, np.int64)
        self.last_t = np.full_like(self.last_id, -GAP - 1)
        self.kernel = np.ones((3, 3), np.uint8)
        self.active: dict[int, Candidate] = {}
        self.best: list[tuple[float, int, Candidate]] = []
        self.total = 0
        self.next_id = 0

    def _finish(self, cid: int) -> None:
        candidate = self.active.pop(cid)
        self.total += 1
        if candidate.peak_area < self.min_area or self.limit == 0:
            return
        item = (candidate.mass, -cid, candidate)
        if len(self.best) < self.limit:
            heapq.heappush(self.best, item)
        elif item > self.best[0]:
            heapq.heapreplace(self.best, item)

    def add(self, t: int, grid: np.ndarray) -> None:
        for cid, candidate in list(self.active.items()):
            if t - candidate.last > GAP:
                self._finish(cid)
        active = (grid > ACTIVE).astype(np.uint8)
        if not active.any():
            return
        count, labels = cv2.connectedComponents(
            cv2.dilate(active, self.kernel), connectivity=8
        )
        for label in range(1, count):
            region = labels == label
            core = region & (active != 0)
            ys, xs = np.nonzero(core)
            area = float(grid[core].sum()) * (self.w / self.gw) * (self.h / self.gh)
            bbox = (
                int(xs.min()) * self.w // self.gw,
                int(ys.min()) * self.h // self.gh,
                ((int(xs.max()) + 1) * self.w + self.gw - 1) // self.gw,
                ((int(ys.max()) + 1) * self.h + self.gh - 1) // self.gh,
            )
            previous = self.last_id[region & (t - self.last_t <= GAP)]
            ids = [int(cid) for cid in np.unique(previous) if cid >= 0]
            if ids:
                cid = ids[0]
                group = self.active[cid]
                for other in ids[1:]:
                    merged = self.active.pop(other)
                    self.last_id[self.last_id == other] = cid
                    self._merge(
                        group,
                        merged.bbox,
                        merged.mass,
                        merged.peak_area,
                        merged.first,
                        merged.last,
                    )
                self._merge(group, bbox, area, area, t, t)
            else:
                cid = self.next_id
                self.next_id += 1
                self.active[cid] = Candidate(t, t, bbox, area, area)
            self.last_id[region] = cid
            self.last_t[region] = t

    @staticmethod
    def _merge(
        group: Candidate,
        bbox: tuple[int, int, int, int],
        mass: float,
        peak: float,
        first: int,
        last: int,
    ) -> None:
        x0, y0, x1, y1 = group.bbox
        group.bbox = (
            min(x0, bbox[0]),
            min(y0, bbox[1]),
            max(x1, bbox[2]),
            max(y1, bbox[3]),
        )
        group.mass += mass
        group.peak_area = max(group.peak_area, peak)
        group.first = min(group.first, first)
        group.last = max(group.last, last)

    def finish(self) -> tuple[list[Candidate], int]:
        for cid in list(self.active):
            self._finish(cid)
        kept = sorted(
            (item[2] for item in self.best),
            key=lambda c: (c.first, c.bbox[1], c.bbox[0]),
        )
        return kept, self.total - len(kept)


def scan(
    source: FrameSource,
    *,
    cell: int = 16,
    threshold: int = 12,
    min_area: float = 120,
    max_segments: int = 12,
    max_frames: int = 1_000_000,
    ambient_ms: float = 3000.0,
) -> Scan:
    times = array("d")
    ambient = None
    history: deque[np.ndarray] = deque(maxlen=LAG)
    ref = None
    first_id = 0
    size = grid_size = (0, 0)
    groups = None
    for frame_id, t, frame in source:
        if len(times) >= max_frames:
            raise ValueError(
                f"selected range exceeds max_frames ({max_frames}); select a smaller interval"
            )
        # Blur before comparing; keep colour changes even when luma is constant.
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        if ref is None:
            first_id = frame_id
            h, w = blurred.shape[:2]
            size = (w, h)
            grid_size = ((w + cell - 1) // cell, (h + cell - 1) // cell)
            groups = _Components(size, grid_size, min_area, max_segments)
            if ambient_ms > 0:
                ambient = _Ambient((grid_size[1], grid_size[0]), ambient_ms, groups.add)
            ref = blurred.copy()
        else:
            mask = changed_mask(ref, blurred, threshold).astype(np.float32)
            grid = cv2.resize(mask, grid_size, interpolation=cv2.INTER_AREA)
            if ambient is not None:
                ambient.push(len(times), t, grid)
            else:
                groups.add(len(times), grid)
            if len(history) == LAG:
                restless = changed_mask(history[-1], blurred, QUIET) | changed_mask(
                    history[0], blurred, QUIET
                )
                moving = cv2.resize(
                    restless.astype(np.float32), grid_size, interpolation=cv2.INTER_AREA
                )
                settled = (
                    cv2.resize(
                        (moving < 0.004).astype(np.uint8),
                        size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0
                )
                ref[settled] = blurred[settled]
        times.append(t)
        history.append(blurred)
    regions = ambient.finish(size) if ambient is not None else []
    candidates, dropped = groups.finish() if groups is not None else ([], 0)
    cells = ambient.frames > 0 if ambient is not None and regions else None
    return Scan(times, first_id, size, candidates, dropped, regions, cells)
