"""Screen tools for agents, kept resident so models and learned state survive between steps.

`ScreenAgent` is the Python API; `serve_mcp()` exposes it as a Model Context Protocol server over
stdio (``motionspec agent-mcp``), so agent frameworks can call it as tools. All coordinates are
physical screen pixels [left, top, right, bottom], the space a click is sent in.

Each step of an agent is one call: ``click_element`` or ``set_field_text`` takes a baseline
frame, acts through UI Automation, waits until the window is still and returns what changed, the
text there and any dialog that opened. ``wait_for_settle`` is for longer processes (a sync, a
page load) and for changes made by the agent's own input tools; ``find_on_screen`` and
``list_ui_elements`` locate targets.

Areas that keep changing (video, animated backgrounds) are learned once per region and then
ignored, because one `SettleWatcher` per region lives as long as the server.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .action_receipts import ActionReceipts, operation_key, receipt_path
from .screen_history import ScreenHistory
from .settle import SettleWatcher, _finite, capture_screen

Box = tuple[int, int, int, int]
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_VERSION = "0.1.0"


def window_rect(title: str) -> Box:
    """Screen rectangle covering every visible window whose title contains `title`.

    Covering all of them keeps the capture area stable when the application opens a dialog with
    a related title, so the dialog shows up as a change instead of starting a new baseline.
    """
    import ctypes
    from ctypes import wintypes

    from svsg.recognition.ui_elements import find_windows

    handles = find_windows(title)
    if not handles:
        raise ValueError(f"no visible window title contains {title!r}")
    boxes = []
    for hwnd in handles:
        rect = wintypes.RECT()
        # Visible bounds, without the invisible resize border that GetWindowRect includes.
        if ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)
        ):
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        boxes.append((rect.left, rect.top, rect.right, rect.bottom))
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _offset(item: dict, dx: int, dy: int) -> dict:
    out = dict(item)
    for key in ("bbox_px", "center_px"):
        if key in out:
            v = out[key]
            out[key] = (
                [v[0] + dx, v[1] + dy, v[2] + dx, v[3] + dy]
                if len(v) == 4
                else [v[0] + dx, v[1] + dy]
            )
    if "words" in out:
        out["words"] = [_offset(w, dx, dy) for w in out["words"]]
    return out


def _draw_marker(frame, area, marker):
    x, y = marker["window_px"]
    h, w = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return
    left, top = max(0, x - 88), max(0, y - 60)
    crop = frame[top : min(h, y + 61), left : min(w, x + 89)].copy()
    center = (x - left, y - top)
    for color, width in [((0, 0, 0), 4), ((20, 220, 255), 2)]:
        cv2.circle(crop, center, 9, color, width, cv2.LINE_AA)
        cv2.drawMarker(crop, center, color, cv2.MARKER_CROSS, 9, width)
    ok, image = cv2.imencode(".png", crop)
    if ok:
        marker.update(
            crop_origin_px=[area[0] + left, area[1] + top],
            image_point_px=list(center),
            pre_action_png=base64.b64encode(image).decode("ascii"),
        )


def _compact_element(element: dict | None) -> dict | None:
    if not element:
        return None
    keep = ("name", "control_type", "enabled", "value", "toggle_state", "center_px", "element_ref")
    return {k: element[k] for k in keep if k in element}


def _control_changes(before: list[dict], after: list[dict], *, locations=False) -> list[dict]:
    """What an action changed in a window's controls: states, values, controls added or removed."""

    def group(c):
        return (c.get("name", ""), c.get("control_type", ""))

    counts = Counter(map(group, before)) | Counter(map(group, after))

    def key(c):
        name, kind = group(c)
        identity = c.get("element_ref") or tuple(c.get("center_px", []))
        return name, kind, identity if counts[name, kind] > 1 else None

    def identity(k, c):
        result = {"name": k[0], "control_type": k[1]}
        if k[2] is not None:
            result.update({f: c[f] for f in ("element_ref", "center_px") if f in c})
        elif locations and "center_px" in c:
            result["center_px"] = c["center_px"]
        return result

    old = {key(c): c for c in before}
    new = {key(c): c for c in after}
    changes = []
    for k, c in new.items():
        if k not in old:
            changes.append({**identity(k, c), "appeared": True})
            continue
        for field in ("enabled", "value", "toggle_state"):
            if old[k].get(field) != c.get(field):
                changes.append(
                    {
                        **identity(k, c),
                        field: c.get(field),
                        "was": old[k].get(field),
                    }
                )
    changes += [{**identity(k, old[k]), "gone": True} for k in old if k not in new]
    return changes[:20]


def _ui_delta(before, after):
    changed_window = before["title"] != after["title"] or before.get("window_id") != after.get(
        "window_id"
    )
    changes = _control_changes(
        [] if changed_window else before["controls"], after["controls"], locations=True
    )
    if not changes and not changed_window:
        return None
    event = {"kind": "window" if changed_window else "controls", "changes": []}
    if changed_window:
        event["window_title"] = after["title"][:160]
        if len(after["title"]) > 160:
            event["truncated"] = True
    for change in changes[:12]:
        bounded = {
            key: value[:160] if isinstance(value, str) else value for key, value in change.items()
        }
        if bounded != change:
            event["truncated"] = True
        event["changes"].append(bounded)
    if len(changes) > 12:
        event["truncated"] = True
    return event


def _window_summary(window: dict) -> dict:
    if len(window["controls"]) <= 30:
        return window
    return {**window, "controls": window["controls"][:30], "truncated": True}


CAPTION_NAMES = {
    "最小化",
    "最大化",
    "还原",
    "关闭",
    "系统",
    "Minimize",
    "Maximize",
    "Restore",
    "Close",
    "System",
}
BLOCKED_NOTE = (
    "Screen capture of this window does not show changes (for example a browser screen "
    "protection feature of security software), so screenshots and OCR of it are "
    "unreliable; effects are read from UI Automation instead."
)


def _without_caption(elements: list[dict], window_box: Sequence[int] | None) -> list[dict]:
    """Drop the title bar's system buttons and menu: always present, rarely what a task needs."""
    if not window_box:
        return elements
    top = window_box[1]
    return [
        e
        for e in elements
        if not (
            e.get("name") in CAPTION_NAMES
            and e.get("bbox_px")
            and e["bbox_px"][1] - top < 64
            and e.get("control_type") in ("button", "menu_bar", "menu_item")
        )
    ]


class DesktopUnavailable(RuntimeError):
    """The user's desktop cannot be seen or driven right now; nothing was done."""


def parse_allow_rule(entry: str) -> tuple[str, str | None]:
    """``title`` or ``title | program.exe`` -> (title part, program name or None), case-folded.

    A rule with a program only matches windows of that executable, so another application
    cannot borrow an allowed title. An empty title part allows every window of the program.
    """
    title, sep, program = entry.rpartition("|")
    if sep and program.strip().casefold().endswith(".exe"):
        return title.strip().casefold(), program.strip().casefold()
    return entry.strip().casefold(), None


class _Stream:
    """A capture generator shared by calls on one area; one caller at a time pulls frames."""

    def __init__(self, generator):
        self.generator = generator
        self.lock = threading.Lock()
        self.last_used = time.monotonic()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            self.last_used = time.monotonic()
            return next(self.generator)

    def close(self):
        with self.lock:
            if hasattr(self.generator, "close"):
                self.generator.close()


class ScreenAgent:
    """Resident screen tools. Dependencies are injectable for tests."""

    def __init__(
        self,
        *,
        capture: Callable[..., Iterator[tuple[float, np.ndarray]]] = capture_screen,
        reader_factory: Callable[[], Any] | None = None,
        elements: Callable[..., dict] | None = None,
        locate_window: Callable[[str], Box] = window_rect,
        act: Callable[..., dict] | None = None,
        windows: Callable[[str], list[dict]] | None = None,
        click: Callable[..., dict] | None = None,
        max_fps: float = 60.0,
        allowed_windows: Sequence[str] = (),
        max_wait_ms: float | None = None,
        images: bool = True,
        history_seconds: float = 20.0,
        process_name: Callable[[int], str] | None = None,
        desktop: Callable[[], str] | None = None,
    ) -> None:
        self.capture = capture
        self.max_fps = max_fps
        self._act = act
        self._windows = windows
        self._click = click
        self.blocked: set[str] = set()  # window titles whose capture never shows changes
        self.last_controls: dict[str, dict] = {}
        self.allow_rules = [parse_allow_rule(w) for w in allowed_windows if w.strip()]
        self.allowed_windows = [title for title, _ in self.allow_rules]  # truthy = actions on
        self._process_name = process_name
        self._desktop = desktop
        self._state_lock = threading.RLock()  # streams, watchers and armed baselines
        self._armed_areas: set = set()
        self.max_wait_ms = max_wait_ms
        self.images = images
        self.history_seconds = min(60.0, _finite("history_seconds", history_seconds))
        self.history = ScreenHistory()
        self._reader_factory = reader_factory
        self._reader = None
        self._elements = elements
        self.locate_window = locate_window
        self.watchers: dict[Box | None, SettleWatcher] = {}
        self.streams: dict[Box | None, Iterator] = {}

    # -- helpers ------------------------------------------------------------------------------
    @property
    def reader(self) -> Any:
        if self._reader is None:
            if self._reader_factory is None:
                from svsg.recognition.screen_text import ScreenTextReader

                self._reader_factory = ScreenTextReader
            self._reader = self._reader_factory()
        return self._reader

    def _list_elements(self, **kw) -> dict:
        if self._elements is None:
            from svsg.recognition.ui_elements import list_elements

            self._elements = list_elements
        return self._elements(**kw)

    def resolve_region(self, region: Sequence[int] | None, window_title: str | None) -> Box | None:
        if region is not None and window_title is not None:
            raise ValueError("give either region or window_title")
        if window_title is not None:
            return tuple(self.locate_window(window_title))
        if region is not None:
            if len(region) != 4 or region[0] >= region[2] or region[1] >= region[3]:
                raise ValueError("region must be [left, top, right, bottom] with positive size")
            return tuple(int(v) for v in region)
        return None

    def _frames(self, area: Box | None) -> Iterator[tuple[float, np.ndarray]]:
        with self._state_lock:
            stream = self.streams.get(area)
            if stream is None:
                if len(self.streams) >= 8:
                    # Evict the least recently used stream that nobody has pulled for 30 s.
                    idle = [
                        (s.last_used, key)
                        for key, s in self.streams.items()
                        if time.monotonic() - s.last_used > 30 and not s.lock.locked()
                    ]
                    if idle:
                        oldest = min(idle, key=lambda row: row[0])[1]
                        self.streams.pop(oldest).close()
                        self.watchers.pop(oldest, None)
                stream = self.streams[area] = _Stream(self.capture(area, max_fps=self.max_fps))
            return stream

    def close(self):
        self.history.close()
        with self._state_lock:
            for stream in self.streams.values():
                stream.close()
            self.streams.clear()
            self.watchers.clear()

    # -- production guards --------------------------------------------------------------------
    def desktop_state(self) -> str:
        if self._desktop is None:
            try:
                from svsg.recognition.ui_elements import desktop_state
            except ImportError:  # pragma: no cover - svsg is installed alongside
                return "interactive"
            self._desktop = desktop_state
        return self._desktop()

    def require_desktop(self) -> None:
        state = self.desktop_state()
        if state == "interactive":
            return
        reason = {
            "locked": "the Windows session is locked or a secure desktop (UAC, sign-in) is in "
            "front; unlock it and retry",
            "service": "the server runs as a service in session 0, which has no user desktop; "
            "start it from the signed-in user's session",
        }.get(state, f"the desktop is not available ({state})")
        raise DesktopUnavailable(f"{reason}. Nothing was captured or clicked.")

    def _permit(self, hwnd: int, actual: str) -> None:
        """Raise unless the window matches an allowed title (and program, when one is given)."""
        title = actual.casefold()
        program = None
        for allowed, exe in self.allow_rules:
            if allowed not in title:
                continue
            if exe is None:
                return
            if program is None:
                if self._process_name is None:
                    from svsg.recognition.ui_elements import window_process_name

                    self._process_name = window_process_name
                program = self._process_name(hwnd)
            if program == exe:
                return
        shown = f" ({program})" if program else ""
        raise PermissionError(f"window {actual!r}{shown} is not in the allowed list")

    def watch_screen_changes(self, *, window_title: str) -> dict:
        if not self.history_seconds or not window_title.strip():
            raise ValueError("screen history requires an enabled recorder and a window title")
        area = self.resolve_region(None, window_title)
        before = self._top_window(window_title)
        try:
            frame = None if window_title.casefold() in self.blocked else self.grab(area)
        except Exception:
            if before is None:
                raise
            frame = None
        return self._start_history(window_title, area, frame, before)

    def _start_history(self, window_title, area, frame, before):
        from svsg.recognition.ui_elements import release_thread_client

        return self.history.start(
            window_title,
            area,
            frame,
            self.history_seconds,
            min(8.0, self.max_fps),
            self.capture,
            self.locate_window,
            observe=lambda: self._top_window(window_title),
            before=before,
            delta=_ui_delta,
            cleanup=release_thread_client,
        )

    def read_screen_history(
        self,
        *,
        history_id: str,
        after: int = 0,
        event_id: int | None = None,
        read_text: bool = False,
        stop: bool = False,
        include_image: bool = False,
        region: Sequence[int] | None = None,
        compact: bool = False,
    ) -> dict:
        read = (lambda image: self.reader.read(image)) if read_text else None
        return self.history.read(
            history_id, after, event_id, read, stop, include_image and self.images, region, compact
        )

    def grab(self, area: Box | None) -> np.ndarray:
        return next(self._frames(area))[1]

    # -- tools --------------------------------------------------------------------------------
    def wait_for_settle(
        self,
        *,
        region: Sequence[int] | None = None,
        window_title: str | None = None,
        quiet_ms: float = 200.0,
        expect_change_ms: float = 1000.0,
        timeout_ms: float = 10_000.0,
        ambient_ms: float = 3000.0,
        read_text: bool = False,
    ) -> dict:
        if region is None and window_title and window_title.casefold() in self.blocked:
            key = window_title.casefold()
            before = self.last_controls.get(key)
            baseline = "last_observation" if before else "first_observation_of_this_call"
            if before is None:
                before = self._top_window(window_title)
            settle, top = self._settle_by_elements(
                window_title,
                before,
                quiet_ms=quiet_ms,
                expect_change_ms=expect_change_ms,
                timeout_ms=timeout_ms,
            )
            settle.update(baseline=baseline, source="ui_automation", note=BLOCKED_NOTE)
            if top:
                self.last_controls[key] = top
                settle["controls_changed"] = _control_changes(
                    before["controls"] if before else [], top["controls"]
                )
                if read_text:
                    settle["window"] = _window_summary(top)
            return settle
        area = self.resolve_region(region, window_title)
        return self._settle(
            area,
            quiet_ms=quiet_ms,
            expect_change_ms=expect_change_ms,
            timeout_ms=timeout_ms,
            ambient_ms=ambient_ms,
            read_text=read_text,
        )

    def _watcher(self, area: Box | None, ambient_ms: float, waits: dict) -> tuple[Any, bool]:
        with self._state_lock:
            watcher = self.watchers.get(area)
            # Only ambient_ms changes what has been learned; the waits can change between calls.
            fresh = watcher is None or watcher.ambient_ms != float(ambient_ms)
            if fresh:
                watcher = self.watchers[area] = SettleWatcher(ambient_ms=ambient_ms, **waits)
            else:
                watcher.update(**waits)
            return watcher, fresh

    def _mark_baseline(self, area: Box | None, ambient_ms: float = 3000.0) -> np.ndarray:
        """Make the current screen the baseline the next settle compares against."""
        watcher, _ = self._watcher(area, ambient_ms, {})
        t_ms, frame = next(self._frames(area))
        watcher.push(frame, t_ms)
        watcher.rearm()
        with self._state_lock:
            self._armed_areas.add(area)
        return frame

    def _settle(
        self,
        area: Box | None,
        *,
        quiet_ms: float,
        expect_change_ms: float,
        timeout_ms: float,
        ambient_ms: float,
        read_text: bool,
    ) -> dict:
        if self.max_wait_ms is not None:  # e.g. the client gives up on a request after 10 s
            timeout_ms = min(timeout_ms, self.max_wait_ms)
            expect_change_ms = min(expect_change_ms, self.max_wait_ms)
        waits = {
            "quiet_ms": quiet_ms,
            "expect_change_ms": expect_change_ms,
            "timeout_ms": timeout_ms,
        }
        watcher, fresh = self._watcher(area, ambient_ms, waits)
        with self._state_lock:
            armed = area in self._armed_areas
            self._armed_areas.discard(area)
        if not fresh and not armed:
            watcher.rearm()  # otherwise _mark_baseline already set the baseline
        for t_ms, frame in self._frames(area):
            if watcher.push(frame, t_ms):
                break
        result = watcher.result()
        dx, dy = (area[0], area[1]) if area else (0, 0)
        for key in ("changed_regions", "ambient_regions"):
            result[key] = [_offset(r, dx, dy) for r in result[key]]
        result["area_px"] = list(area) if area else None
        result["baseline"] = "first_frame_of_this_call" if fresh else "last_frame_of_previous_call"
        if read_text:
            from svsg.recognition.screen_text import lines_touching, widen_regions

            boxes = [
                [
                    r["bbox_px"][0] - dx,
                    r["bbox_px"][1] - dy,
                    r["bbox_px"][2] - dx,
                    r["bbox_px"][3] - dy,
                ]
                for r in result["changed_regions"]
            ]
            result["text"] = None
            if boxes:
                frame = watcher.last_frame
                text = self.reader.read(frame, widen_regions(boxes, frame.shape[1]))
                result["text"] = [_offset(line, dx, dy) for line in lines_touching(text, boxes)]
        return result

    def read_screen_text(
        self, *, region: Sequence[int] | None = None, window_title: str | None = None
    ) -> dict:
        area = self.resolve_region(region, window_title)
        result = self.reader.read(self.grab(area))
        dx, dy = (area[0], area[1]) if area else (0, 0)
        out = {
            "area_px": list(area) if area else None,
            "lines": [_offset(line, dx, dy) for line in result["lines"]],
            "elapsed_ms": result["elapsed_ms"],
        }
        if window_title and window_title.casefold() in self.blocked:
            out["note"] = BLOCKED_NOTE
        return out

    def list_ui_elements(self, *, window_title: str | None = None) -> dict:
        result = self._list_elements(title=window_title)
        window = result.get("window") or {}
        elements = _without_caption(result["elements"], window.get("bbox_px"))
        out = {
            "window": window,
            "elements": [_compact_element(e) for e in elements],
            "truncated": result.get("truncated", False),
        }
        if window_title and window_title.casefold() in self.blocked:
            out["note"] = BLOCKED_NOTE
        return out

    def find_on_screen(
        self,
        *,
        query: str,
        window_title: str | None = None,
        exact: bool = False,
        use_ocr: bool = True,
    ) -> dict:
        """Click targets for `query`: UI Automation names first, then visible text."""
        from svsg.recognition.screen_text import find_text
        from svsg.recognition.ui_elements import find_element

        targets: list[dict] = []
        notes: list[str] = []
        try:
            elements = self._list_elements(title=window_title)
            for e in find_element(elements, query, exact=exact)[:5]:
                targets.append(
                    {
                        "source": "ui_automation",
                        **({"element_ref": e["element_ref"]} if "element_ref" in e else {}),
                        "text": e["name"],
                        "control_type": e["control_type"],
                        "enabled": e["enabled"],
                        "bbox_px": e["bbox_px"],
                        "center_px": e["center_px"],
                    }
                )
        except (OSError, ValueError, ImportError) as exc:
            notes.append(f"ui_automation unavailable: {exc}")
        if use_ocr and not any(t["source"] == "ui_automation" and t["enabled"] for t in targets):
            area = self.resolve_region(None, window_title) if window_title else None
            text = self.reader.read(self.grab(area))
            dx, dy = (area[0], area[1]) if area else (0, 0)
            for hit in find_text(text, query, exact=exact)[:5]:
                hit = _offset(hit, dx, dy)
                targets.append(
                    {
                        "source": "ocr",
                        "text": hit["text"],
                        "bbox_px": hit["bbox_px"],
                        "center_px": hit["center_px"],
                        "score": hit["score"],
                        "low_confidence": hit["low_confidence"],
                    }
                )
        if window_title and window_title.casefold() in self.blocked and use_ocr:
            notes.append(BLOCKED_NOTE)
        return {"query": query, "targets": targets, "ambiguous": len(targets) > 1, "notes": notes}

    def _act_on(
        self, window_title: str, query: str, action: str, *, marker_image=False, **kw
    ) -> dict:
        """Act inside an allowed window, then wait and report the effect in the same call."""
        if not self.allowed_windows:
            raise PermissionError("actions are disabled; start the server with --allow-window")
        self.require_desktop()
        if not window_title or not query:
            raise ValueError("window_title and target are required")
        if self._act is None:
            from svsg.recognition import ui_elements

            def act(query, action, *, title, **kw):
                hwnd = ui_elements.find_window(title)
                actual = ui_elements.window_title(hwnd)
                self._permit(hwnd, actual)
                return ui_elements.perform(query, action, hwnd=hwnd, **kw)

            self._act = act
        return self._with_effect(
            window_title,
            lambda: self._act(query, action, title=window_title, **kw),
            query,
            marker_image=marker_image,
        )

    def click_point(self, *, window_title: str, x: int, y: int, marker_image: bool = False) -> dict:
        """Click at a screen point inside an allowed window, for controls without names."""
        if not self.allowed_windows:
            raise PermissionError("actions are disabled; start the server with --allow-window")
        self.require_desktop()
        if not window_title:
            raise ValueError("window_title is required")
        x, y = int(x), int(y)
        if self._click is None:
            from svsg.recognition import ui_elements

            def click(title, x, y):
                hwnd = ui_elements.find_window(title)
                actual = ui_elements.window_title(hwnd)
                self._permit(hwnd, actual)
                done = ui_elements.post_click(hwnd, x, y)
                return {
                    "performed": "mouse_down_at"
                    if done["dispatch_state"] == "partial"
                    else "click_at",
                    "dispatch_state": done["dispatch_state"],
                    "window": actual,
                    "element": {"name": f"({x}, {y})"},
                    "after": None,
                    "target_window": done["target_window"],
                    "client_point": done["client_point"],
                }

            self._click = click
        return self._with_effect(
            window_title,
            lambda: self._click(window_title, x, y),
            f"({x}, {y})",
            point=(x, y),
            marker_image=marker_image,
        )

    def _with_effect(
        self, window_title: str, do, label: str, *, point=None, marker_image=False
    ) -> dict:
        """Baseline, act, wait until the window is still, and report what the action did."""
        area = self.resolve_region(None, window_title)
        self.history.finish_window(window_title, area)
        before = self._top_window(window_title)
        blocked = window_title.casefold() in self.blocked
        baseline = None if blocked else self._mark_baseline(area)
        outcome = do()
        report = {
            "execution_state": outcome.get("dispatch_state", "completed"),
            "done": f"{outcome['performed']} {outcome['element']['name'] or label!r}",
            "element_after": _compact_element(outcome.get("after")),
        }
        pos = point or outcome["element"].get("center_px")
        if pos is not None:
            marker = {
                "kind": "posted_click" if point else "uia_target_center",
                "screen_px": list(pos),
                "window_px": [pos[0] - area[0], pos[1] - area[1]],
            }
            if "client_point" in outcome:
                marker["target_client_px"] = outcome["client_point"]
            report["operation_marker"] = marker
            if marker_image and self.images and baseline is not None:
                try:
                    _draw_marker(baseline, area, marker)
                except (ValueError, cv2.error) as exc:
                    marker["image_error"] = str(exc)
        if self.history_seconds:
            try:
                report["screen_history"] = self._start_history(window_title, area, baseline, before)
            except Exception as exc:
                report["screen_history"] = {"state": "capture_unavailable", "detail": str(exc)}
        try:
            self._observe_effect(report, window_title, area, before, outcome, blocked)
        except Exception as exc:  # The side effect already happened, even if its window is gone.
            with self._state_lock:
                self._armed_areas.discard(area)
            report.update(
                observation_state="failed",
                screen={"state": "unavailable"},
                observation_error=f"{type(exc).__name__}: {exc}",
                next_step="Inspect current controls or wait for a fresh observation. "
                "Do not repeat this action based on this observation error.",
            )
        if report["execution_state"] == "partial":
            report["operation_marker"]["kind"] = "posted_mouse_down"
            report["next_step"] = (
                "Button down was posted but release failed. Inspect the window before acting again."
            )
        return report

    def _observe_effect(self, report, window_title, area, before, outcome, blocked):
        if blocked:
            settle, top = self._settle_by_elements(window_title, before)
        else:
            settle = self._settle(
                area,
                quiet_ms=250.0,
                expect_change_ms=1500.0,
                timeout_ms=5000.0,
                ambient_ms=3000.0,
                read_text=True,
            )
            top = self._top_window(window_title)
        report["screen"] = {
            "state": settle["state"],
            "settled_after_ms": settle["elapsed_ms"],
            "changed_regions": [r["bbox_px"] for r in settle["changed_regions"][:6]],
            "text_in_changed_regions": [
                {"text": line["text"], "center_px": line["center_px"]}
                for line in (settle.get("text") or [])[:20]
            ],
        }
        report["observation_state"] = "failed" if settle["state"] == "unavailable" else "observed"
        if top and top["title"] != outcome.get("window"):
            report["new_top_window"] = _window_summary(top)
        elif top and before and top["title"] == before["title"]:
            report["controls_changed"] = _control_changes(before["controls"], top["controls"])
        saw_change = bool(report.get("new_top_window") or report.get("controls_changed"))
        if not blocked and settle["state"] == "unchanged" and saw_change:
            # The application changed but the screen did not: capture is being blocked.
            self.blocked.add(window_title.casefold())
            blocked = True
        if blocked:
            if top:
                self.last_controls[window_title.casefold()] = top
            report["screen"]["source"] = "ui_automation"
            report["screen"]["note"] = BLOCKED_NOTE

    def _settle_by_elements(
        self,
        window_title: str,
        before: dict | None,
        *,
        quiet_ms: float = 250.0,
        expect_change_ms: float = 1000.0,
        timeout_ms: float = 3000.0,
    ) -> tuple[dict, dict | None]:
        """Wait until a window's controls stop changing, when its pixels can't be read."""
        import time as _time

        quiet_ms = _finite("quiet_ms", quiet_ms, positive=True)
        timeout_ms = _finite("timeout_ms", timeout_ms, positive=True)
        expect_change_ms = _finite("expect_change_ms", expect_change_ms)
        if self.max_wait_ms is not None:
            timeout_ms = min(timeout_ms, self.max_wait_ms)
        start = _time.perf_counter()
        last, stable_since, changed = before, start, False
        while True:
            snapshot = self._top_window(window_title)
            now = _time.perf_counter()
            if snapshot is None:
                state, last = "unavailable", None
                break
            if snapshot != last:
                last, stable_since, changed = snapshot, now, True
            elif changed and (now - stable_since) * 1000 >= quiet_ms:
                state = "settled"
                break
            elif not changed and (now - start) * 1000 >= expect_change_ms:
                state = "unchanged"
                break
            if (now - start) * 1000 >= timeout_ms:
                state = "timeout"
                break
            _time.sleep(min(0.12, max(0, timeout_ms / 1000 - (now - start))))
        elapsed = round((_time.perf_counter() - start) * 1000, 1)
        return {"state": state, "elapsed_ms": elapsed, "changed_regions": [], "text": []}, last

    def _top_window(self, window_title: str) -> dict | None:
        """The topmost window matching the title with its controls, e.g. a dialog that opened."""
        if self._windows is None:
            from svsg.recognition import ui_elements

            def windows(title):
                hwnd = ui_elements.find_window(title)
                listed = ui_elements.list_elements(hwnd=hwnd)
                controls = [
                    _compact_element(e)
                    for e in _without_caption(listed["elements"], listed["window"]["bbox_px"])
                    if e["control_type"] in ui_elements.INTERACTIVE or e["control_type"] == "text"
                ]
                return {"title": listed["window"]["title"], "window_id": hwnd, "controls": controls}

            self._windows = windows
        try:
            return self._windows(window_title)
        except (OSError, ValueError):
            return None

    def click_element(
        self,
        *,
        window_title: str,
        target: str,
        control_type: str | None = None,
        element_ref: str | None = None,
        marker_image: bool = False,
    ) -> dict:
        selector = {"element_ref": element_ref} if element_ref is not None else {}
        return self._act_on(
            window_title,
            target,
            "click",
            control_type=control_type,
            marker_image=marker_image,
            **selector,
        )

    def set_field_text(
        self, *, window_title: str, field: str, text: str, element_ref: str | None = None
    ) -> dict:
        selector = {"element_ref": element_ref} if element_ref is not None else {}
        return self._act_on(window_title, field, "set_text", text=text, **selector)

    def tool_specs(self) -> list[dict]:
        tools = [t for t in TOOLS if t["name"] not in ACTION_TOOLS and t["name"] != "screenshot"]
        if self.images:
            tools += [t for t in TOOLS if t["name"] == "screenshot"]
        if self.allowed_windows:
            tools += [t for t in TOOLS if t["name"] in ACTION_TOOLS]
        if not self.history_seconds:
            tools = [
                t for t in tools if t["name"] not in ("watch_screen_changes", "read_screen_history")
            ]
        return tools

    def screenshot(
        self,
        *,
        region: Sequence[int] | None = None,
        window_title: str | None = None,
        max_side: int = 1568,
    ) -> tuple[dict, bytes]:
        area = self.resolve_region(region, window_title)
        frame = self.grab(area)
        h, w = frame.shape[:2]
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            frame = cv2.resize(
                frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError("could not encode the screenshot")
        info = {
            "area_px": list(area) if area else [0, 0, w, h],
            "size_px": [w, h],
            "image_scale": round(scale, 6),
            "note": "image pixel / image_scale + area origin = screen pixel",
        }
        return info, png.tobytes()


# -- MCP over stdio -------------------------------------------------------------------------
_REGION = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 4,
    "maxItems": 4,
    "description": "physical screen pixels [left,top,right,bottom]",
}
_WINDOW = {"type": "string", "description": "visible window title substring"}
TOOLS = [
    {
        "name": "wait_for_settle",
        "description": (
            "Wait for stillness after external input or a longer operation. Actions already wait. "
            "For periodic updates set quiet_ms above their interval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "region": _REGION,
                "window_title": _WINDOW,
                "quiet_ms": {"type": "number", "exclusiveMinimum": 0, "default": 200},
                "expect_change_ms": {"type": "number", "minimum": 0, "default": 1000},
                "timeout_ms": {"type": "number", "exclusiveMinimum": 0, "default": 10000},
                "ambient_ms": {
                    "type": "number",
                    "minimum": 0,
                    "default": 3000,
                    "description": "0 waits for everything, including loading spinners",
                },
                "read_text": {
                    "type": "boolean",
                    "default": False,
                    "description": "also read the text inside the changed regions",
                },
            },
        },
    },
    {
        "name": "find_on_screen",
        "description": (
            "Find click targets by UI Automation name, then OCR. Best center_px first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "window_title": _WINDOW,
                "exact": {"type": "boolean", "default": False},
                "use_ocr": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_screen_text",
        "description": "Read all visible text in a region or window, with a box and click point "
        "per line.",
        "inputSchema": {
            "type": "object",
            "properties": {"region": _REGION, "window_title": _WINDOW},
        },
    },
    {
        "name": "list_ui_elements",
        "description": (
            "List controls, coordinates, values and states. Defaults to foreground window."
        ),
        "inputSchema": {"type": "object", "properties": {"window_title": _WINDOW}},
    },
    {
        "name": "screenshot",
        "description": "Capture a region or window as PNG, scaled so the longest side is at most "
        "max_side.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "region": _REGION,
                "window_title": _WINDOW,
                "max_side": {"type": "integer", "default": 1568},
            },
        },
    },
]


ACTION_TOOLS = ("click_element", "set_field_text", "click_point")
TOOLS += [
    {
        "name": "watch_screen_changes",
        "description": "Start sampling one visible window between calls. Actions already return a "
        "screen_history.history_id. Use read_screen_history to inspect transient changes.",
        "inputSchema": {
            "type": "object",
            "properties": {"window_title": _WINDOW},
            "required": ["window_title"],
        },
    },
    {
        "name": "read_screen_history",
        "description": "Read transient UI changes by history_id. Prefer compact=true. "
        "Controls include text; "
        "read_text also OCRs two pixel crops. Page using after=next_after.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "history_id": {"type": "string"},
                "compact": {"type": "boolean"},
                "after": {"type": "integer", "minimum": 0},
                "event_id": {"type": "integer", "minimum": 1},
                "region": _REGION,
                "read_text": {"type": "boolean"},
                "stop": {"type": "boolean"},
                "include_image": {
                    "type": "boolean",
                    "description": "Return the historical crop for one event_id as an image.",
                },
            },
            "required": ["history_id"],
        },
    },
]
_MARKER_IMAGE = {
    "type": "boolean",
    "description": "Include a marked pre-action crop on image-capable servers.",
}
_ELEMENT_REF = {
    "type": "string",
    "description": "Use the current element_ref for same-name controls.",
}
TOOLS += [
    {
        "name": "click_element",
        "description": (
            "Activate a named control in an allowed window via UI Automation. "
            "Returns effect and history_id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "window_title": _WINDOW,
                "target": {
                    "type": "string",
                    "description": "control name",
                },
                "control_type": {"type": "string", "description": "optional, e.g. button"},
                "marker_image": _MARKER_IMAGE,
                "element_ref": _ELEMENT_REF,
            },
            "required": ["window_title", "target"],
        },
    },
    {
        "name": "set_field_text",
        "description": (
            "Replace a field's text by name or current value in an allowed window. Returns effect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "window_title": _WINDOW,
                "field": {"type": "string"},
                "element_ref": _ELEMENT_REF,
                "text": {"type": "string"},
            },
            "required": ["window_title", "field", "text"],
        },
    },
]


TOOLS.append(
    {
        "name": "click_point",
        "description": (
            "Click center_px from find_on_screen or read_screen_text in an allowed window. "
            "Prefer click_element for named controls. Returns effect and history_id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "window_title": _WINDOW,
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "marker_image": _MARKER_IMAGE,
            },
            "required": ["window_title", "x", "y"],
        },
    }
)


NO_DESKTOP_TOOLS = {"read_screen_history"}  # reads what was recorded; needs no live desktop


class McpServer:
    def __init__(self, agent: ScreenAgent, *, receipts=None, log=None) -> None:
        self.agent = agent
        self.receipts = receipts if receipts is not None else ActionReceipts()
        self.scope = uuid4().hex
        self.log = log
        self._locks: dict = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, args: dict) -> threading.Lock:
        """Calls on one window run one at a time; different windows run side by side."""
        title = args.get("window_title") if isinstance(args, dict) else None
        key = ("window", str(title).casefold()) if title else ("screen",)
        with self._locks_guard:
            if key not in self._locks:
                if len(self._locks) > 256:  # forget idle windows
                    for stale in [k for k, lock in self._locks.items() if not lock.locked()]:
                        del self._locks[stale]
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _record(self, **event) -> None:
        if self.log is not None:
            try:
                self.log(event)
            except Exception:  # noqa: BLE001 - logging must never break a tool call
                pass

    def close(self):
        self.agent.close()
        self.receipts.close()

    def call(self, name: str, args: dict) -> list[dict]:
        if name == "screenshot":
            info, png = self.agent.screenshot(**args)
            return [
                {"type": "text", "text": json.dumps(info)},
                {"type": "image", "data": base64.b64encode(png).decode(), "mimeType": "image/png"},
            ]
        method = {
            "wait_for_settle": self.agent.wait_for_settle,
            "find_on_screen": self.agent.find_on_screen,
            "read_screen_text": self.agent.read_screen_text,
            "list_ui_elements": self.agent.list_ui_elements,
            "watch_screen_changes": self.agent.watch_screen_changes,
            "read_screen_history": self.agent.read_screen_history,
            "click_element": self.agent.click_element,
            "set_field_text": self.agent.set_field_text,
            "click_point": self.agent.click_point,
        }.get(name)
        if method is None:
            raise KeyError(name)
        if name not in NO_DESKTOP_TOOLS:
            self.agent.require_desktop()
        report = method(**args)
        image = report.get("operation_marker", {}).pop("pre_action_png", None)
        event_image = report.pop("event_jpeg", None)
        content = [
            {"type": "text", "text": json.dumps(report, ensure_ascii=False, separators=(",", ":"))}
        ]
        if image:
            content.append({"type": "image", "mimeType": "image/png", "data": image})
        if event_image:
            content.append({"type": "image", "mimeType": "image/jpeg", "data": event_image})
        return content

    def _action(self, key, name, args):
        replay = self.receipts.reserve(key, name, args)
        if replay is not None:
            history_id = replay.get("screen_history", {}).get("history_id")
            if history_id and not self.agent.history.contains(history_id):
                replay.pop("screen_history")
            return [{"type": "text", "text": json.dumps(replay)}]
        # A crash from here onwards leaves a reservation. The next caller must inspect the UI.
        try:
            content = self.call(name, args)
        except Exception as exc:
            raise RuntimeError(
                f"{exc}. Dispatch unconfirmed; inspect the window before acting again."
            ) from exc
        report = json.loads(content[0]["text"])
        receipt = {k: report[k] for k in ("execution_state", "observation_state") if k in report}
        if "history_id" in report.get("screen_history", {}):
            receipt["screen_history"] = {"history_id": report["screen_history"]["history_id"]}
        if report.get("next_step"):
            receipt["next_step"] = report["next_step"]
        try:
            self.receipts.finish(key, receipt)
        except Exception:
            # Reservation already committed. Preserve the original dispatch result.
            report["receipt_warning"] = "Final status unavailable on replay; inspect current state."
            content[0]["text"] = json.dumps(report, ensure_ascii=False)
        return content

    def handle(self, message: dict, *, session=True, operation_id=None) -> dict | None:
        """Answer one JSON-RPC message; notifications get no reply."""
        method, mid = message.get("method"), message.get("id")
        if mid is None:
            return None
        if method == "initialize":
            asked = (message.get("params") or {}).get("protocolVersion")
            result = {
                "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "visfunc-screen", "version": SERVER_VERSION},
                "instructions": __doc__.split("\n\n", 1)[1].strip(),
            }
        elif method == "ping":
            result = {}
            if ((message.get("params") or {}).get("_meta") or {}).get("visfunc/cacheStats") is True:
                result["_meta"] = {
                    "visfunc/cacheStats": {
                        "actions": self.receipts.stats(),
                        "history": self.agent.history.stats(),
                    }
                }
        elif method == "tools/list":
            result = {"tools": self.agent.tool_specs()}
        elif method == "tools/call":
            params = message.get("params") or {}
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in {t["name"] for t in self.agent.tool_specs()}:
                return {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "error": {"code": -32602, "message": f"unknown tool {name!r}"},
                }
            started = time.perf_counter()
            window = args.get("window_title") if isinstance(args, dict) else None
            try:
                if name in ACTION_TOOLS:
                    # Refuse before reserving: a locked desktop must not leave an unknown receipt.
                    self.agent.require_desktop()
                    scope = self.scope if session is True else session
                    meta_id = (params.get("_meta") or {}).get("visfunc/operationId")
                    if operation_id and meta_id and operation_id != meta_id:
                        raise ValueError("conflicting operation IDs")
                    key = operation_key(
                        scope, mid, operation_id if operation_id is not None else meta_id
                    )
                    with self._lock_for(args):
                        content = self._action(key, name, args)
                else:
                    with self._lock_for(args):
                        content = self.call(name, args)
                result = {"content": content, "isError": False}
                outcome = {}
                try:
                    report = json.loads(content[0]["text"])
                    outcome = {
                        k: report[k]
                        for k in ("execution_state", "observation_state", "replayed")
                        if isinstance(report, dict) and k in report
                    }
                except (ValueError, KeyError, IndexError, TypeError):
                    pass
                self._record(
                    tool=name,
                    window=window,
                    ok=True,
                    ms=round((time.perf_counter() - started) * 1000),
                    **outcome,
                )
            except Exception as exc:  # noqa: BLE001 - a failing tool must not end the server
                result = {
                    "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                    "isError": True,
                }
                self._record(
                    tool=name,
                    window=window,
                    ok=False,
                    ms=round((time.perf_counter() - started) * 1000),
                    error=f"{type(exc).__name__}: {str(exc)[:200]}",
                )
        else:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": mid, "result": result}


def json_log(path: str | None, *, max_bytes: int = 5_000_000, backups: int = 5):
    """A rotating JSON-lines writer for tool calls, or None when no path is given.

    Each line holds time, tool, window title, duration and outcome. Arguments such as typed
    text or search queries are never written.
    """
    if not path:
        return None
    import logging
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"motionspec.agent.{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    def write(event: dict) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        logger.info(json.dumps({"time": stamp, **event}, ensure_ascii=False))

    return write


def serve_mcp(
    agent: ScreenAgent | None = None, stdin=None, stdout=None, *, log_file: str | None = None
) -> None:
    """Serve MCP over newline-delimited JSON-RPC on stdin/stdout until stdin closes."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    sys.stdout = sys.stderr  # Anything printed by libraries must not corrupt the protocol stream.
    server = McpServer(
        agent or ScreenAgent(), receipts=ActionReceipts(receipt_path()), log=json_log(log_file)
    )
    try:
        _serve_lines(server, stdin, stdout)
    finally:
        server.close()


def _serve_lines(server, stdin, stdout):
    for raw in stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            reply = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            reply = (
                server.handle(message)
                if isinstance(message, dict)
                else {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "invalid request"},
                }
            )
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False).encode("utf-8") + b"\n")
            stdout.flush()


MAX_REQUEST_BYTES = 1024 * 1024


def serve_http(
    agent: ScreenAgent | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    token: str = "",
    ready=None,
    workers: int = 4,
    log_file: str | None = None,
) -> None:
    """Serve MCP over Streamable HTTP (JSON responses, no server-sent events) at /mcp.

    A small pool of worker threads serves requests. Calls on the same window run one at a
    time, so they never race over its frames or watcher; calls on different windows proceed
    side by side. Only loopback browsers may call it (Origin check), and a bearer token can be
    required.
    """
    import hmac
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlsplit

    log = json_log(log_file)
    server_logic = McpServer(
        agent or ScreenAgent(), receipts=ActionReceipts(receipt_path()), log=log
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep stdout clean; one line per request on stderr
            sys.stderr.write("agent-mcp " + fmt % args + "\n")

        def _reply(self, status: int, body: dict | None = None, headers: dict | None = None):
            data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _allowed(self) -> bool:
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).hostname not in {"localhost", "127.0.0.1", "::1"}:
                self._reply(403, {"error": "origin not allowed"})
                return False
            if token:
                given = self.headers.get("Authorization", "")
                if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                    if log is not None:
                        log({"event": "unauthorized", "client": self.client_address[0]})
                    self._reply(401, {"error": "unauthorized"})
                    return False
            if urlsplit(self.path).path not in ("/mcp", "/mcp/"):
                self._reply(404, {"error": "use /mcp"})
                return False
            return True

        def do_GET(self):
            if self._allowed():
                self._reply(
                    405,
                    {"error": "this server does not open event streams"},
                    {"Allow": "POST, DELETE"},
                )

        def do_DELETE(self):
            if self._allowed():
                self._reply(200)

        def do_POST(self):
            if not self._allowed():
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._reply(413 if length > MAX_REQUEST_BYTES else 400, {"error": "bad length"})
                return
            try:
                message = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._reply(
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "parse error"},
                    },
                )
                return
            if not isinstance(message, dict):
                self._reply(
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32600, "message": "batches are not supported"},
                    },
                )
                return
            session = self.headers.get("Mcp-Session-Id")
            if session:
                try:
                    session = uuid.UUID(session).hex
                except ValueError:
                    self._reply(400, {"error": "invalid MCP session ID"})
                    return
            if (
                message.get("method") == "tools/list"
                and not session
                and server_logic.agent.allowed_windows
                and self.headers.get("MCP-Protocol-Version") not in (*PROTOCOL_VERSIONS, None)
            ):
                self._reply(400, {"error": "initialize to negotiate a supported MCP version"})
                return
            reply = server_logic.handle(
                message, session=session, operation_id=self.headers.get("Idempotency-Key")
            )
            if reply is None:
                self._reply(202)
                return
            headers = {}
            if message.get("method") == "initialize":
                headers["Mcp-Session-Id"] = uuid.uuid4().hex
            self._reply(200, reply, headers)

    class PoolServer(HTTPServer):
        def __init__(self, address, handler):
            super().__init__(address, handler)
            self.pool = ThreadPoolExecutor(max(1, workers), thread_name_prefix="agent-mcp")

        def process_request(self, request, client_address):
            self.pool.submit(self._work, request, client_address)

        def _work(self, request, client_address):
            try:
                self.finish_request(request, client_address)
            except Exception:  # noqa: BLE001 - report like the base server, keep serving
                self.handle_error(request, client_address)
            finally:
                self.shutdown_request(request)

        def server_close(self):
            super().server_close()
            self.pool.shutdown(wait=True, cancel_futures=True)

    httpd = PoolServer((host, port), Handler)
    if ready is not None:
        ready(httpd)
    sys.stderr.write(f"agent-mcp listening on http://{host}:{httpd.server_port}/mcp\n")
    if log is not None:
        log(
            {
                "event": "start",
                "port": httpd.server_port,
                "workers": workers,
                "actions": bool(server_logic.agent.allowed_windows),
                "token": bool(token),
            }
        )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        server_logic.close()
        if log is not None:
            log({"event": "stop"})
