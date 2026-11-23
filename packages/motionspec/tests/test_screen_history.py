import base64
import threading
import time

import cv2
import numpy as np
import pytest
from motionspec.agent import _ui_delta
from motionspec.screen_history import ScreenHistory, WindowHistory

AREA = (-40, 20, 200, 120)
BASE = np.full((100, 240, 3), 240, np.uint8)


def test_transient_returns_to_baseline_and_is_readable_by_cursor():
    watch = WindowHistory("App", AREA, BASE, 10, 8)
    changed = BASE.copy()
    changed[20:55, 80:170] = (40, 30, 210)
    for frame, ms in [(BASE, 100), (changed, 200), (changed, 300), (BASE, 400)]:
        watch.sample(frame, ms)
    pool = ScreenHistory()
    pool.watches[watch.id] = watch

    def read(after=0, event_id=None, reader=None, include_image=False):
        return pool.read(watch.id, after, event_id, reader, False, include_image=include_image)

    all_events = read()
    assert all_events["samples"] == 5
    assert [r["at_ms"] for r in all_events["events"]] == [200, 400]
    box = all_events["events"][0]["changed_regions_px"][0]
    assert box[0] <= 40 and box[1] <= 40 and box[2] >= 130 and box[3] >= 75
    assert read(after=1)["events"] == all_events["events"][1:]
    image = read(event_id=1, include_image=True, reader=lambda img: {"lines": [{"text": "notice"}]})
    assert image["events"][0]["text"] == ["notice"]
    data = np.frombuffer(base64.b64decode(image["event_jpeg"]), np.uint8)
    decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    np.testing.assert_allclose(
        decoded[decoded.shape[0] // 2, decoded.shape[1] // 2], [40, 30, 210], atol=5
    )
    assert read() == all_events  # read-time OCR and encoding do not mutate the ring
    with pytest.raises(ValueError, match="event_id"):
        read(include_image=True)
    pool.finish_window("App", AREA)
    watch.sample(changed, 500)
    assert read()["state"] == "stopped" and read()["latest_event_id"] == 2


def test_history_is_bounded_and_marks_lost_events(monkeypatch):
    monkeypatch.setattr("motionspec.screen_history.MAX_CROP_BYTES", 2400)
    watch = WindowHistory("App", AREA, BASE, 2, 8)
    other = np.zeros_like(BASE)
    for i in range(20):
        for j in range(4):
            watch.sample(other if i % 2 == 0 else BASE, (4 * i + j) * 125)
    summary, rows = watch.snapshot(1, None)
    assert len(rows) == 12 and summary["missed_events"] == 2 and summary["has_more"]
    assert watch.crop_bytes <= 2400 and summary["evicted_crops"] > 0
    page, tail = watch.snapshot(rows[-1][0]["event_id"], None)
    assert len(tail) == 5 and not page["has_more"]
    watch.run(lambda *a, **kw: iter([]), lambda _: (0, 0, 1, 1))
    assert watch.snapshot(19, None)[0]["state"] == "window_changed"
    with pytest.raises(ValueError, match="history_id"):
        ScreenHistory().read(watch.id, 0, None, None, False)


def test_playing_video_does_not_evict_an_adjacent_short_notice():
    watch = WindowHistory("App", AREA, BASE, 10, 8)
    for i in range(1, 48):
        frame = BASE.copy()
        frame[:50, :120] = 30 if i % 2 else 230
        if 10 <= i <= 13:
            frame[70:84, 168:221] = (25, 120, 25)
        watch.sample(frame, i * 125)
    info, rows = watch.snapshot(0, None)
    assert info["repeated_motion_samples"] >= 40
    assert info["missed_events"] == 0
    assert [r["at_ms"] for r, _ in rows if r["at_ms"] >= 1000] == [1250, 1750]
    _, notice = watch.snapshot(0, None, region=[130, 90, 160, 106])
    assert [r["at_ms"] for r, _ in notice] == [1250, 1750]


def test_semantic_only_stream_retains_transient_and_expires(monkeypatch):
    pool = ScreenHistory()
    monkeypatch.setattr("motionspec.screen_history.RETENTION_SECONDS", 0.15)

    def window(label):
        return {
            "title": "App",
            "controls": [{"name": label, "control_type": "text", "center_px": [23, 57]}],
        }

    baseline = window("Ready")
    notices = iter([window("PULSE 17029"), baseline])
    cleanup = threading.Event()
    handle = pool.start(
        "App",
        AREA,
        None,
        0.55,
        8,
        None,
        None,
        observe=lambda: next(notices, baseline),
        before=baseline,
        delta=_ui_delta,
        cleanup=cleanup.set,
    )
    key = handle["history_id"]
    assert cleanup.wait(2)
    data = pool.read(key, 0, None, None, False, region=[20, 50, 27, 60], compact=True)
    assert data["state"] == "capture_unavailable" and data["ui_state"] == "completed"
    assert [ev["changes"][0]["name"] for ev in data["events"]] == ["PULSE 17029", "Ready"]
    assert not pool.read(key, data["next_after"], None, None, False)["events"]
    outside = pool.read(key, 0, None, None, False, region=[0, 0, 10, 10])
    assert not outside["events"] and outside["next_after"] == data["next_after"]
    deadline = time.monotonic() + 2
    while pool.contains(key) and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(ValueError, match="history_id"):
        pool.read(key, 0, None, None, False)
    assert pool.stats() == {
        "reads": 4,
        "matches": 1,
        "unavailable": 1,
        "ocr_hits": 0,
        "ocr_calls": 0,
    }


def test_ocr_empty_results_memoized_after_failure_and_per_event():
    watch = WindowHistory("App", AREA, BASE, 10, 8)
    other = np.zeros_like(BASE)
    watch.sample(other, 100)
    watch.sample(BASE, 200)
    pool = ScreenHistory()
    pool.watches[watch.id] = watch
    calls = []

    def reader(image):
        calls.append(image.shape)
        if len(calls) == 1:
            raise RuntimeError("busy")
        return {"lines": []}

    def read():
        return pool.read(watch.id, 0, None, reader, False, compact=True)

    assert read()["ocr_cache"] == {"hits": 0, "calls": 2}
    assert read()["ocr_cache"] == {"hits": 1, "calls": 1}
    assert read()["ocr_cache"] == {"hits": 2, "calls": 0}
    assert len(calls) == 3


def test_compact_byte_budget_pages_without_skips_and_detail_is_queryable():
    import json

    from motionspec.screen_history import COMPACT_EVENT_BYTES

    pool = ScreenHistory()
    watch = WindowHistory("App", AREA, BASE, 10, 8)
    pool.watches[watch.id] = watch
    change = {"name": "字" * 160, "was": "字" * 160, "value": "字" * 160}
    with watch.lock:
        for index in range(14):
            watch._append({"kind": "controls", "changes": [change] * (index % 3 + 1)}, index * 80)
    seen = []
    while True:
        data = pool.read(watch.id, seen[-1] if seen else 0, None, None, False, compact=True)
        assert (
            sum(len(json.dumps(ev, ensure_ascii=False).encode()) for ev in data["events"])
            <= COMPACT_EVENT_BYTES
        )
        seen.extend(ev["event_id"] for ev in data["events"])
        assert data["next_after"] == seen[-1]
        if not data["has_more"]:
            break
    assert seen == list(range(1, 15))
    assert pool.read(watch.id, 0, 3, None, False)["events"][0]["changes"] == [change] * 3


def test_window_delta_and_repeated_snapshot():
    a = {"title": "App", "controls": []}
    b = {"title": "App confirm", "controls": [{"name": "Okay", "control_type": "button"}]}
    assert _ui_delta(a, b) == {
        "kind": "window",
        "window_title": "App confirm",
        "changes": [{"name": "Okay", "control_type": "button", "appeared": True}],
    }
    assert _ui_delta(b, b) is None
    replaced = dict(b, window_id=7)
    assert _ui_delta(dict(b, window_id=6), replaced)["kind"] == "window"
    long_label = {"title": "App", "controls": [{"name": "x" * 200, "control_type": "text"}]}
    clipped = _ui_delta(a, long_label)
    assert clipped["truncated"] and len(clipped["changes"][0]["name"]) == 160
