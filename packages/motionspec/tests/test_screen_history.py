import base64

import cv2
import numpy as np
import pytest
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
