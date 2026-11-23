import cv2
import numpy as np
import pytest
from motionspec.frames import FrameSource, InvalidTimestamps


def test_sparse_evidence_keeps_pts_and_source_pixels(tmp_path):
    video = tmp_path / "回放.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (96, 64))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    try:
        for i in range(30):
            writer.write(np.full((64, 96, 3), (2 * i, i, 4 * i), np.uint8))
    finally:
        writer.release()
    full = list(FrameSource(video).selected_raw())
    source = FrameSource(video, start_ms=554, end_ms=924)
    wanted = {15, 21}
    sparse = list(source.selected_samples(wanted.__contains__))
    assert [(i, t) for i, t, _ in sparse] == [(i, t) for i, t, _ in full if 554 <= t <= 924]
    for i, _, raw in sparse:
        if i in wanted:
            np.testing.assert_array_equal(raw, full[i][2])
    cap = cv2.VideoCapture(str(video))
    try:
        if cap.getBackendName() == "FFMPEG":
            assert {i for i, _, frame in sparse if frame is not None} == wanted
    finally:
        cap.release()


def test_bad_clock_outside_requested_pixels_is_still_rejected(monkeypatch):
    frames = [np.ones((32, 48, 3), np.uint8)] * 3
    source = FrameSource(frames, start_ms=1000)
    source.info.timestamps = "container"
    monkeypatch.setattr(source, "_raw", lambda: iter([(0, 0, None), (1, 0, None)]))
    with pytest.raises(InvalidTimestamps, match="source frame 1"):
        list(source.selected_samples(lambda i: False))
