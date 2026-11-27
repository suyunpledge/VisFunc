"""Exercise four-edge measurements through the ordinary analysis and export paths."""

import json

import numpy as np
import pytest
from motionspec import analyze
from motionspec.reflow import measure_reflow
from motionspec.spec import summarize, validate


def panel_frames():
    frames = []
    for height in [16] * 7 + list(range(18, 54, 2)) + [54] * 8:
        frame = np.full((140, 200, 3), 24, np.uint8)
        frame[15 : 15 + height, 35:145] = (61, 71, 85)
        frame[20 + height : 45 + height, 35:145] = (61, 71, 85)
        frames.append(frame)
    return frames


def test_reflow_keeps_resized_and_displaced_elements_and_exports_their_evidence(tmp_path):
    pytest.importorskip("svsg.video_functions")
    spec = analyze(panel_frames(), tmp_path, fps=60, layout_edges=True, functions=True)
    assert validate(spec) == []
    segments = spec["segments"]
    assert len(segments) == 2
    (panel,) = [s for s in segments if "size" in s["channels"]]
    (following,) = [s for s in segments if "translate" in s["channels"]]
    size = panel["channels"]["size"]
    assert size["height_from"] == 16 and size["height_to"] == 54 and size["anchor"] == "top"
    assert following["channels"]["translate"]["delta"] == [0, 38]
    assert "size 110x16 -> 110x54" in summarize(spec)
    for s in segments:
        assert (tmp_path / s["sheet"]).is_file()
    from motionspec.functions import catalogue

    entries = catalogue(tmp_path / "functions.json")["segments"]
    actor = next(e for e in entries if e["segment_id"] == panel["id"])
    assert actor["sampled"]["height"]["range"] == [16, 54]
    assert "size.progress" in actor["fitted"]
    assert json.loads((tmp_path / "spec.json").read_text(encoding="utf8")) == spec


def test_pure_translation_and_clipped_resize_fall_back():
    frames = panel_frames()
    # Neither edge of the panel along x is visible in this cropped strip.
    clipped = [f[:, 50:130] for f in frames]
    times = np.arange(len(frames)) * 16.0
    assert measure_reflow(clipped, times, (50, 0), 1.0, (0, 0, 80, 140), 12) is None
    moving = []
    for i, f in enumerate(frames):
        copy = np.full_like(f, 24)
        copy[30:55, 20 + i : 80 + i] = 120
        moving.append(copy)
    assert measure_reflow(moving, times, (0, 0), 1.0, (0, 0, 200, 140), 12) is None
