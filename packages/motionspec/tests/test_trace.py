"""Qt 探针的离线一半：轨迹转规格。这里手工构造轨迹，不需要 Qt。"""

import json

import numpy as np
import pytest

from motionspec import check, validate
from motionspec.cli import main
from motionspec.fit import bezier_ease
from motionspec.trace import spec_from_trace

FRAME = 1000 / 60
FIELDS = ["tick", "x", "y", "width", "height", "opacity", "scale", "rotation", "visible", "scene_x", "scene_y"]


def rows(start_tick, frames, make):
    """模仿录制器：变化前一帧、每个变化帧、变化后第一帧静止。make(p) 返回 10 个属性值。"""
    out = [[start_tick - 1, *make(0.0)]]
    for i in range(1, frames + 1):
        out.append([start_tick - 1 + i, *make(i / frames)])
    out.append([start_tick + frames, *make(1.0)])
    return out


def trace(elements, ticks=70, ratio=1.0):
    return {"schema": "motionspec-trace/0.1", "title": "test", "size": [640, 400], "device_pixel_ratio": ratio,
            "ticks_ms": [i * FRAME for i in range(ticks)], "elements_seen": 5, "truncated": False,
            "elements": [{"id": i, "fields": FIELDS, **e} for i, e in enumerate(elements)]}


def slide(p):
    x = 40 + round(300 * float(bezier_ease(0.33, 1, 0.68, 1, p)))  # 控件位置是整数像素
    return [x, 60, 200, 100, 1.0, 1.0, 0.0, 1.0, x, 60]


def fade(p):
    return [300, 300, 220, 44, p, 1.0, 0.0, 1.0, 300, 300]


def test_slide_is_exact_and_named_with_its_design_duration():
    spec = spec_from_trace(trace([{"class": "QFrame", "name": "card", "rows": rows(11, 18, slide)}]))
    assert validate(spec) == []
    (seg,) = spec["segments"]
    assert seg["model"] == "qt-probe" and seg["kind"] == "transform"
    assert seg["channels"]["translate"]["delta"] == [300.0, 0.0]
    assert seg["element"] == {"class": "QFrame", "name": "card"}
    best = seg["easing"]["named"][0]
    assert best["name"] == "out-cubic"
    assert best["duration_ms"] == pytest.approx(300, abs=FRAME)


def test_opacity_only_change_is_a_fade_with_exact_endpoints():
    spec = spec_from_trace(trace([{"class": "QFrame", "name": "toast", "rows": rows(20, 12, fade)}]))
    (seg,) = spec["segments"]
    assert seg["kind"] == "fade"
    opacity = seg["channels"]["opacity"]
    assert (opacity["from"], opacity["to"]) == (0.0, 1.0)
    assert opacity["duration_ms"] == pytest.approx(200, abs=FRAME)
    assert opacity["easing"]["named"][0]["name"] == "linear"
    assert seg["t_start_ms"] == pytest.approx(19 * FRAME, abs=0.01)


def test_device_pixel_ratio_scales_positions():
    spec = spec_from_trace(trace([{"class": "QFrame", "rows": rows(11, 18, slide)}], ratio=1.5))
    (seg,) = spec["segments"]
    assert seg["channels"]["translate"]["delta"] == [450.0, 0.0]
    assert spec["source"]["width"] == 960 and spec["source"]["device_pixel_ratio"] == 1.5


def test_two_elements_and_claims():
    spec = spec_from_trace(trace([{"class": "QFrame", "name": "card", "rows": rows(11, 18, slide)},
                                  {"class": "QFrame", "name": "toast", "rows": rows(40, 12, fade)}]))
    assert [s["element"]["name"] for s in spec["segments"]] == ["card", "toast"]
    verdict = check(spec, [
        {"field": "count", "value": 2},
        {"segment": "s1", "field": "direction", "value": "right"},
        {"segment": "s1", "field": "distance_px", "value": 300},
        {"segment": "s1", "field": "easing", "value": "out-cubic"},
        {"segment": "s2", "field": "duration_ms", "value": 200, "channel": "opacity"},
    ])
    assert verdict["verdict"] == "passed", verdict


def test_separate_runs_of_one_element_stay_separate():
    back = [[r[0] + 30, *slide(1 - (r[0] - 10) / 18 if 10 <= r[0] <= 28 else (1.0 if r[0] < 10 else 0.0))] for r in rows(11, 18, slide)]
    spec = spec_from_trace(trace([{"class": "QFrame", "rows": rows(11, 18, slide) + back}]))
    deltas = [s["channels"]["translate"]["delta"][0] for s in spec["segments"]]
    assert deltas == [300.0, -300.0]


def test_smaller_changes_are_dropped_and_counted():
    elements = [{"class": "QFrame", "name": "card", "rows": rows(11, 18, slide)}]
    elements += [{"class": "QLabel", "rows": rows(11, 6, lambda p, i=i: [10, 10 + 2 * p, 50, 20, 1.0, 1.0, 0.0, 1.0, 10, 100 + 30 * i + 2 * p])}
                 for i in range(3)]
    spec = spec_from_trace(trace(elements), max_segments=1)
    assert [s["element"].get("name") for s in spec["segments"]] == ["card"]
    assert spec["coverage"]["dropped_candidates"] == 3
    assert check(spec, [{"field": "count", "value": 1}])["verdict"] == "insufficient_evidence"


def test_cli_converts_a_trace(tmp_path, capsys):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace([{"class": "QFrame", "name": "card", "text": "快速配置", "rows": rows(11, 18, slide)}]),
                               ensure_ascii=False), encoding="utf-8")
    assert main(["trace", str(path), "-o", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert "element QFrame#card" in printed and "快速配置" in printed
    assert json.loads((tmp_path / "out" / "spec.json").read_text(encoding="utf-8"))["source"]["timestamps"] == "probe"


def test_bad_trace_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        spec_from_trace({"schema": "something-else"})
    with pytest.raises(ValueError):
        spec_from_trace({**trace([]), "ticks_ms": [0.0]})
    assert np.isfinite(spec_from_trace(trace([]))["time_resolution_ms"])
