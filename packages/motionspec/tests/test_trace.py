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
    elements += [{"class": "QLabel", "rows": rows(11, 6, lambda p, i=i: [10, 10 + (2 + i) * p, 50, 20, 1.0, 1.0, 0.0, 1.0, 10, 100 + 30 * i + (2 + i) * p])}
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


def test_scale_from_or_to_zero_and_static_origin_have_no_fake_translation():
    pop_in = lambda p: [300, 300, 220, 44, p, p, 0.0, 1.0, 300, 300]
    pop_out = lambda p: [300, 300, 220, 44, 1 - p, 1 - p, 0.0, 1.0, 300, 300]
    for fn, expected in ((pop_in, (0.0, 1.0)), (pop_out, (1.0, 0.0))):
        spec = spec_from_trace(trace([{"class": "QQuickRectangle", "name": "toast", "rows": rows(15, 12, fn)}]))
        assert validate(spec) == []
        (seg,) = spec["segments"]
        assert "translate" not in seg["channels"]
        assert seg["dominant"] == "scale"
        assert (seg["channels"]["scale"]["from"], seg["channels"]["scale"]["to"]) == expected


def test_size_channel_can_be_claimed():
    expand = lambda p: [100, 100, 200 + round(120 * p), 80, 1.0, 1.0, 0.0, 1.0, 100, 100]
    spec = spec_from_trace(trace([{"class": "QFrame", "name": "panel", "rows": rows(10, 12, expand)}]))
    assert validate(spec) == []
    verdict = check(spec, [{"segment": "s1", "field": "duration_ms", "channel": "size", "value": 200}])
    assert verdict["verdict"] == "passed"


@pytest.mark.parametrize("origin", ["top_left", "center"])
@pytest.mark.parametrize("ratio", [1.0, 1.5])
def test_scale_preserves_scene_geometry_without_independent_translation(origin, ratio):
    def zoom(p):
        scale = 1 + p
        # mapToScene(0, 0) moves under centre-origin scaling, but the item's
        # position properties remain fixed for either transform origin.
        scene_x = 100 if origin == "top_left" else 150 - 50 * scale
        scene_y = 100 if origin == "top_left" else 125 - 25 * scale
        return [100, 100, 100, 50, 1, scale, 0, 1, scene_x, scene_y]

    spec = spec_from_trace(trace([{"rows": rows(10, 12, zoom)}], ratio=ratio))
    assert validate(spec) == []
    (segment,) = spec["segments"]
    assert set(segment["channels"]) == {"scale"}
    expected_region = [100, 100, 300, 200] if origin == "top_left" else [50, 75, 250, 175]
    assert segment["region"] == [
        int(np.floor(value * ratio)) if i < 2 else int(np.ceil(value * ratio))
        for i, value in enumerate(expected_region)
    ]
    start, end = segment["samples"][0], segment["samples"][-1]
    assert [start["x"], start["y"]] == pytest.approx([150 * ratio, 125 * ratio], abs=0.05)
    expected_end = [200, 150] if origin == "top_left" else [150, 125]
    assert [end["x"], end["y"]] == pytest.approx([v * ratio for v in expected_end], abs=0.05)


def test_resize_updates_bounds_and_center_without_independent_translation():
    def expand(p):
        return [100, 100, 100 + 100 * p, 50, 1, 1, 0, 1, 100, 100]

    (segment,) = spec_from_trace(trace([{"rows": rows(10, 12, expand)}]))["segments"]
    assert set(segment["channels"]) == {"size"}
    assert segment["region"] == [100, 100, 300, 150]
    assert [segment["samples"][-1][key] for key in ("x", "y")] == [200, 125]


STILL = [300, 300, 220, 44, 1.0, 1.0, 0.0, 1.0, 300, 300]


def test_colour_and_custom_properties_become_segments():
    colour = [[19, 0.14, 0.16, 0.22, 1.0]] + [[19 + i, *(round(a + (b - a) * i / 12, 4) for a, b in
                                                       zip((0.14, 0.16, 0.22), (0.17, 0.44, 0.95))), 1.0] for i in range(1, 13)]
    hover = [[39, 0.0]] + [[39 + i, i / 9] for i in range(1, 10)]
    spec = spec_from_trace(trace([{"class": "QQuickRectangle", "name": "glass", "rows": [[0, *STILL]],
                                   "props": {"color": colour, "hoverAmount": hover}}]))
    assert validate(spec) == []
    first, second = spec["segments"]
    assert first["kind"] == second["kind"] == "fade"
    assert first["channels"]["color"]["from"][:3] == [0.14, 0.16, 0.22]
    assert first["channels"]["color"]["to"][:3] == [0.17, 0.44, 0.95]
    assert first["channels"]["color"]["easing"]["named"][0]["name"] == "linear"
    prop = second["channels"]["property"]
    assert (prop["name"], prop["from"], prop["to"]) == ("hoverAmount", 0.0, 1.0)
    assert second["duration_ms"] == pytest.approx(150, abs=FRAME)
    assert second["region"] == [300, 300, 520, 344]


def test_bad_property_rows_are_rejected():
    for props in ({"color": [[5, 0.1, 0.2]]}, {"color": [[5, 2.0, 0.2, 0.3, 1.0]]}, {"x": [[5, 1.0], [4, 2.0]]}, {"": [[5, 1.0]]}):
        with pytest.raises(ValueError):
            spec_from_trace(trace([{"class": "QFrame", "rows": [[0, *STILL]], "props": props}]))


def test_growing_from_zero_size_is_a_size_change_not_a_fade():
    """展开、进度条从 0 开始时控件先是隐藏的：这不该被报成透明度 0→1。"""
    def grow(p):
        h = round(180 * p)
        return [210, 140, 400, h, 1.0, 1.0, 0.0, 1.0 if h > 0 else 0.0, 210, 140]
    (seg,) = spec_from_trace(trace([{"class": "QFrame", "name": "panel", "rows": rows(11, 18, grow)}]))["segments"]
    assert "opacity" not in seg["channels"]
    size = seg["channels"]["size"]
    assert (size["height_from"], size["height_to"]) == (0.0, 180.0)


def test_elements_moving_together_are_grouped():
    def row(offset):
        return lambda p: [210, offset + round(180 * p), 400, 40, 1.0, 1.0, 0.0, 1.0, 210, offset + round(180 * p)]
    spec = spec_from_trace(trace([{"class": "QFrame", "name": "a", "rows": rows(11, 18, row(150))},
                                  {"class": "QFrame", "name": "b", "rows": rows(11, 18, row(200))}]))
    first, second = spec["segments"]
    assert [s["element"]["name"] for s in spec["segments"]] == ["a", "b"]
    assert first["element_id"] != second["element_id"]
    assert first["samples"][0]["y"] != second["samples"][0]["y"]
    assert spec["motion_groups"] == [{"id": "g1", "segment_ids": ["s1", "s2"],
                                      "relation": "equivalent_sampled_motion"}]
