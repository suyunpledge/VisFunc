"""进入 / 退出类动效：元素一边移动或缩放、一边改变透明度。由图层模型处理。"""

import numpy as np
import pytest

from motionspec import analyze, check, validate
from motionspec.fit import bezier_ease, fit_open
from motionspec.synth import background, card, compose, enter, fade, leave, listing, pop, popover, slide

FRAME = 1000 / 60


def only(spec):
    assert validate(spec) == []
    assert len(spec["segments"]) == 1, [s["kind"] for s in spec["segments"]]
    return spec["segments"][0]


def best(channel):
    return channel["easing"]["named"][0]


def test_enter_recovers_full_distance_although_the_start_is_invisible():
    frames, truth = enter()
    seg = only(analyze(frames, fps=60))
    assert seg["kind"] == "transform" and seg["model"] == "layer"
    assert seg["layer"]["direction"] == "enter"
    move, opacity = seg["channels"]["translate"], seg["channels"]["opacity"]
    # 透明度为 0 的起点位置看不见：看得见的只有约 19px，完整位移是 24px
    assert move["visible_distance_px"] < 21
    assert move["delta"][1] == pytest.approx(-24, abs=1.0) and abs(move["delta"][0]) < 0.5
    assert move["follows_opacity"]
    assert (opacity["from"], opacity["to"]) == (0.0, 1.0)
    for channel in (move, opacity):
        assert best(channel)["name"] == "material-decelerate"
        assert best(channel)["duration_ms"] == pytest.approx(300, abs=2 * FRAME)
    assert seg["t_start_ms"] == pytest.approx(truth["start_ms"], abs=FRAME)
    assert seg["confidence"] == "high"


def test_enter_with_separate_curves_reports_each_channel():
    frames, _ = enter(opacity_curve=(0.0, 0.0, 1.0, 1.0))
    seg = only(analyze(frames, fps=60))
    move, opacity = seg["channels"]["translate"], seg["channels"]["opacity"]
    assert not move["follows_opacity"]
    assert best(opacity)["name"] == "linear"
    assert best(move)["name"] == "material-decelerate"
    assert move["distance_px"] == pytest.approx(24, abs=1.5)


def test_leave_is_the_time_reverse_of_enter():
    frames, _ = leave()
    seg = only(analyze(frames, fps=60))
    assert seg["layer"]["direction"] == "exit"
    move, opacity = seg["channels"]["translate"], seg["channels"]["opacity"]
    assert move["delta"][1] == pytest.approx(24, abs=1.5)
    assert (opacity["from"], opacity["to"]) == (1.0, 0.0)
    assert best(opacity)["name"] == "ease-in"
    assert best(opacity)["duration_ms"] == pytest.approx(250, abs=2 * FRAME)


def test_popover_scale_is_not_mistaken_for_a_plain_fade():
    frames, _ = popover()
    seg = only(analyze(frames, fps=60))
    assert seg["kind"] == "transform" and seg["dominant"] == "scale"
    scale = seg["channels"]["scale"]
    assert scale["from"] == pytest.approx(0.9, abs=0.012) and scale["to"] == 1.0
    assert "translate" not in seg["channels"]
    assert "opacity" in seg["channels"]


def test_list_items_entering_one_after_another_are_separate_segments():
    frames, truth = listing()
    spec = analyze(frames, fps=60)
    assert validate(spec) == []
    assert len(spec["segments"]) == 3
    tops = [seg["region"][1] for seg in spec["segments"]]
    assert tops == sorted(tops) and len(set(tops)) == 3
    for seg, start in zip(spec["segments"], truth["starts_ms"]):
        assert seg["t_start_ms"] == pytest.approx(start, abs=FRAME)
        assert seg["channels"]["translate"]["delta"][1] == pytest.approx(-20, abs=1.0)
        assert best(seg["channels"]["opacity"])["duration_ms"] == pytest.approx(300, abs=2 * FRAME)


def test_noise_does_not_break_the_layer_model():
    frames, _ = enter()
    rng = np.random.default_rng(1)
    noisy = [np.clip(f.astype(np.float32) + rng.normal(0, 2.0, f.shape), 0, 255).astype(np.uint8) for f in frames]
    seg = only(analyze(noisy, fps=60, threshold=16))
    assert seg.get("model") == "layer"
    assert seg["channels"]["translate"]["distance_px"] == pytest.approx(24, abs=1.5)


@pytest.mark.parametrize("make", [slide, pop, fade])
def test_layer_model_stays_out_of_what_it_does_not_explain(make):
    """已有元素的平移和缩放、纯淡入，都不是"进入"，不该被图层模型接手。"""
    frames, _ = make()
    assert "model" not in only(analyze(frames, fps=60))


def test_identical_static_neighbour_does_not_block_an_entering_item():
    """旁边有一个外观相同但没动的元素时，新进入的那个仍按进入处理。"""
    bg, row = background(), card(180, 60)
    still = compose(bg, row, 320, 110)
    tau = np.r_[np.zeros(10), np.arange(19) / 18, np.ones(12)]
    frames = [compose(still, row, 320, 180 + 20 * (1 - v), alpha=float(v)) for v in bezier_ease(0, 0, 0.2, 1, tau)]
    seg = only(analyze(frames, fps=60))
    assert seg.get("model") == "layer"
    assert seg["channels"]["translate"]["delta"][1] == pytest.approx(-20, abs=1.0)


def test_opacity_channel_can_be_claimed():
    frames, _ = enter()
    spec = analyze(frames, fps=60)
    good = [
        {"segment": "s1", "field": "direction", "value": "up"},
        {"segment": "s1", "field": "distance_px", "value": 24},
        {"segment": "s1", "field": "easing", "value": "material-decelerate", "channel": "opacity"},
        {"segment": "s1", "field": "duration_ms", "value": 300, "channel": "opacity", "easing": "material-decelerate"},
    ]
    assert check(spec, good)["verdict"] == "passed"
    assert check(spec, [{"segment": "s1", "field": "distance_px", "value": 16}])["verdict"] == "failed"


@pytest.mark.parametrize("curve", [(0, 0, 0.2, 1), (0.42, 0, 0.58, 1), (0, 0, 1, 1)])
def test_open_fit_recovers_the_hidden_amplitude(curve):
    tau = np.arange(19) / 18
    p = bezier_ease(*curve, tau)
    hidden_start = fit_open(tau[2:], 24 * (1 - p)[2:], True)
    hidden_end = fit_open(tau[:-2], (120 * p)[:-2], False)
    assert hidden_start["amplitude"] == pytest.approx(24, abs=0.3)
    assert hidden_end["amplitude"] == pytest.approx(120, abs=1.5)
