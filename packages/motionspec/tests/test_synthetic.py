"""用已知真值的合成动效检验整条管线。帧间隔 16.67ms，时长容差取两帧。"""

import json

import numpy as np
import pytest

from motionspec import analyze, check, validate
from motionspec.fit import bezier_ease, fit_bezier, fit_spring, spring_response
from motionspec.synth import fade, pop, slide, stagger, write_video

FRAME = 1000 / 60


def only(spec):
    assert validate(spec) == []
    assert len(spec["segments"]) == 1, [s["kind"] for s in spec["segments"]]
    return spec["segments"][0]


def test_slide_recovers_distance_curve_and_duration():
    frames, truth = slide()
    seg = only(analyze(frames, fps=60))
    assert seg["kind"] == "transform"
    dx, dy = seg["channels"]["translate"]["delta"]
    assert abs(dx - 300) < 2 and abs(dy) < 1
    assert abs(seg["t_start_ms"] - truth["start_ms"]) <= FRAME
    best = seg["easing"]["named"][0]
    assert best["name"] == "out-cubic"
    assert abs(best["duration_ms"] - 400) <= 2 * FRAME
    assert seg["easing"]["rmse"] < 0.01


@pytest.mark.parametrize("name,curve", [("ease-in-out", (0.42, 0, 0.58, 1)), ("linear", (0, 0, 1, 1)), ("material-standard", (0.4, 0, 0.2, 1))])
def test_slide_identifies_named_curve(name, curve):
    frames, _ = slide(duration_ms=350, distance=220, curve=curve)
    seg = only(analyze(frames, fps=60))
    best = seg["easing"]["named"][0]
    assert best["name"] == name
    assert abs(best["duration_ms"] - 350) <= 2 * FRAME


def test_fade_is_not_mistaken_for_motion():
    frames, _ = fade()
    seg = only(analyze(frames, fps=60))
    assert seg["kind"] == "fade"
    assert abs(seg["duration_ms"] - 250) <= 2 * FRAME
    assert seg["easing"]["named"][0]["name"] == "linear"
    assert "translate" not in seg["channels"]


def test_slow_low_contrast_fade_is_detected():
    frames, _ = fade(duration_ms=800)
    seg = only(analyze(frames, fps=60))
    assert seg["kind"] == "fade"
    assert abs(seg["duration_ms"] - 800) <= 4 * FRAME


def test_spring_pop_recovers_spring_parameters():
    frames, truth = pop()
    seg = only(analyze(frames, fps=60))
    assert seg["kind"] == "transform" and seg["dominant"] == "scale"
    easing = seg["easing"]
    assert easing["model"] == "spring"
    assert abs(easing["damping_ratio"] - truth["zeta"]) < 0.04
    assert abs(easing["angular_frequency"] - truth["omega"]) < 2
    assert abs(seg["channels"]["scale"]["to"] - truth["scale_to"]) < 0.04
    assert "translate" not in seg["channels"]  # 绕中心缩放不应报出位移


def test_two_elements_become_two_segments():
    frames, _ = stagger()
    spec = analyze(frames, fps=60)
    kinds = [s["kind"] for s in spec["segments"]]
    assert kinds == ["transform", "fade"]
    first, second = spec["segments"]
    assert abs(first["duration_ms"] - 300) <= 2 * FRAME
    assert abs(second["duration_ms"] - 200) <= 2 * FRAME
    assert second["t_start_ms"] > first["t_end_ms"]


def test_static_clip_has_no_segments():
    frames, _ = slide()
    spec = analyze([frames[0]] * 30, fps=60)
    assert spec["segments"] == []


def test_instant_change():
    frames, _ = fade()
    spec = analyze([frames[0]] * 12 + [frames[-1]] * 12, fps=60)
    seg = only(spec)
    assert seg["kind"] == "instant"
    assert abs(seg["t_start_ms"] - 11 * FRAME) <= FRAME


def test_out_and_back_is_flagged_instead_of_fitted():
    frames, _ = slide(distance=60, duration_ms=200, tail=0)
    seg = only(analyze(frames + frames[::-1], fps=60))
    assert seg.get("pattern") == "out-and-back"
    assert seg["easing"] is None
    assert abs(seg["channels"]["translate"]["distance_px"] - 60) < 3


def test_outputs_are_written(tmp_path):
    frames, _ = slide()
    spec = analyze(frames, tmp_path, fps=60)
    saved = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert saved["segments"][0]["id"] == spec["segments"][0]["id"]
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("# Motion spec")
    assert (tmp_path / "s1.png").stat().st_size > 1000
    assert (tmp_path / "overview.png").exists()


def test_video_file_round_trip(tmp_path):
    frames, _ = slide()
    path = tmp_path / "clip.mp4"
    write_video(path, frames)
    seg = only(analyze(path))
    assert seg["kind"] == "transform"
    assert abs(seg["channels"]["translate"]["distance_px"] - 300) < 4
    assert abs(seg["t_start_ms"] - 10 * FRAME) <= 2 * FRAME


def test_claims_are_checked_against_measurements():
    frames, _ = slide()
    spec = analyze(frames, fps=60)
    good = [
        {"field": "count", "value": 1},
        {"segment": "s1", "field": "kind", "value": "transform"},
        {"segment": "s1", "field": "direction", "value": "right"},
        {"segment": "s1", "field": "distance_px", "value": 300},
        {"segment": "s1", "field": "easing", "value": "out-cubic"},
        {"segment": "s1", "field": "duration_ms", "value": 400, "easing": "out-cubic"},
    ]
    assert check(spec, good)["passed"]
    bad = [
        {"segment": "s1", "field": "direction", "value": "left"},
        {"segment": "s1", "field": "easing", "value": "ease-in"},
        {"segment": "s1", "field": "duration_ms", "value": 900},
        {"segment": "s9", "field": "kind", "value": "fade"},
        {"segment": "s1", "field": "colour", "value": "blue"},
    ]
    verdict = check(spec, bad)
    assert not verdict["passed"]
    assert [v["claim"] for v in verdict["violations"]] == [0, 1, 2, 3, 4]
    assert verdict["violations"][0]["measured"] == "right"


def test_bezier_fit_reproduces_curve():
    tau = np.linspace(0, 1, 25)
    fitted = fit_bezier(tau, bezier_ease(0.2, 0.0, 0.0, 1.0, tau))
    assert fitted["rmse"] < 0.002
    assert np.abs(bezier_ease(*fitted["params"], tau) - bezier_ease(0.2, 0.0, 0.0, 1.0, tau)).max() < 0.01


def test_spring_fit_reproduces_parameters():
    t = np.arange(0, 0.8, 1 / 60)
    fitted = fit_spring(t, spring_response(0.5, 20.0, t))
    assert abs(fitted["damping_ratio"] - 0.5) < 0.02
    assert abs(fitted["angular_frequency"] - 20.0) < 0.5
