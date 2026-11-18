"""第二批界面动效：遮罩上的对话框、互相交错的元素、只沿一个方向变尺寸的色块、只是挪动的纯色卡片。"""

import numpy as np
import pytest

from motionspec import analyze, validate
from motionspec.fit import bezier_ease
from motionspec.synth import background, card, compose, indicator, modal, progress, swap

FRAME = 1000 / 60


def design(seg):
    return (seg.get("timing") or {}).get("estimated_design") or {}


def test_modal_backdrop_and_dialog_are_measured_separately():
    frames, truth = modal()
    spec = analyze(frames, fps=60)
    assert validate(spec) == []
    backdrop, dialog = spec["segments"]
    assert backdrop["kind"] == "fade"
    assert design(backdrop)["easing"] == "linear"
    assert design(backdrop)["duration_ms"] == pytest.approx(truth["dim_ms"], abs=2 * FRAME)
    assert dialog["kind"] == "transform"
    move = dialog["channels"]["translate"]
    assert move["delta"][1] == pytest.approx(-24, abs=1.5) and abs(move["delta"][0]) < 0.5
    assert (dialog["channels"]["opacity"]["from"], dialog["channels"]["opacity"]["to"]) == (0.0, 1.0)
    assert design(dialog)["duration_ms"] == pytest.approx(300, abs=2 * FRAME)
    assert any("dimming" in note for note in dialog["notes"])


def test_two_elements_crossing_each_other_are_separated():
    frames, truth = swap()
    spec = analyze(frames, fps=60)
    assert validate(spec) == []
    moves = sorted(seg["channels"]["translate"]["delta"][1] for seg in spec["segments"])
    assert len(moves) == 2
    assert moves[0] == pytest.approx(-70, abs=2) and moves[1] == pytest.approx(70, abs=2)
    assert all(seg["model"] == "multibody" for seg in spec["segments"])


def test_progress_bar_grows_from_its_left_edge():
    frames, truth = progress()
    (seg,) = analyze(frames, fps=60)["segments"]
    size = seg["channels"]["size"]
    assert size["width_from"] == pytest.approx(0, abs=1) and size["width_to"] == pytest.approx(360, abs=2)
    assert size["anchor"] == "left"
    assert "translate" not in seg["channels"]
    assert design(seg)["easing"] == "linear"
    assert design(seg)["duration_ms"] == pytest.approx(500, abs=2 * FRAME)


def test_indicator_moves_and_widens_at_once():
    frames, truth = indicator()
    (seg,) = analyze(frames, fps=60)["segments"]
    size, move = seg["channels"]["size"], seg["channels"]["translate"]
    assert size["width_from"] == pytest.approx(80, abs=2) and size["width_to"] == pytest.approx(120, abs=2)
    assert size["height_from"] == size["height_to"]
    assert move["delta"][0] == pytest.approx(truth["centre_shift"], abs=2)
    assert design(seg)["easing"] == "out-cubic"


def test_plain_card_moving_is_not_an_entering_layer():
    """纯色卡片整体上移：变化区域被拆成上下两条，不能被当成"新出现的元素"。"""
    bg = background()
    sprite = card(220, 120, (60, 50, 45))
    tau = np.r_[np.zeros(10), np.arange(25) / 24, np.ones(12)]
    frames = [compose(bg, sprite, 320, 240 - 70 * v) for v in bezier_ease(0.34, 1.56, 0.64, 1.0, tau)]
    (seg,) = analyze(frames, fps=60)["segments"]
    assert seg.get("model") != "layer"
    assert "opacity" not in seg["channels"]
    assert seg["channels"]["translate"]["delta"][1] == pytest.approx(-70, abs=2)
