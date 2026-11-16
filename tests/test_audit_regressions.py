import copy
import hashlib
import importlib
import json
import weakref

import numpy as np
import pytest

from motionspec import analyze, check, validate
from motionspec.cli import main
from motionspec.frames import FrameSource, read_image, write_image
from motionspec.synth import background, card, compose, stagger


def _color_frames():
    red = np.full((128, 160, 3), 25, np.uint8)
    green = red.copy()
    red[20:100, 20:140] = [0, 0, 255]
    green[20:100, 20:140] = [0, 130, 0]
    return red, green


def test_equal_luma_change_has_full_size_hashed_evidence(tmp_path):
    red, green = _color_frames()
    spec = analyze(
        [red] * 12 + [green] * 12, tmp_path, fps=60, max_dim=80, sheets=False
    )
    assert [segment["kind"] for segment in spec["segments"]] == ["instant"]
    first, *_, last = spec["evidence_frames"]
    for row, expected in ((first, red), (last, green)):
        path = tmp_path / row["image"]
        assert np.array_equal(read_image(path), expected)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert not (tmp_path / "s1.png").exists()


def test_static_clips_keep_distinguishable_images(tmp_path):
    black = np.zeros((20, 32, 3), np.uint8)
    white = np.full_like(black, 255)
    a = analyze([black] * 4, tmp_path / "black")
    b = analyze([white] * 4, tmp_path / "white")
    assert a["segments"] == b["segments"] == []
    assert a["evidence_frames"][0]["sha256"] != b["evidence_frames"][0]["sha256"]


@pytest.mark.parametrize("bad_stamp", [0.0, float("nan")])
def test_timestamp_fallback_applies_before_crop(monkeypatch, tmp_path, bad_stamp):
    red, green = _color_frames()
    source = [red] * 12 + [green] * 18
    init = FrameSource.__init__

    def init_with_container_clock(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.info.timestamps = "container"

    monkeypatch.setattr(FrameSource, "__init__", init_with_container_clock)
    monkeypatch.setattr(
        FrameSource,
        "_raw",
        lambda self: ((i, bad_stamp, f) for i, f in enumerate(source)),
    )
    spec = analyze(source, tmp_path, fps=30, start_ms=100, end_ms=600)
    assert spec["source"]["timestamps"] == "index"
    assert spec["source"]["t_start_ms"] == pytest.approx(100)
    assert spec["segments"][0]["frames"] == [11, 12]
    assert spec["evidence_frames"][0]["frame"] == 3
    assert spec["evidence_frames"][-1]["frame"] == 18
    assert any("rescanned" in warning for warning in spec["warnings"])


def test_image_names_have_natural_order_and_require_fps(tmp_path):
    for i in range(1, 13):
        write_image(tmp_path / f"{i}.png", np.full((24, 24, 3), i, np.uint8))
    with pytest.raises(ValueError, match="explicit fps"):
        FrameSource(tmp_path)
    assert [int(f[0, 0, 0]) for _, _, f in FrameSource(tmp_path, fps=30)] == list(
        range(1, 13)
    )


@pytest.mark.parametrize(
    "options",
    [
        {"fps": -1},
        {"fps": float("nan")},
        {"max_dim": 0},
        {"max_frames": 2},
        {"threshold": 256},
    ],
)
def test_bad_options_fail_without_invalid_output(tmp_path, options):
    with pytest.raises(ValueError):
        analyze([np.zeros((16, 16, 3), np.uint8)] * 3, tmp_path, **options)
    assert not (tmp_path / "spec.json").exists()


def test_schema_and_claims_reject_malformed_evidence(tmp_path, capsys):
    a, b = _color_frames()
    spec = analyze(
        [a] * 10 + [b] * 10, tmp_path, fps=60, raw_frames=False, sheets=False
    )
    malformed = copy.deepcopy(spec)
    malformed["segments"].append(None)
    malformed["segments"][0].update(duration_ms=-1, region=[1], samples="invalid")
    assert len(validate(malformed)) >= 4
    for claim in [
        {"field": "duration_ms", "value": "NaN"},
        {"field": "duration_ms", "value": float("nan")},
        {"field": "duration_ms", "value": 99999, "tol": 99999},
        {"field": "overshoot", "value": "false"},
    ]:
        result = check(spec, [{"segment": "s1", **claim}])
        assert result["verdict"] == "failed"
        json.dumps(result, allow_nan=False)
    assert check(spec, [])["verdict"] == "failed"
    assert (
        check(spec, [{"segment": "s1", "field": "scale_to", "value": 1}])["verdict"]
        == "insufficient_evidence"
    )
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(
        '[{"segment":"s1","field":"duration_ms","value":NaN}]', encoding="utf-8"
    )
    assert main(["check", str(tmp_path / "spec.json"), str(claims_path)]) == 2
    assert "invalid_input" in json.loads(capsys.readouterr().out)


def test_window_crops_release_after_render(monkeypatch, tmp_path):
    module = importlib.import_module("motionspec.analyze")
    original = module.analyze_window
    refs = []

    def observe(crops, *args, **kwargs):
        assert all(ref() is None for ref in refs)
        refs.extend(weakref.ref(crop) for crop in crops)
        return original(crops, *args, **kwargs)

    monkeypatch.setattr(module, "analyze_window", observe)
    frames, _ = stagger()
    spec = analyze(frames, tmp_path, max_crop_bytes=8_000_000, raw_frames=False)
    assert len(spec["segments"]) == 2
    assert all((tmp_path / s["sheet"]).is_file() for s in spec["segments"])


def test_overlapping_windows_split_to_respect_shared_budget(monkeypatch):
    module = importlib.import_module("motionspec.analyze")
    original = module._decode_frames
    passes = []

    def observe(*args):
        passes.append(args[-1])
        yield from original(*args)

    monkeypatch.setattr(module, "_decode_frames", observe)
    bg, sprite = background(), card(110, 65)
    frames = []
    for x in np.r_[np.full(10, 215), np.linspace(215, 345, 31), np.full(12, 345)]:
        top = compose(bg, sprite, x, 88)
        frames.append(compose(top, sprite, x, 290))
    spec = analyze(frames, fps=60, max_crop_bytes=5_000_000)
    assert len(spec["segments"]) == len(passes) == 2
    assert all(
        s["channels"]["translate"]["delta"][0] == pytest.approx(130, abs=2)
        for s in spec["segments"]
    )
    assert spec["coverage"]["truncated_segments"] == 0


def test_dropped_segments_do_not_validate_completeness():
    a, b = _color_frames()
    spec = analyze([a] * 10 + [b] * 10, max_segments=0)
    assert spec["coverage"]["dropped_candidates"] == 1
    assert (
        check(spec, [{"field": "count", "value": 0}])["verdict"]
        == "insufficient_evidence"
    )


@pytest.mark.parametrize("exponent", [1, 3])
def test_combined_transform_keeps_anchor_semantics_and_channel_timing(exponent):
    bg, sprite = background(), card()
    p = np.r_[np.zeros(10), np.linspace(0, 1, 31), np.ones(12)]
    frames = [
        compose(bg, sprite, 250 + 60 * v, 190, scale=1 + 0.5 * v**exponent) for v in p
    ]
    (seg,) = analyze(frames, fps=60)["segments"]
    ref = seg["transform_reference"]
    assert ref["feature_displacement_px"][0] > 35
    assert seg["duration_ms"] == pytest.approx(500, abs=17)
    if exponent == 1:
        assert ref["type"] == "fitted_fixed_point"
        assert "translate" not in seg["channels"]
        assert any("design pivot are unknown" in note for note in seg["notes"])
    else:
        assert ref["type"] == "tracked_feature_anchor"
        move, scale = seg["channels"]["translate"], seg["channels"]["scale"]
        assert move["delta"][0] > 35
        assert scale["t_start_ms"] > move["t_start_ms"]
        # The anchor is off-center, so its trajectory includes a contribution from scaling.
        assert move["reference"] == "tracked_feature_anchor"
        assert move["easing"]["named"][0]["name"] != scale["easing"]["named"][0]["name"]
        assert scale["easing"]["named"][0]["name"] == "in-cubic"
