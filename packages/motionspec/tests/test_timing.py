"""Timing semantics are shared, auditable and independent of model responses."""

import copy
import json

import pytest
from motionspec.spec import describe_design, design_timing, summarize
from motionspec.timing import motion_timing, with_timing


def segment():
    easing = {
        "model": "cubic-bezier",
        "params": [0.3, 0, 0.7, 1],
        "css": "cubic-bezier(0.3,0,0.7,1)",
        "rmse": 0.01,
        "named": [
            {"name": "in-out-cubic", "rmse": 0.01, "start_shift_ms": -75, "duration_ms": 600}
        ],
    }
    return {
        "id": "s1",
        "kind": "transform",
        "confidence": "high",
        "region": [0, 0, 50, 50],
        "t_start_ms": 325,
        "t_end_ms": 775,
        "duration_ms": 450,
        "easing": easing,
        "channels": {
            "translate": {
                "t_start_ms": 325,
                "t_end_ms": 775,
                "duration_ms": 450,
                "delta": [0, 90],
                "easing": easing,
            }
        },
        "samples": [{"t_ms": 325, "x": 0, "y": 0}, {"t_ms": 775, "x": 0, "y": 90}],
        "notes": [],
    }


def test_visible_and_design_intervals_are_separate_and_recomputable():
    seg = segment()
    before = copy.deepcopy(seg)
    timing = motion_timing(seg)
    assert seg == before
    assert timing["visible_change"] == {"t_start_ms": 325, "t_end_ms": 775, "duration_ms": 450}
    design = timing["estimated_design"]
    assert (design["t_start_ms"], design["t_end_ms"], design["duration_ms"]) == (250, 850, 600)
    assert design["t_start_ms"] == design["basis_t_start_ms"] + design["start_shift_ms"]
    assert design["t_end_ms"] == design["t_start_ms"] + design["duration_ms"]
    assert design["is_ground_truth"] is False
    assert timing["channels"]["translate"]["estimated_design"] == design
    assert design_timing(seg) == design
    assert describe_design(seg) == "in-out-cubic, 250-850 (600 ms)"


def test_alignment_uses_the_primary_channel_clock_after_json_roundtrip():
    seg = segment()
    seg.update(t_start_ms=300, t_end_ms=900, duration_ms=600)
    seg["channels"] = {"opacity": {"t_start_ms": 300, "t_end_ms": 900}, **seg["channels"]}
    seg = json.loads(json.dumps(seg))  # Easing no longer shares Python object identity.
    timing = motion_timing(seg)
    assert timing["primary_channel"] == "translate"
    assert timing["estimated_design"]["t_start_ms"] == 250
    assert timing["channels"]["opacity"]["estimated_design"] is None


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("truncated", "truncated_segment"),
        ("poor", "poor_named_fit"),
        ("missing_shift", "invalid_alignment"),
        ("missing_curve", "no_named_fit"),
        ("negative", "estimate_before_timeline"),
        ("nan", "poor_named_fit"),
    ],
)
def test_unreliable_design_is_explicitly_unavailable(mutation, reason):
    seg = segment()
    if mutation == "truncated":
        seg["truncated"] = True
    if mutation == "poor":
        seg["easing"]["named"][0]["rmse"] = 0.051
    if mutation == "missing_shift":
        del seg["easing"]["named"][0]["start_shift_ms"]
    if mutation == "missing_curve":
        seg["easing"]["named"] = []
    if mutation == "negative":
        seg["easing"]["named"][0]["start_shift_ms"] = -400
    if mutation == "nan":
        seg["easing"]["named"][0]["rmse"] = float("nan")
    timing = motion_timing(seg)
    assert timing["estimated_design"] is None
    assert timing["design_unavailable_reason"] == reason
    assert timing["visible_change"]["duration_ms"] == 450
    assert reason in describe_design(seg)


def test_summary_and_json_use_identical_timing_calculation():
    spec = {
        "schema": "motionspec/0.1",
        "source": {"width": 100, "height": 100, "duration_ms": 1000, "frames_analysed": 40},
        "time_resolution_ms": 25,
        "segments": [segment()],
    }
    enriched = with_timing(spec)
    assert "timing" not in spec["segments"][0]
    assert enriched == with_timing(enriched)
    text = summarize(enriched)
    assert "325-775 (450 ms) | in-out-cubic, 250-850 (600 ms)" in text
    assert enriched["segments"][0]["timing"] == motion_timing(spec["segments"][0])
