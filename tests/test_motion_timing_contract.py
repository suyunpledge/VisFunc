"""SVSG consumes timing evidence; it never refits the MotionSpec measurements."""

import copy

import pytest
from pydantic import ValidationError

from svsg.contracts.video_functions import MotionFunctions, MotionTiming
from svsg.video_functions.temporal import compile_motion


def packet():
    design = {
        "t_start_ms": 250,
        "t_end_ms": 850,
        "duration_ms": 600,
        "basis_t_start_ms": 325,
        "start_shift_ms": -75,
        "easing": "in-out-cubic",
        "rmse": 0.01,
        "rmse_limit": 0.05,
        "provenance": "named_curve_alignment",
        "is_ground_truth": False,
    }
    return {
        "visible_change": {"t_start_ms": 325, "t_end_ms": 775, "duration_ms": 450},
        "estimated_design": design,
        "design_unavailable_reason": None,
    }


def segment():
    return {
        "id": "s1",
        "kind": "transform",
        "confidence": "high",
        "region": [0, 0, 50, 50],
        "t_start_ms": 325,
        "t_end_ms": 775,
        "samples": [{"t_ms": 325, "x": 0}, {"t_ms": 775, "x": 90}],
        "channels": {},
        "timing": packet(),
    }


def test_compiler_keeps_design_outside_visible_support_without_relabelling_samples():
    seg = segment()
    function = compile_motion(seg)
    assert function.timing.estimated_design.duration_ms == 600
    assert function.t_start_ms == 325 and function.t_end_ms == 775
    assert function.samples["x"].knots == [(325, 0), (775, 90)]
    assert MotionFunctions.model_validate_json(function.model_dump_json()) == function


def test_legacy_data_remains_readable_and_does_not_fabricate_design_timing():
    seg = segment()
    del seg["timing"]
    function = compile_motion(seg)
    assert function.timing.estimated_design is None
    assert function.timing.design_unavailable_reason == "legacy_missing_timing"
    legacy = function.model_dump(mode="json", exclude={"timing"})
    assert MotionFunctions.model_validate(legacy).timing is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("duration_ms", 450),
        ("start_shift_ms", -70),
        ("rmse", 0.051),
        ("t_end_ms", float("inf")),
        ("is_ground_truth", True),
        ("duration_ms", True),
    ],
)
def test_fabricated_or_inconsistent_design_metadata_is_rejected(key, value):
    timing = packet()
    timing["estimated_design"][key] = value
    with pytest.raises(ValidationError):
        MotionTiming.model_validate(timing)


def test_missing_design_needs_a_reason_and_primary_channel_must_agree():
    timing = packet()
    timing["estimated_design"] = None
    with pytest.raises(ValidationError):
        MotionTiming.model_validate(timing)
    timing = packet()
    timing["primary_channel"] = "translate"
    timing["channels"] = {"translate": copy.deepcopy(packet())}
    timing["channels"]["translate"]["estimated_design"]["easing"] = "linear"
    with pytest.raises(ValidationError):
        MotionTiming.model_validate(timing)


def test_truncation_or_wrong_visible_domain_rejects_claimed_design():
    seg = segment()
    seg["truncated"] = True
    with pytest.raises(ValidationError, match="truncated"):
        compile_motion(seg)
    seg = segment()
    seg["timing"]["visible_change"].update(t_end_ms=800, duration_ms=475)
    with pytest.raises(ValidationError, match="visible timing"):
        compile_motion(seg)
