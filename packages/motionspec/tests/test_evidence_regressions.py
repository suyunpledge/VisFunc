"""Review findings are executable guards, independent of paid APIs or Qt."""

import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryFile
from types import SimpleNamespace

import numpy as np
import pytest
from motionspec import qt_probe as qt
from motionspec.measurement import decode_measurement
from motionspec.segment import Candidate, _Components
from motionspec.trace import spec_from_trace
from motionspec.trace_contract import FIELDS


def sample_trace():
    return {
        "schema": "motionspec-trace/0.1",
        "size": [640, 400],
        "ticks_ms": [i * 20.0 for i in range(30)],
        "elements": [
            {
                "id": "a",
                "fields": list(FIELDS),
                "rows": [
                    [i, 20 + i * 5, 10, 50, 20, 1, 1, 0, 1, 20 + i * 5, 10] for i in range(3, 16)
                ],
            }
        ],
    }


def test_recorder_idle_gap_preserves_two_durations(monkeypatch):
    node = SimpleNamespace(values=(0, 20, 50, 20, 1, 1, 0, 1, 0, 20))
    monkeypatch.setattr(qt, "_snapshot", lambda obj: obj.values)
    monkeypatch.setattr(qt, "_children", lambda obj: [])
    monkeypatch.setattr(qt, "_pointer", id)
    monkeypatch.setattr(qt, "_describe", lambda obj: {"name": "card"})
    monkeypatch.setattr(
        qt.Recorder, "_note_window", lambda self, obj: setattr(self, "_size", (640, 400))
    )
    recorder = qt.Recorder([node])
    for t, x in [(0, 0), (20, 10), (40, 20), (60, 20), (1000, 30), (1020, 40), (1040, 40)]:
        node.values = (x, 20, 50, 20, 1, 1, 0, 1, x, 20)
        recorder.sample(t)
    result = spec_from_trace(recorder.trace())
    assert len(result["segments"]) == 2
    for segment in result["segments"]:
        assert segment["channels"]["translate"]["delta"] == [20.0, 0.0]
        assert segment["duration_ms"] == pytest.approx(40, abs=20)
    assert result["segments"][0]["t_end_ms"] < 100
    assert result["segments"][1]["t_start_ms"] > 900


def test_declared_field_order_controls_interpretation():
    original = sample_trace()
    reordered = copy.deepcopy(original)
    element = reordered["elements"][0]
    element["fields"] = list(reversed(FIELDS))
    element["rows"] = [list(reversed(row)) for row in element["rows"]]
    assert spec_from_trace(reordered) == spec_from_trace(original)


def test_static_birth_death_and_reused_pointer_have_distinct_identities(monkeypatch):
    root = SimpleNamespace(values=(0, 0, 640, 400, 1, 1, 0, 1, 0, 0))
    child = SimpleNamespace(values=(10, 10, 50, 20, 1, 1, 0, 1, 10, 10))
    children = []
    monkeypatch.setattr(qt, "_snapshot", lambda obj: obj.values)
    monkeypatch.setattr(qt, "_children", lambda obj: children if obj is root else [])
    monkeypatch.setattr(qt, "_pointer", lambda obj: 1 if obj is root else 2)
    monkeypatch.setattr(qt, "_describe", lambda obj: {"name": "card"})
    monkeypatch.setattr(
        qt.Recorder, "_note_window", lambda self, obj: setattr(self, "_size", (640, 400))
    )
    recorder = qt.Recorder([root])
    recorder.sample(0)
    children.append(child)
    recorder.sample(20)
    children.clear()
    recorder.sample(40)
    children.append(child)  # Simulate a different widget reusing the native address.
    recorder.sample(60)
    result = spec_from_trace(recorder.trace())
    assert result["segments"] == []  # Birth/death does not fabricate a fade animation.
    assert [event["kind"] for event in result["lifecycle"]] == [
        "appeared",
        "disappeared",
        "appeared",
    ]
    assert [event["timestamp_ms"] for event in result["lifecycle"]] == [20, 40, 60]
    assert result["lifecycle"][0]["element_id"] != result["lifecycle"][-1]["element_id"]


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "shape",
        "negative_tick",
        "tick_bounds",
        "repeat_tick",
        "nan",
        "fields",
        "time",
        "ratio",
        "opacity",
    ],
)
def test_malformed_trace_is_explicitly_rejected(case):
    trace = sample_trace()
    element = trace["elements"][0]
    if case == "empty":
        element["rows"] = []
    if case == "shape":
        element["rows"][0].pop()
    if case == "negative_tick":
        element["rows"][0][0] = -1
    if case == "tick_bounds":
        element["rows"][-1][0] = 30
    if case == "repeat_tick":
        element["rows"][1][0] = 3
    if case == "nan":
        element["rows"][0][1] = float("nan")
    if case == "fields":
        element["fields"][1] = "scene_x"
    if case == "time":
        trace["ticks_ms"][1] = 0
    if case == "ratio":
        trace["device_pixel_ratio"] = 0
    if case == "opacity":
        element["rows"][0][5] = 2
    with pytest.raises(ValueError):
        spec_from_trace(trace)


def answer():
    return {
        "element": "card",
        "start_ms": 0,
        "duration_ms": 100,
        "direction": "right",
        "distance_px": 20,
        "opacity_change": "none",
        "easing": "linear",
    }


def test_array_is_delivered_but_format_failure_remains_visible():
    row = answer()
    result = decode_measurement(json.dumps([row]))
    assert result["accepted"] and not result["strict_format_pass"]
    assert result["rows"] == [row]
    assert result["normalizations"] == ["array_envelope"]
    result = decode_measurement("```json\n" + json.dumps({"animations": [row]}) + "\n```")
    assert result["accepted"] and result["normalizations"] == ["markdown_fence"]


@pytest.mark.parametrize(
    "field,value",
    [("easing", None), ("start_ms", True), ("duration_ms", "100"), ("distance_px", -1)],
)
def test_schema_failures_are_never_silently_repaired(field, value):
    row = answer()
    row[field] = value
    result = decode_measurement(json.dumps({"animations": [row]}))
    assert not result["accepted"] and result["errors"]
    assert result["rows"] == [row]  # Retained for separate diagnostic field scoring.


def load_runner():
    path = Path(__file__).parents[1] / "bench" / "run.py"
    spec = importlib.util.spec_from_file_location("evidence_bench_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_attempt_contributes_tokens_and_latency(monkeypatch):
    run = load_runner()
    monkeypatch.setattr(
        run, "find", lambda _: SimpleNamespace(base_url="https://test.invalid", key="unused")
    )
    monkeypatch.setattr(run, "build", lambda *args: [])
    monkeypatch.setattr(run.time, "sleep", lambda _: None)
    replies = iter(
        [
            {
                "ok": True,
                "status": 200,
                "text": "",
                "seconds": 3,
                "error": None,
                "usage": {
                    "total_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            },
            {
                "ok": True,
                "status": 200,
                "text": json.dumps([answer()]),
                "seconds": 5,
                "error": None,
                "usage": {"total_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 5}},
            },
        ]
    )
    monkeypatch.setattr(run.client, "chat", lambda *args, **kw: next(replies))
    result = run.ask(("test", "model", "pattern", 5), "scene", {"truth": [answer()]}, "video")
    assert len(result["attempts"]) == 2
    assert result["request_seconds"] == 8
    assert result["usage"] == {
        "total_tokens": 50,
        "completion_tokens_details": {"reasoning_tokens": 15},
    }
    assert result["format_fail"] and not result["answer_wrong"]
    assert result["delivery"]["accepted"]
    assert result["usage_unknown_attempts"] == 0


def test_quota_failure_is_not_retried(monkeypatch):
    run = load_runner()
    monkeypatch.setattr(
        run, "find", lambda _: SimpleNamespace(base_url="https://test.invalid", key="unused")
    )
    monkeypatch.setattr(run, "build", lambda *args: [])
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        return {"ok": False, "status": 402, "text": "", "seconds": 2, "usage": {}, "error": "quota"}

    monkeypatch.setattr(run.client, "chat", fail)
    result = run.ask(("test", "model", "pattern", 5), "scene", {"truth": [answer()]}, "video")
    assert len(calls) == 1 and result["usage_unknown_attempts"] == 1
    assert result["answer_wrong"] is None


def test_completed_candidates_stay_bounded_without_losing_late_clean_motion():
    with TemporaryFile(mode="w+b") as store:
        groups = _Components((100, 100), (10, 10), 0, 1, store)
        for i in range(20_000):
            groups.active[i] = Candidate(i, i + 1, (0, 0, 10, 10), 1000, 10)
            groups._finish(i)
            assert len(groups.best) <= 1
        groups.active[20_000] = Candidate(20_000, 20_001, (90, 90, 100, 100), 1, 1)
        ambient = np.zeros((10, 10), dtype=bool)
        ambient[0, 0] = True
        kept, dropped = groups.finish(ambient)
        assert len(kept) == 1 and kept[0].first == 20_000
        assert dropped == 20_000


def test_candidate_disk_budget_is_explicit(monkeypatch):
    import motionspec.segment as module

    monkeypatch.setattr(module, "MAX_CANDIDATE_BYTES", 0)
    with TemporaryFile(mode="w+b") as store:
        groups = _Components((100, 100), (10, 10), 0, 1, store)
        groups.active[0] = Candidate(0, 1, (0, 0, 10, 10), 1, 1)
        with pytest.raises(ValueError, match="budget"):
            groups._finish(0)


def test_report_total_is_item_weighted_and_unknown_cost_stays_unknown():
    path = Path(__file__).parents[1] / "bench" / "report.py"
    spec = importlib.util.spec_from_file_location("evidence_bench_report", path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows = [
        {
            "model": "a",
            "condition": "test",
            "scene": "clip",
            "score": {"count": [1, 1]},
            "seconds": 1,
            "usage": {},
        },
        {
            "model": "b",
            "condition": "test",
            "scene": "clip",
            "score": {"count": [0, 9]},
            "seconds": 1,
            "usage": {},
        },
    ]
    overall = report.aggregate(rows)["ALL", "test"]
    assert overall["score"] == [1, 10]  # 10%, not the 50% average of model percentages.
    assert overall["strict"] == [0, 0] and overall["unknown_usage"] == 2
    assert "1/10 (10.0%)" in report.render(rows)


def test_provider_ambiguity_never_silently_selects_coding_or_metered_endpoint(monkeypatch):
    load_runner()
    import providers

    slots = [
        providers.Provider("plan", "https://token.invalid/v1", "model", "test", "unused"),
        providers.Provider("metered", "https://api.invalid/v1", "model", "test", "unused"),
    ]
    monkeypatch.setattr(providers, "load", lambda: slots)
    with pytest.raises(ValueError, match="ambiguous"):
        providers.find("model")
    assert providers.find("^plan$").slot == "plan"
