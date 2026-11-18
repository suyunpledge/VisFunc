"""Hard-case benchmarks use the same delivery and cost guarantees as the first suite."""
import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
run = importlib.import_module("run_hard")
report = importlib.import_module("report_hard")
retry = importlib.import_module("retry_hard")


def answer():
    return {"element": "card", "start_ms": 166.7, "duration_ms": 400,
            "dx_px": 0, "dy_px": -90, "dw_px": 0, "dh_px": 0,
            "opacity_change": "none", "color_change": "no", "easing": "overshoot"}


def test_design_measurement_never_substitutes_visible_support(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    compare = importlib.import_module("qt_hard_compare")
    want = {"start_ms": 100, "duration_ms": 250, "easing": "linear"}
    segment = {"channels": {}, "t_start_ms": 100, "duration_ms": 250}
    marks = compare.judge(want, segment, 1)
    assert marks["found"] and not marks["start"] and not marks["duration"]
    segment["timing"] = {"estimated_design": {"t_start_ms": 100, "duration_ms": 250}}
    marks = compare.judge(want, segment, 1)
    assert marks["start"] and marks["duration"]


def test_array_delivery_retains_signed_values_and_format_failure():
    response = run.decode(json.dumps([answer()]))
    assert response["accepted"] and not response["strict_format_pass"]
    assert response["rows"][0]["dy_px"] == -90
    assert response["normalizations"] == ["array_envelope"]


@pytest.mark.parametrize("field,value", [("dx_px", True), ("dy_px", "-90"), ("duration_ms", -1), ("easing", "unknown")])
def test_schema_failures_are_distinct_from_readable_field_values(field, value):
    row = answer()
    row[field] = value
    response = run.decode(json.dumps({"animations": [row]}))
    assert not response["accepted"] and response["rows"] == [row]


def setup_client(monkeypatch, replies):
    monkeypatch.setattr(run, "find", lambda _: SimpleNamespace(base_url="https://test.invalid", key="unused", slot="test"))
    monkeypatch.setattr(run, "build", lambda *args: [])
    monkeypatch.setattr(run.time, "sleep", lambda _: None)
    monkeypatch.setattr(run.client, "chat", lambda *args, **kwargs: next(replies))


def reply(ok, status, text="", tokens=None, seconds=2):
    return {"ok": ok, "status": status, "text": text, "seconds": seconds,
            "usage": {} if tokens is None else {"total_tokens": tokens, "completion_tokens_details": {"reasoning_tokens": 1}},
            "error": None if ok else "test failure"}


def test_attempt_costs_and_identity_are_complete(monkeypatch):
    setup_client(monkeypatch, iter([reply(False, 503, tokens=5), reply(True, 200, json.dumps([answer()]), tokens=10, seconds=3)]))
    row = run.ask(("test", "model", "test", 5), "overshoot", {}, "video", 1000, {}, repeat=2, run_id="r")
    assert row["usage"]["total_tokens"] == 15
    assert row["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2
    assert row["request_seconds"] == 5 and len(row["attempts"]) == 2
    assert row["format_fail"] and not row["answer_wrong"]
    assert row["repeat"] == 2 and row["run_id"] == "r" and row["task_id"]


def test_402_is_not_retried_and_unknown_usage_stays_visible(monkeypatch):
    setup_client(monkeypatch, iter([reply(False, 402)]))
    row = run.ask(("test", "model", "test", 5), "overshoot", {}, "video", 1000, {})
    assert len(row["attempts"]) == 1 and row["usage_unknown_attempts"] == 1
    assert row["transport_fail"] and row["format_fail"] is None


def pair_row(condition, repeat=0, accepted=True):
    return {"model": "m", "scene": "clip", "run_id": "r", "repeat": repeat,
            "provider_slot": "slot", "condition": condition, "ok": True,
            "delivery": {"accepted": accepted}}


def test_pairing_never_borrows_a_success_from_another_repeat():
    rows = [pair_row("a", 0), pair_row("b", 0, False), pair_row("a", 1, False), pair_row("b", 1)]
    assert report.paired_rows(rows, ["a", "b"]) == []
    assert report.paired_rows([pair_row("a"), pair_row("b")], ["a", "b"]) != []
    assert report.paired_rows([pair_row("a"), pair_row("a"), pair_row("b")], ["a", "b"]) == []
    legacy = [pair_row("a"), pair_row("b")]
    for row in legacy:
        del row["repeat"]
    assert report.paired_rows(legacy, ["a", "b"]) == []


def test_report_uses_item_denominators_and_exposes_count_error(monkeypatch):
    setup_client(monkeypatch, iter([reply(True, 200, json.dumps({"animations": [answer(), answer()]}), tokens=10)]))
    row = run.ask(("test", "model", "test", 5), "overshoot", {}, "video", 1000, {}, run_id="r")
    assert row["answer_wrong"] and row["score"]["extra"] == 1
    assert "7/7 (100.0%)" in report.render([row])
    assert "0/1 (0.0%)" in report.render([row])


def test_manual_rerun_never_replaces_original_evidence(tmp_path, monkeypatch):
    failed = {"model": "mimo-v2.6-flash", "scene": "overshoot", "condition": "video", "score": {"parsed": False}}
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(failed) + "\n", encoding="utf-8")
    original = path.read_bytes()
    monkeypatch.setattr(retry.R, "find", lambda _: None)
    monkeypatch.setattr(retry.R, "prepare", lambda *args: {"overshoot": {}})
    monkeypatch.setattr(retry.R, "ask", lambda *args, **kwargs: {**copy.deepcopy(failed), **kwargs})
    out = tmp_path / "rerun"
    retry.main([str(tmp_path), "--models", "mimo-v2.6-flash", "--out", str(out)])
    assert path.read_bytes() == original
    result = json.loads((out / "raw.jsonl").read_text(encoding="utf-8"))
    assert result["retry_of"] == {"source_line": 1}
    assert json.loads((out / "protocol.json").read_text(encoding="utf-8"))["retry_source_sha256"]
