"""Keep the model benchmark reproducible and charge every attempt to the same budget."""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from analyzer_eval import fields
from motion_corpus import cases
from strong_client import Ledger, metered_usd
from strong_eval import predictions
from strong_inputs import select
from strong_report import histories


@pytest.mark.parametrize("source", ["qt", "browser"])
def test_selects_two_distinct_holdouts_per_family(source):
    chosen = select(source)
    assert len(chosen) == len(set(chosen)) == 12
    assert set(
        Counter(c["family"] for c in cases(source, "test") if c["id"] in chosen).values()
    ) == {2}
    assert set(chosen).isdisjoint(c["id"] for c in cases(source, "dev"))


def test_model_unknown_is_distinct_from_zero_translation_and_format_failure():
    row = {
        "region": [0, 0, 50, 10],
        "start_ms": 0,
        "duration_ms": 100,
        "resize": [0, 10],
        "delta": None,
        "opacity": None,
        "easing": "linear",
    }
    truth = {**row, "delta": [0, 0], "rect": row["region"]}
    del truth["opacity"]
    parsed, bare = predictions(json.dumps([row]))
    assert (
        bare
        and fields(truth, parsed[0], implicit_stationary=False)["direction"]["state"] == "missing"
    )
    parsed, bare = predictions(json.dumps({"segments": [{**row, "delta": [0, 0]}]}))
    assert (
        not bare
        and fields(truth, parsed[0], implicit_stationary=False)["direction"]["state"] == "correct"
    )
    with pytest.raises(ValueError):
        predictions(json.dumps([dict(row, region=[1, 2, 0, 3])]))


def test_parallel_ledgers_reserve_before_call_and_release_only_settled_difference(tmp_path):
    path = tmp_path / "budget.sqlite"
    ledgers = [Ledger(path, 1) for _ in range(2)]

    def reserve(ledger):
        try:
            return ledger.reserve("frame", 0.7)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(reserve, ledgers))
    assert results.count(None) == 1
    key = next(k for k in results if k)
    ledgers[0].settle(key, 0.15)
    assert ledgers[1].reserve("next", 0.8)


def test_uses_billed_read_and_write_categories_and_keeps_previous_tool_turns():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "input_tokens_details": {
            "cached_tokens": 400,
            "cache_write_tokens": 250,
            "cache_write_tokens_details": {"ttl_12h_tokens": 50, "ttl_30m_tokens": 200},
        },
    }
    assert metered_usd("gpt", usage) == pytest.approx(
        (350 * 2 + 400 * 0.1 + 200 * 2.5 + 50 * 5 + 100 * 10) / 1e6
    )
    assert metered_usd("mimo", usage) == 0
    receipt = {"attempts": [2, 3], "previous": {"attempts": [1]}}
    assert [t for run in histories(receipt) for t in run["attempts"]] == [2, 3, 1]
