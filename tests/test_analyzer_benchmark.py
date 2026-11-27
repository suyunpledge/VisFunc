"""Guard the independent motion benchmark's holdout and scoring contract."""

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
from analyzer_eval import match, score, summarize  # noqa: E402
from motion_corpus import cases  # noqa: E402


@pytest.mark.parametrize("source", ["qt", "browser"])
def test_frozen_corpus_has_disjoint_scenes_and_media(source):
    dev, test = (cases(source, split) for split in ("dev", "test"))
    assert not {c["id"] for c in dev} & {c["id"] for c in test}
    if source == "qt":
        assert sum("legacy" in c for c in dev) == 15
        assert len(dev) - 15 == len(test) == 30
        assert all("legacy" not in c for c in test)
    with ZipFile(BENCH / "corpus" / f"{source}.zip") as archive:
        seen = set()
        for case in dev + test:
            base = case["id"] + "/"
            digest = hashlib.sha256(archive.read(base + "video.mp4")).digest()
            assert digest not in seen
            seen.add(digest)
            assert json.loads(archive.read(base + "truth.json"))["segments"]
            if source == "browser":
                trace = json.loads(archive.read(base + "trace.json"))
                pts = trace["encoded_pts_ms"]
                assert pts == [
                    round(1000 * (t - trace["capture_epoch_s"][0]))
                    for t in trace["capture_epoch_s"]
                ]
                assert pts == sorted(set(pts))
                assert trace["cdp_animations"] and trace["states"] and trace["animations"]


def test_matching_keeps_two_objects_even_when_one_region_covers_both():
    boxes = [[0, 0, 20, 20], [15, 0, 30, 20]]
    preds = [{"region": [0, 0, 30, 20]}, {"region": [0, 0, 12, 20]}]
    assert match([{"rect": b} for b in boxes], preds) == [1, 0]


def test_missing_and_invalid_outputs_both_stay_in_denominator():
    want = {
        "rect": [0, 0, 10, 10],
        "element": "panel",
        "start_ms": 100,
        "duration_ms": 200,
        "delta": [0, 0],
        "resize": [0, 30],
        "easing": "linear",
    }
    case = {"id": "fixture", "family": "panel"}
    missing = score(case, [want], [])
    invalid = score(case, [want], [], "format_fail")
    summary = summarize([missing, invalid])
    assert summary["missed_objects"] == 1 and summary["format_fail"] == 1
    assert summary["scene_macro"]["mean"] == 0
    assert summary["fields"]["height_px"]["missing"] == 1
    assert summary["fields"]["height_px"]["format_fail"] == 1
