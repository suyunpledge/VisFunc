"""A duplicate can overlap a valid pedestrian and an ignored person in a crowd."""

import json
from pathlib import Path

import pytest


def test_overlapping_ignore_annotation_preserves_duplicate_penalty(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "bench"))
    from pedestrian_eval import score

    (tmp_path / "source.json").write_text(json.dumps({"size": [100, 100], "last_frame": 1}))
    (tmp_path / "gt.txt").write_text("1,1,11,11,30,60,1,1,0.5\n1,2,19,11,30,60,0,7,0.3\n")
    predictions = [
        {"label": "person", "bbox_px": [10, 10, 40, 70]},
        {"label": "person", "bbox_px": [18, 10, 48, 70]},
        {"label": "person", "bbox_px": [25, 10, 53, 70]},
    ]
    result = score(tmp_path, predictions)
    assert (result["tp"], result["fp"], result["ignored_predictions"]) == (1, 1, 1)
    assert result["duplicates_on_ignore"] == result["duplicates"] == 1
    assert result["false_positives"][0]["best_ignore_iou"] == 1.0
