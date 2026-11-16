"""One-to-one, label-and-location scoring against independently supplied boxes."""

from __future__ import annotations

from .geometry import canonical_label, iou


def score_objects(predictions: list[dict], expected: list[dict], threshold: float = 0.5) -> dict:
    """Maximum cardinality matching; duplicate predictions cannot reuse a ground truth."""
    adjacency = []
    for pred in predictions:
        candidates = [
            (index, iou(tuple(pred["bbox_px"]), tuple(gt["bbox_px"])))
            for index, gt in enumerate(expected)
            if canonical_label(pred["label"]) == canonical_label(gt["label"])
        ]
        adjacency.append(
            [
                index
                for index, overlap in sorted(candidates, key=lambda p: -p[1])
                if overlap >= threshold
            ]
        )
    matched: dict[int, int] = {}

    def augment(pred_index: int, visited: set[int]) -> bool:
        for gt_index in adjacency[pred_index]:
            if gt_index in visited:
                continue
            visited.add(gt_index)
            if gt_index not in matched or augment(matched[gt_index], visited):
                matched[gt_index] = pred_index
                return True
        return False

    for index in range(len(predictions)):
        augment(index, set())
    tp = len(matched)
    fp, fn = len(predictions) - tp, len(expected) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "matches": [
            {
                "expected_index": g,
                "prediction_index": p,
                "iou": iou(tuple(expected[g]["bbox_px"]), tuple(predictions[p]["bbox_px"])),
            }
            for g, p in sorted(matched.items())
        ],
    }


def aggregate(scores: list[dict]) -> dict:
    tp, fp, fn = (sum(score[key] for score in scores) for key in ("tp", "fp", "fn"))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }
