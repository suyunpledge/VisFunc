"""Score the unchanged video analyzer against a frozen Qt or Chromium split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from motion_corpus import BENCH, TOOLS, cases

sys.path.insert(0, str(TOOLS.parent))
from motionspec import analyze
from motionspec.spec import validate

sys.path.insert(0, str(TOOLS))
from qt_hard_compare import FAMILY, family


def iou(a, b):
    common = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return common / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - common)


def match(truth, predictions):
    """Maximize number of eligible pairs, then total box IoU. Do not match on scored fields."""

    @lru_cache(None)
    def best(i, used):
        if i == len(truth):
            return 0, 0.0, ()
        count, total, rest = best(i + 1, used)
        winner = (count, total, (None, *rest))
        for j, pred in enumerate(predictions):
            if used & (1 << j):
                continue
            overlap = iou(truth[i]["rect"], pred["region"])
            if overlap < 0.15:
                continue
            n, score, tail = best(i + 1, used | (1 << j))
            if (n + 1, score + overlap) > winner[:2]:
                winner = (n + 1, score + overlap, (j, *tail))
        return winner

    return list(best(0, 0)[2])


def direction(x, y):
    if math.hypot(x, y) <= 2:
        return "still"
    if abs(x) >= abs(y):
        return "right" if x > 0 else "left"
    return "down" if y > 0 else "up"


def measured(seg):
    ch = seg["channels"]
    design = seg["timing"]["estimated_design"] or {}
    shift = ch.get("translate", {}).get("delta")
    size = ch.get("size", {})
    resize = [
        size.get(axis + "_to", 0) - size.get(axis + "_from", 0) for axis in ("width", "height")
    ]
    return {
        "id": seg["id"],
        "region": seg["region"],
        "start_ms": design.get("t_start_ms"),
        "duration_ms": design.get("duration_ms"),
        "delta": shift,
        "resize": resize if size else None,
        "opacity": [ch["opacity"]["from"], ch["opacity"]["to"]] if "opacity" in ch else None,
        "easing": family(seg),
    }


def fields(want, pred):
    dx, dy = want["delta"]
    distance = math.hypot(dx, dy)
    target = {
        "start_ms": want["start_ms"],
        "duration_ms": want["duration_ms"],
        "direction": direction(dx, dy),
        "distance_px": distance,
        "easing": FAMILY.get(want["easing"], want["easing"]),
    }
    tol = {
        "start_ms": 34.0,
        "duration_ms": max(17.0, 0.15 * want["duration_ms"]),
        "distance_px": max(3.0, 0.10 * distance),
    }
    got = {}
    if pred:
        got = {k: pred[k] for k in ("start_ms", "duration_ms", "easing")}
        shift = pred["delta"]
        if shift is not None or distance == 0:
            got["direction"] = direction(*(shift or (0, 0)))
            got["distance_px"] = math.hypot(*(shift or (0, 0)))
    for i, axis in enumerate(("width_px", "height_px")):
        if want["resize"][i] != 0:
            target[axis] = want["resize"][i]
            tol[axis] = max(3.0, 0.1 * abs(target[axis]))
            if pred and pred["resize"] is not None:
                got[axis] = pred["resize"][i]
    if "opacity" in want:
        for i, key in enumerate(("opacity_from", "opacity_to")):
            target[key] = want["opacity"][i]
            tol[key] = 0.1
            if pred and pred["opacity"] is not None:
                got[key] = pred["opacity"][i]
    result = {}
    for key, expected in target.items():
        actual = got.get(key)
        error = None
        if actual is None:
            state = "missing"
        elif key in tol:
            error = round(actual - expected, 6)
            state = "correct" if abs(error) <= tol[key] else "wrong"
        else:
            state = "correct" if actual == expected else "wrong"
        result[key] = {"expected": expected, "actual": actual, "error": error, "state": state}
    return result


def score(case, truth, preds, failure=None):
    pairs = match(truth, preds)
    rows = []
    for want, j in zip(truth, pairs, strict=True):
        row = fields(want, preds[j] if j is not None else None)
        if failure:
            for item in row.values():
                item["state"] = failure
        rows.append({"element": want["element"], "region": want["rect"], "match": j, "fields": row})
    extras = [j for j in range(len(preds)) if j not in pairs]
    count_ok = not failure and len(truth) == len(preds)
    outcomes = [m["state"] == "correct" for row in rows for m in row["fields"].values()]
    correct = int(count_ok) + sum(outcomes)
    return {
        "id": case["id"],
        "family": case.get("family", case.get("legacy")),
        "failure": failure,
        "predictions": preds,
        "objects": rows,
        "extra": extras,
        "expected_count": len(truth),
        "count_correct": count_ok,
        "missed": pairs.count(None),
        "score": correct / (1 + len(outcomes)),
        "all_correct": count_ok and all(outcomes) and not extras,
    }


def summarize(rows):
    by_field = {}
    for row in rows:
        for obj in row["objects"]:
            for name, item in obj["fields"].items():
                field = by_field.setdefault(
                    name,
                    {
                        "correct": 0,
                        "wrong": 0,
                        "missing": 0,
                        "format_fail": 0,
                        "analyzer_error": 0,
                        "errors": [],
                    },
                )
                field[item["state"]] += 1
                if item["error"] is not None:
                    field["errors"].append(item["error"])
    for field in by_field.values():
        errors = field.pop("errors")
        if errors:
            field["signed_median"] = round(float(np.median(errors)), 3)
            field["absolute_p50_p90"] = np.round(
                np.percentile(np.abs(errors), [50, 90]), 3
            ).tolist()
    rng = np.random.default_rng(61127)
    indices = rng.integers(0, len(rows), (10000, len(rows)))

    def interval(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return {
            "mean": round(float(vals.mean()), 6),
            "ci95": np.round(np.percentile(vals[indices].mean(axis=1), [2.5, 97.5]), 6).tolist(),
        }

    return {
        "scenes": len(rows),
        "objects": sum(r["expected_count"] for r in rows),
        "missed_objects": sum(r["missed"] for r in rows if not r["failure"]),
        "extra_segments": sum(len(r["extra"]) for r in rows),
        "scene_macro": interval("score"),
        "all_fields_correct": interval("all_correct"),
        "format_fail": sum(r["failure"] == "format_fail" for r in rows),
        "analyzer_error": sum(r["failure"] == "analyzer_error" for r in rows),
        "fields": by_field,
    }


def evaluate(source, split, corpus):
    rows = []
    cv2.setNumThreads(1)
    cv2.setRNGSeed(61127)
    with ZipFile(corpus) as archive, tempfile.TemporaryDirectory(prefix="visfunc-eval-") as tmp:
        video = Path(tmp) / "clip.mp4"
        for case in cases(source, split):
            prefix = case["id"] + "/"
            pixels = archive.read(prefix + "video.mp4")
            labels = archive.read(prefix + "truth.json")
            video.write_bytes(pixels)
            failure, detail, preds = None, None, []
            try:
                spec = analyze(video, raw_frames=False, sheets=False)
            except Exception as exc:
                failure = (
                    "format_fail"
                    if isinstance(exc, ValueError)
                    and str(exc).startswith("generated invalid motion spec:")
                    else "analyzer_error"
                )
                detail = type(exc).__name__ + ": " + str(exc)
            else:
                try:
                    problems = validate(spec)
                    if problems:
                        raise ValueError("; ".join(problems))
                    preds = [measured(s) for s in spec["segments"]]
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    failure, detail = "format_fail", type(exc).__name__ + ": " + str(exc)
            row = score(case, json.loads(labels)["segments"], preds, failure)
            if detail:
                row["detail"] = detail
            row.update(
                video_sha256=hashlib.sha256(pixels).hexdigest(),
                truth_sha256=hashlib.sha256(labels).hexdigest(),
            )
            rows.append(row)
    return {
        "protocol": "motion-analyzer/1",
        "source": source,
        "split": split,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "summary": summarize(rows),
        "scenes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("qt", "browser"), required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.source, args.split, args.corpus or BENCH / "corpus" / (args.source + ".zip")
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf8"
        )
    print(json.dumps(report if not args.output else report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
