"""Re-score frozen independent sessions. No provider configuration or network is needed."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from analyzer_eval import score
from motion_corpus import BENCH, cases, save
from strong_client import metered_usd
from strong_inputs import select

PAIRS = (("V1", "V0"), ("V2", "V3"), ("V4", "V3"), ("V3", "V0"), ("V5", "V3"))


def interval(values):
    values = np.array(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(61127)
    draws = rng.integers(len(values), size=(10000, len(values)))
    return {
        "mean": round(values.mean(), 6),
        "ci95": np.percentile(values[draws].mean(axis=1), [2.5, 97.5]).round(6).tolist(),
    }


def histories(receipt):
    while receipt is not None:
        yield receipt
        receipt = receipt.get("previous")


def metrics(receipt, case, truth):
    current = receipt.get("failure", "ok")
    result = score(
        case,
        truth,
        receipt["predictions"],
        None if current == "ok" else current,
        implicit_stationary=False,
    )
    runs = list(histories(receipt))
    calls = [a for run in runs for a in run["attempts"]]
    tools = receipt["tool_results"]
    tools_ok = [r["status"] == 0 for r in tools if isinstance(r, dict) and "status" in r]
    fields = Counter(f["state"] for o in result["objects"] for f in o["fields"].values())
    return {
        "scene": case["id"],
        "family": case["family"],
        "model": receipt["model"],
        "condition": receipt["condition"],
        "status": current,
        "initial_status": runs[-1].get("failure", "ok"),
        "score": round(result["score"], 8),
        "strict": bool(result["all_correct"]),
        "fields": dict(fields),
        "extra_segments": len(result["extra"]),
        "missed_objects": result["missed"] if current == "ok" else None,
        "input_tokens": sum(a["input_tokens"] for a in calls),
        "output_tokens": sum(a["output_tokens"] for a in calls),
        "final_input": sum(a["input_tokens"] for a in receipt["attempts"]),
        "metered_usd": round(
            sum(metered_usd(receipt["model"], a.get("usage", {})) for a in calls), 9
        ),
        "unconfirmed_bound_usd": round(
            sum(a["budget_usd"] for a in calls if not a["usage_known"]), 7
        ),
        "seconds": round(sum(r["seconds"] for r in runs), 3),
        "requests": len(calls),
        "served_models": sorted({a["served_model"] for a in calls if a.get("served_model")}),
        "errors": [
            {"status": a["status"], "code": a.get("error_code", "")}
            for a in calls
            if a["status"] != 200
        ],
        "bare_array": receipt.get("bare_array", False),
        "frames": receipt["frame_count_supplied"],
        "tool_turns": len(tools),
        "scripts_ok": sum(tools_ok),
        "scripts_run": len(tools_ok),
        "function_queries": sum(
            1
            for a in receipt["attempts"][: len(tools)]
            if '"query"' in a.get("answer", "") and '"calls"' in a["answer"]
        ),
        "retry_count": len(runs) - 1,
    }


def summarize(rows):
    complete = [r for r in rows if r["status"] == "ok"]
    return {
        "n": len(rows),
        "macro": interval([r["score"] for r in rows]),
        "valid_only": interval([r["score"] for r in complete]),
        "strict": interval([r["strict"] for r in rows]),
        "strict_count": sum(r["strict"] for r in rows),
        "status": dict(Counter(r["status"] for r in rows)),
        "bare_arrays": sum(r["bare_array"] for r in rows),
        **{
            key: round(sum(r[key] for r in rows), 7)
            for key in (
                "input_tokens",
                "output_tokens",
                "metered_usd",
                "unconfirmed_bound_usd",
                "scripts_ok",
                "scripts_run",
                "function_queries",
                "retry_count",
            )
        },
        "median_seconds": round(float(np.median([r["seconds"] for r in rows])), 3),
    }


def report(trials, corpus):
    all_rows = []
    by_scene = defaultdict(list)
    missing = []
    for source in ("qt", "browser"):
        chosen = select(source)
        with ZipFile(corpus / f"{source}.zip") as clips:
            for case in cases(source, "test"):
                scene = case["id"]
                if scene not in chosen:
                    continue
                truth = json.loads(clips.read(scene + "/truth.json"))["segments"]
                digest = hashlib.sha256(clips.read(scene + "/video.mp4")).hexdigest()
                for model in ("mimo", "seed", "gpt"):
                    for cond in (f"V{i}" for i in range(6 if source == "qt" else 5)):
                        name = f"{model}-{scene}-{cond}.json"
                        try:
                            receipt = json.loads(trials.read(name))
                        except KeyError:
                            missing.append(name)
                            continue
                        if (
                            receipt["scene"],
                            receipt["model"],
                            receipt["condition"],
                            receipt["media_sha256"],
                        ) != (scene, model, cond, digest):
                            raise ValueError(f"Receipt identity mismatch: {name}")
                        row = metrics(receipt, case, truth)
                        row["source"] = source
                        all_rows.append(row)
                        by_scene[source, model, scene].append(row)
    cells = defaultdict(list)
    paired = defaultdict(list)
    for (source, model, _), rows in by_scene.items():
        scores = {r["condition"]: r for r in rows}
        for r in rows:
            cond = r["condition"]
            cells[source, model, cond].append(r)
            if cond == "V1" and r["final_input"] and "V3" in scores and scores["V3"]["final_input"]:
                r["v1_v3_input_ratio"] = round(r["final_input"] / scores["V3"]["final_input"], 4)
        for a, b in PAIRS:
            if a in scores and b in scores:
                paired[source, model, a + "-" + b].append(scores[a]["score"] - scores[b]["score"])
    combined = defaultdict(list)
    for (source, _, scene), rows in by_scene.items():
        for row in rows:
            combined[source, row["condition"], scene].append(row["score"])
    cluster = defaultdict(list)
    scene_scores = defaultdict(dict)
    for (source, condition, scene), values in sorted(combined.items()):
        if len(values) == 3:
            mean = float(np.mean(values))
            cluster[source, condition].append(mean)
            scene_scores[source, scene][condition] = mean
    cluster_diffs = defaultdict(list)
    for (source, _), values in scene_scores.items():
        for a, b in PAIRS:
            if a in values and b in values:
                cluster_diffs[source, a + "-" + b].append(values[a] - values[b])
    return {
        "protocol": "motion-strong/1",
        "missing": missing,
        "trials": all_rows,
        "three_model_scene_macro": {"/".join(k): interval(v) for k, v in sorted(cluster.items())},
        "three_model_paired_differences": {
            "/".join(k): interval(v) for k, v in sorted(cluster_diffs.items())
        },
        "groups": {"/".join(k): summarize(rows) for k, rows in sorted(cells.items())},
        "paired_scene_differences": {"/".join(k): interval(v) for k, v in sorted(paired.items())},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts", type=Path, help="ZIP with the raw JSON receipts at its root")
    p.add_argument("output", type=Path)
    p.add_argument("--corpus", type=Path, default=BENCH / "corpus")
    args = p.parse_args()
    with ZipFile(args.receipts) as z:
        result = report(z, args.corpus)
    save(args.output, result)
    print(len(result["trials"]), "trials,", len(result["missing"]), "absent")


if __name__ == "__main__":
    main()
