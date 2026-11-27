"""Archive the exact M2 evidence and reproduce paired field scores without an API call."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from analyzer_eval import measured, score
from motion_corpus import BENCH, cases, save
from motionspec.spec import summarize
from motionspec.uncertainty import summary_uncertainties
from strong_client import MODELS, metered_usd
from strong_eval import media, predictions
from strong_report import histories, interval
from uncertainty_eval import cohort

ARMS = ("M2-base", "M2-marked")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_receipt(archive, work, path, model, expected, seen):
    raw = (work / path).read_bytes()
    for attempt in histories(json.loads(raw)):
        if attempt["payload_sha256"] != expected:
            raise ValueError(f"Input has changed since dispatch: {path}")
        for call in attempt["attempts"]:
            rid = call.get("response_id")
            if rid and (model, rid) in seen:
                raise ValueError(f"Repeated provider response ID: {path}")
            if rid:
                seen.add((model, rid))
    archive.writestr(path, raw)


def archive_run(work, output):
    response_ids = set()
    with ZipFile(output.with_suffix(".partial"), "w", ZIP_DEFLATED) as archive:
        for scenes in cohort().values():
            for scene in scenes:
                folder = work / "inputs" / scene
                spec = json.loads((folder / "spec.json").read_bytes())
                if summary_uncertainties(spec) != json.loads((folder / "flags.json").read_bytes()):
                    raise ValueError(f"Field masks changed: {scene}")
                for arm in ARMS:
                    expected_text = summarize(spec, mark_uncertain=arm == "M2-marked")
                    if expected_text != (folder / (arm + ".txt")).read_text(encoding="utf8"):
                        raise ValueError(f"Summary changed: {scene}/{arm}")
                # The local source filename is not sent to providers.
                spec["source"]["path"] = "clip.mp4"
                archive.writestr(f"{scene}/spec.json", json.dumps(spec, ensure_ascii=False))
                for name in ("media.json", "flags.json", *(a + ".txt" for a in ARMS)):
                    archive.write(folder / name, scene + "/" + name)
                for model in MODELS:
                    for arm in ARMS:
                        name = f"{model}-{scene}-{arm}.json"
                        parts, _ = media(folder, model, arm, 0)
                        expected = digest(json.dumps(parts, sort_keys=True).encode())
                        archive_receipt(
                            archive, work, "trials/" + name, model, expected, response_ids
                        )
                        if (work / "recovery" / name).exists():
                            archive_receipt(
                                archive, work, "recovery/" + name, model, expected, response_ids
                            )
    output.with_suffix(".partial").replace(output)


def cells(result):
    yield ("count",), "correct" if result["count_correct"] else result["failure"] or "wrong"
    for i, obj in enumerate(result["objects"]):
        for key, field in obj["fields"].items():
            yield (i, key), field["state"]


def mask(result, flags):
    marked = {("count",)} if "count" in flags else set()
    for i, obj in enumerate(result["objects"]):
        if obj["match"] is not None:
            segment = result["predictions"][obj["match"]]["id"]
            fields = flags.get("segments", {}).get(segment, {})
            marked.update((i, key) for key in obj["fields"] if key in fields)
    return marked


def evaluate(archive, *, recover=False):
    pairs = []
    for source, scenes in cohort().items():
        with ZipFile(BENCH / "corpus" / (source + ".zip")) as corpus:
            for case in cases(source, "dev"):
                scene = case["id"]
                if scene not in scenes:
                    continue
                labels = corpus.read(scene + "/truth.json")
                truth = json.loads(labels)["segments"]
                video_hash = digest(corpus.read(scene + "/video.mp4"))
                spec = json.loads(archive.read(scene + "/spec.json"))
                flags = json.loads(archive.read(scene + "/flags.json"))
                baseline = score(case, truth, [measured(s) for s in spec["segments"]])
                marked = mask(baseline, flags)
                for model in MODELS:
                    states = {}
                    arms = {}
                    for arm in ARMS:
                        name = f"{model}-{scene}-{arm}.json"
                        rec = json.loads(archive.read("trials/" + name))
                        if recover and "recovery/" + name in archive.namelist():
                            if rec.get("failure") != "transport_fail":
                                raise ValueError(f"Recovery on a non-transport outcome: {name}")
                            rec = {**json.loads(archive.read("recovery/" + name)), "previous": rec}
                        identity = (
                            rec["model"],
                            rec["condition"],
                            rec["scene"],
                            rec["media_sha256"],
                        )
                        if identity != (model, arm, scene, video_hash):
                            raise ValueError(f"Receipt identity mismatch: {identity}")
                        if not rec.get("failure"):
                            parsed, bare = predictions(rec["attempts"][-1]["answer"])
                            if parsed != rec["predictions"] or bare != rec["bare_array"]:
                                raise ValueError(f"Predictions disagree with answer: {identity}")
                        result = score(
                            case,
                            truth,
                            rec["predictions"],
                            rec.get("failure"),
                            implicit_stationary=False,
                        )
                        states[arm] = dict(cells(result))
                        calls = [a for run in histories(rec) for a in run["attempts"]]
                        arms[arm] = {
                            "score": result["score"],
                            "failure": rec.get("failure"),
                            "strict": bool(result["all_correct"]),
                            "extra": len(result["extra"]),
                            "missed": result["missed"],
                            "attempts": len(calls),
                            "bare_array": rec.get("bare_array", False),
                            "input_tokens": sum(a["input_tokens"] for a in calls),
                            "output_tokens": sum(a["output_tokens"] for a in calls),
                            "metered_usd": sum(
                                metered_usd(model, a.get("usage", {})) for a in calls
                            ),
                            "unknown_usd": sum(
                                a["budget_usd"] for a in calls if not a["usage_known"]
                            ),
                        }
                    items = [
                        {
                            "object": key[0] if len(key) == 2 else None,
                            "field": key[-1],
                            "analyzer": state,
                            "marked": key in marked,
                            "base": states[ARMS[0]][key],
                            "after": states[ARMS[1]][key],
                        }
                        for key, state in cells(baseline)
                    ]
                    pairs.append(
                        {
                            "model": model,
                            "scene": scene,
                            "source": source,
                            "truth_sha256": digest(labels),
                            "video_sha256": video_hash,
                            "changed": bool(flags),
                            "arms": arms,
                            "items": items,
                        }
                    )
    return pairs


def field_counts(pairs, predicate):
    counts = []
    for pair in pairs:
        items = [i for i in pair["items"] if predicate(i)]
        if items:
            a = sum(i["base"] == "correct" for i in items)
            b = sum(i["after"] == "correct" for i in items)
            gains = sum(i["base"] != "correct" and i["after"] == "correct" for i in items)
            counts.append([len(items), a, b, gains, gains + a - b])
    if not counts:
        return {"n": 0}
    data = np.array(counts)
    n, a, b, gains, losses = data.sum(axis=0).tolist()
    # Resample contributing scenes as clusters, retaining all fields of each scene.
    rng = np.random.default_rng(61127)
    draws = data[rng.integers(len(data), size=(10000, len(data)))].sum(axis=1)
    delta = (draws[:, 2] - draws[:, 1]) / draws[:, 0]
    return {
        "n": n,
        "base": a,
        "marked": b,
        "gains": gains,
        "losses": losses,
        "scenes": len(counts),
        "delta_ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
    }


def aggregate(pairs):
    summary = {}
    for model in MODELS:
        rows = [p for p in pairs if p["model"] == model]

        def difference(subset):
            return interval(
                [p["arms"][ARMS[1]]["score"] - p["arms"][ARMS[0]]["score"] for p in subset]
            )

        groups = {
            "all": lambda i: True,
            "marked": lambda i: i["marked"],
            "marked_fields": lambda i: i["marked"] and i["field"] != "count",
            "marked_gap": lambda i: i["marked"] and i["analyzer"] != "correct",
            "marked_field_gap": lambda i: (
                i["marked"] and i["analyzer"] != "correct" and i["field"] != "count"
            ),
            "unmarked_gap": lambda i: not i["marked"] and i["analyzer"] != "correct",
            "marked_already_correct": lambda i: i["marked"] and i["analyzer"] == "correct",
        }
        valid = [p for p in rows if all(not a["failure"] for a in p["arms"].values())]
        item = {
            "paired_scene_delta": difference(rows),
            "valid_pair_delta": difference(valid),
            "valid_pairs": len(valid),
            "changed_delta": difference([p for p in rows if p["changed"]]),
            "identical_delta": difference([p for p in rows if not p["changed"]]),
            "fields": {name: field_counts(rows, pred) for name, pred in groups.items()},
            "valid_gap": field_counts(valid, groups["marked_field_gap"]),
        }
        for arm in ARMS:
            recs = [p["arms"][arm] for p in rows]
            item[arm] = {
                "scene_score": interval([r["score"] for r in recs]),
                "status": dict(Counter(r["failure"] or "ok" for r in recs)),
                **{
                    k: sum(r[k] for r in recs)
                    for k in (
                        "strict",
                        "extra",
                        "attempts",
                        "bare_array",
                        "input_tokens",
                        "output_tokens",
                        "metered_usd",
                        "unknown_usd",
                    )
                },
            }
        summary[model] = item
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive-work", type=Path)
    args = parser.parse_args()
    if args.archive_work:
        archive_run(args.archive_work, args.evidence)
    with ZipFile(args.evidence) as archive:
        pairs = evaluate(archive)
        recovery = [n for n in archive.namelist() if n.startswith("recovery/")]
        recovered = aggregate(evaluate(archive, recover=True)) if recovery else {}
    save(
        args.output,
        {
            "protocol": "motion-uncertainty-dev/1",
            "evidence_sha256": digest(args.evidence.read_bytes()),
            "summary": aggregate(pairs),
            "transport_recovery": {"receipts": recovery, "summary": recovered},
            "pairs": pairs,
        },
    )
