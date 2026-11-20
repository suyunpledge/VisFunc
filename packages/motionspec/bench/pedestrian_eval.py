"""Stateless video/parameter ablation, scored on the last decoded frame of each clip."""

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import client
import providers
from pedestrian_clips import SPLITS, read, save

from svsg.recognition.evaluation import score_objects
from svsg.recognition.geometry import iou

MODEL = "mimo-v2.6-flash"
PROMPT = (
    "Locate every visible upright pedestrian in the LAST frame, including distant and partially "
    "occluded people. Use the final frame only for coordinates. Estimate each pedestrian's full "
    "body box, clipped to the frame. Exclude riders, seated people, reflection, posters and people "
    "with no visible evidence. Coordinates are integers normalized independently to 0..1000 in "
    '[left,top,right,bottom] order. Return JSON only: {"boxes":[[x1,y1,x2,y2],...]}. '
    "Each person once. Return an empty list if evidence is insufficient."
)
BASELINE = ("video", "geometry", "video_geometry")
ENHANCED = ("proposals", "video_proposals")


def call(root, seq, condition, provider, *, output=None):
    folder = root / seq
    output = folder if output is None else output
    log = output / f"response-{condition}.json"
    parts = [client.text(PROMPT)]
    video = condition.startswith("video")
    if video:
        parts.append(client.video(folder / "clip.mp4", "mimo", fps=5))
    if condition != "video":
        payload = read(folder / "geometry.json")
        if condition in ENHANCED:
            payload["detector_candidates"] = read(output / "proposals.json")
        parts.append(client.text(json.dumps(payload, separators=(",", ":"))))
    digest = sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    if log.exists():
        past = read(log)
        if past["parts_sha256"] != digest or past["model"] != MODEL:
            raise ValueError(f"existing result has different inputs: {log}")
        return
    result = client.chat(
        provider.base_url,
        provider.key,
        MODEL,
        parts,
        max_tokens=3500,
        extra={"temperature": 0, "thinking": {"type": "disabled"}},
    )
    save(
        log,
        {
            **result,
            "model": MODEL,
            "parts_sha256": digest,
            "condition": condition,
            "clip_sha256": read(folder / "source.json")["clip_sha256"],
            "media_parts": int(video),
        },
    )
    print(seq, condition, result["status"], result["seconds"], flush=True)


def truth(folder):
    source = read(folder / "source.json")
    w, h = source["size"]
    expected, ignore = [], []
    for line in (folder / "gt.txt").read_text().splitlines():
        idx, ident, x, y, bw, bh, active, label, visible = map(float, line.split(",")[:9])
        if idx != source["last_frame"]:
            continue
        box = [max(0, x - 1), max(0, y - 1), min(w, x - 1 + bw), min(h, y - 1 + bh)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if active == 1 and label == 1 and visible > 0:
            expected.append(
                {
                    "label": "person",
                    "bbox_px": box,
                    "id": int(ident),
                    "visibility": visible,
                    "small": (box[2] - box[0]) * (box[3] - box[1]) < 0.0025 * w * h,
                }
            )
        elif label in (1, 2, 7, 8, 12) or not active:
            ignore.append(box)
    return w, h, expected, ignore


def predictions(answer, w, h):
    raw = answer.strip()
    adaptations = []
    if raw.startswith("```") and raw.endswith("```"):
        adaptations.append("code_fence")
        raw = raw.split("\n", 1)[-1][:-3]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # One redundant outer list, with its closing bracket omitted. Require an
        # independently complete, schema-valid inner object. Never repair coordinates.
        if not raw.startswith("["):
            raise
        obj = json.loads(raw[1:].strip())
        if not isinstance(obj, dict) or set(obj) != {"boxes"}:
            raise ValueError("invalid outer list") from None
        adaptations.append("unclosed_outer_list")
    if isinstance(obj, list):
        adaptations.append("list_wrapper")
        if len(obj) == 1 and isinstance(obj[0], dict) and set(obj[0]) == {"boxes"}:
            obj = obj[0]
        elif obj and all(
            isinstance(b, dict) and b.get("label") in ("pedestrian", "person") and "bbox_2d" in b
            for b in obj
        ):
            obj = [b["bbox_2d"] for b in obj]
    boxes = obj["boxes"] if isinstance(obj, dict) else obj
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    result = []
    for b in boxes:
        if (
            not isinstance(b, list)
            or len(b) != 4
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in b)
            or not (0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000)
        ):
            raise ValueError("invalid box")
        result.append(
            {
                "label": "person",
                "bbox_px": [v * d / 1000 for v, d in zip(b, (w, h, w, h), strict=True)],
            }
        )
    return result, adaptations


def score(folder, predictions_px):
    _, _, gt, ignore = truth(folder)
    matches = score_objects(predictions_px, gt)["matches"]
    hit = {m["expected_index"] for m in matches}
    used = {m["prediction_index"] for m in matches}
    errors = []
    for i, p in enumerate(predictions_px):
        if i in used:
            continue
        best = max((iou(p["bbox_px"], g["bbox_px"]) for g in gt), default=0)
        if best < 0.5 and any(iou(p["bbox_px"], b) >= 0.5 for b in ignore):
            continue
        errors.append(
            {
                "bbox_px": p["bbox_px"],
                "best_iou": best,
                "kind": "duplicate"
                if best >= 0.5
                else "localization"
                if best >= 0.1
                else "background",
            }
        )
    return {
        "tp": len(matches),
        "fp": len(errors),
        "fn": len(gt) - len(matches),
        "ignored_predictions": len(predictions_px) - len(matches) - len(errors),
        "small_gt": sum(g["small"] for g in gt),
        "small_tp": sum(gt[i]["small"] for i in hit),
        "occluded_gt": sum(g["visibility"] <= 0.5 for g in gt),
        "occluded_tp": sum(gt[i]["visibility"] <= 0.5 for i in hit),
        "misses": [g for i, g in enumerate(gt) if i not in hit],
        "false_positives": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("phase", choices=("baseline", "enhanced"))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    conditions = BASELINE if args.phase == "baseline" else ENHANCED
    jobs = [(s, c) for s in SPLITS[args.split] for c in conditions]
    random.Random(3127).shuffle(jobs)
    if args.live:
        slots = read(Path(__file__).with_name("providers-hard.json"))
        p = providers.find(slots[MODEL])
        if p.model != MODEL or "token-plan" not in p.base_url:
            raise ValueError("expected configured MiMo Flash Token Plan")
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda job: call(args.root, *job, p), jobs))
    scored = []
    for s, c in jobs:
        folder = args.root / s
        response = read(folder / f"response-{c}.json")
        row = {"seq": s, "condition": c, "ok": False}
        if response["ok"]:
            try:
                w, h = read(folder / "source.json")["size"]
                boxes, adaptations = predictions(response["text"], w, h)
                row.update(score(folder, boxes), ok=True, adaptations=adaptations)
            except (ValueError, KeyError, TypeError) as exc:
                row["error"] = f"format: {exc}"
        else:
            row["error"] = f"transport: {response['status']}"
        if not row["ok"]:
            row.update(score(folder, []))  # No usable boxes were delivered to the caller.
        scored.append(row)
    save(args.root / f"scores-{args.split}-{args.phase}.json", scored)
    for c in conditions:
        rows = [r for r in scored if r["condition"] == c]
        print(
            c,
            f"delivered {sum(r['ok'] for r in rows)}/{len(rows)}",
            {
                k: sum(r[k] for r in rows)
                for k in ("tp", "fp", "fn", "small_tp", "small_gt", "occluded_tp", "occluded_gt")
            },
        )


if __name__ == "__main__":
    main()
