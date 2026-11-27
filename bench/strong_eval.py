"""V0–V5 motion comparison. One isolated message history and receipt per trial."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from analyzer_eval import FAMILY
from motionspec.functions import catalogue, run_queries
from strong_client import MODELS, Ledger, PaidModels, client
from strong_inputs import select

QUESTION = (
    "Measure all distinct UI animations in the supplied clip (not static decor). "
    """Return JSON
{"segments":[{"region":[left,top,right,bottom],"start_ms":0,"duration_ms":250,"delta":[dx,dy],
"resize":[width_change,height_change],"opacity":null,"easing":"ease-out"}]}.
Use source screen pixels for the swept bounding rectangle. Give designed start and duration on the
clip timeline. Displacement is change in top-left, size is signed end minus start. A stationary
edge expanding is zero translation. Use [0,0] for known absent movement or resizing, null for
unknown. Opacity is [from,to] only for fades. Easing family is linear, ease-in, ease-out,
ease-in-out or overshoot. List simultaneously moving parts separately. Use no external sources."""
)
AUX = {
    "V2": """You have a local offline Python/OpenCV sandbox. The full video file is clip.mp4 in
your working directory. First respond with {"python":"..."} containing a complete Python script
that measures pixels and frame timestamps and prints useful numbers. No functions.json or truth
is available. You may issue one additional script after seeing output, then answer the task.""",
    "V4": """A local functions.json runtime is available. First return {"tool":"catalogue"} to
inspect it. Then either answer or return {"tool":"query","calls":[...]} with up to 12 of the
listed operations and use their results before answering. All results are on the source clock.""",
}


def decode(answer):
    raw = answer.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def predictions(answer):
    value = decode(answer)
    bare = isinstance(value, list)
    rows = value if bare else value["segments"]
    if not isinstance(rows, list) or len(rows) > 40:
        raise ValueError("invalid segment list")

    def vector(x, n):
        return (
            isinstance(x, list)
            and len(x) == n
            and all(type(v) in (int, float) and math.isfinite(v) for v in x)
        )

    parsed = []
    for index, row in enumerate(rows):
        box = row["region"]
        if not vector(box, 4) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("invalid region")
        record = {"id": str(index), "region": box}
        for key in ("start_ms", "duration_ms"):
            v = row[key]
            if v is not None and not (type(v) in (int, float) and math.isfinite(v) and v >= 0):
                raise ValueError(key)
            record[key] = v
        for key in ("delta", "resize", "opacity"):
            v = row[key]
            if v is not None and not vector(v, 2):
                raise ValueError(key)
            record[key] = v
        es = row["easing"]
        if es is not None and not isinstance(es, str):
            raise ValueError("easing")
        record["easing"] = FAMILY.get(es, es)
        parsed.append(record)
    return parsed, bare


def sandbox(image, video, code):
    if not isinstance(code, str) or len(code.encode()) > 20000:
        raise ValueError("Script exceeds the sandbox input limit")
    name = "visfunc-measure-" + uuid4().hex
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/work:rw,noexec,nosuid,nodev,mode=1777,size=32m",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "48",
        "--memory",
        "512m",
        "--cpus",
        "1",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "OMP_NUM_THREADS=1",
        "-i",
        image,
    ]
    job = json.dumps({"video": base64.b64encode(video.read_bytes()).decode(), "code": code})
    try:
        raw = subprocess.run(
            command, input=job, text=True, encoding="utf8", capture_output=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        return {"status": 124, "output": "Sandbox time limit reached."}
    if raw.returncode:
        raise RuntimeError("Sandbox container failed to start/complete")
    return json.loads(raw.stdout)


def media(folder, model, condition, dense_index):
    meta = json.loads((folder / "media.json").read_text(encoding="utf8"))
    parts = [client.text(QUESTION + "\nSource dimensions: " + str(meta["size"]) + " px.")]
    if condition == "V1":
        dense = meta["dense"]
        sample = dense["variants"][dense_index]
        parts.append(
            client.text(
                f"Timestamped tiles share source crop xyxy {dense['crop_xyxy']}; "
                f"one tile spans {dense['tile_px']} px excluding timestamp header."
            )
        )
        parts.extend(client.image(folder / n) for n in sample["images"])
        return parts, sample["frames"]
    if model == "gpt":
        for name, stamp in meta["sparse"]:
            parts.extend((client.text(f"{stamp} ms"), client.image(folder / name)))
        n = len(meta["sparse"])
    else:
        vendor = "mimo" if model == "mimo" else "ark"
        parts.append(client.video(folder / "clip.mp4", vendor, 10 if vendor == "mimo" else 5))
        n = None
    if condition in ("V3", "V5"):
        file = "summary.txt" if condition == "V3" else "probe.txt"
        parts.append(client.text((folder / file).read_text(encoding="utf8")))
    if condition in AUX:
        parts.append(client.text(AUX[condition]))
    return parts, n


def trial(api, folder, model, cond, image, variant):
    parts, n = media(folder, model, cond, variant)
    task = f"{model}-{folder.name}-{cond}"
    turns = []
    msgs = [{"role": "user", "content": parts}]
    result = {
        "model": model,
        "requested_model": MODELS[model],
        "condition": cond,
        "scene": folder.name,
        "media_sha256": hashlib.sha256((folder / "clip.mp4").read_bytes()).hexdigest(),
        "payload_sha256": hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest(),
        "frame_count_supplied": n,
        "dense_variant": variant if cond == "V1" else None,
        "attempts": turns,
        "tool_results": [],
        "predictions": [],
    }
    began = time.monotonic()
    try:
        for step in range(3 if cond in AUX else 1):
            reply = api.call(model, msgs, task, max_output=3000)
            turns.append(reply)
            if reply["status"] != 200:
                result["failure"] = "transport_fail"
                break
            answer = reply["answer"]
            if cond in AUX and step < 2:
                obj = decode(answer)
                if isinstance(obj, dict) and cond == "V2" and "python" in obj:
                    ret = sandbox(image, folder / "clip.mp4", obj["python"])
                elif isinstance(obj, dict) and cond == "V4" and obj.get("tool") == "catalogue":
                    ret = catalogue(folder / "evidence/functions.json")
                    # Frames list does not convey measurements to this task; keep sample queries.
                    ret = {k: v for k, v in ret.items() if k != "frames"}
                elif (
                    isinstance(obj, dict)
                    and cond == "V4"
                    and obj.get("tool") == "query"
                    and isinstance(obj.get("calls"), list)
                    and 1 <= len(obj["calls"]) <= 12
                ):
                    ret = run_queries(folder / "evidence/functions.json", obj["calls"])
                elif step == 0:
                    result["failure"] = "tool_protocol_fail"
                    break
                else:
                    ret = None
                if ret is not None:
                    tool_text = json.dumps(ret, ensure_ascii=False)
                    result["tool_results"].append(ret)
                    msgs.extend(
                        [
                            {"role": "assistant", "content": answer},
                            {
                                "role": "user",
                                "content": "Local tool result: "
                                + tool_text
                                + (
                                    "\nNow return the final segments JSON."
                                    if step == 1
                                    else "\nReturn the final segments JSON or one more tool call."
                                ),
                            },
                        ]
                    )
                    continue
            result["predictions"], result["bare_array"] = predictions(answer)
            break
    except (ValueError, TypeError, KeyError, IndexError):
        result["failure"] = "format_fail"
    except RuntimeError as exc:
        if str(exc) == "budget_reservation_unavailable":
            result["failure"] = "budget_limit"
        else:
            raise
    result["seconds"] = round(time.monotonic() - began, 3)
    return result


def write_result(path, obj):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--limit-usd", type=float, required=True)
    parser.add_argument("--providers", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--conditions", nargs="+", choices=[f"V{i}" for i in range(6)], required=True
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sandbox-image", default="visfunc-measure:20261127")
    parser.add_argument("--dense-calibration", type=Path)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument(
        "--retry-infra",
        action="store_true",
        help="Retry one infrastructure failure, retaining its attempts and usage",
    )
    args = parser.parse_args()
    if args.limit_usd > 10 or args.limit_usd <= 0 or not 1 <= args.workers <= 8:
        parser.error("Limit must be in (0,10] and workers 1..8")
    args.output.mkdir(parents=True, exist_ok=True)
    if "V1" in args.conditions and args.dense_calibration is None:
        parser.error("V1 requires --dense-calibration")
    api = PaidModels(args.providers, Ledger(args.ledger, args.limit_usd), args.models)
    tasks = [
        (m, s, c)
        for source in ("qt", "browser")
        for s in select(source)
        for m in args.models
        for c in args.conditions
        if not (source == "browser" and c == "V5") and (not args.scenes or s in args.scenes)
    ]
    random.Random(61127).shuffle(tasks)

    def run(task):
        model, scene, cond = task
        path = args.output / f"{model}-{scene}-{cond}.json"
        prior = None
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf8"))
            earlier = prior.get("previous", {})
            if (
                not args.retry_infra
                or prior.get("failure") not in {"budget_limit", "transport_fail"}
                or earlier.get("failure") == "transport_fail"
                or "previous" in earlier
            ):
                return
        variant = 0
        if cond == "V1":
            baseline = args.output / f"{model}-{scene}-V3.json"
            sample = json.loads(baseline.read_text(encoding="utf8")) if baseline.is_file() else {}
            budget = next(
                (a["input_tokens"] for a in sample.get("attempts", []) if a["input_tokens"]), 0
            )
            if budget <= 0:
                print(path.stem, "pending paired V3 input usage", flush=True)
                return
            cal = json.loads(
                (args.dense_calibration / (model + ".json")).read_text(encoding="utf8")
            )
            dense = json.loads((args.inputs / scene / "media.json").read_text(encoding="utf8"))[
                "dense"
            ]
            anchors = []
            for case in ("q000", "q002"):
                points = sorted(
                    (r for r in cal if r["case"] == case and r["input_tokens"] > 0),
                    key=lambda r: r["sheets"],
                )
                if len(points) == 2:
                    a, b = points
                    unit = (b["input_tokens"] - a["input_tokens"]) / (b["sheets"] - a["sheets"])
                    anchors.append(
                        (a["tiles_height"], a["input_tokens"] - unit * a["sheets"], unit)
                    )
            if not anchors:
                raise ValueError("no successful dev calibration")
            _, base, unit = min(anchors, key=lambda a: abs(a[0] - dense["tile_px"][1]))
            opts = dense["variants"]
            eligible = [
                i for i, v in enumerate(opts) if base + unit * len(v["images"]) <= 1.2 * budget
            ]
            variant = (
                max(eligible, key=lambda i: (opts[i]["frames"], -len(opts[i]["images"])))
                if eligible
                else 0
            )
        result = trial(api, args.inputs / scene, model, cond, args.sandbox_image, variant)
        if prior is not None:
            result["previous"] = prior
        write_result(path, result)
        print(path.stem, result.get("failure", "ok"), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for dense in (False, True):
            list(pool.map(run, [t for t in tasks if (t[2] == "V1") == dense]))


if __name__ == "__main__":
    main()
