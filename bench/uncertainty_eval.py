"""Paired V3 summaries on frozen development recordings. Each trial starts a fresh history."""

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from motion_corpus import BENCH, cases, save
from strong_eval import MODELS, Ledger, PaidModels, trial, write_result
from strong_inputs import analyze, frame_data, select


def cohort():
    hard = [c["id"] for c in cases("qt", "dev") if c.get("legacy") == "hard"]
    return {"qt": hard + select("qt", "dev", 2), "browser": select("browser", "dev", 1)}


def prepare(work):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(61127)
    for source, scenes in cohort().items():
        with ZipFile(BENCH / "corpus" / (source + ".zip")) as archive:
            for scene in scenes:
                folder = work / "inputs" / scene
                folder.mkdir(parents=True, exist_ok=True)
                video = folder / "clip.mp4"
                video.write_bytes(archive.read(scene + "/video.mp4"))
                spec = analyze(video, sheets=False, raw_frames=False, layout_edges=True)
                save(folder / "spec.json", spec)
                frames, times = frame_data(video)
                sparse = []
                indices = np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int)
                for i in indices:
                    name = f"frame-{i}.png"
                    ok, png = cv2.imencode(".png", frames[i])
                    if not ok:
                        raise ValueError("Could not encode frame")
                    (folder / name).write_bytes(png.tobytes())
                    sparse.append([name, round(float(times[i]), 3)])
                save(
                    folder / "media.json", {"size": list(frames[0].shape[1::-1]), "sparse": sparse}
                )


def render(work):
    from motionspec.spec import summarize
    from motionspec.uncertainty import summary_uncertainties

    for scenes in cohort().values():
        for scene in scenes:
            folder = work / "inputs" / scene
            spec = json.loads((folder / "spec.json").read_text(encoding="utf8"))
            for cond, marked in (("M2-base", False), ("M2-marked", True)):
                (folder / (cond + ".txt")).write_text(
                    summarize(spec, mark_uncertain=marked), encoding="utf8"
                )
            save(folder / "flags.json", summary_uncertainties(spec))


def run(work, model, providers, workers, limit, *, recover_transport=False):
    out = work / "trials"
    out.mkdir(parents=True, exist_ok=True)
    api = PaidModels(providers, Ledger(work / "cost.sqlite", limit), [model])
    tasks = [
        (scene, cond)
        for scenes in cohort().values()
        for scene in scenes
        for cond in ("M2-base", "M2-marked")
    ]
    random.Random(61127).shuffle(tasks)

    def one(task):
        scene, cond = task
        path = out / f"{model}-{scene}-{cond}.json"
        if recover_transport:
            if (
                not path.exists()
                or json.loads(path.read_bytes()).get("failure") != "transport_fail"
            ):
                return
            path = work / "recovery" / path.name
            path.parent.mkdir(exist_ok=True)
        if path.exists():
            return
        result = trial(api, work / "inputs" / scene, model, cond, None, 0)
        if not recover_transport and result.get("failure") == "transport_fail":
            retry = trial(api, work / "inputs" / scene, model, cond, None, 0)
            retry["previous"] = result
            result = retry
        write_result(path, result)
        print(path.stem, result.get("failure", "ok"), flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, tasks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "render", "run", "recover"))
    parser.add_argument("work", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--providers", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--limit-usd", type=float, default=3)
    args = parser.parse_args()
    if args.step in ("run", "recover"):
        if not args.model or not args.providers:
            parser.error("A model and provider slots are required")
        if not math.isfinite(args.limit_usd) or args.limit_usd <= 0:
            parser.error("USD limit must be finite and positive")
        run(
            args.work,
            args.model,
            args.providers,
            args.workers,
            args.limit_usd,
            recover_transport=args.step == "recover",
        )
    elif args.step == "prepare":
        prepare(args.work)
    else:
        render(args.work)
