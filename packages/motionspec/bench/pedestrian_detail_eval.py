"""Paired pedestrian-detail benchmark using the same media and prompt on both profiles."""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from pedestrian_clips import read, save
from pedestrian_eval import BASELINE, MODEL, call, predictions, providers, score
from pedestrian_proposals import make

from svsg.l1_compiler.detector_adapter import YoloDetector
from svsg.recognition.crowd_detector import CrowdHumanDetector


def run(
    root,
    sequences,
    weights,
    *,
    live=False,
    controls=False,
    proposal_control=False,
    adaptive=False,
    crowd_weights=None,
    compare_overlap=False,
    head_focus=True,
):
    model = YoloDetector(str(weights), imgsz=1280)
    digest = sha256(weights.read_bytes()).hexdigest()
    jobs = []
    records = []
    profiles = {"whole": False, "detail": True}
    proposal_profile, prefix = "detail", ""
    if adaptive:
        profiles["auto"] = None
    specialist = CrowdHumanDetector(crowd_weights) if crowd_weights else None
    if specialist:
        profiles = {"auto": None, "crowd": None}
        proposal_profile, prefix = "crowd", "crowd-"
    if compare_overlap:
        if not specialist:
            raise ValueError("the paired overlap comparison requires crowd weights")
        profiles = {"before": None, "after": None}
        proposal_profile, prefix = "after", "overlap-"
    for seq in sequences:
        src = root / seq
        for profile, use_detail in profiles.items():
            dest = src / profile
            dest.mkdir(exist_ok=True)
            start = perf_counter()
            make(
                src,
                model,
                digest,
                detail=use_detail,
                output=dest,
                crowd_detector=specialist if profile in ("crowd", "before", "after") else None,
                overlap_review=profile != "before",
                head_focus=head_focus,
            )
            seconds = perf_counter() - start
            packet = read(dest / "candidates-px.json")
            boxes = [{"label": "person", "bbox_px": b["bbox_px"]} for b in packet["boxes"]]
            records.append(
                {
                    "seq": seq,
                    "input": profile,
                    "seconds": seconds,
                    "detail_views": packet["detail_views"],
                    **score(src, boxes),
                }
            )
            jobs.append((seq, profile, "video_proposals"))
        if controls:
            jobs.extend((seq, "control", c) for c in BASELINE)
        if controls or proposal_control:
            jobs.append((seq, proposal_profile, "proposals"))
    save(root / f"{prefix}detector-scores.json", records)
    random.Random(917).shuffle(jobs)
    if live:
        provider = providers.find(read(Path(__file__).with_name("providers-hard.json"))[MODEL])
        if provider.model != MODEL or "token-plan" not in provider.base_url:
            raise ValueError("expected configured MiMo Flash Token Plan")

        def request(job):
            seq, profile, condition = job
            dest = root / seq / profile
            dest.mkdir(exist_ok=True)
            call(root, seq, condition, provider, output=dest)
            print(seq, profile, condition, flush=True)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(request, jobs))
    outputs = []
    for seq, profile, condition in jobs:
        src = root / seq
        path = src / profile / f"response-{condition}.json"
        if not path.exists():
            continue
        response = read(path)
        w, h = read(src / "source.json")["size"]
        boxes, adaptations, error = [], [], None
        if response["ok"]:
            try:
                boxes, adaptations = predictions(response["text"], w, h)
            except (ValueError, TypeError, KeyError) as exc:
                error = f"format: {exc}"
        else:
            error = f"transport: {response['status']}"
        outputs.append(
            {
                "seq": seq,
                "profile": profile,
                "input": condition,
                "delivered": error is None,
                "error": error,
                "adaptations": adaptations,
                "response_id": response["response_id"] if response["ok"] else None,
                "parts_sha256": response["parts_sha256"],
                **score(src, boxes),
            }
        )
    save(root / f"{prefix}model-scores.json", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--sequences", nargs="+", default=["09", "10", "13", "11"])
    parser.add_argument(
        "--controls",
        action="store_true",
        help="also run raw video, geometry and detail-only controls",
    )
    parser.add_argument("--live", action="store_true", help="make paid requests in fresh sessions")
    parser.add_argument(
        "--proposal-control",
        action="store_true",
        help="also send geometry and proposed boxes without video",
    )
    parser.add_argument("--auto", action="store_true", help="also evaluate automatic cropping")
    parser.add_argument(
        "--no-head-focus", action="store_true", help="skip close-ups of unassigned heads"
    )
    parser.add_argument(
        "--crowd-weights", type=Path, help="compare automatic crops with/without CrowdHuman ONNX"
    )
    parser.add_argument(
        "--compare-overlap",
        action="store_true",
        help="pair the same crowd detector with overlap review off and on",
    )
    args = parser.parse_args()
    run(
        args.root,
        args.sequences,
        args.weights,
        live=args.live,
        controls=args.controls,
        proposal_control=args.proposal_control,
        adaptive=args.auto,
        crowd_weights=args.crowd_weights,
        compare_overlap=args.compare_overlap,
        head_focus=not args.no_head_focus,
    )
