"""Live benchmark. Configuration from SVSG_VISION_*; no credentials in artifacts.

python scripts/evaluate_recognition.py --manifest cases.json --output results.json
Manifest: {"cases":[{"id":"A","file":"image.png","sha256":"...",
"labels":["person","cup"],"expected":[{"label":"person","bbox_px":[0,0,20,40]}]}]}
Freeze annotations BEFORE execution. Paths are relative to the manifest. No auto retries.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from svsg.config import Settings
from svsg.contracts.recognition import FrameReference
from svsg.recognition import RecognitionService
from svsg.recognition.adaptive import AdaptiveRecognitionService
from svsg.recognition.evaluation import aggregate, score_objects
from svsg.recognition.provider import OpenAIRecognizer


async def run(manifest: Path, output: Path, *, concurrency: int = 1) -> None:
    raw = manifest.read_bytes()
    data = json.loads(raw)
    settings = Settings()
    if settings.vision_provider != "openai":
        raise ValueError("set SVSG_VISION_PROVIDER=openai and configure its endpoint/model/key")
    service_type = (
        AdaptiveRecognitionService
        if settings.vision_strategy == "multiview"
        else (RecognitionService)
    )
    budget = (
        {"max_calls": settings.vision_max_calls, "discovery_count": settings.vision_discovery_views}
        if (settings.vision_strategy == "multiview")
        else {}
    )
    service = service_type(
        OpenAIRecognizer(
            settings.vision_base_url,
            settings.vision_api_key,
            settings.vision_model,
            timeout_s=settings.vision_timeout_s,
            thinking_disabled=settings.vision_thinking_disabled,
            json_mode=settings.vision_json_mode,
        ),
        reviewer=OpenAIRecognizer(
            settings.vision_base_url,
            settings.vision_api_key,
            settings.vision_review_model,
            timeout_s=settings.vision_timeout_s,
            thinking_disabled=settings.vision_thinking_disabled,
            json_mode=settings.vision_json_mode,
        )
        if settings.vision_review_model
        else None,
        total_timeout_s=settings.vision_timeout_s,
        max_objects=settings.vision_max_objects,
        **budget,
    )
    results: dict = {
        "started_at": datetime.now(UTC).isoformat(),
        "model": settings.vision_model,
        "review_model": settings.vision_review_model or settings.vision_model,
        "strategy": settings.vision_strategy,
        "json_mode": settings.vision_json_mode,
        "budget": budget,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "annotations": "Pre-inference labels and boxes; metrics limited to the supplied classes",
        "scoring": "one-to-one label + IoU >= 0.5; also report IoU >= 0.75",
        "review_iou": service.min_review_iou,
        "max_objects": service.max_objects,
        "cases": [],
    }
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, case: dict) -> dict:
        source = (manifest.parent / case["file"]).read_bytes()
        if hashlib.sha256(source).hexdigest() != case["sha256"]:
            raise ValueError(f"source hash mismatch: {case['id']}")
        async with semaphore:
            start = time.perf_counter()
            result = await service.recognize(
                source,
                labels=case["labels"],
                frame=FrameReference(
                    sequence_id="benchmark-independent-images",
                    frame_index=index,
                    timestamp_ms=index * 1000,
                ),
            )
        row = {
            "id": case["id"],
            "elapsed_s": round(time.perf_counter() - start, 3),
            "group": case.get("group", "original"),
            "expected": case["expected"],
            "recognition": result.model_dump(mode="json"),
        }
        proposed = [
            o.model_dump() for o in result.observations if o.reason != "outside_label_scope"
        ]
        if result.views:
            proposed = [
                o.model_dump()
                for view in result.views
                if view.purpose == "primary"
                for o in view.objects
                if not result.label_scope or o.label in result.label_scope
            ]
        accepted = [o.model_dump() for o in result.observations if o.status == "corroborated"]
        for name, objects in (("proposed", proposed), ("corroborated", accepted)):
            row[name] = score_objects(objects, case["expected"])
            row[name + "_iou75"] = score_objects(objects, case["expected"], 0.75)
        candidates = [o for o in result.observations if o.reason != "outside_label_scope"]
        row["review_coverage"] = len(accepted) / len(candidates) if candidates else None
        row["confirmed_discoveries"] = sum(
            o.status == "corroborated" and o.origin != "primary" for o in result.observations
        )
        results["cases"].append(row)
        results["aggregate"] = {
            name: aggregate([c[name] for c in results["cases"]])
            for name in ("proposed", "corroborated", "proposed_iou75", "corroborated_iou75")
        }
        results["total_tokens"] = sum(
            (call["prompt_tokens"] or 0) + (call["completion_tokens"] or 0)
            for c in results["cases"]
            for call in c["recognition"]["calls"]
        )
        results["unknown_usage_calls"] = sum(
            call["prompt_tokens"] is None or call["completion_tokens"] is None
            for c in results["cases"]
            for call in c["recognition"]["calls"]
        )
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "id": case["id"],
                    "status": result.status,
                    "proposed": row["proposed"],
                    "corroborated": row["corroborated"],
                    "tokens_so_far": results["total_tokens"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return row

    await asyncio.gather(*(one(i, case) for i, case in enumerate(data["cases"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    asyncio.run(run(args.manifest, args.output, concurrency=args.concurrency))
