"""Full image proposals → blind context crops → conservative observation ledger."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import hypot
from typing import Literal

from svsg.contracts.recognition import (
    FrameReference,
    ModelCall,
    ObjectObservation,
    ObjectRelation,
    RecognitionResult,
    ReviewObjects,
    SceneObjects,
)
from svsg.image_functions.compiler import describe_raster
from svsg.image_functions.raster import RasterFunction

from .geometry import Box, canonical_label, context_box, iou, pixel_box
from .images import canonical_raster, image_part
from .provider import ProviderOutputError, VisionBackend


def _messages(raster: RasterFunction, crops: dict[int, Box] | None = None) -> list[dict]:
    with raster.to_image() as image:
        if crops is None:
            return [
                {"type": "text", "text": "Identify visible objects in this image."},
                image_part(image, 1536),
            ]
        messages = []
        for cid, box in crops.items():
            with image.crop(box) as crop:
                messages.extend(
                    [{"type": "text", "text": f"candidate_id={cid}"}, image_part(crop, 768)]
                )
        return messages


def _observation(
    raster: RasterFunction,
    digest: str,
    cid: int,
    label: str,
    box: Box,
) -> ObjectObservation:
    with raster.to_image() as image, image.crop(box) as crop:
        crop_digest = sha256(crop.tobytes()).hexdigest()
    oid = sha256(f"{digest}:{label}:{box}".encode()).hexdigest()[:20]
    return ObjectObservation(
        observation_id=f"obs_{oid}",
        candidate_id=cid,
        label=label,
        bbox_px=box,
        bbox_normalized=(
            box[0] / raster.width,
            box[1] / raster.height,
            box[2] / raster.width,
            box[3] / raster.height,
        ),
        status="needs_review",
        reason="review_missing",
        crop_pixel_sha256=crop_digest,
        crop_mean_rgba=raster.mean(box),
    )


def _relations(objects: list[ObjectObservation], width: int, height: int) -> list[ObjectRelation]:
    pairs = []
    for a, b in combinations((o for o in objects if o.status == "corroborated"), 2):
        ax, ay, ax1, ay1 = a.bbox_px
        bx, by, bx1, by1 = b.bbox_px
        dx, dy = (bx + bx1 - ax - ax1) / 2, (by + by1 - ay - ay1) / 2
        pairs.append(
            ObjectRelation(
                source_id=a.observation_id,
                target_id=b.observation_id,
                center_delta_normalized=(dx / width, dy / height),
                center_distance_px=hypot(dx, dy),
                bbox_iou=iou(a.bbox_px, b.bbox_px),
            )
        )
    return pairs


def _clipped_review(box: Box, crop: Box, width: int, height: int) -> bool:
    """A crop that cuts off the re-detected object cannot corroborate its full extent."""
    return (
        (crop[0] > 0 and box[0] <= crop[0] + 1)
        or (crop[1] > 0 and box[1] <= crop[1] + 1)
        or (crop[2] < width and box[2] >= crop[2] - 1)
        or (crop[3] < height and box[3] >= crop[3] - 1)
    )


@dataclass(frozen=True)
class RecognitionService:
    backend: VisionBackend
    reviewer: VisionBackend | None = None
    total_timeout_s: float = 90
    max_objects: int = 8
    min_review_iou: float = 0.5

    def __post_init__(self) -> None:
        if not 1 <= self.max_objects <= 12 or not 0 < self.min_review_iou <= 1:
            raise ValueError("invalid recognition limits")
        if not 0 < self.total_timeout_s <= 300:
            raise ValueError("recognition deadline must be in (0, 300]")

    async def recognize(
        self,
        data: bytes,
        *,
        labels: list[str] | None = None,
        frame: FrameReference | None = None,
    ) -> RecognitionResult:
        start = time.perf_counter()
        raster = await asyncio.to_thread(canonical_raster, data)
        functions = await asyncio.to_thread(describe_raster, raster)
        digest = sha256(data).hexdigest()
        scope = sorted({canonical_label(label) for label in (labels or [])})
        calls: list[ModelCall] = []
        warnings: list[str] = [
            "Model agreement is not a calibrated probability or ground-truth guarantee.",
            "Counts cover corroborated observations only; missed objects remain possible.",
            "Object boxes are estimates; color-region contours are not object masks.",
        ]
        objects: list[ObjectObservation] = []
        crops: dict[int, Box] = {}
        failed = False
        try:
            content = await asyncio.to_thread(_messages, raster)
            result, call = await asyncio.wait_for(
                self.backend.infer("detect", content, scope, self.max_objects),
                timeout=max(0, self.total_timeout_s - (time.perf_counter() - start)),
            )
            calls.append(call)
            if not isinstance(result, SceneObjects):
                raise ValueError("invalid detection response")
            if len(result.objects) >= self.max_objects:
                warnings.append("Candidate limit reached; coverage may be truncated.")
            view = (0, 0, raster.width, raster.height)
            for cid, obj in enumerate(result.objects[: self.max_objects], 1):
                label, box = canonical_label(obj.label), pixel_box(obj.bbox, view)
                observation = await asyncio.to_thread(_observation, raster, digest, cid, label, box)
                if scope and label not in scope:
                    observation.reason = "outside_label_scope"
                elif any(o.label == label and iou(o.bbox_px, box) >= 0.8 for o in objects):
                    observation.reason = "duplicate_proposal"
                else:
                    crops[cid] = context_box(box, raster.width, raster.height)
                objects.append(observation)
        except Exception as exc:
            failed = True
            calls.append(
                exc.call
                if isinstance(exc, ProviderOutputError)
                else ModelCall(
                    stage="detect",
                    model=self.backend.model,
                    elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
                    error=type(exc).__name__,
                )
            )
            objects, crops = [], {}

        reviewer = self.reviewer or self.backend
        if crops:
            review_start = time.perf_counter()
            try:
                content = await asyncio.to_thread(_messages, raster, crops)
                review, call = await asyncio.wait_for(
                    reviewer.infer("review", content, scope, self.max_objects),
                    timeout=max(0, self.total_timeout_s - (time.perf_counter() - start)),
                )
                calls.append(call)
                if not isinstance(review, ReviewObjects):
                    raise ValueError("invalid review response")
                if any(row.candidate_id not in crops for row in review.results):
                    raise ValueError("review references a crop that was not sent")
                by_id = {row.candidate_id: row.objects for row in review.results}
                for obs in objects:
                    local_objects = by_id.get(obs.candidate_id, [])
                    if not local_objects:
                        continue
                    candidates = [
                        (canonical_label(o.label), pixel_box(o.bbox, crops[obs.candidate_id]))
                        for o in local_objects
                    ]
                    label, box = max(candidates, key=lambda pair: iou(obs.bbox_px, pair[1]))
                    obs.review_label, obs.review_bbox_px = label, box
                    obs.review_iou = iou(obs.bbox_px, box)
                    if label != obs.label:
                        obs.reason = "label_disagreement"
                    elif obs.review_iou < self.min_review_iou:
                        obs.reason = "localization_disagreement"
                    elif _clipped_review(box, crops[obs.candidate_id], raster.width, raster.height):
                        obs.reason = "review_object_clipped_by_crop"
                    else:
                        obs.status, obs.reason = "corroborated", "label_and_box_agree"
            except Exception as exc:
                calls.append(
                    exc.call
                    if isinstance(exc, ProviderOutputError)
                    else ModelCall(
                        stage="review",
                        model=reviewer.model,
                        elapsed_ms=round((time.perf_counter() - review_start) * 1000, 2),
                        error=type(exc).__name__,
                    )
                )
                for obs in objects:
                    if obs.candidate_id in crops:
                        obs.status, obs.reason = "needs_review", "review_unavailable"

        accepted = [o for o in objects if o.status == "corroborated"]
        counts = dict(sorted(Counter(o.label for o in accepted).items()))
        status: Literal["corroborated", "needs_review", "unavailable"]
        status = (
            "unavailable"
            if failed
            else ("corroborated" if objects and len(accepted) == len(objects) else "needs_review")
        )
        if counts:
            summary = "、".join(f"{k} × {v}" for k, v in counts.items())
            answer = "复核一致的可见对象：" + summary + "。"
        else:
            answer = "暂未获得复核一致的物体证据。"
        answer += "这不是全图完整计数；未检出不代表不存在。"
        if len(accepted) < len(objects):
            answer += f"另有 {len(objects) - len(accepted)} 个候选需要复核。"
        return RecognitionResult(
            status=status,
            image_sha256=digest,
            pixel_sha256=raster.pixel_sha256,
            width=raster.width,
            height=raster.height,
            frame=frame,
            label_scope=scope,
            review_mode="different_backend"
            if reviewer.model != self.backend.model
            else "same_model_different_view",
            observations=objects,
            corroborated_counts=counts,
            relations=_relations(objects, raster.width, raster.height),
            image_functions=functions,
            calls=calls,
            warnings=warnings,
            final_answer=answer,
        )
