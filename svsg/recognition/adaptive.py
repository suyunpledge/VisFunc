"""Bounded discovery and independent cross-view confirmation for difficult images."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256

from pydantic import TypeAdapter

from svsg.contracts.recognition import (
    FrameReference,
    Labels,
    ModelCall,
    ObjectObservation,
    RecognitionResult,
    RecognitionView,
    SceneObjects,
)
from svsg.image_functions.compiler import describe_raster

from .consensus import Candidate, associate, consensus_box
from .geometry import canonical_label, iou
from .images import canonical_raster
from .provider import ProviderOutputError, VisionBackend
from .service import RecognitionService, _observation, _relations
from .views import View, confirmation_view, discovery_views, focus_small_object


@dataclass(frozen=True)
class AdaptiveRecognitionService(RecognitionService):
    max_objects: int = 12
    max_calls: int = 5
    discovery_count: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 2 <= self.max_calls <= 8 or not 0 <= self.discovery_count <= 2:
            raise ValueError("invalid adaptive call budget")

    async def recognize(
        self,
        data: bytes,
        *,
        labels: list[str] | None = None,
        frame: FrameReference | None = None,
    ) -> RecognitionResult:
        start = time.perf_counter()
        scope = sorted(
            {canonical_label(s) for s in TypeAdapter(Labels).validate_python(labels or [])}
        )
        raster = await asyncio.to_thread(canonical_raster, data)
        functions = await asyncio.to_thread(describe_raster, raster)
        digest = sha256(data).hexdigest()
        calls: list[ModelCall] = []
        views: list[RecognitionView] = []
        groups: list[Candidate] = []
        warnings = [
            "Cross-view agreement is not ground truth; missed objects remain possible.",
            "Model input interpolation does not recover lost source-image detail.",
            "Object boxes are estimates; color-region contours are not object masks.",
        ]
        reviewer = self.reviewer or self.backend
        two_models = reviewer.model != self.backend.model

        async def inspect(view: View, backend: VisionBackend) -> bool:
            remaining = self.total_timeout_s - (time.perf_counter() - start)
            if len(calls) >= self.max_calls or remaining <= 0:
                return False
            call_start = time.perf_counter()
            record: RecognitionView | None = None
            try:
                content, record = await asyncio.to_thread(view.encode, raster)
                assert record is not None
                result, call = await asyncio.wait_for(
                    backend.infer("detect", content, scope, self.max_objects),
                    timeout=max(0, self.total_timeout_s - (time.perf_counter() - start)),
                )
                if not isinstance(result, SceneObjects):
                    raise ValueError("single-image detection response required")
                if len(result.objects) >= self.max_objects:
                    warnings.append(f"{view.id}: candidate limit reached")
                record.objects = [
                    view.map_object(obj, backend.model, raster.width, raster.height)
                    for obj in result.objects[: self.max_objects]
                ]
                for hit in record.objects:
                    hit.evidence_key = record.encoded_sha256
            except Exception as exc:
                call = (
                    exc.call
                    if isinstance(exc, ProviderOutputError)
                    else ModelCall(
                        stage="detect",
                        model=backend.model,
                        elapsed_ms=round((time.perf_counter() - call_start) * 1000, 2),
                        error=type(exc).__name__,
                    )
                )
            call.view_id, call.purpose = view.id, view.purpose
            if view.purpose in ("cross_view", "confirmation"):
                call.stage = "review"
            calls.append(call)
            if record is not None:
                views.append(record)
                # A malformed or failed response must not contribute partial objects.
                if call.error is None:
                    associate(
                        groups,
                        [obj for obj in record.objects if not scope or obj.label in scope],
                        self.min_review_iou,
                    )
            return call.error is None

        whole = (0, 0, raster.width, raster.height)
        await inspect(View("primary", "primary", whole), self.backend)
        await inspect(View("mirror", "cross_view", whole, mirrored=True), reviewer)
        details = discovery_views(raster.width, raster.height)
        focus = focus_small_object(raster.width, raster.height, [g.first.bbox_px for g in groups])
        if focus is not None:
            details[0] = focus
            warnings.append("One discovery half replaced by a tiny-object context zoom.")
        # Keep a broad discovery view as well as an optional tiny-object neighbourhood.
        for index, view in enumerate(details[: self.discovery_count]):
            # The second scanner gets its own discovery view, so both overviews missing
            # a small object does not leave discovery entirely to the first detector.
            await inspect(view, reviewer if index % 2 else self.backend)

        attempted: set[int] = set()
        while (
            len(calls) < self.max_calls and self.total_timeout_s - (time.perf_counter() - start) > 0
        ):
            unresolved = [
                (index, group)
                for index, group in enumerate(groups)
                if index not in attempted
                and group.agreement(self.min_review_iou, reviewer.model, two_models) is None
            ]
            if not unresolved:
                break
            # Prefer small, partly occluded or newly discovered instances. This chooses
            # where to spend evidence, never lowers the corroboration threshold.
            index, group = min(
                unresolved,
                key=lambda pair: (
                    (pair[1].first.bbox_px[2] - pair[1].first.bbox_px[0])
                    * (pair[1].first.bbox_px[3] - pair[1].first.bbox_px[1]),
                    -len({s.view_id for s in pair[1].supports}),
                ),
            )
            attempted.add(index)
            view = confirmation_view(
                group.first.bbox_px, raster.width, raster.height, len(attempted)
            )
            # A proposal first seen by the reviewer needs a different-model confirmation.
            backend = (
                self.backend
                if two_models and all(s.model == reviewer.model for s in group.supports)
                else reviewer
            )
            await inspect(view, backend)

        objects: list[ObjectObservation] = []
        origins = {view.view_id: view.purpose for view in views}
        for cid, group in enumerate(groups, 1):
            agreement = group.agreement(self.min_review_iou, reviewer.model, two_models)
            box = (
                consensus_box(agreement[1].bbox_px, agreement[2].bbox_px)
                if agreement
                else (group.first.bbox_px)
            )
            obs = await asyncio.to_thread(_observation, raster, digest, cid, group.first.label, box)
            obs.origin = origins[group.first.view_id]
            obs.supports = group.supports
            if agreement:
                overlap, _, support = agreement
                obs.status, obs.reason = "corroborated", "independent_views_agree"
                obs.review_iou, obs.review_label, obs.review_bbox_px = (
                    overlap,
                    support.label,
                    support.bbox_px,
                )
            else:
                obs.reason = "insufficient_independent_support"
                if any(s.clipped for s in group.supports):
                    obs.reason = "view_boundary_truncation"
            objects.append(obs)

        # Two association groups can converge after refinement. Never count them twice.
        accepted: list[ObjectObservation] = []
        for obs in objects:
            if obs.status == "corroborated":
                if any(
                    a.label == obs.label and iou(a.bbox_px, obs.bbox_px) >= 0.8 for a in accepted
                ):
                    obs.status, obs.reason = "needs_review", "duplicate_consensus"
                else:
                    accepted.append(obs)
        if len(accepted) < len(objects):
            warnings.append(
                "Unresolved candidates retained; more evidence or human review required."
            )
        if len(calls) >= self.max_calls:
            warnings.append("Call budget reached; no automatic retries or unbounded discovery.")
        if self.total_timeout_s - (time.perf_counter() - start) <= 0:
            warnings.append("Total recognition deadline reached.")
        counts = dict(sorted(Counter(o.label for o in accepted).items()))
        answer = (
            "复核一致的可见对象：" + "、".join(f"{k} × {v}" for k, v in counts.items()) + "。"
            if counts
            else ("暂未获得复核一致的物体证据。")
        )
        answer += "这不是全图完整计数；未检出不代表不存在。"
        if len(accepted) < len(objects):
            answer += f"另有 {len(objects) - len(accepted)} 个候选需要复核。"
        return RecognitionResult(
            status="unavailable"
            if not any(c.error is None for c in calls)
            else ("corroborated" if objects and len(accepted) == len(objects) else "needs_review"),
            image_sha256=digest,
            pixel_sha256=raster.pixel_sha256,
            width=raster.width,
            height=raster.height,
            frame=frame,
            label_scope=scope,
            review_mode="different_backend" if two_models else "same_model_different_view",
            observations=objects,
            corroborated_counts=counts,
            relations=_relations(objects, raster.width, raster.height),
            image_functions=functions,
            calls=calls,
            warnings=warnings,
            final_answer=answer,
            pipeline="multiview",
            views=views,
        )
