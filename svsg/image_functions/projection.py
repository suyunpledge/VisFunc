"""Bound the contour evidence sent to a text model while retaining the full IR."""

from __future__ import annotations

from itertools import combinations

from svsg.contracts.contour import twice_area
from svsg.contracts.image_functions import ImageFunctionEvidence, RegionParameters

from .spatial import evaluate_relation

MAX_CONTEXT_REGIONS = 12


def _region_summary(region: RegionParameters, preview: bool) -> dict:
    data = region.model_dump(mode="json", exclude={"contour"})
    data.update(shape=region.shape, fill_ratio=round(region.fill_ratio, 6))
    if region.contour is not None:
        c = region.contour
        angle, ratio = c.axes
        data.update(
            outline_kind=region.outline_kind,
            holes=c.holes,
            solidity=round(c.solidity, 6),
            circularity=round(c.circularity, 6),
            elongation=round(ratio, 6),
            axis_angle_deg=None if angle is None else round(angle, 6),
        )
        if preview:
            exterior = next(ring for ring in c.rings if twice_area(ring) > 0)
            count = min(12, len(exterior))
            data["outline_preview"] = [exterior[i * len(exterior) // count] for i in range(count)]
    return data


def _region_list(regions: list[RegionParameters]) -> dict:
    visible = regions[:MAX_CONTEXT_REGIONS]
    pairs = []
    for source, target in combinations(visible[:4], 2):
        pair = {
            "a": source.id,
            "b": target.id,
            "center_distance_px": round(
                float(evaluate_relation(source, target, "center_distance_px")), 6
            ),
            "bbox_iou": round(float(evaluate_relation(source, target, "bbox_iou")), 6),
        }
        if source.contour is not None and target.contour is not None:
            pair["shared_edge_px"] = evaluate_relation(source, target, "shared_edge_px")
        pairs.append(pair)
    return {
        "regions": [_region_summary(r, i < 4) for i, r in enumerate(visible)],
        "regions_omitted_from_ir": len(regions) - len(visible),
        "example_pairs": pairs,
    }


def function_context(evidence: ImageFunctionEvidence) -> dict:
    payload = evidence.model_dump(mode="json", exclude={"regions", "quantized"})
    payload["top_color_rgba"] = evidence.top_colors[0].rgba if evidence.top_colors else None
    payload.update(_region_list(evidence.regions))
    if evidence.quantized is not None:
        view = evidence.quantized
        payload["quantized"] = {
            "width": view.width,
            "height": view.height,
            "method": view.method,
            "component_count": view.component_count,
            **_region_list(view.regions),
        }
    return payload
