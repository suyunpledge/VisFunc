"""Resolve the function vocabulary shared by evidence checks and answer rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from svsg.contracts import Claim
from svsg.contracts.image_functions import MAX_RASTER_PIXELS, ImageFunctionEvidence, PixelQuery

from .spatial import RELATIONS, evaluate_relation

IMAGE_METRICS = {
    "width_px": "图像宽度（像素）",
    "height_px": "图像高度（像素）",
    "unique_colors": "不同 RGBA 颜色数",
    "component_count": "同色四连通区域数",
    "top_color_rgba": "颜色表第一项（按像素数排序）",
}
REGION_METRICS = {
    "area_px": "面积（像素数）",
    "perimeter_px": "周长（像素边长）",
    "centroid_x": "重心 X（像素）",
    "centroid_y": "重心 Y（像素）",
    "fill_ratio": "包围框填充比例",
    "shape": "几何形状",
    "rgba": "RGBA 颜色",
}
CONTOUR_METRICS = {
    "outline_kind": "轮廓类型",
    "holes": "孔洞数",
    "solidity": "面积与凸包面积之比",
    "circularity": "等周紧致度",
    "axis_angle_deg": "主轴角（度）",
    "elongation": "主轴与次轴尺度比",
}


@dataclass(frozen=True)
class FunctionFact:
    caption: str
    value: int | float | str | bool


def resolve_function_fact(evidence: ImageFunctionEvidence | None, claim: Claim) -> FunctionFact:
    if evidence is None:
        raise ValueError("请求没有函数证据")
    if claim.instance_id is not None or claim.distance_px is not None:
        raise ValueError("函数断言使用 region_id 或测量 ID")
    metric = claim.field.removeprefix("function:")
    relative = metric.removeprefix("spatial:") if metric.startswith("spatial:") else None
    if relative is None and claim.target_region_id is not None:
        raise ValueError("只有空间函数接受 target_region_id")
    quantized = claim.region_basis == "quantized"
    view = evidence.quantized if quantized else None
    if quantized and view is None:
        raise ValueError("没有量化网格证据")
    basis = "量化网格" if quantized else "原图像素"
    regions = view.regions if view is not None else evidence.regions
    value: int | float | str | None
    if metric.startswith("measurement:") and claim.region_id is None:
        if quantized:
            raise ValueError("测量查询使用原图坐标")
        mid = metric.removeprefix("measurement:")
        for measurement in evidence.measurements:
            query = measurement.query
            if query.id == mid:
                if isinstance(query, PixelQuery):
                    caption = f"像素 ({query.x}, {query.y}) 的 RGBA"
                else:
                    caption = f"区域 {list(query.box)} 的 RGBA 通道均值"
                return FunctionFact(caption, measurement.value)
        raise ValueError("没有此测量 ID")
    if claim.region_id is None and metric in IMAGE_METRICS:
        if view is not None:
            value = {
                "width_px": view.width,
                "height_px": view.height,
                "component_count": view.component_count,
            }.get(metric)
        elif metric == "top_color_rgba":
            value = evidence.top_colors[0].rgba if evidence.top_colors else None
        else:
            value = getattr(evidence, metric)
        if value is None:
            raise ValueError("该参数超出本次分析范围")
        return FunctionFact(f"{basis}的{IMAGE_METRICS[metric]}", value)
    region = next((r for r in regions if r.id == claim.region_id), None)
    if region is None:
        raise ValueError("没有此区域的参数")
    if relative is not None:
        target = next((r for r in regions if r.id == claim.target_region_id), None)
        if relative not in RELATIONS or target is None or target.id == region.id:
            raise ValueError("空间函数需要同一坐标基准的两个不同区域和受支持的指标")
        return FunctionFact(
            f"{basis}区域 {region.id} 相对区域 {target.id} 的{RELATIONS[relative]}",
            evaluate_relation(region, target, relative),
        )
    if metric in REGION_METRICS:
        value = getattr(region, metric)
        label = REGION_METRICS[metric]
    elif metric in CONTOUR_METRICS:
        if region.contour is None:
            raise ValueError("轮廓未保留，证据不足")
        values = {
            "outline_kind": region.outline_kind,
            "holes": region.contour.holes,
            "solidity": region.contour.solidity,
            "circularity": region.contour.circularity,
            "axis_angle_deg": region.contour.axes[0],
            "elongation": region.contour.axes[1],
        }
        value, label = values[metric], CONTOUR_METRICS[metric]
        if value is None:
            raise ValueError("形状没有可确定的主轴角")
    else:
        raise ValueError("未知函数断言")
    return FunctionFact(f"{basis}区域 {region.id} 的{label}", value)


def check_function_claim(evidence: ImageFunctionEvidence | None, claim: Claim) -> str | None:
    try:
        expected = resolve_function_fact(evidence, claim).value
    except ValueError as exc:
        return str(exc)
    value = claim.value
    if isinstance(expected, float):
        matches = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value <= MAX_RASTER_PIXELS
            and isclose(value, expected, rel_tol=0, abs_tol=0.0005)
        )
    else:
        matches = type(value) is type(expected) and value == expected
    return None if matches else f"函数测量结果为 {expected!r}"
