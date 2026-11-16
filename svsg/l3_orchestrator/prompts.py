"""Instructions and structured evidence for the L3 claim generator."""

from __future__ import annotations

import json
from collections.abc import Iterable

from svsg.contracts import IR, VerificationReport, VerificationStatus
from svsg.image_functions.claims import CONTOUR_METRICS, IMAGE_METRICS, REGION_METRICS
from svsg.image_functions.projection import function_context
from svsg.image_functions.spatial import RELATIONS

SYSTEM_PROMPT_TEMPLATE = (
    """你是 SVSG 视觉问答主控引擎（L3）。你看不到图像，
只能基于 L2 提供的结构化证据作答。

断言生成规则：
1. 只允许使用【结构化证据】中的事实作答；禁止推测、编造或用外部知识补充视觉细节。
2. 输出 JSON 对象，包含非空 claims 数组和简短 final_answer。
   交付文本由网关从通过校验的断言生成。把回答所需的全部事实写进 claims。
3. claims 字段词表（全局断言省略 instance_id）：
   - {{"instance_id": <id>, "field": "class", "value": "<类别>"}}
   - {{"field": "count", "value": <有效实例总数>}}
   - {{"field": "count:<类别>", "value": <该类别实例数>}}
   - {{"instance_id": <id>, "field": "exists", "value": true 或 false}}
   - {{"instance_id": <id>, "field": "attribute:<名称>", "value": <属性值>}}
   - {{"instance_id": <id>, "field": "relation:<类型>", "value": <目标 instance_id>}}
   relation 类型词表：left_of / right_of / above / below / contains /
   overlaps / nearest_to / distance_to。
   对距离类关系，value 始终是目标实例 ID。可省略 distance_px，由网关从两个框中心计算。
4. {distance_rule}
5. verification_results 中 exists=false 的实例不计入计数，也不能用于其他视觉断言。
   可以用 exists=false 表达该实例的验证结果。
6. 验证结果为 ambiguous 的实例只能参与总数，不支持其具体类别以及精确分类计数。
7. class_overrides 中给出的类别是冲突解决后的最终结论，claims 必须采用该类别而非检测器原始类别。
"""
)

UNSAFE_DISTANCE_RULE = (
    "距离和最近邻使用图像平面内的像素，不能用像素推断物理距离。"
)

SAFE_DISTANCE_RULE = (
    "当前断言协议只支持从框中心复算的图像平面像素距离。"
)


def build_system_prompt(physical_metrics_available: bool, *, image_functions: bool = False) -> str:
    """Build the model instructions for the structured claim protocol."""
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        distance_rule=(
            SAFE_DISTANCE_RULE if physical_metrics_available else UNSAFE_DISTANCE_RULE
        )
    )
    if image_functions:
        prompt += (
            "\n函数证据：使用 field=\"function:<参数名>\"，value 为实际参数。"
            f"全图参数 {', '.join(IMAGE_METRICS)}。"
            f"区域参数 {', '.join(REGION_METRICS | CONTOUR_METRICS)}，须指定 region_id。"
            f"空间函数 field=\"function:spatial:<指标>\"，指标 {', '.join(RELATIONS)}，"
            "须指定 region_id 和 target_region_id。"
            "量化数据使用 region_basis=quantized，面积、距离和颜色属于量化网格。"
            "精确区与量化区的 ID 是独立的。outline_preview 是采样轮廓，不能视作完整边界。"
            "指定位置的测量使用 function:measurement:<query.id> 和原样的 value。"
            "null 表示未测得。连通区域是像素证据，不能当成物体或语义类别。"
            "仅有函数证据时只输出 function 断言。"
        )
    return prompt


def build_context(
    query: str,
    ir: IR,
    report: VerificationReport | None = None,
    *,
    degraded: Iterable[int] = (),
    disputed: Iterable[int] = (),
    corrective_feedback: list[dict] | None = None,
) -> str:
    """构建单轮用户上下文：结构化证据 JSON。

    corrective_feedback：上一轮被锚定否决的断言及原因（重生成时注入），
    LLM 必须修正后重新作答。
    """
    degraded_set = set(degraded)
    disputed_set = set(disputed)

    detections = []
    for d in ir.detections:
        digest = {
            "instance_id": d.instance_id,
            "class": d.object_class,
            "bbox_px": list(d.bbox_px),
            "conf_level": "lowest" if d.instance_id in degraded_set else d.conf_level.value,
            "relations": [
                {
                    "type": r.type.value,
                    "target_id": r.target_id,
                    **({"distance_px": r.distance_px} if r.distance_px is not None else {}),
                }
                for r in d.relations
            ],
        }
        detections.append(digest)

    payload: dict = {
        "query": query,
        "scene_type": ir.scene_type.value,
        "image": {"width": ir.image_width, "height": ir.image_height},
        "detections": detections,
        "degraded_instances": sorted(degraded_set),
        "disputed_instances": sorted(disputed_set),
    }
    if report is not None and report.status == VerificationStatus.SUCCESS:
        payload["verification_results"] = [
            {
                "instance_id": r.instance_id,
                "class": r.class_label,
                "attributes": r.attributes,
                "exists": r.exists,
                "confidence": r.confidence,
            }
            for r in report.results
        ]
    if ir.image_functions is not None:
        payload["image_functions"] = function_context(ir.image_functions)
    if corrective_feedback:
        payload["corrective_feedback"] = {
            "notice": "上一轮答案被证据锚定校验否决，以下断言与结构化证据不一致，必须修正",
            "violations": corrective_feedback,
        }

    return json.dumps(payload, ensure_ascii=False, indent=1)
