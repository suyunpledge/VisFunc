"""Render the API answer from accepted claims and recomputed measurements."""

from __future__ import annotations

import json

from svsg.contracts import IR, L3Answer, RelationType
from svsg.image_functions.claims import resolve_function_fact
from svsg.l1_compiler.geometry_engine import center_distance_px

RELATIONS = {
    RelationType.LEFT_OF: "位于实例 {} 的左侧",
    RelationType.RIGHT_OF: "位于实例 {} 的右侧",
    RelationType.ABOVE: "位于实例 {} 的上方",
    RelationType.BELOW: "位于实例 {} 的下方",
    RelationType.CONTAINS: "包含实例 {}",
    RelationType.OVERLAPS: "与实例 {} 重叠",
}


def render_anchored_answer(answer: L3Answer, ir: IR, review_ids: set[int]) -> L3Answer:
    by_id = {d.instance_id: d for d in ir.detections}
    claims, lines = [], []
    for claim in answer.claims:
        field, value, iid = claim.field, claim.value, claim.instance_id
        name, _, suffix = field.partition(":")
        display = json.dumps(value, ensure_ascii=False)
        updates: dict = {"confidence": None}
        if name == "function":
            fact = resolve_function_fact(ir.image_functions, claim)
            value = fact.value
            updates["value"] = value
            rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
            line = f"{fact.caption}为 {rendered}"
        elif field == "count":
            line = f"证据中记录了 {value} 个有效实例"
        elif name == "count":
            label = json.dumps(suffix, ensure_ascii=False)
            line = f"证据中记录了 {value} 个 {label} 实例"
        elif field == "class":
            line = f"实例 {iid} 的类别为 {display}"
        elif field == "exists":
            line = f"实例 {iid} 存在" if value else f"验证结果判定实例 {iid} 不存在"
        elif name == "attribute":
            attribute = json.dumps(suffix, ensure_ascii=False)
            line = f"实例 {iid} 的属性 {attribute} 为 {display}"
        else:
            assert iid is not None and type(value) is int
            relation = RelationType(suffix)
            if relation in (RelationType.DISTANCE_TO, RelationType.NEAREST_TO):
                distance = round(center_distance_px(by_id[iid].bbox_px, by_id[value].bbox_px), 2)
                updates["distance_px"] = distance
                line = f"实例 {iid} 与实例 {value} 的框中心在图像平面内相距 {distance:g} 像素"
                if relation == RelationType.NEAREST_TO:
                    line += f"，实例 {value} 是像素距离最近的有效实例之一"
            else:
                line = f"实例 {iid} " + RELATIONS[relation].format(value)
        claims.append(claim.model_copy(update=updates))
        lines.append(line + "。")
    if review_ids:
        ids = "、".join(map(str, sorted(review_ids)))
        lines.append(f"实例 {ids} 的证据已降级或存在争议，待人工复审。")
    return L3Answer(claims=claims, final_answer="\n".join(lines))
