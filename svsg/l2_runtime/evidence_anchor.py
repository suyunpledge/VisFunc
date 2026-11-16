"""Validate structured claims against image detections and verification results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isclose, isfinite

from svsg.contracts import (
    IR,
    Claim,
    Detection,
    ErrorCode,
    L3Answer,
    RelationType,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from svsg.image_functions.claims import check_function_claim
from svsg.l1_compiler.geometry_engine import (
    center_distance_px,
    distance_relations_allowed,
    nearest_neighbor,
    relation_holds,
)

AMBIGUOUS_LABEL = "ambiguous"


@dataclass(frozen=True)
class Violation:
    """单条锚定违规。"""

    code: ErrorCode
    field: str
    reason: str
    instance_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "field": self.field,
            "instance_id": self.instance_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AnchorVerdict:
    """锚定校验裁决。"""

    passed: bool
    violations: list[Violation]

    @property
    def primary_code(self) -> ErrorCode | None:
        """映射 FSM 事件用：E3003 → VERIFICATION_SKIPPED，否则 E3002。"""
        if self.passed:
            return None
        if any(v.code == ErrorCode.VERIFICATION_SKIPPED for v in self.violations):
            return ErrorCode.VERIFICATION_SKIPPED
        return ErrorCode.EVIDENCE_MISMATCH


def missing_verifications(ir: IR, executed: Iterable[int], exempt: Iterable[int] = ()) -> set[int]:
    """工具调用完整性：应验证而未验证的实例集合。

    exempt：S5 静默降级实例（已请求验证但 L1.5 超时），不参与完整性
    检查——其 claims 仅锚定 IR（见 anchor_claims 的降级语义）。
    """
    executed_set = set(executed)
    exempt_set = set(exempt)
    return {
        d.instance_id
        for d in ir.detections
        if d.verification_required
        and d.instance_id not in executed_set
        and d.instance_id not in exempt_set
    }


def evidence_results(ir: IR, report: VerificationReport | None) -> dict[int, VerificationResult]:
    if (
        report is None
        or report.image_id != ir.image_id
        or report.status != VerificationStatus.SUCCESS
    ):
        return {}
    return {r.instance_id: r for r in report.results}


def _check_relation(
    claim: Claim, source: Detection, active: dict[int, Detection], ir: IR
) -> str | None:
    name = claim.field.partition(":")[2]
    try:
        relation = RelationType(name)
    except ValueError:
        return f"未知关系类型 {name!r}"
    target = active.get(claim.value) if type(claim.value) is int else None
    if target is None or target.instance_id == source.instance_id:
        return "关系必须引用另一个有存在证据的实际实例"
    if relation not in (RelationType.DISTANCE_TO, RelationType.NEAREST_TO):
        if claim.distance_px is not None:
            return "只有距离类关系可以携带 distance_px"
        if not relation_holds(relation, source.bbox_px, target.bbox_px):
            return f"关系断言 {name}(→{target.instance_id}) 未通过几何引擎复核"
        return None

    if not distance_relations_allowed(ir.scene_type, ir.camera_intrinsics is not None):
        return "当前场景未启用距离关系"
    if relation == RelationType.NEAREST_TO:
        ids = list(active)
        nearest = nearest_neighbor(
            ids.index(source.instance_id), [d.bbox_px for d in active.values()]
        )
        if nearest is None or not isclose(
            center_distance_px(source.bbox_px, target.bbox_px), nearest[1], abs_tol=1e-9
        ):
            return f"实例 {target.instance_id} 并非最近的有效实例"
    actual = center_distance_px(source.bbox_px, target.bbox_px)
    if claim.distance_px is not None and not isclose(
        claim.distance_px, actual, rel_tol=0.0, abs_tol=0.5
    ):
        return f"distance_px 与图像平面实测距离 {actual:.2f} px 不一致"
    return None


def _check_claim(
    claim: Claim,
    ir: IR,
    by_id: dict[int, Detection],
    active: dict[int, Detection],
    results: dict[int, VerificationResult],
) -> str | None:
    field, value, iid = claim.field, claim.value, claim.instance_id
    if field.startswith("function:"):
        return check_function_claim(ir.image_functions, claim)
    if any(v is not None for v in (claim.region_id, claim.region_basis, claim.target_region_id)):
        return "区域参数只适用于函数断言"
    if ir.image_functions is not None and not by_id:
        return "当前证据描述像素和几何区域，只支持 function 断言"
    name, _, suffix = field.partition(":")
    if claim.distance_px is not None and name != "relation":
        return "distance_px 只适用于距离类关系"
    if field == "count" or field.startswith("count:"):
        if (
            iid is not None
            or type(value) is not int
            or value < 0
            or (field != "count" and not suffix.strip())
        ):
            return "count 需要全局范围及非负整数值"
        if field == "count":
            actual = len(active)
            return None if value == actual else f"count 断言 {value} != 有效实例总数 {actual}"
        if any(
            r.class_label is not None
            and (r.class_label == AMBIGUOUS_LABEL or r.class_label != active[rid].object_class)
            for rid, r in results.items()
            if rid in active
        ):
            return "存在类别未确定的实例，证据不足以确认精确分类计数"
        actual = sum(d.object_class == suffix for d in active.values())
        return None if value == actual else f"count:{suffix} 断言 {value} != 有效实例数 {actual}"

    if field not in ("class", "exists") and name not in ("attribute", "relation"):
        return f"不支持的断言字段 {field!r}"
    if iid is None:
        return "实例断言缺少 instance_id"
    det = by_id.get(iid)
    if det is None:
        return f"断言引用了 IR 中不存在的实例 {iid}"
    result = results.get(iid)
    if field == "exists":
        if type(value) is not bool:
            return "exists 需要布尔值"
        actual_exists = iid in active
        return None if value == actual_exists else f"断言与证据的 exists={actual_exists} 冲突"
    if iid not in active:
        return "验证报告已判定该实例不存在"
    if field == "class":
        if value != det.object_class:
            return f"class 断言 {value!r} 与 IR 检测结果 {det.object_class!r} 不一致"
        if result is not None and result.class_label is not None and result.class_label != value:
            return f"class 断言 {value!r} 与验证报告 {result.class_label!r} 不一致"
        if value == AMBIGUOUS_LABEL:
            return "类别仍为 ambiguous"
        return None
    if name == "attribute" and suffix.strip():
        if isinstance(value, float) and not isfinite(value):
            return "属性值必须为有限数值"
        if value is None or result is None or suffix not in result.attributes:
            return f"属性 {suffix!r} 缺少 L1.5 验证证据"
        if str(value) != result.attributes[suffix]:
            return f"属性 {suffix}={value!r} 与验证报告 {result.attributes[suffix]!r} 不一致"
        return None
    if name == "relation" and suffix:
        return _check_relation(claim, det, active, ir)
    return f"断言字段 {field!r} 缺少名称"


def anchor_claims(
    ir: IR,
    answer: L3Answer,
    report: VerificationReport | None = None,
) -> list[Violation]:
    """Return violations for all claims, including unsupported fields and empty sets."""
    if not answer.claims:
        return [Violation(ErrorCode.EVIDENCE_MISMATCH, "<claims>", "缺少可校验的结构化断言")]
    by_id = {d.instance_id: d for d in ir.detections}
    results = evidence_results(ir, report)
    active = {
        iid: d for iid, d in by_id.items() if iid not in results or results[iid].exists is not False
    }
    violations = []
    for claim in answer.claims:
        reason = _check_claim(claim, ir, by_id, active, results)
        if reason is not None:
            violations.append(
                Violation(ErrorCode.EVIDENCE_MISMATCH, claim.field, reason, claim.instance_id)
            )
    return violations


def nl_claims_coverage(
    ir: IR,
    answer: L3Answer,
    threshold: float = 0.8,
) -> float:
    """NL-claims 关键词覆盖率（V5.2.4 §4.5.2 第 5 步，Phase 1 简化实现）。

    提取自然语言中的视觉关键词（IR 类别词 + claim 值词），检查每个
    关键词是否至少出现在一个 claim 的 value 中。返回覆盖率 [0,1]。

    Phase 1 语义：覆盖率 < threshold 时由调用方记录告警日志，不触发
    S7（已知风险敞口，Phase 2 关闭：完整 NL-claims 结构化对齐）。
    """
    # 关键词集 = IR 中出现过的类别名（检测器词汇表）
    keywords: set[str] = {d.object_class.lower() for d in ir.detections}
    if not keywords:
        return 1.0  # 无检测实例时无关键词可覆盖
    # claims 值文本（小写化）
    claim_text = " ".join(str(cl.value).lower() for cl in answer.claims if cl.value is not None)
    hit = sum(1 for kw in keywords if kw in claim_text)
    return hit / len(keywords)


def run_anchor(
    ir: IR,
    answer: L3Answer,
    executed_verifications: Iterable[int],
    report: VerificationReport | None = None,
    exempt: Iterable[int] = (),
) -> AnchorVerdict:
    """执行完整锚定校验（完整性 → 忠实度），输出裁决。

    编排器根据 primary_code 映射 FSM 事件：
    None → ANCHOR_PASSED；E2003 → VERIFICATION_SKIPPED；E3002 → ANCHOR_FAILED。
    exempt：S5 静默降级实例（L1.5 超时），豁免完整性检查。
    """
    violations: list[Violation] = []

    missing = missing_verifications(ir, executed_verifications, exempt=exempt)
    if missing:
        violations.append(
            Violation(
                code=ErrorCode.VERIFICATION_SKIPPED,
                field="<completeness>",
                reason=f"应验证而未调用 VerifyRegion 的实例: {sorted(missing)}",
            )
        )

    violations.extend(anchor_claims(ir, answer, report))
    return AnchorVerdict(passed=not violations, violations=violations)
