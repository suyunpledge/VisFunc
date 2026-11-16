"""L1.5 视觉验证服务：超时隔离 + 幂等调用。

V5.2.1 §5.1 的机器可执行化：
- 超时阈值必须小于 L3 主控总超时时间的 1/3（由 fallback.assert_timeout_budget
  在编排器装配期强制断言）。
- verify_instances 绝不抛异常、绝不重试：超时/后端故障统一返回
  status=timeout / error 的报告（results 为空），由 L2 走静默降级
  （conf_level → lowest，转 S6）。
- 幂等性：本服务无内部状态，同一 (image_id, instance_ids) 的重复调用
  相互独立、无副作用；降级决策由调用方（L2 会话）持有。

保守性决策（生产锁定）：整体批调用超时后，后端的任何部分输出都不采信
（status != success 时 results 恒为空）——极端负载下的服务状态不可信，
宁可多降级、不可透传未经证实的结果。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from svsg.contracts import VerificationReport, VerificationResult, VerificationStatus
from svsg.l1_compiler.geometry_engine import BBox


class VerifierBackend(Protocol):
    """真实视觉验证后端协议（细粒度分类 / 属性 / 存在性）。"""

    async def verify_batch(
        self, image_id: str, instance_ids: list[int]
    ) -> list[VerificationResult]: ...


class ImageVerifierBackend(Protocol):
    """Verify only the regions supplied for the current image."""

    async def verify_regions(
        self, image: bytes, regions: dict[int, BBox]
    ) -> list[VerificationResult]: ...


@dataclass(frozen=True)
class VerifierConfig:
    """L1.5 服务配置。

    timeout_s：单次批验证的强制超时。装配期断言
    timeout_s < l3_budget_s / 3（V5.2.1 §5.1）。
    """

    timeout_s: float = 1.0
    report_id_prefix: str = "vr"


@dataclass
class StubVerifierBackend:
    """确定性桩后端。

    batches：按调用次序弹出的脚本化结果（{instance_id: 结果字段}），
    列表耗尽后返回空批次；delay_s 模拟服务延迟（用于超时注入）；
    fail_with 模拟后端故障。
    """

    batches: list[dict[int, dict]] = field(default_factory=list)
    delay_s: float = 0.0
    fail_with: Exception | None = None
    calls: list[list[int]] = field(default_factory=list)

    async def verify_batch(
        self, image_id: str, instance_ids: list[int]
    ) -> list[VerificationResult]:
        self.calls.append(list(instance_ids))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with
        batch = self.batches.pop(0) if self.batches else {}
        return [
            VerificationResult.model_validate({**payload, "instance_id": iid})
            for iid, payload in batch.items()
        ]


class L15Verifier:
    """L1.5 服务门面：超时包装 + 结构化报告输出（绝不抛异常）。"""

    def __init__(
        self,
        backend: VerifierBackend | None = None,
        config: VerifierConfig | None = None,
        *,
        image_backend: ImageVerifierBackend | None = None,
    ) -> None:
        if backend is not None and image_backend is not None:
            raise ValueError("configure one verification backend")
        self._backend = backend
        self._image_backend = image_backend
        self.config = config or VerifierConfig()

    async def verify_instances(
        self,
        image_id: str,
        instance_ids: list[int],
        *,
        image: bytes | None = None,
        regions: dict[int, BBox] | None = None,
        timeout_s: float | None = None,
    ) -> VerificationReport:
        """执行一批实例验证；超时/故障 → 非成功状态报告（静默降级入口）。"""
        report_id = f"{self.config.report_id_prefix}_{uuid4().hex[:12]}"
        if not instance_ids:
            return VerificationReport(
                report_id=report_id,
                image_id=image_id,
                status=VerificationStatus.SUCCESS,
                results=[],
                elapsed_ms=0.0,
            )

        started = time.perf_counter()
        timeout = (
            self.config.timeout_s if timeout_s is None else min(timeout_s, self.config.timeout_s)
        )
        if timeout <= 0:
            return self._report(report_id, image_id, VerificationStatus.TIMEOUT, [], started)
        if self._image_backend is not None and image and regions:
            candidates = {iid: regions[iid] for iid in instance_ids if iid in regions}
            job = self._image_backend.verify_regions(image, candidates)
        elif self._backend is not None:
            job = self._backend.verify_batch(image_id, list(instance_ids))
        else:
            return self._report(report_id, image_id, VerificationStatus.DEGRADED, [], started)
        try:
            results = await asyncio.wait_for(job, timeout=timeout)
        except TimeoutError:
            return self._report(report_id, image_id, VerificationStatus.TIMEOUT, [], started)
        except Exception:
            # 后端故障与超时同路径处理（保守：不采信任何部分输出）
            return self._report(report_id, image_id, VerificationStatus.ERROR, [], started)
        requested = set(instance_ids)
        returned = {r.instance_id for r in results}
        if len(returned) != len(results) or not returned <= requested:
            return self._report(report_id, image_id, VerificationStatus.ERROR, [], started)
        results = [r for r in results if r.class_label or r.exists is not None or r.attributes]
        if not results:
            return self._report(report_id, image_id, VerificationStatus.DEGRADED, [], started)
        return self._report(report_id, image_id, VerificationStatus.SUCCESS, results, started)

    @staticmethod
    def _report(
        report_id: str,
        image_id: str,
        status: VerificationStatus,
        results: list[VerificationResult],
        started: float,
    ) -> VerificationReport:
        return VerificationReport(
            report_id=report_id,
            image_id=image_id,
            status=status,
            results=results,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
