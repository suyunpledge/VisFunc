"""FastAPI 入口测试：请求封装、错误映射、trace 头、三档采样留痕。"""

from __future__ import annotations

import random
import sys

import pytest
from fastapi.testclient import TestClient

from svsg.api import create_app
from svsg.contracts import L3Answer
from svsg.l3_orchestrator import Orchestrator, OrchestratorConfig, StubLLM
from svsg.l15_verifier import L15Verifier, StubVerifierBackend, VerifierConfig
from svsg.observability import SampleMode, SamplingPolicy, TraceStore


def _payload(*, verify2: bool = True):
    return {
        "schema_version": "5.2.1",
        "image_id": "img_api_001",
        "image_width": 1920,
        "image_height": 1080,
        "timestamp": "2026-09-02T10:00:00Z",
        "scene_type": "orthographic",
        "detector_model": "stub_detector_v0",
        "detections": [
            {
                "instance_id": 1,
                "class": "screw",
                "bbox_px": [420, 150, 480, 210],
                "detection_score": 0.92,
                "conf_level": "high",
                "verification_required": False,
                "uncertainty": {
                    "classification_entropy": 0.08,
                    "local_edge_sharpness": 0.73,
                    "tta_variance": None,
                },
                "relations": [],
            },
            {
                "instance_id": 2,
                "class": "screw",
                "bbox_px": [900, 400, 1000, 500],
                "detection_score": 0.55,
                "conf_level": "low",
                "verification_required": verify2,
                "uncertainty": {
                    "classification_entropy": 0.6,
                    "local_edge_sharpness": 0.3,
                    "tta_variance": None,
                },
                "relations": [],
            },
            {
                "instance_id": 3,
                "class": "washer",
                "bbox_px": [1100, 400, 1160, 460],
                "detection_score": 0.88,
                "conf_level": "high",
                "verification_required": False,
                "uncertainty": {
                    "classification_entropy": 0.1,
                    "local_edge_sharpness": 0.8,
                    "tta_variance": None,
                },
                "relations": [],
            },
        ],
    }


def _answer(claims, text="答案文本"):
    return L3Answer.model_validate({"claims": claims, "final_answer": text})


def _client(answers, backend=None):
    verifier = L15Verifier(
        backend or StubVerifierBackend(), VerifierConfig(timeout_s=0.05)
    )
    llm = StubLLM(answers=answers)
    orch = Orchestrator(verifier, llm, OrchestratorConfig(llm_timeout_s=0.5))
    app = create_app(orchestrator=orch)
    return TestClient(app), app


def test_analyze_delivers_with_trace_header():
    claims = [{"instance_id": 1, "field": "class", "value": "screw"},
              {"field": "count", "value": 3}]
    client, _ = _client([_answer(claims, text="画面里有 999 颗螺丝。")])
    resp = client.post(
        "/v1/analyze", json={"query": "图里有什么", "ir": _payload(verify2=False)}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "delivered"
    assert body["evidence_mode"] == "submitted_ir"
    assert '实例 1 的类别为 "screw"。' in body["final_answer"]
    assert "3 个有效实例" in body["final_answer"]
    assert "999" not in body["final_answer"]
    assert body["human_review_required"] is False
    assert body["claims"][0]["value"] == "screw"
    assert body["trace_id"]
    # 响应头回传 trace_id，供调用方关联日志与采样记录
    assert resp.headers["X-SVSG-Trace-Id"] == body["trace_id"]


@pytest.mark.parametrize("claims", [
    [],
    [{"field": "class", "value": "screw"}],
    [{"instance_id": 1, "field": "material", "value": "gold"}],
    [{"instance_id": 1, "field": "relation:unknown", "value": 2}],
    [{"instance_id": 1, "field": "exists", "value": "true"}],
    [{"field": "count", "value": True}],
])
def test_invalid_claims_go_to_review(claims):
    client, _ = _client([_answer(claims), _answer(claims)])
    response = client.post("/v1/analyze", json={"ir": _payload(verify2=False), "query": "q"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "human_review"
    assert body["final_answer"] is None
    assert body["anchor_violations"]


def test_distance_and_nearest_recompute_pixels_despite_untrusted_draft():
    claims = [
        {"instance_id": 2, "field": "relation:distance_to", "value": 3},
        {"instance_id": 2, "field": "relation:nearest_to", "value": 3},
    ]
    client, _ = _client([_answer(claims, "两个物体相距一米")])
    response = client.post("/v1/analyze", json={"ir": _payload(verify2=False), "query": "q"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "delivered"
    assert [c["distance_px"] for c in body["claims"]] == [pytest.approx(181.11)] * 2
    assert "像素" in body["final_answer"] and "一米" not in body["final_answer"]


def test_wrong_distance_and_nearest_send_feedback_for_regeneration():
    bad = [
        {"instance_id": 2, "field": "relation:distance_to", "value": 3, "distance_px": 12},
        {"instance_id": 2, "field": "relation:nearest_to", "value": 1},
    ]
    good = [{"instance_id": 2, "field": "relation:nearest_to", "value": 3}]
    client, _ = _client([_answer(bad), _answer(good)])
    response = client.post("/v1/analyze", json={"ir": _payload(verify2=False), "query": "q"})
    assert response.status_code == 200
    assert response.json()["status"] == "delivered"
    assert response.json()["claims"][0]["value"] == 3


def test_rejected_detection_is_excluded_from_count_and_relations():
    backend = StubVerifierBackend(batches=[{2: {"exists": False}}])
    invalid = [
        {"field": "count:screw", "value": 2},
        {"instance_id": 1, "field": "relation:nearest_to", "value": 2},
    ]
    valid = [
        {"field": "count", "value": 2},
        {"field": "count:screw", "value": 1},
        {"instance_id": 2, "field": "exists", "value": False},
        {"instance_id": 1, "field": "relation:nearest_to", "value": 3},
    ]
    client, _ = _client([_answer(invalid), _answer(valid)], backend)
    body = client.post("/v1/analyze", json={"ir": _payload(), "query": "q"}).json()
    assert body["status"] == "delivered"
    assert body["verified_instances"] == [2]
    assert [claim["value"] for claim in body["claims"]] == [2, 1, False, 3]


def test_analyze_rejected_is_business_level_200():
    client, _ = _client([])
    bad = _payload(verify2=False)
    bad["detections"][0]["bbox_px"] = [10, 10, 50, 5000]
    resp = client.post("/v1/analyze", json={"query": "q", "ir": bad})
    assert resp.status_code == 200  # 业务级结果，非 HTTP 错误
    body = resp.json()
    assert body["status"] == "rejected"
    assert body["error"]["code"] == "E1002"
    assert body["final_answer"] is None


def test_analyze_degraded_marks_human_review():
    backend = StubVerifierBackend(batches=[{2: {"class": "screw"}}], delay_s=0.3)
    claims = [{"field": "count", "value": 3}]
    client, _ = _client([_answer(claims)], backend)
    resp = client.post(
        "/v1/analyze", json={"query": "描述一下", "ir": _payload()}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "delivered_with_review"
    assert body["human_review_required"] is True
    assert body["degraded_instances"] == [2]


def test_anomaly_always_recorded_in_steady_mode():
    backend = StubVerifierBackend(batches=[{2: {"class": "screw"}}], delay_s=0.3)
    client, app = _client([_answer([{"field": "count", "value": 3}])], backend)
    # 稳态 + 费率 0：正常流量全丢，异常（降级）恒采样
    app.state.trace_store = TraceStore(
        SamplingPolicy(SampleMode.STEADY, steady_rate=0.0, rng=random.Random(0))
    )
    client.post("/v1/analyze", json={"query": "q", "ir": _payload()})
    retained = app.state.trace_store.retained()
    assert len(retained) == 1
    assert retained[0]["anomaly"] is True
    assert retained[0]["session"]["final_state"] == "S6"
    assert retained[0]["violations"] == []


def test_normal_traffic_dropped_in_steady_mode():
    claims = [{"field": "count", "value": 3}]
    client, app = _client([_answer(claims)])
    app.state.trace_store = TraceStore(
        SamplingPolicy(SampleMode.STEADY, rng=random.Random(42))
    )
    client.post(
        "/v1/analyze", json={"query": "q", "ir": _payload(verify2=False)}
    )
    assert app.state.trace_store.retained() == []
    assert app.state.trace_store.dropped == 1


def test_request_validation_returns_422():
    client, _ = _client([])
    resp = client.post("/v1/analyze", json={"query": "", "ir": {}})
    assert resp.status_code == 422  # FastAPI 请求模型校验


def test_healthz():
    client, _ = _client([])
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["version"] == "5.2.1"


if __name__ == "__main__":
    failures = 0
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
