from __future__ import annotations

import base64
import io
import json
from hashlib import sha256

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from svsg.api import create_app
from svsg.config import Settings
from svsg.contracts import Claim, L3Answer
from svsg.l1_compiler import L1Compiler, RawDetection, StubDetector
from svsg.l3_orchestrator import Orchestrator, OrchestratorConfig, StubLLM
from svsg.l15_verifier import L15Verifier, VerifierConfig
from svsg.l15_verifier.openai_vision import OpenAIVisionVerifier


def _test_image() -> bytes:
    image = Image.new("RGB", (80, 64), "blue")
    image.paste((255, 0, 0), (8, 8, 32, 32))
    with io.BytesIO() as buf:
        image.save(buf, "PNG")
        return buf.getvalue()


@pytest.mark.parametrize("provider_id", [1, 2, None])
def test_image_api_sends_requested_crop_and_tracks_verified_regions(provider_id):
    requests = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        parts = json.loads(request.content)["messages"][1]["content"]
        assert parts[0]["text"] == "Crop for instance_id 1"
        url = parts[1]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2]))) as crop:
            assert crop.size == (24, 24)
            assert crop.getpixel((12, 12))[0] >= 240
            assert crop.getpixel((12, 12))[2] <= 12
        results = []
        if provider_id is not None:
            results.append({"instance_id": provider_id, "class": "card", "exists": True})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "results": results,
                                }
                            )
                        },
                    }
                ],
            },
        )

    detection = RawDetection("card", (8, 8, 32, 32), detection_score=0.55, local_edge_sharpness=0.1)
    compiler = L1Compiler(StubDetector([detection]))
    image_backend = OpenAIVisionVerifier(
        base_url="https://example.invalid/v1",
        api_key="test-only",
        model="vision-test",
        timeout_s=1,
        transport=httpx.MockTransport(provider),
    )
    verifier = L15Verifier(config=VerifierConfig(timeout_s=0.25), image_backend=image_backend)
    llm = StubLLM(
        answers=[
            L3Answer(
                claims=[Claim(instance_id=1, field="class", value="card")], final_answer="untrusted"
            )
        ]
    )
    orch = Orchestrator(verifier, llm, OrchestratorConfig(llm_timeout_s=1.0))
    app = create_app(orchestrator=orch, compiler=compiler, settings=Settings(_env_file=None))
    image = _test_image()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/analyze-image",
            files={"file": ("frame.png", image, "image/png")},
            data={"query": "对象是什么？"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_sha256"] == sha256(image).hexdigest()
    assert body["evidence_mode"] == "image"
    assert len(requests) == 1
    if provider_id == 1:
        assert body["status"] == "delivered"
        assert body["verified_instances"] == [1]
    else:
        assert body["status"] == "delivered_with_review"
        assert body["verified_instances"] == []
        assert body["degraded_instances"] == [1]
        assert "待人工复审" in body["final_answer"]
