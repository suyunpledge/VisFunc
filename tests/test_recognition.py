from __future__ import annotations

import asyncio
import base64
import io
import json
from hashlib import sha256

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from svsg.api import build_recognizer, create_app
from svsg.config import Settings
from svsg.contracts.recognition import FrameReference, ModelCall, ReviewObjects, SceneObjects
from svsg.recognition import RecognitionService
from svsg.recognition.geometry import context_box, iou, pixel_box
from svsg.recognition.images import canonical_raster
from svsg.recognition.provider import OpenAIRecognizer, ProviderOutputError


def picture():
    with Image.new("RGBA", (100, 80), "white") as img, io.BytesIO() as output:
        img.paste((255, 0, 0, 255), (10, 20, 50, 60))
        img.save(output, "PNG")
        return output.getvalue()


class Backend:
    model = "test-model"

    def __init__(self, review_label="cup", box=(150, 150, 850, 850), *, fail=None):
        self.review_label, self.box, self.fail = review_label, box, fail
        self.seen = []

    async def infer(self, stage, content, labels, max_objects):
        self.seen.append((stage, content, labels))
        if stage == self.fail:
            raise TimeoutError()
        if stage == "detect":
            result = SceneObjects.model_validate(
                {
                    "objects": [
                        {"label": "cup", "bbox": [100, 250, 500, 750]},
                    ]
                }
            )
        else:
            result = ReviewObjects.model_validate(
                {
                    "results": [
                        {
                            "candidate_id": 1,
                            "objects": [{"label": self.review_label, "bbox": self.box}],
                        },
                    ]
                }
            )
        return result, ModelCall(stage=stage, model=self.model, elapsed_ms=1)


def test_multimodal_api_blind_review_and_frame_provenance():
    backend = Backend()
    service = RecognitionService(backend)
    app = create_app(settings=Settings(_env_file=None), recognizer=service)
    data = picture()
    with TestClient(app) as client:
        response = client.post(
            "/v1/recognize-image",
            files={"file": ("frame.png", data)},
            data={
                "labels": '["cup","dog"]',
                "frame": '{"sequence_id":"clip","frame_index":7,"timestamp_ms":241.5}',
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "corroborated"
    assert result["corroborated_counts"] == {"cup": 1}
    assert result["image_sha256"] == sha256(data).hexdigest()
    assert result["pixel_sha256"] == result["image_functions"]["pixel_sha256"]
    assert result["frame"]["timestamp_ms"] == 241.5
    assert result["id_scope"] == "image_observation_not_track"
    assert result["coverage"] == "not_exhaustive"
    observation = result["observations"][0]
    assert observation["bbox_px"] == [10, 20, 50, 60]
    assert observation["crop_mean_rgba"] == [255, 0, 0, 255]
    assert observation["review_iou"] >= 0.5
    review_parts = backend.seen[1][1]
    assert review_parts[0]["text"] == "candidate_id=1"  # No proposed label/answer leakage.
    url = review_parts[1]["image_url"]["url"]
    with Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2]))) as crop:
        assert crop.size == (52, 52)
    assert response.headers["X-SVSG-Trace-Id"]


@pytest.mark.parametrize(
    "label,box,reason",
    [
        ("dog", (150, 150, 850, 850), "label_disagreement"),
        ("cup", (0, 0, 150, 150), "localization_disagreement"),
    ],
)
def test_wrong_class_or_location_never_enters_counts(label, box, reason):
    result = asyncio.run(RecognitionService(Backend(label, box)).recognize(picture()))
    assert result.corroborated_counts == {}
    assert result.observations[0].reason == reason
    assert result.status == "needs_review"
    assert result.relations == []


@pytest.mark.parametrize("stage,status", [("detect", "unavailable"), ("review", "needs_review")])
def test_provider_failures_are_visible_and_never_confirmed(stage, status):
    result = asyncio.run(RecognitionService(Backend(fail=stage)).recognize(picture()))
    assert result.status == status
    assert result.corroborated_counts == {}
    assert result.calls[-1].error == "TimeoutError"


@pytest.mark.parametrize("ids", [[99], [1, 1], []])
def test_missing_foreign_and_duplicate_review_ids_fail_closed(ids):
    class BadReview(Backend):
        async def infer(self, stage, content, labels, max_objects):
            if stage == "detect":
                return await super().infer(stage, content, labels, max_objects)
            result = ReviewObjects.model_validate(
                {
                    "results": [
                        {
                            "candidate_id": cid,
                            "objects": [{"label": "cup", "bbox": [0, 0, 1000, 1000]}],
                        }
                        for cid in ids
                    ]
                }
            )
            return result, ModelCall(stage=stage, model=self.model, elapsed_ms=1)

    result = asyncio.run(RecognitionService(BadReview()).recognize(picture()))
    assert result.corroborated_counts == {}
    assert result.status == "needs_review"


def test_no_proposals_does_not_claim_absence():
    class Empty(Backend):
        async def infer(self, stage, content, labels, max_objects):
            assert stage == "detect"
            return SceneObjects(objects=[]), ModelCall(stage=stage, model=self.model, elapsed_ms=1)

    result = asyncio.run(RecognitionService(Empty()).recognize(picture()))
    assert result.status == "needs_review"
    assert result.corroborated_counts == {}
    assert "未检出不代表不存在" in result.final_answer
    assert len(result.calls) == 1


def test_scope_and_duplicates_cannot_inflate_counts():
    class Duplicate(Backend):
        async def infer(self, stage, content, labels, max_objects):
            if stage == "detect":
                result = SceneObjects.model_validate(
                    {
                        "objects": [
                            {"label": label, "bbox": [100, 250, 500, 750]}
                            for label in ("cup", "cup", "dog")
                        ]
                    }
                )
                return result, ModelCall(stage=stage, model=self.model, elapsed_ms=1)
            return await super().infer(stage, content, labels, max_objects)

    result = asyncio.run(RecognitionService(Duplicate()).recognize(picture(), labels=["cup"]))
    assert result.corroborated_counts == {"cup": 1}
    assert [o.reason for o in result.observations] == [
        "label_and_box_agree",
        "duplicate_proposal",
        "outside_label_scope",
    ]


def test_exif_orientation_and_transparency_are_explicit():
    with Image.new("RGB", (30, 20), "red") as img, io.BytesIO() as buf:
        exif = img.getexif()
        exif[274] = 6
        img.save(buf, "JPEG", exif=exif)
        raster = canonical_raster(buf.getvalue())
    assert (raster.width, raster.height) == (20, 30)
    assert pixel_box((100, 200, 900, 800), (10, 20, 30, 50)) == (12, 26, 28, 44)
    assert context_box((0, 0, 100, 80), 100, 80) == (0, 0, 100, 80)
    assert iou((0, 0, 20, 20), (10, 0, 30, 20)) == pytest.approx(1 / 3)
    from svsg.recognition.images import image_part

    with Image.new("RGBA", (8, 8), (255, 0, 0, 0)) as transparent:
        part = image_part(transparent, 64)
    with Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",")[1]))) as sent:
        assert sent.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize(
    "bbox",
    [[-1, 0, 500, 500], [0, 0, 1001, 100], [True, 0, 500, 500], [0, 0, 0, 100], [0.1, 0, 500, 500]],
)
def test_bad_model_boxes_rejected(bbox):
    with pytest.raises(ValidationError):
        SceneObjects.model_validate({"objects": [{"label": "cup", "bbox": bbox}]})


def test_provider_rejects_truncation_and_does_not_retry():
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "length", "message": {"content": '{"objects":[]}'}}]
            },
        )

    backend = OpenAIRecognizer(
        "https://example.invalid/v1", "test", "test", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderOutputError) as exc:
        asyncio.run(backend.infer("detect", [], [], 8))
    assert exc.value.call.error == "invalid_model_output"
    assert len(requests) == 1


def test_cropped_fragment_is_not_corrobated_as_a_whole_object():
    result = asyncio.run(RecognitionService(Backend(box=(0, 0, 1000, 1000))).recognize(picture()))
    assert result.corroborated_counts == {}
    assert result.observations[0].reason == "review_object_clipped_by_crop"


@pytest.mark.parametrize("invalid", [False, True])
def test_bare_array_dialect_keeps_strict_boxes_and_error_usage(invalid):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                [{"label": "cup", "bbox": [0, 0, 1001 if invalid else 1000, 1000]}]
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 25},
            },
        )

    backend = OpenAIRecognizer(
        "https://example.invalid/v1", "test", "test", transport=httpx.MockTransport(handler)
    )
    if invalid:
        with pytest.raises(ProviderOutputError) as exc:
            asyncio.run(backend.infer("detect", [], [], 8))
        assert exc.value.call.completion_tokens == 25
        assert exc.value.call.validation_errors == ["objects.0.bbox.2:less_than_equal"]
    else:
        result, call = asyncio.run(backend.infer("detect", [], [], 8))
        assert result.objects[0].label == "cup"
        assert call.output_format == "bare_array"


def test_required_backend_auth_and_frame_validation():
    with TestClient(create_app(settings=Settings(_env_file=None))) as client:
        assert (
            client.post("/v1/recognize-image", files={"file": ("f.png", picture())}).status_code
            == 503
        )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vision_provider="openai")
    with pytest.raises(ValidationError):
        FrameReference(sequence_id="clip", frame_index=1, timestamp_ms=float("nan"))
    app = create_app(
        settings=Settings(_env_file=None, api_key="test"), recognizer=RecognitionService(Backend())
    )
    with TestClient(app) as client:
        assert (
            client.post("/v1/recognize-image", files={"file": ("f.png", picture())}).status_code
            == 401
        )
        response = client.post(
            "/v1/recognize-image",
            files={"file": ("f.png", picture())},
            headers={"X-API-Key": "test"},
            data={"frame": '{"frame_index":1}'},
        )
        assert response.status_code == 422


def test_separate_review_model_configuration():
    settings = Settings(
        _env_file=None,
        vision_provider="openai",
        vision_base_url="https://example.invalid/v1",
        vision_api_key="test",
        vision_model="first",
        vision_review_model="second",
    )
    service = build_recognizer(settings)
    assert service.backend.model == "first"
    assert service.reviewer.model == "second"


@pytest.mark.parametrize("bad", [False, True])
def test_bbox_dialect_and_single_envelope_never_repair_invalid_coordinates(bad):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                [
                                    {
                                        "objects": [
                                            {
                                                "label": "cup",
                                                "bbox_2d": [-1 if bad else 0, 0, 500, 500],
                                            }
                                        ]
                                    }
                                ]
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    backend = OpenAIRecognizer(
        "https://example.invalid/v1", "test", "test", transport=httpx.MockTransport(handler)
    )
    if bad:
        with pytest.raises(ProviderOutputError) as exc:
            asyncio.run(backend.infer("detect", [], [], 8))
        assert "greater_than_equal" in exc.value.call.validation_errors[0]
        assert exc.value.raw_response
    else:
        result, call = asyncio.run(backend.infer("detect", [], [], 8))
        assert result.objects[0].bbox == (0, 0, 500, 500)
        assert call.normalizations == ["single_wrapped_envelope", "bbox_2d"]


def test_json_mode_is_explicit_and_does_not_replace_schema_validation():
    def handler(request):
        assert json.loads(request.content)["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"objects":[{"label":"cup"}]}'},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    backend = OpenAIRecognizer(
        "https://example.invalid/v1",
        "test",
        "test",
        json_mode=True,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderOutputError) as exc:
        asyncio.run(backend.infer("detect", [], [], 8))
    assert exc.value.call.response_format == "json_object"
    assert exc.value.call.validation_errors == ["objects.0.bbox:missing"]
