from __future__ import annotations

import asyncio
import base64
import io
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from svsg.api import create_app
from svsg.config import Settings
from svsg.contracts.recognition import ModelCall, SceneObjects, VisualObject
from svsg.recognition.adaptive import AdaptiveRecognitionService
from svsg.recognition.consensus import Candidate, associate
from svsg.recognition.views import View, discovery_views, focus_small_object


def picture():
    with Image.new("RGBA", (100, 80), "red") as image, io.BytesIO() as buf:
        image.save(buf, "PNG")
        return buf.getvalue()


def obj(label="cup", box=(100, 250, 500, 750)):
    return VisualObject(label=label, bbox=box)


class Backend:
    def __init__(self, model, scenes):
        self.model, self.scenes, self.seen = model, scenes, []

    async def infer(self, stage, content, labels, max_objects):
        self.seen.append(content)
        scene = self.scenes.pop(0)
        if isinstance(scene, Exception):
            raise scene
        assert len([p for p in content if p["type"] == "image_url"]) == 1
        return SceneObjects(objects=scene), ModelCall(stage=stage, model=self.model, elapsed_ms=1)


def test_cross_view_api_restores_original_coordinates_and_upsamples_only_model_input():
    first, second = (
        Backend("first", [[obj()]]),
        Backend("second", [[obj(box=(500, 250, 900, 750))]]),
    )
    service = AdaptiveRecognitionService(first, second, max_calls=2, discovery_count=0)
    app = create_app(settings=Settings(_env_file=None), recognizer=service)
    data = picture()
    with TestClient(app) as client:
        response = client.post(
            "/v1/recognize-image", files={"file": ("f.png", data)}, data={"labels": '["cup"]'}
        )
    assert response.status_code == 200
    result = response.json()
    assert result["corroborated_counts"] == {"cup": 1}
    assert result["observations"][0]["bbox_px"] == [10, 20, 50, 60]
    assert result["pipeline"] == "multiview"
    assert result["width"] == result["image_functions"]["width_px"] == 100
    assert result["image_sha256"] == sha256(data).hexdigest()
    assert result["views"][1]["mirrored"]
    assert len(result["observations"][0]["supports"]) == 2
    encoded = base64.b64decode(first.seen[0][1]["image_url"]["url"].split(",")[1])
    assert sha256(encoded).hexdigest() == result["views"][0]["encoded_sha256"]
    with Image.open(io.BytesIO(encoded)) as sent:
        assert sent.width == 768


def test_small_object_absent_in_both_overviews_can_be_discovered_and_confirmed():
    first = Backend("first", [[], [obj(box=(200, 200, 400, 400))]])
    second = Backend("second", [[], [obj(box=(310, 333, 619, 666))]])
    result = asyncio.run(
        AdaptiveRecognitionService(
            first,
            second,
            max_calls=4,
            discovery_count=1,
        ).recognize(picture())
    )
    assert result.corroborated_counts == {"cup": 1}
    assert result.observations[0].origin == "discovery"
    assert [c.purpose for c in result.calls] == [
        "primary",
        "cross_view",
        "discovery",
        "confirmation",
    ]
    assert result.observations[0].review_iou >= 0.5


def test_repeated_detector_without_reviewer_agreement_cannot_pass():
    first = Backend("first", [[obj()], [obj(box=(154, 250, 769, 750))]])
    second = Backend("second", [[]])
    result = asyncio.run(
        AdaptiveRecognitionService(first, second, max_calls=3, discovery_count=1).recognize(
            picture()
        )
    )
    assert result.corroborated_counts == {}
    assert len(result.calls) == 3
    assert any("budget" in warning for warning in result.warnings)


def test_wrong_label_is_not_confirmed_even_when_location_agrees():
    first = Backend("first", [[obj("person")]])
    second = Backend("second", [[obj("tower", (500, 250, 900, 750))]])
    result = asyncio.run(
        AdaptiveRecognitionService(first, second, max_calls=2, discovery_count=0).recognize(
            picture(), labels=["person"]
        )
    )
    assert result.corroborated_counts == {}
    assert result.views[1].objects[0].label == "tower"


def test_internal_tile_boundary_cannot_confirm_full_object_extent():
    view = View("tile", "discovery", (0, 0, 65, 80))
    hit = view.map_object(obj(box=(100, 250, 1000, 750)), "first", 100, 80)
    assert hit.clipped
    group = Candidate([hit, hit.model_copy(update={"view_id": "other", "model": "second"})])
    assert group.agreement(0.5, "second", True) is None
    at_source_edge = View("full", "primary", (0, 0, 100, 80)).map_object(
        obj(box=(0, 250, 500, 750)), "first", 100, 80
    )
    assert not at_source_edge.clipped


def test_association_does_not_confirm_duplicate_outputs_from_one_view():
    view = View("one", "primary", (0, 0, 100, 80))
    hit = view.map_object(obj(), "first", 100, 80)
    groups = []
    associate(groups, [hit, hit], 0.5)
    assert len(groups) == 1
    assert groups[0].agreement(0.5, "first", False) is None
    first = hit.model_copy(update={"evidence_key": "same-encoded-pixels"})
    repeat = first.model_copy(update={"view_id": "renamed-view"})
    assert Candidate([first, repeat]).agreement(0.5, "first", False) is None


def test_empty_and_failed_views_are_not_absence_evidence_and_no_retry():
    first, second = Backend("first", [TimeoutError()]), Backend("second", [[]])
    result = asyncio.run(
        AdaptiveRecognitionService(first, second, max_calls=2, discovery_count=0).recognize(
            picture()
        )
    )
    assert result.status == "needs_review"
    assert result.corroborated_counts == {}
    assert result.calls[0].error == "TimeoutError"
    assert len(first.seen) == 1
    assert "未检出不代表不存在" in result.final_answer


def test_discovery_covers_both_axes_and_invalid_budgets_rejected():
    for width, height in ((100, 80), (80, 100)):
        views = discovery_views(width, height)
        assert views[0].box[:2] == (0, 0)
        assert views[-1].box[2:] == (width, height)
        assert views[0].box[2] >= views[1].box[0]
        assert views[0].box[3] >= views[1].box[1]
    with pytest.raises(ValueError):
        AdaptiveRecognitionService(Backend("first", []), max_calls=1)


def test_total_deadline_cancels_network_and_stops_remaining_views():
    class Slow(Backend):
        async def infer(self, stage, content, labels, max_objects):
            self.seen.append(content)
            await asyncio.sleep(2)
            raise AssertionError("should have been cancelled")

    first = Slow("first", [])
    second = Backend("second", [])
    result = asyncio.run(
        AdaptiveRecognitionService(first, second, total_timeout_s=0.3).recognize(picture())
    )
    assert result.status == "unavailable"
    assert len(result.calls) == 1
    assert result.calls[0].error == "TimeoutError"
    assert second.seen == []


def test_tiny_object_zoom_expands_neighbourhood_without_changing_source_coordinates():
    focused = focus_small_object(640, 480, [(320, 300, 330, 320)])
    assert focused is not None
    assert focused.box == (261, 246, 389, 374)
    assert focus_small_object(640, 480, [(0, 0, 500, 400)]) is None
    edge = focus_small_object(640, 480, [(635, 475, 640, 480)])
    assert edge.box == (512, 352, 640, 480)
