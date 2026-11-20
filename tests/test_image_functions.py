from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from svsg.api import create_app
from svsg.config import Settings
from svsg.contracts import Claim, L3Answer
from svsg.contracts.image_functions import ImageFunctionEvidence
from svsg.image_functions.claims import check_function_claim
from svsg.image_functions.compiler import describe_raster
from svsg.image_functions.raster import RasterFunction
from svsg.image_functions.spatial import evaluate_relation
from svsg.l3_orchestrator import Orchestrator, StubLLM
from svsg.l15_verifier import L15Verifier, StubVerifierBackend, VerifierConfig


def _shape() -> tuple[bytes, bytes]:
    with Image.new("RGBA", (7, 6), (0, 0, 255, 255)) as image, BytesIO() as buf:
        for y in range(1, 4):
            for x in range(2, 5):
                if (x, y) != (3, 2):
                    image.putpixel((x, y), (255, 0, 0, 128))
        image.save(buf, "PNG")
        return buf.getvalue(), image.tobytes()


def test_pixel_digest_and_samples_remain_bound_to_the_same_bytes():
    buffer = bytearray([10, 20, 30, 40])
    raster = RasterFunction(1, 1, buffer)
    original = raster.pixel_sha256
    buffer[0] = 100
    assert raster.pixel(0, 0) == (10, 20, 30, 40)
    assert raster.pixel_sha256 == original == sha256(raster.samples).hexdigest()


def test_raster_round_trip_preserves_all_channels_and_validates_hash():
    source, pixels = _shape()
    raster = RasterFunction.from_image(source)
    program = raster.to_program()
    assert "row_intervals" in program
    restored = RasterFunction.from_program(json.loads(json.dumps(program)))
    assert restored.samples == pixels
    assert restored.pixel(2, 1) == (255, 0, 0, 128)
    assert restored.pixel(3, 2) == (0, 0, 255, 255)
    with pytest.raises(ValueError, match="digest mismatch"):
        RasterFunction.from_program({**program, "pixel_sha256": "0" * 64})
    with pytest.raises(ValueError, match="payload"):
        RasterFunction.from_program({**program, "width_px": 1})


def test_dense_texture_uses_packed_exact_raster_when_summary_exceeds_budgets():
    pixels = b"".join((0xFF000000 | i).to_bytes(4, "little") for i in range(100 * 90))
    raster = RasterFunction(100, 90, pixels)
    program = raster.to_program()
    assert "rgba8_zlib_base64" in program
    assert RasterFunction.from_program(program).samples == pixels
    evidence = describe_raster(raster, span_limit=4)
    assert evidence.unique_colors is None and not evidence.top_colors
    assert evidence.component_count is None and not evidence.regions


def test_region_parameters_include_hole_and_four_connected_topology():
    source, _ = _shape()
    evidence = describe_raster(RasterFunction.from_image(source))
    assert evidence.component_count == 3  # Outer blue, red ring, isolated blue center.
    assert evidence.unique_colors == 2
    assert evidence.top_colors[0].area_px == 34
    red = next(r for r in evidence.regions if r.rgba == "#FF000080")
    assert red.box == (2, 1, 5, 4)
    assert red.area_px == 8
    assert red.perimeter_px == 16
    assert red.centroid_x == 3.5 and red.centroid_y == 2.5
    assert red.fill_ratio == pytest.approx(8 / 9)
    assert red.shape == "connected_region"


def test_contour_replays_shape_statistics_and_spatial_relations():
    source, _ = _shape()
    evidence = describe_raster(RasterFunction.from_image(source))
    evidence = ImageFunctionEvidence.model_validate_json(evidence.model_dump_json())
    outer, red, center = evidence.regions
    contour = red.contour
    assert contour is not None
    assert contour.holes == 1 and red.outline_kind == "holed_region"
    assert contour.area2 == 16 and contour.perimeter == 16
    assert contour.centroid == (3.5, 2.5)
    assert contour.solidity == pytest.approx(8 / 9)
    assert contour.axes == (None, 1.0)
    assert evaluate_relation(outer, red, "shared_edge_px") == 12
    assert evaluate_relation(red, center, "shared_edge_px") == 4
    assert evaluate_relation(red, center, "bbox_contains") is True
    assert evaluate_relation(red, center, "bbox_iou") == pytest.approx(1 / 9)
    assert evaluate_relation(red, center, "center_distance_px") == 0


def test_thin_rectangle_has_direction_and_corner_contact_is_not_adjacency():
    samples = bytearray(bytes((0, 0, 0, 255)) * 49)
    for x in range(1, 6):
        i = (2 * 7 + x) * 4
        samples[i : i + 4] = bytes((255, 0, 0, 255))
    i = (3 * 7 + 6) * 4
    samples[i : i + 4] = bytes((0, 255, 0, 255))
    evidence = describe_raster(RasterFunction(7, 7, bytes(samples)))
    _, line, dot = evidence.regions
    assert line.contour is not None
    assert line.outline_kind == "filled_rectangle"
    assert line.contour.axes == pytest.approx((0, 5))
    assert evaluate_relation(line, dot, "left_of") is True
    assert evaluate_relation(line, dot, "touching") is False


def test_quantized_spatial_claim_is_scoped_to_its_own_grid():
    source, _ = _shape()
    evidence = describe_raster(RasterFunction.from_image(source), span_limit=1)
    assert evidence.component_count is None and not evidence.regions
    assert evidence.quantized is not None
    first, second = evidence.quantized.regions[:2]
    distance = evaluate_relation(first, second, "center_distance_px")
    claim = Claim(
        field="function:spatial:center_distance_px",
        region_id=first.id,
        target_region_id=second.id,
        region_basis="quantized",
        value=distance,
    )
    assert check_function_claim(evidence, claim) is None
    assert check_function_claim(evidence, claim.model_copy(update={"region_basis": "exact"}))
    assert check_function_claim(evidence, claim.model_copy(update={"target_region_id": first.id}))


def test_interval_budget_keeps_exact_raster_and_marks_region_count_unknown():
    source, pixels = _shape()
    raster = RasterFunction.from_image(source)
    evidence = describe_raster(raster, span_limit=1)
    assert evidence.component_count is None and not evidence.regions
    assert evidence.pixel_sha256 == sha256(pixels).hexdigest()


def test_compiler_endpoint_runs_without_models_and_accepts_bounded_calls():
    source, pixels = _shape()
    queries = [
        {"id": "sample", "op": "pixel_rgba", "x": 2, "y": 1},
        {"id": "mean", "op": "mean_rgba", "box": [2, 1, 5, 4]},
    ]
    with TestClient(create_app(settings=Settings(_env_file=None))) as client:
        result = client.post(
            "/v1/compile-functions",
            files={"file": ("ring.png", source, "image/png")},
            data={"queries": json.dumps(queries)},
        )
        bad = client.post(
            "/v1/compile-functions",
            files={"file": ("ring.png", source, "image/png")},
            data={"queries": '[{"id":"bad","op":"pixel_rgba","x":7,"y":0}]'},
        )
    assert result.status_code == 200
    ir = result.json()
    assert ir["image_id"] == "sha256:" + sha256(source).hexdigest()
    assert ir["detections"] == []
    evidence = ir["image_functions"]
    assert evidence["pixel_sha256"] == sha256(pixels).hexdigest()
    assert [m["value"] for m in evidence["measurements"]] == [
        "#FF000080",
        "226.67,0.00,28.33,142.11",
    ]
    assert bad.status_code == 422


def test_function_claims_use_the_same_parameters_in_prompt_validation_and_delivery():
    source, _ = _shape()
    wrong = L3Answer(
        claims=[Claim(field="count", value=0), Claim(field="function:component_count", value=8)],
        final_answer="untrusted",
    )
    good = L3Answer(
        claims=[
            Claim(field="function:component_count", value=3),
            Claim(field="function:perimeter_px", region_id=2, value=16),
            Claim(field="function:measurement:sample", value="#FF000080"),
        ],
        final_answer="untrusted",
    )
    llm = StubLLM(answers=[wrong, good])
    backend = StubVerifierBackend()
    orch = Orchestrator(L15Verifier(backend, VerifierConfig(timeout_s=0.1)), llm)
    app = create_app(orchestrator=orch, settings=Settings(detector="functions", _env_file=None))
    with TestClient(app) as client:
        result = client.post(
            "/v1/analyze-image",
            files={"file": ("ring.png", source, "image/png")},
            data={
                "query": "测量周长和颜色",
                "queries": json.dumps(
                    [
                        {"id": "sample", "op": "pixel_rgba", "x": 2, "y": 1},
                    ]
                ),
            },
        )
    assert result.status_code == 200
    answer = result.json()
    assert answer["evidence_mode"] == "functions" and answer["status"] == "delivered"
    assert "16" in answer["final_answer"] and "#FF000080" in answer["final_answer"]
    assert "untrusted" not in answer["final_answer"]
    assert backend.calls == []
    first, retry = (json.loads(call["context"]) for call in llm.calls)
    assert "rgba8_zlib_base64" not in llm.calls[0]["context"]
    assert "rings" not in llm.calls[0]["context"]
    assert first["image_functions"]["regions"][1]["perimeter_px"] == 16
    assert len(retry["corrective_feedback"]["violations"]) == 2
