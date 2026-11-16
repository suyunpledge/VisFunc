from __future__ import annotations

import copy
import json
import math
from hashlib import sha256

import pytest
from PIL import Image
from pydantic import ValidationError

from svsg.contracts.recognition import FrameReference, RecognitionResult
from svsg.contracts.video_functions import FittedProgress, SampledFunction, VideoFunctions
from svsg.video_functions import VideoRuntime, attach_recognition, evaluate, export_motion_spec


def source(tmp_path):
    refs = []
    for index, stamp, color in [
        (10, 100, (10, 20, 30, 255)),
        (11, 137.5, (50, 20, 5, 255)),
        (16, 250, (50, 20, 5, 100)),
    ]:
        image = Image.new("RGBA", (16, 12), (10, 20, 30, 255))
        image.putpixel((4, 5), color)
        path = tmp_path / "input.png"
        image.save(path)
        digest = sha256(path.read_bytes()).hexdigest()
        filename = f"frame-{index:08d}-{digest}.png"
        path.replace(tmp_path / filename)
        refs.append(
            {"frame": index, "t_ms": stamp, "roles": ["test"], "image": filename, "sha256": digest}
        )
    return {
        "schema": "motionspec/0.1",
        "source": {
            "width": 16,
            "height": 12,
            "t_start_ms": 100,
            "t_end_ms": 250,
            "timestamps": "container",
        },
        "time_resolution_ms": 75,
        "evidence_frames": refs,
        "segments": [
            {
                "id": "s1",
                "kind": "transform",
                "confidence": "medium",
                "region": [1, 1, 15, 11],
                "t_start_ms": 100,
                "t_end_ms": 250,
                "truncated": True,
                "samples": [
                    {"t_ms": 100, "x": 1, "y": 2},
                    {"t_ms": 137.5, "x": 4, "y": 2},
                    {"t_ms": 250, "x": 8, "y": 2},
                ],
                "channels": {
                    "translate": {
                        "extrapolated": True,
                        "lower_bound": True,
                        "t_start_ms": 100,
                        "t_end_ms": 250,
                        "easing": {"model": "cubic-bezier", "params": [0, 0, 1, 1], "rmse": 0.1},
                    }
                },
            }
        ],
    }


def test_export_reuses_exact_image_functions_and_nonuniform_clock(tmp_path):
    spec = source(tmp_path)
    out = tmp_path / "out"
    bundle = export_motion_spec(spec, tmp_path, out, sequence_id="test")
    runtime = VideoRuntime.load(out / "functions.json")
    assert runtime.bundle == bundle
    for frame in bundle.frames:
        assert frame.raster_file == f"raster-{frame.raster_sha256}.json"
        assert sha256((out / frame.raster_file).read_bytes()).hexdigest() == frame.raster_sha256
    assert [f.timestamp_ms for f in bundle.frames] == [100, 137.5, 250]
    assert [d.elapsed_ms for d in bundle.differences] == [37.5, 112.5]
    for ref in spec["evidence_frames"]:
        with Image.open(tmp_path / ref["image"]) as image:
            assert runtime.raster(ref["frame"]).samples == image.convert("RGBA").tobytes()
    assert runtime.delta_pixel(10, 11, 4, 5) == (40, 0, -25, 0)
    assert runtime.delta_pixel(11, 16, 4, 5) == (0, 0, 0, -155)
    assert [d.changed_pixels for d in bundle.differences] == [1, 1]
    assert bundle.differences[0].changed_bbox == (4, 5, 5, 6)
    assert bundle.differences[1].changed_bbox == (4, 5, 5, 6)
    assert bundle.differences[0].mean_abs_rgba[0] == pytest.approx(40 / 192)
    assert bundle.frames[0].image_functions.version == "svsg.functions/1"
    sample = runtime.motion("s1", "x", 137.5)
    assert sample["value"] == 4 and sample["evaluation"] == "sample"
    assert sample["provenance"] == "motion_estimate" and sample["truncated"]
    assert sample["inferred_channels"] == sample["lower_bound_channels"] == ["translate"]
    assert runtime.motion("s1", "x", 193.75)["value"] == 6
    assert runtime.motion("s1", "translate.progress", 175, fitted=True)["value"] == pytest.approx(
        0.5
    )
    with pytest.raises(ValueError, match="not exported"):
        runtime.pixel(12, 0, 0)
    with pytest.raises(ValueError, match="support"):
        runtime.motion("s1", "x", 99)
    with pytest.raises(ValueError, match="chain"):
        runtime.delta_pixel(10, 16, 0, 0)


def test_bezier_uses_inverse_x_and_spring_uses_seconds():
    params = [0.15, 0.9, 0.8, 1.4]
    u = 0.3
    x = 3 * (1 - u) ** 2 * u * params[0] + 3 * (1 - u) * u**2 * params[2] + u**3
    y = 3 * (1 - u) ** 2 * u * params[1] + 3 * (1 - u) * u**2 * params[3] + u**3
    curve = FittedProgress(op="cubic_bezier", params=params, t_start_ms=20, t_end_ms=220, rmse=0)
    assert evaluate(curve, 20 + 200 * x)["value"] == pytest.approx(y)
    spring = FittedProgress(
        op="damped_spring", params=[0.3, 20], t_start_ms=100, t_end_ms=900, rmse=0
    )
    wd = 20 * math.sqrt(1 - 0.3**2)
    expected = 1 - math.exp(-0.3 * 20 * 0.2) * (
        math.cos(wd * 0.2) + 0.3 * 20 / wd * math.sin(wd * 0.2)
    )
    assert evaluate(spring, 300)["value"] == pytest.approx(expected)
    assert evaluate(spring, 300)["value"] > 1  # Preserve overshoot rather than clamp to 1.


@pytest.mark.parametrize(
    "knots",
    [[], [(1, 1), (1, 2)], [(2, 1), (1, 2)], [(float("nan"), 1)], [(0, float("inf"))], [(True, 0)]],
)
def test_invalid_temporal_parameters_rejected(knots):
    with pytest.raises(ValidationError):
        SampledFunction(unit="ratio", knots=knots)


def test_corruption_fails_without_publishing_new_manifest(tmp_path):
    spec = source(tmp_path)
    out = tmp_path / "out"
    bundle = export_motion_spec(spec, tmp_path, out, sequence_id="test")
    old_manifest = (out / "functions.json").read_bytes()
    (tmp_path / spec["evidence_frames"][1]["image"]).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash"):
        export_motion_spec(spec, tmp_path, out, sequence_id="changed")
    assert (out / "functions.json").read_bytes() == old_manifest
    raster_path = out / bundle.frames[0].raster_file
    raster_path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        VideoRuntime.load(out / "functions.json").pixel(10, 0, 0)


@pytest.mark.parametrize("mutation", ["duplicate_frame", "duplicate_time", "wrong_size", "path"])
def test_bad_source_alignment_is_rejected(tmp_path, mutation):
    spec = source(tmp_path)
    if mutation == "duplicate_frame":
        spec["evidence_frames"][1] = copy.deepcopy(spec["evidence_frames"][0])
        spec["evidence_frames"][1]["t_ms"] = 137.5
    elif mutation == "duplicate_time":
        spec["evidence_frames"][1]["t_ms"] = 100
    elif mutation == "wrong_size":
        spec["source"]["width"] = 17
    else:
        spec["evidence_frames"][0]["image"] = "../input.png"
    with pytest.raises(ValueError):
        export_motion_spec(spec, tmp_path, tmp_path / "out", sequence_id="test")
    assert not (tmp_path / "out" / "functions.json").exists()


def test_budgets_and_tampered_manifest_references(tmp_path):
    spec = source(tmp_path)
    with pytest.raises(ValueError, match="frame budget"):
        export_motion_spec(spec, tmp_path, tmp_path / "out", sequence_id="test", max_frames=2)
    with pytest.raises(ValueError, match="pixel budget"):
        export_motion_spec(spec, tmp_path, tmp_path / "out", sequence_id="test", max_total_pixels=1)
    bundle = export_motion_spec(spec, tmp_path, tmp_path / "out", sequence_id="test")
    bad = bundle.model_dump(mode="json")
    bad["differences"][0]["after_frame"] = 16
    with pytest.raises(ValueError, match="references"):
        VideoFunctions.model_validate(bad)
    bad = json.loads(bundle.model_dump_json())
    bad["motions"][0]["samples"]["x"]["knots"][-1][0] = 300
    with pytest.raises(ValueError, match="segment"):
        VideoFunctions.model_validate(bad)


def test_attach_recognition_requires_exact_sequence_frame_and_pixel_identity(tmp_path):
    spec = source(tmp_path)
    bundle = export_motion_spec(spec, tmp_path, tmp_path / "out", sequence_id="test")
    frame = bundle.frames[0]
    result = RecognitionResult(
        status="needs_review",
        image_sha256=frame.image_sha256,
        pixel_sha256=frame.image_functions.pixel_sha256,
        width=16,
        height=12,
        frame=FrameReference(sequence_id="test", frame_index=10, timestamp_ms=100),
        label_scope=[],
        review_mode="same_model_different_view",
        observations=[],
        corroborated_counts={},
        relations=[],
        image_functions=frame.image_functions,
        calls=[],
        warnings=[],
        final_answer="No confirmed object",
    )
    attached = attach_recognition(bundle, result)
    assert attached.frames[0].semantics.status == "needs_review"
    assert bundle.frames[0].semantics is None
    for field, value in [("sequence_id", "other"), ("timestamp_ms", 101), ("frame_index", 12)]:
        bad = result.model_copy(deep=True)
        setattr(bad.frame, field, value)
        with pytest.raises(ValueError):
            attach_recognition(bundle, bad)
    for field in ["image_sha256", "pixel_sha256"]:
        with pytest.raises(ValueError, match="match"):
            attach_recognition(bundle, result.model_copy(update={field: "0" * 64}))


def test_data_only_calls_reuse_image_ops_and_return_bound_source(tmp_path):
    export_motion_spec(source(tmp_path), tmp_path, tmp_path / "out", sequence_id="test")
    runtime = VideoRuntime.load(tmp_path / "out" / "functions.json")
    pixel = runtime.call({"id": "p", "op": "pixel_rgba", "frame_index": 11, "x": 4, "y": 5})
    assert pixel["value"] == (50, 20, 5, 255)
    assert pixel["frames"][0]["timestamp_ms"] == 137.5
    mean = runtime.call({"id": "m", "op": "mean_rgba", "frame_index": 11, "box": [4, 5, 5, 6]})
    assert mean["value"] == pixel["value"]
    delta = runtime.call(
        {"id": "d", "op": "delta_rgba", "before_frame": 10, "after_frame": 11, "x": 4, "y": 5}
    )
    assert delta["value"] == (40, 0, -25, 0)
    motion = runtime.call(
        {"id": "x", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 193.75}
    )
    assert motion["value"] == 6 and motion["evaluation"] == "interpolated"
    with pytest.raises(ValueError):
        runtime.call({"id": "evil", "op": "eval", "code": "print('no')"})
    with pytest.raises(ValueError):
        runtime.call({"id": "p", "op": "pixel_rgba", "frame_index": True, "x": 0, "y": 0})
