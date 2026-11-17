"""Integration tests activate when the optional local SVSG package is installed."""
import copy
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("svsg.video_functions")

from motionspec import analyze, validate
from motionspec.cli import main
from motionspec.functions import export_functions
from svsg.video_functions import VideoRuntime


def test_analyze_exports_same_raster_runtime_and_periodic_frames(tmp_path):
    black = np.zeros((24, 32, 3), np.uint8)
    spec = analyze([black] * 8, tmp_path, fps=20, functions=True,
                   evidence_every_ms=100, sequence_id="static-test", sheets=False)
    assert validate(spec) == []
    assert spec["segments"] == []
    assert [f["frame"] for f in spec["evidence_frames"]] == [0, 2, 4, 6, 7]
    ref = spec["function_evidence"]
    assert ref["sha256"] == hashlib.sha256((tmp_path / "functions.json").read_bytes()).hexdigest()
    runtime = VideoRuntime.load(tmp_path / ref["manifest"])
    assert runtime.pixel(4, 2, 2) == (0, 0, 0, 255)
    assert all(d.changed_pixels == 0 for d in runtime.bundle.differences)
    assert len(list(tmp_path.glob("raster-*.json"))) == 1  # Static frames share one program.
    assert runtime.bundle.sequence_id == "static-test"
    assert runtime.bundle.source_spec_sha256 == hashlib.sha256(json.dumps(
        {k: v for k, v in spec.items() if k != "function_evidence"}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    second = tmp_path / "reexport"
    assert main(["functions", str(tmp_path / "spec.json"), "-o", str(second),
                 "--sequence-id", "static-test"]) == 0
    assert (second / "functions.json").read_bytes() == (tmp_path / "functions.json").read_bytes()
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"id": "p", "op": "pixel_rgba", "frame_index": 4, "x": 2, "y": 2}))
    assert main(["query", str(second / "functions.json"), str(query)]) == 0


def test_export_existing_motion_preserves_samples_and_fitted_easing(tmp_path):
    from motionspec.synth import slide

    frames, _ = slide(distance=80, duration_ms=200)
    spec = analyze(frames, tmp_path, fps=60, sheets=False, functions=True)
    runtime = VideoRuntime.load(tmp_path / "functions.json")
    for segment in spec["segments"]:
        exported = next(m for m in runtime.bundle.motions if m.segment_id == segment["id"])
        assert exported.timing.model_dump(mode="json") == segment["timing"]
        for sample in segment["samples"]:
            for name in ("x", "y", "scale", "rot_deg", "luma", "progress"):
                if name in sample:
                    assert runtime.motion(segment["id"], name, sample["t_ms"])["value"] == sample[name]
        for name, channel in segment["channels"].items():
            if channel.get("easing"):
                from motionspec.fit import bezier_ease, spring_response

                t = channel["t_start_ms"] + .37 * channel["duration_ms"]
                curve = channel["easing"]
                if curve["model"] == "cubic-bezier":
                    tau = (t-channel["t_start_ms"])/(channel["t_end_ms"]-channel["t_start_ms"])
                    expected = float(bezier_ease(*curve["params"], tau))
                else:
                    expected = float(spring_response(curve["damping_ratio"],
                                                     curve["angular_frequency"],
                                                     (t-channel["t_start_ms"])/1000))
                assert runtime.motion(segment["id"], name+".progress", t, fitted=True)["value"] == pytest.approx(expected, abs=2e-5)
    changed = copy.deepcopy(spec)
    changed["segments"][0]["timing"]["visible_change"]["duration_ms"] += 1
    assert any("timing disagrees" in issue for issue in validate(changed))
    with pytest.raises(ValueError, match="timing disagrees"):
        export_functions(changed, tmp_path, tmp_path / "invalid")
    legacy = copy.deepcopy(spec)
    for segment in legacy["segments"]:
        segment.pop("timing")
    export_functions(legacy, tmp_path, tmp_path / "legacy-upgraded", sequence_id=runtime.bundle.sequence_id)
    assert VideoRuntime.load(tmp_path / "legacy-upgraded" / "functions.json").bundle == runtime.bundle


def test_missing_frames_limits_and_invalid_references(tmp_path):
    frame = np.zeros((20, 24, 3), np.uint8)
    with pytest.raises(ValueError, match="raw_frames"):
        analyze([frame]*2, tmp_path, functions=True, raw_frames=False)
    with pytest.raises(ValueError, match="out_dir"):
        analyze([frame]*2, functions=True)
    with pytest.raises(ValueError, match="budget"):
        analyze([frame]*10, tmp_path, fps=20, functions=True,
                evidence_every_ms=50, function_max_frames=2)
    assert not (tmp_path / "spec.json").exists()
    spec = analyze([frame]*2, tmp_path, sheets=False)
    wrong = copy.deepcopy(spec)
    wrong["evidence_frames"][1]["t_ms"] = 0
    assert any("strictly increasing" in p for p in validate(wrong))
    (tmp_path / spec["evidence_frames"][0]["image"]).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash"):
        export_functions(spec, tmp_path, tmp_path / "out")
