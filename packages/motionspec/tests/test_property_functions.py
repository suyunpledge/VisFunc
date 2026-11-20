"""Values, identities and units must survive the probe-to-function boundary."""
import copy
import json

import pytest

from motionspec import check, validate
from motionspec.cli import main
from motionspec.functions import catalogue, run_queries
from motionspec.qt_probe import Recorder
from motionspec.spec import summarize
from motionspec.trace import spec_from_trace
from motionspec.trace_contract import FIELDS
pytest.importorskip("svsg.video_functions")
from svsg.contracts.video_functions import VideoFunctions
from svsg.video_functions import VideoRuntime, evaluate
from svsg.video_functions.temporal import compile_motion


def property_trace(props, *, moving=False):
    return {
        "schema": "motionspec-trace/0.1", "size": [640, 400],
        "ticks_ms": [i * 20 for i in range(30)],
        "elements": [{"id": "button", "name": "button", "fields": list(FIELDS),
                      "rows": [[i, 100 + (i * 3 if moving else 0), 100, 80, 40, 1, 1, 0, 1,
                                100 + (i * 3 if moving else 0), 100] for i in range(30)],
                      "props": props}],
    }


def test_alpha_only_colour_exports_values_and_matching_prose():
    trace = property_trace({"color": [[i, .2, .3, .4, (i - 5) / 12] for i in range(5, 18)]})
    spec = spec_from_trace(trace)
    (segment,) = spec["segments"]
    assert validate(spec) == []
    assert "colour alpha 0 -> 1" in summarize(spec)
    function = compile_motion(segment)
    for row in segment["samples"]:
        for name in ("color.r", "color.g", "color.b", "color.a"):
            result = evaluate(function.samples[name], row["t_ms"])
            assert result["value"] == row[name]
            assert result["provenance"] == "qt_property_sample"
    assert function.reference["element_id"] == "button"
    assert "color.progress" in function.samples
    assert check(spec, [{"segment": "s1", "field": "duration_ms", "channel": "color", "value": 240}])["verdict"] == "passed"


def test_cli_probe_manifest_is_queryable_without_raster(tmp_path, capsys):
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(property_trace({"color": [[tick, .2, .3, .4, (tick-5)/12]
                                                         for tick in range(5, 18)]})))
    assert main(["trace", str(source), "-o", str(tmp_path)]) == 0
    assert main(["functions", str(tmp_path / "spec.json"), "-o", str(tmp_path)]) == 0
    path = tmp_path / "functions.json"
    runtime = VideoRuntime.load(path)
    assert runtime.bundle.version == "svsg.probe-functions/1"
    assert runtime.bundle.frames == runtime.bundle.differences == []
    assert not list(tmp_path.glob("raster-*.json"))
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"id": "alpha", "op": "motion_value", "segment_id": "s1",
                                 "name": "color.a", "t_ms": 220}))
    capsys.readouterr()
    assert main(["query", str(path), str(query)]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == pytest.approx(.5)
    with pytest.raises(ValueError, match="not exported"):
        runtime.pixel(0, 0, 0)
    data = json.loads(path.read_bytes())
    data.update(version="svsg.video-functions/1", clock="index", coverage="sampled_frames_only",
                coordinate_space="decoded_source_rgba8")
    with pytest.raises(ValueError, match="decoded frame"):
        VideoFunctions.model_validate(data)


def test_custom_properties_keep_names_negative_values_and_unknown_units():
    trace = property_trace({"hoverAmount": [[i, (i - 5) / 12] for i in range(5, 18)],
                            "offset": [[i, -2 + (i - 5) / 12] for i in range(5, 18)]})
    spec = spec_from_trace(trace)
    assert len(spec["segments"]) == 2 and spec["motion_groups"] == []
    for segment in spec["segments"]:
        function = compile_motion(segment)
        curve = function.samples["property.value"]
        assert curve.unit == "property_units"
        assert function.reference["property_name"] == segment["channels"]["property"]["name"]
        a, b = curve.knots[:2]
        assert evaluate(curve, (a[0] + b[0]) / 2)["value"] == pytest.approx((a[1] + b[1]) / 2)
        with pytest.raises(ValueError, match="outside"):
            evaluate(curve, curve.knots[-1][0] + 1)


def test_probe_directory_exposes_property_identity_and_recorded_rotation(tmp_path):
    trace = property_trace({"offset": [[i, -3 + i / 8] for i in range(5, 18)]})
    for row in trace["elements"][0]["rows"]:
        row[FIELDS.index("rotation")] = -30 + row[0] * 2.25
    spec_from_trace(trace, tmp_path)
    assert main(["functions", str(tmp_path / "spec.json"), "-o", str(tmp_path)]) == 0
    path = tmp_path / "functions.json"
    page = catalogue(path)
    assert page["coverage"] == "qt_properties_only"
    assert page["coordinate_space"] == "qt_window_device_px"
    assert all(q["op"] == "motion_value" for q in page["query_examples"])
    assert all("value" in r for r in run_queries(path, page["query_examples"]))
    prop = next(s for s in page["segments"] if "property.value" in s["sampled"])
    assert prop["reference"]["element_id"] == "button"
    assert prop["reference"]["property_name"] == "offset"
    rotating = next(s for s in page["segments"] if "rot_deg" in s["sampled"])
    angle, between = run_queries(path, [
        {"id": f"t{t}", "op": "motion_value", "segment_id": rotating["segment_id"], "name": "rot_deg", "t_ms": t}
        for t in (200, 210)
    ])
    assert angle["unit"] == "degrees" and angle["value"] == -7.5
    assert angle["provenance"] == "qt_property_sample" and angle["evaluation"] == "sample"
    assert between["evaluation"] == "interpolated" and between["value"] == pytest.approx(-6.375)


def test_colour_region_covers_motion_during_property_change():
    spec = spec_from_trace(property_trace({"color": [[i, (i - 5) / 12, 0, 0, 1] for i in range(5, 18)]}, moving=True))
    color = next(s for s in spec["segments"] if "color" in s["channels"])
    assert color["region"][2] >= 100 + 17 * 3 + 80


@pytest.mark.parametrize("damage", ["missing_from", "bad_rgba", "negative_distance", "property_name"])
def test_new_channels_reject_malformed_payloads(damage):
    prop = "offset" if damage == "property_name" else "color"
    values = [[i, (i - 5) / 12] if prop == "offset" else [i, (i - 5) / 12, 0, 0, 1] for i in range(5, 18)]
    spec = copy.deepcopy(spec_from_trace(property_trace({prop: values})))
    channel = spec["segments"][0]["channels"]["property" if prop == "offset" else "color"]
    if damage == "missing_from":
        del channel["from"]
    elif damage == "bad_rgba":
        channel["to"][3] = 2
    elif damage == "negative_distance":
        channel["distance"] = -1
    else:
        channel["name"] = ""
    assert validate(spec)


@pytest.mark.parametrize("names", ["hoverAmount", ["hoverAmount", "hoverAmount"]])
def test_recorder_rejects_ambiguous_property_lists(names):
    with pytest.raises(ValueError, match="properties"):
        Recorder(properties=names)


def test_size_sample_values_are_compiled_without_inventing_values():
    segment = spec_from_trace(property_trace({"hoverAmount": [[i, i / 20] for i in range(5, 18)]}, moving=True))["segments"][0]
    function = compile_motion(segment)
    assert function.samples["width"].knots[0][1] == 80
    assert function.samples["height"].knots[0][1] == 40
