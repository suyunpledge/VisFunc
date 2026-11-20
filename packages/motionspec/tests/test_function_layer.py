"""函数层：各条测量路径导出的采样曲线是否齐全、准确，以及给模型用的目录和批量查询。"""

import json

import pytest

from motionspec import analyze, validate
from motionspec.synth import background, enter, progress

pytest.importorskip("svsg.video_functions")

from motionspec.cli import main  # noqa: E402
from motionspec.functions import catalogue, run_queries  # noqa: E402

FRAME = 1000 / 60


def rows(spec):
    (seg,) = spec["segments"]
    return seg, seg["samples"]


def test_entering_element_has_sub_pixel_size():
    """合成卡片的可见范围是 192 × 102（掩码画在 4..w-5、模糊后半高处）。整数外接框每边会少一点。"""
    frames, _ = enter()
    seg, samples = rows(analyze(frames, fps=60))
    assert seg["model"] == "layer"
    assert all(abs(r["width"] - 192) <= 1.5 and abs(r["height"] - 102) <= 1.5 for r in samples)
    assert {"opacity", "x", "y", "width", "height"} <= set(samples[0])


def test_progress_bar_curves_cover_every_frame():
    frames, truth = progress()
    seg, samples = rows(analyze(frames, fps=60))
    assert seg["model"] == "solid-box"
    # 高度在长度为 0 的帧也保持不变（原先报成 0）
    assert {round(r["height"]) for r in samples} == {11}
    widths = [r["width"] for r in samples]
    assert widths == sorted(widths) and widths[-1] == pytest.approx(360, abs=2)
    assert all("size" in r["channel_progress"] for r in samples)


def test_translating_box_keeps_motion_before_and_after_resize(tmp_path):
    frames = []
    for tick in range(34):
        frame = background()
        left = 100 + 4 * min(20, max(0, tick - 5))
        width = 80 + 3 * min(10, max(0, tick - 8))
        frame[80:86, left:left + width] = (240, 140, 40)
        frames.append(frame)
    spec = analyze(frames, tmp_path, fps=50, functions=True, sheets=False)
    segment, samples = rows(spec)
    assert segment["model"] == "solid-box" and segment["frames"] == [5, 25]
    assert segment["keyframes"][0]["t_ms"] == samples[0]["t_ms"] == 100
    assert segment["keyframes"][-1]["t_ms"] == samples[-1]["t_ms"] == 500
    size = segment["channels"]["size"]
    move = segment["channels"]["translate"]
    assert (size["t_start_ms"], size["t_end_ms"]) == (160, 360)
    assert (move["t_start_ms"], move["t_end_ms"]) == (100, 500)
    assert move["delta"] == [95, 0]
    calls = [{"id": str(t), "op": "motion_value", "segment_id": segment["id"], "name": "x", "t_ms": t}
             for t in (100, 500)]
    results = run_queries(tmp_path / "functions.json", calls)
    assert [r["value"] for r in results] == [move["from"][0], move["to"][0]]
    segment.update(t_start_ms=160, t_end_ms=360, duration_ms=200)
    assert any("'translate' timing exceeds the segment" in error for error in validate(spec))


def test_catalogue_and_batch_queries(tmp_path, capsys):
    frames, _ = progress()
    spec = analyze(frames, tmp_path, fps=60, functions=True, sequence_id="bar")
    manifest = tmp_path / spec["function_evidence"]["manifest"]
    page = catalogue(manifest)
    assert {call["id"] for call in page["query_examples"]} == {"sampled", "fitted", "pixel"}
    assert all("value" in r for r in run_queries(manifest, page["query_examples"]))
    (segment,) = page["segments"]
    assert {"width", "height", "size.progress"} <= set(segment["sampled"])
    width = segment["sampled"]["width"]
    assert width["unit"] == "source_px" and width["range"][1] == pytest.approx(360, abs=2)
    assert len(json.dumps(page)) < manifest.stat().st_size / 5
    lo, hi = width["t_ms"]
    results = run_queries(manifest, [
        {"id": "w", "op": "motion_value", "segment_id": segment["segment_id"], "name": "width", "t_ms": (lo + hi) / 2},
        {"id": "late", "op": "motion_value", "segment_id": segment["segment_id"], "name": "width", "t_ms": hi + 1000},
        {"id": "unknown", "op": "no_such_op"},
    ])
    assert results[0]["value"] == pytest.approx(180, abs=12)
    assert "outside" in results[1]["error"] and results[2]["id"] == "unknown" and "error" in results[2]
    with pytest.raises(ValueError):
        run_queries(manifest, [])
    assert main(["query", str(manifest), "--list"]) == 0
    assert json.loads(capsys.readouterr().out)["sequence_id"] == "bar"


def test_legacy_catalogue_and_batch_with_missing_raster(tmp_path, capsys):
    frames, _ = progress()
    analyze(frames, tmp_path, fps=60, functions=True, sheets=False)
    path = tmp_path / "functions.json"
    data = json.loads(path.read_bytes())
    for motion in data["motions"]:
        del motion["timing"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["query", str(path), "--list"]) == 0
    page = json.loads(capsys.readouterr().out)
    (segment,) = page["segments"]
    start, end = segment["visible_t_ms"]
    assert segment["visible_duration_ms"] == pytest.approx(end - start)
    assert segment["design"] is None
    assert segment["design_unavailable_reason"] == "legacy_missing_timing"

    (tmp_path / data["frames"][0]["raster_file"]).unlink()
    examples = {q["id"]: q for q in page["query_examples"]}
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([examples["pixel"], examples["sampled"]]), encoding="utf-8")
    assert main(["query", str(path), str(queries)]) == 0
    failed, succeeded = json.loads(capsys.readouterr().out)
    assert failed["id"] == "pixel" and failed["error"]
    assert succeeded["id"] == "sampled" and "value" in succeeded
