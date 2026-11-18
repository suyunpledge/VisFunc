"""函数层：各条测量路径导出的采样曲线是否齐全、准确，以及给模型用的目录和批量查询。"""

import json

import pytest

from motionspec import analyze
from motionspec.synth import enter, progress

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


def test_catalogue_and_batch_queries(tmp_path, capsys):
    frames, _ = progress()
    spec = analyze(frames, tmp_path, fps=60, functions=True, sequence_id="bar")
    manifest = tmp_path / spec["function_evidence"]["manifest"]
    page = catalogue(manifest)
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
