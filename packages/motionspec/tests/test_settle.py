"""等界面稳定：状态判定、持续变化区域、忽略区、净变化区域和命令行。"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from motionspec.cli import main
from motionspec.settle import SettleWatcher, watch

W, H = 1920, 1080
STEP = 1000 / 60


def blank() -> np.ndarray:
    return np.full((H, W, 3), 245, np.uint8)


def clip(n: int, draw) -> list[tuple[float, np.ndarray]]:
    frames = []
    for i in range(n):
        frame = blank()
        draw(frame, i * STEP, i)
        frames.append((i * STEP, frame))
    return frames


def test_static_screen_reports_unchanged():
    result, _ = watch(clip(90, lambda f, t, i: None), expect_change_ms=500)
    assert result["state"] == "unchanged"
    assert result["first_change_ms"] is None
    assert result["changed_regions"] == []
    assert 500 <= result["elapsed_ms"] < 500 + STEP


def slide(frame, t, _):
    x = 200 + round(300 * min(1.0, max(0.0, (t - 100) / 300)))
    cv2.rectangle(frame, (x, 400), (x + 100, 460), (40, 40, 40), -1)


def test_settles_after_last_change_and_reports_net_regions():
    result, last = watch(clip(120, slide), quiet_ms=200)
    assert result["state"] == "settled"
    assert abs(result["last_change_ms"] - 400) <= STEP
    assert result["elapsed_ms"] - result["last_change_ms"] >= 200
    assert last.shape == (H, W, 3)
    boxes = [r["bbox_px"] for r in result["changed_regions"]]
    # The rectangle left its old place and arrived at its new one; both are net changes.
    assert any(b[0] <= 205 and b[2] >= 295 and b[1] <= 405 and b[3] >= 455 for b in boxes)
    assert any(b[0] <= 505 and b[2] >= 595 and b[1] <= 405 and b[3] >= 455 for b in boxes)
    assert all(0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H for b in boxes)


def test_transient_change_leaves_no_net_region():
    def flash(frame, t, _):
        if 100 <= t < 200:
            cv2.rectangle(frame, (800, 500), (1000, 560), (20, 20, 200), -1)

    result, _ = watch(clip(60, flash), quiet_ms=200)
    assert result["state"] == "settled"
    assert result["changed_regions"] == []


def noise(frame, i):
    rng = np.random.default_rng(i)
    frame[100:160, 1600:1700] = rng.integers(0, 255, (60, 100, 3), dtype=np.uint8)


def test_restless_area_becomes_ambient_and_real_change_still_counts():
    def scene(frame, t, i):
        noise(frame, i)
        if t >= 400:
            cv2.rectangle(frame, (300, 700), (700, 760), (200, 80, 20), -1)

    result, _ = watch(clip(120, scene), quiet_ms=200, ambient_ms=300)
    assert result["state"] == "settled"
    assert abs(result["last_change_ms"] - 400) <= STEP
    ambient = [r["bbox_px"] for r in result["ambient_regions"]]
    assert any(b[0] <= 1600 and b[2] >= 1700 and b[1] <= 100 and b[3] >= 160 for b in ambient)
    changed = [r["bbox_px"] for r in result["changed_regions"]]
    assert any(b[0] <= 305 and b[2] >= 695 for b in changed)
    assert not any(b[0] >= 1500 for b in changed)


def test_waiting_for_everything_times_out_on_restless_area():
    result, _ = watch(clip(90, lambda f, t, i: noise(f, i)), ambient_ms=0, timeout_ms=1000)
    assert result["state"] == "timeout"


def test_ignored_area_does_not_count():
    result, _ = watch(
        clip(90, lambda f, t, i: noise(f, i)),
        ambient_ms=0,
        expect_change_ms=500,
        ignore=[(1590, 90, 1710, 170)],
    )
    assert result["state"] == "unchanged"


def test_frames_running_out_report_ended():
    result, _ = watch(clip(20, slide), quiet_ms=200)
    assert result["state"] == "ended"
    assert result["first_change_ms"] is not None


def test_rejects_bad_input():
    watcher = SettleWatcher()
    watcher.push(blank(), 0.0)
    with pytest.raises(ValueError, match="increase"):
        watcher.push(blank(), 0.0)
    with pytest.raises(ValueError, match="size"):
        watcher.push(np.zeros((10, 10, 3), np.uint8), 20.0)
    with pytest.raises(ValueError):
        SettleWatcher(quiet_ms=0)
    with pytest.raises(ValueError):
        SettleWatcher(ignore=[(10, 10, 5, 20)])


def test_cli_reads_a_frame_directory(tmp_path, capsys):
    for i, (_, frame) in enumerate(clip(60, slide)):
        cv2.imwrite(str(tmp_path / f"{i:04d}.png"), frame)
    code = main(
        [
            "settle",
            str(tmp_path),
            "--fps",
            "60",
            "--quiet-ms",
            "150",
            "--save-frame",
            str(tmp_path / "final.png"),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["state"] == "settled"
    assert result["version"] == "motionspec.settle/1"
    assert (tmp_path / "final.png").is_file()


def test_cli_region_needs_screen(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.png"), blank())
    with pytest.raises(SystemExit) as exc:
        main(["settle", str(tmp_path), "--fps", "60", "--region", "0", "0", "10", "10"])
    assert exc.value.code == 2


def test_rearm_keeps_learned_restless_areas_across_a_pause():
    watcher = SettleWatcher(quiet_ms=200, ambient_ms=300)
    for i in range(40):  # Video plays while the agent looks at the screen.
        frame = blank()
        noise(frame, i)
        watcher.push(frame, i * STEP)
    assert watcher.state == "settled"
    watcher.rearm()
    start = 40 * STEP + 2000  # The agent spends two seconds acting without pushing frames.
    for j in range(40):
        frame = blank()
        noise(frame, 100 + j)
        if j >= 6:
            cv2.rectangle(frame, (300, 700), (700, 760), (200, 80, 20), -1)
        if watcher.push(frame, start + j * STEP):
            break
    result = watcher.result()
    assert result["state"] == "settled"
    # Measured from the last frame before rearm (39) to the frame the box appears (40 + 6).
    assert abs(result["first_change_ms"] - (2000 + 7 * STEP)) < 0.01
    assert result["last_change_ms"] == result["first_change_ms"]
    changed = [r["bbox_px"] for r in result["changed_regions"]]
    assert any(b[0] <= 305 and b[2] >= 695 for b in changed)
    assert not any(b[0] >= 1500 for b in changed)


def test_slow_ease_out_tail_is_not_mistaken_for_stillness():
    def dim(frame, t, _):
        p = min(1.0, max(0.0, (t - 100) / 800))
        frame[:] = round(245 - 90 * (1 - (1 - p) ** 3))

    frames = clip(150, dim)
    result, last = watch(frames, quiet_ms=200)
    # Every step is under 6 grey levels, below the threshold of 12; only drift reveals the fade.
    assert result["state"] == "settled"
    assert result["first_change_ms"] is not None
    assert int(cv2.absdiff(last, frames[-1][1]).max()) <= 6  # within half a threshold of the end


def test_capture_backend_and_region_are_validated():
    from motionspec.settle import _grabber

    with pytest.raises(ValueError, match="backend"):
        _grabber(None, "gdi")
    with pytest.raises(ValueError, match="positive"):
        _grabber((10, 10, 5, 20), "pil")
