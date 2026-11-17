"""持续变化的区域（动态背景、视频、转圈）不该把旁边的界面动效吞掉。"""

import numpy as np
import pytest

from motionspec import analyze, validate
from motionspec.fit import bezier_ease
from motionspec.synth import background, card, compose

FPS = 30


def busy_clip(seconds=8.0, slide_at=5.0):
    """左上角一块不停流动的区域；slide_at 秒时，紧挨着它的卡片右移 150px（400ms，out-cubic）。"""
    bg, sprite = background(), card()
    h, w = bg.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    band = np.exp(-(((yy - 120) / 60.0) ** 2)) * (xx < 300)
    frames = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        wave = 0.5 + 0.5 * np.sin(xx / 30.0 + 9 * t) * np.cos(yy / 25.0 - 6.3 * t)
        glow = (band * wave)[..., None] * np.array([170, 110, 50], np.float32)
        frame = np.clip(bg.astype(np.float32) + glow, 0, 255).astype(np.uint8)
        v = float(bezier_ease(0.33, 1, 0.68, 1, np.clip((t - slide_at) / 0.4, 0, 1)))
        frames.append(compose(frame, sprite, 260 + 150 * v, 250))
    return frames


@pytest.fixture(scope="module")
def clip():
    return busy_clip()


def slides(spec):
    return [s for s in spec["segments"] if s["kind"] == "transform" and s["confidence"] == "high"]


def test_motion_next_to_a_restless_area_is_still_found(clip):
    spec = analyze(clip, fps=FPS)
    assert validate(spec) == []
    (slide,) = slides(spec)
    assert slide["t_start_ms"] == pytest.approx(5000, abs=2 * 1000 / FPS)
    assert slide["channels"]["translate"]["delta"][0] == pytest.approx(150, abs=3)
    assert slide["easing"]["named"][0]["name"] == "out-cubic"
    assert not slide["truncated"]


def test_restless_area_is_reported_separately(clip):
    spec = analyze(clip, fps=FPS)
    assert spec["coverage"]["ambient_regions"] == len(spec["ambient"]) >= 1
    x1, y1, x2, y2 = spec["ambient"][0]["region"]
    assert x1 < 60 and x2 <= 340 and y1 < 110 and y2 > 130  # 流动区域在左上角
    assert spec["ambient"][0]["t_end_ms"] - spec["ambient"][0]["t_start_ms"] > 6000
    assert spec["coverage"]["truncated_segments"] == 0


def test_without_ambient_handling_the_motion_is_lost(clip):
    """对照：关掉之后，流动区域连成贯穿全片的候选，卡片的滑动混在里面测不出来。"""
    spec = analyze(clip, fps=FPS, ambient_ms=0)
    assert spec["ambient"] == []
    assert slides(spec) == []
    longest = max(s["duration_ms"] for s in spec["segments"])
    assert longest > 4000  # 一个片段覆盖了大半个片子


def test_short_clips_are_unaffected():
    from motionspec.synth import stagger

    frames, _ = stagger()
    spec = analyze(frames, fps=60)
    assert spec["ambient"] == [] and len(spec["segments"]) == 2


def test_bad_ambient_option_is_rejected():
    with pytest.raises(ValueError, match="ambient_ms"):
        analyze([np.zeros((16, 16, 3), np.uint8)] * 3, ambient_ms=-1)


def test_contaminated_ambient_bursts_do_not_evict_clean_motion():
    """当环境动画区域在长连续段之外还有短促闪烁时，不应挤掉 max_segments 里的干净界面动效。"""
    bg, sprite = background(), card(120, 60)
    h, w = bg.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    band = np.exp(-(((yy - 120) / 60.0) ** 2)) * (xx < 300)
    frames = []
    for i in range(int(7.0 * FPS)):
        t = i / FPS
        # 0..3.6s 连续流动（触发 ambient），3.6..4.5s 停顿超过 pause_ms，4.5..5.8s 再次短促流动
        phase_t = t if t <= 3.6 else 3.6 if t < 4.5 else 3.6 + (t - 4.5)
        wave = 0.5 + 0.5 * np.sin(xx / 30.0 + 9 * phase_t) * np.cos(yy / 25.0 - 6.3 * phase_t)
        glow = (band * wave)[..., None] * np.array([170, 110, 50], np.float32)
        frame = np.clip(bg.astype(np.float32) + glow, 0, 255).astype(np.uint8)
        v = float(bezier_ease(0.33, 1, 0.68, 1, np.clip((t - 5.0) / 0.4, 0, 1)))
        frames.append(compose(frame, sprite, 430 + 100 * v, 280))
    spec = analyze(frames, fps=FPS, max_segments=1)
    assert len(spec["segments"]) == 1
    seg = spec["segments"][0]
    assert seg["kind"] == "transform" and seg["confidence"] == "high"
    assert seg["channels"]["translate"]["delta"][0] == pytest.approx(100, abs=3)
