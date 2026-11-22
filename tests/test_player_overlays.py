import pytest

from svsg.recognition import overlays
from svsg.recognition.icon_matching import load_icons, match_icons
from svsg.recognition.subtitle_ocr import _review_code_spaces, split_caption_band

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_title_role_retains_source_observation_and_respects_explicit_region(monkeypatch, tmp_path):
    from PIL import Image

    title = dict(text="SUMMER", bbox_px=(65, 108, 135, 120), score=0.98)
    bottom = dict(text="STOP", bbox_px=(70, 157, 130, 171), score=0.99)
    small = dict(text="wait", bbox_px=(88, 108, 112, 116), score=0.95)
    assert split_caption_band([title], 200) == ([], [title])
    assert split_caption_band([bottom], 200) == ([bottom], [])
    assert split_caption_band([small], 200) == ([small], [])
    monkeypatch.setattr(overlays.SubtitleReader, "__init__", lambda self: None)
    monkeypatch.setattr(overlays.SubtitleReader, "read", lambda *a, **kw: [title])
    monkeypatch.setattr(overlays, "version", lambda _: "test")
    path = tmp_path / "frame.png"
    Image.new("RGB", (200, 200)).save(path)
    frame = overlays.measure_overlays(path, only="subtitles")["frames"][0]
    assert frame["subtitles"] == [] and frame["detached_text_candidates"] == [title]
    explicit = overlays.measure_overlays(path, only="subtitles", subtitle_region=(0, 90, 200, 140))
    assert explicit["frames"][0]["subtitles"] == [title]


def test_clocks_and_readout_columns_require_player_context():
    clock = dict(text="00:03 / 01:45", bbox_px=(80, 170, 120, 180))
    cue = dict(text="At 07:35", bbox_px=(80, 150, 120, 160))
    left = dict(text="x0.5", bbox_px=(18, 130, 42, 140))
    right = dict(text="x2", bbox_px=(144, 130, 156, 140))
    count = dict(text="022", bbox_px=(135, 110, 165, 125))
    lines = [count, left, right, cue, clock]
    assert split_caption_band(lines, 200, player_toolbar=True) == (
        [cue],
        [count, left, right, clock],
    )
    assert split_caption_band([clock], 200) == ([clock], [])
    assert split_caption_band([right], 200, player_toolbar=True) == ([right], [])


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize(
    "tight,suffix,expected", [(False, "12", "AB 12"), (False, "I2", "AB12"), (True, "12", "AB12")]
)
def test_insert_space_needs_measured_gap_and_both_original_substrings(
    negative, tight, suffix, expected
):
    crop = np.full((19, 38, 3), 15, np.uint8)
    shift = 0 if tight else 5
    for x in (4, 10, 16 + shift, 22 + shift):
        crop[4:15, x : x + 4] = 230
    if negative:
        crop = 255 - crop
    reads = iter(["AB", suffix])

    def engine(*args, **kwargs):
        return [[next(reads), 0.98]], None

    line = dict(text="AB12", score=0.99, bbox_px=(0, 0, 38, 19))
    out = _review_code_spaces(engine, crop, line)
    assert out["text"] == expected
    if expected == "AB 12":
        assert 14 <= out["spacing_review"]["gap_x_px"][0] <= 21


def test_muted_variant_requires_visible_cross_not_just_speaker_body():
    bank = load_icons(player_toolbar=True)
    mute = next(i for i in bank if i.template_id == "mute/speaker-cross")
    assert all(i.template_id != mute.template_id for i in load_icons())
    glyph = cv2.resize(mute.pixels, (24, 22), interpolation=cv2.INTER_AREA)
    frame = np.full((42, 60, 3), 20, np.uint8)
    frame[10:32, 15:39] = (20 + 0.85 * glyph)[:, :, None].astype(np.uint8)

    def read():
        return match_icons(frame, (0, 0, 60, 42), [mute], sizes=(24,), review_row=True)

    assert len(read()) == 1
    frame[:, 30:] = 20
    assert read() == []
