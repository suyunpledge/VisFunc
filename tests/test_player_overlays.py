import pytest

from svsg.recognition import overlays
from svsg.recognition.icon_matching import load_icons, match_icons
from svsg.recognition.subtitle_ocr import _review_word_spacing, split_caption_band

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
    out = _review_word_spacing(engine, crop, line)
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


@pytest.mark.parametrize(
    ("gap", "second", "expected"),
    [(7, "NOW", "EXIT NOW"), (2, "NOW", "EXITNOW"), (7, "N0W", "EXITNOW")],
)
def test_english_word_space_uses_pixel_gap_and_exact_letters(gap, second, expected):
    crop = np.full((19, 62, 3), 12, np.uint8)
    for x in [4, 10, 16, 22, 26 + gap, 32 + gap, 38 + gap]:
        crop[4:15, x : x + 4] = 241
    reads = iter([["EXIT", 0.99], [second, 0.99]])
    line = dict(text="EXITNOW", score=0.99, bbox_px=(0, 0, 62, 19))
    out = _review_word_spacing(lambda *a, **kw: ([next(reads)], None), crop, line)
    assert out["text"] == expected


def test_bare_speaker_rejects_wave_outside_template():
    icon = next(i for i in load_icons(player_toolbar=True) if i.template_id == "mute/speaker-only")
    frame = np.full((44, 80, 3), 18, np.uint8)
    h, w = icon.pixels.shape
    glyph = cv2.resize(icon.pixels, (round(24 * w / h), 24), interpolation=cv2.INTER_AREA)
    frame[10:34, 16 : 16 + glyph.shape[1]] = (18 + 0.89 * glyph)[:, :, None]
    found = match_icons(frame, (0, 0, 80, 44), [icon], sizes=(24,))
    assert [i["label"] for i in found] == ["mute"]
    frame[16:28, 19 + glyph.shape[1] : 21 + glyph.shape[1]] = 230
    assert not match_icons(frame, (0, 0, 80, 44), [icon], sizes=(24,))


def test_fallback_fills_row_gaps_without_overriding_volume_or_accepting_isolated_marks():
    from svsg.recognition.icon_matching import supplement_toolbar

    def hit(label, box, score=0.98):
        return dict(label=label, bbox_px=box, score=score)

    volume = hit("volume", (20, 40, 46, 60))
    play = hit("play", (70, 40, 86, 60))
    complete = hit("fullscreen", (110, 40, 126, 60))
    inside = hit("mute", (21, 43, 29, 56))
    stray = hit("mute", (110, 2, 126, 22))
    low = hit("mute", (145, 40, 161, 60), 0.9)
    result = supplement_toolbar([volume, play], [inside, complete, stray, low])
    assert result[:2] == [volume, play]
    assert len(result) == 3 and result[2]["template_id"] == "reference/fullscreen"
    assert result[2]["row_support_px"]
    assert supplement_toolbar([], [complete]) == []


def test_letterspaced_title_cannot_invent_word_inside_track():
    from svsg.recognition.subtitle_ocr import _word_cuts

    crop = np.zeros((25, 154), np.uint8)
    x = 3
    for gap in (5, 5, 3, 13, 5, 6, 7, 5, 4, 0):
        crop[4:21, x : x + 7] = 235
        x += 7 + gap
    cuts = _word_cuts(crop, True)
    assert len(cuts) == 1 and 44 <= cuts[0] <= 56


def test_word_boundary_cannot_pass_through_clipped_ascender():
    from svsg.recognition.subtitle_ocr import _word_cuts

    crop = np.zeros((20, 48), np.uint8)
    for x in (2, 8, 14, 29, 35, 41):
        crop[4:15, x : x + 4] = 235
    (gap,) = _word_cuts(crop, True)
    crop[:15, gap : gap + 2] = 235
    assert _word_cuts(crop, True) == []


def test_volume_outline_requires_controls_on_the_same_row_and_clear_right_edge():
    bank = load_icons(player_toolbar=True)
    icons = [
        next(i for i in bank if i.label == "play"),
        next(i for i in bank if i.label == "fullscreen"),
        next(i for i in bank if i.outline_speaker),
    ]
    frame = np.full((52, 160, 3), 16, np.uint8)

    def draw(icon, x):
        h, w = icon.pixels.shape
        shape = (round(24 * w / max(h, w)), round(24 * h / max(h, w)))
        glyph = cv2.resize(icon.pixels, shape, interpolation=cv2.INTER_AREA)
        frame[14 : 14 + shape[1], x : x + shape[0]] = (16 + 0.85 * glyph)[:, :, None]
        return x + shape[0]

    def labels():
        return [
            row["label"]
            for row in match_icons(frame, (0, 0, 160, 52), icons, sizes=(24,), review_row=True)
        ]

    end = draw(icons[-1], 65)
    assert "volume" not in labels()
    draw(icons[0], 15)
    draw(icons[1], 120)
    assert labels().count("volume") == 1
    frame[20:32, end + 3 : end + 5] = 224
    assert "volume" not in labels()


def test_gradient_shape_needs_matching_margins_and_a_matching_glyph():
    from svsg.recognition.icon_matching import _gradient_shape

    icon = next(i for i in load_icons() if i.label == "fullscreen")
    glyph = np.pad(cv2.resize(icon.pixels, (16, 16), interpolation=cv2.INTER_AREA), 3)
    background = np.linspace(20, 140, len(glyph))[:, None]
    observed = np.uint8(background + glyph * 0.4)
    assert _gradient_shape(observed, glyph, 3) > 0.99
    observed[:, :3] += 25
    assert _gradient_shape(observed, glyph, 3) is None
    assert _gradient_shape(np.uint8(np.broadcast_to(background, glyph.shape)), glyph, 3) is None
