from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from svsg.image_functions.raster import RasterFunction
from svsg.recognition import overlays
from svsg.recognition.icon_matching import load_icons, match_icons
from svsg.recognition.subtitle_ocr import SubtitleReader

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_subtitle_views_map_to_source_and_keep_bilingual_lines():
    calls = []
    reader = SubtitleReader.__new__(SubtitleReader)

    def engine(view, **options):
        if options.get("use_det") is False:
            return [], None
        height, width = view.shape[:2]
        calls.append((width, height))
        scale = width / 120

        def row(x, y, text, score):
            coords = [(x, y), (x + 40, y), (x + 40, y + 6), (x, y + 6)]
            return [[(a * scale, b * scale) for a, b in coords], text, score]

        return [
            row(25, 5, "下一站。", 0.96),
            row(25, 18, "Next stop.", 0.97),
            row(25, 30, "uncertain", 0.6),
            row(25, 36, "|", 0.99),
        ], None

    reader.engine = engine
    result = reader.read(np.zeros((100, 180, 3), np.uint8), (30, 45, 150, 90))
    assert calls == [(120, 45), (240, 90)]
    assert [r["text"] for r in result] == ["下一站。", "Next stop."]
    assert result[0]["bbox_px"] == (55, 50, 96, 57)


def test_template_polarity_and_repeated_icons(tmp_path):
    icon = Image.new("RGBA", (20, 20))
    ImageDraw.Draw(icon).polygon([(4, 3), (15, 10), (4, 16)], fill="white")
    icon.save(tmp_path / "play.png")
    canvas = Image.new("RGB", (120, 60), "#444444")
    canvas.paste(icon, (15, 20), icon)
    canvas.paste("black", (58, 20, 78, 40), icon)
    bgr = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    found = match_icons(bgr, (8, 12, 90, 49), load_icons(tmp_path))
    assert sorted((r["label"], r["bbox_px"]) for r in found) == [
        ("play", (19, 23, 31, 37)),
        ("play", (62, 23, 74, 37)),
    ]


def test_color_identity_and_transparent_rgb(tmp_path):
    folder = tmp_path / "refs"
    folder.mkdir()
    for name, color in [("red", "#ff2222"), ("blue", "#2222ff")]:
        ref = Image.new("RGBA", (24, 24))
        ImageDraw.Draw(ref).ellipse((2, 2, 21, 21), fill=color)
        ref.save(folder / f"{name}.png")
    with Image.open(folder / "red.png") as ref:
        canvas = Image.new("RGBA", (90, 40), "white")
        canvas.alpha_composite(ref, (8, 8))
        # Retain red RGB but remove its alpha so no hidden glyph becomes evidence.
        hidden = ref.copy()
        hidden.putalpha(0)
        canvas.paste(hidden, (50, 8))
        hidden.close()
        canvas.save(tmp_path / "scene.png")
    packet = overlays.measure_overlays(tmp_path / "scene.png", only="icons", icon_dir=folder)
    assert packet["icon_templates"]["red"] != packet["icon_templates"]["blue"]
    assert [(r["label"], r["bbox_px"]) for r in packet["frames"][0]["icons"]] == [
        ("red", (10, 10, 30, 30))
    ]
    with pytest.raises(ValueError, match="source frame"):
        overlays.measure_overlays(tmp_path / "scene.png", only="icons", icon_region=(0, -1, 10, 20))


def test_rgba_pattern_template_ignores_transparent_padding(tmp_path):
    ref = Image.new("RGBA", (16, 16))
    d = ImageDraw.Draw(ref)
    d.rectangle((2, 2, 13, 13), fill="red")
    d.rectangle((8, 2, 13, 7), fill="green")
    d.rectangle((2, 8, 7, 13), fill="blue")
    ref.save(tmp_path / "badge.png")
    bg = Image.new("RGB", (68, 44), "#404040")
    bg.paste(ref, (24, 12), ref)
    bgr = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR)
    (found,) = match_icons(bgr, (0, 0, 68, 44), load_icons(tmp_path))
    assert found["label"] == "badge" and found["bbox_px"] == (26, 14, 38, 26)


def test_sampled_clock_and_duplicate_frame_reuse(monkeypatch, tmp_path):
    from svsg.video_functions import VideoRuntime

    raster = RasterFunction(40, 30, bytes([255]) * (40 * 30 * 4))
    refs = [SimpleNamespace(frame_index=i, timestamp_ms=t) for i, t in [(3, 100), (7, 268)]]
    bundle = SimpleNamespace(frames=refs, sequence_id="sample")
    monkeypatch.setattr(
        VideoRuntime, "load", lambda _: SimpleNamespace(bundle=bundle, raster=lambda _: raster)
    )
    calls = []
    monkeypatch.setattr(overlays, "match_icons", lambda *args, **kwargs: calls.append(1) or [])
    manifest = tmp_path / "functions.json"
    result = overlays.measure_overlays(manifest, only="icons")
    assert [(r["sequence_id"], r["frame_index"], r["timestamp_ms"]) for r in result["frames"]] == [
        ("sample", 3, 100),
        ("sample", 7, 268),
    ]
    assert len(calls) == 1
    with pytest.raises(ValueError, match="not exported"):
        overlays.measure_overlays(manifest, only="icons", frame=4)


@pytest.mark.parametrize("second_read, kept", [("下一站", "下一站"), ("下车站", "站")])
def test_edge_recovery_needs_real_continuing_ink_and_two_matching_reads(second_read, kept):
    from svsg.recognition.subtitle_ocr import _review_edges

    image = np.zeros((30, 80, 3), np.uint8)
    for x in range(12, 53, 3):
        image[10:20, x] = (90, 220, 250)
    line = dict(text="站", bbox_px=(28, 9, 54, 21), score=0.99)

    def engine(patch, **kwargs):
        return [[second_read if patch.ndim == 2 else "下一站", 0.97]], None

    result = _review_edges(engine, image, (0, 0, 80, 30), line, [line])
    assert result["text"] == kept
    if kept != "站":
        assert result["bbox_px"][0] <= 12 and result["edge_review"]["original_text"] == "站"
    # A caller's explicit crop excludes the putative prefix.
    assert _review_edges(engine, image, (28, 0, 80, 30), line, [line]) == line


@pytest.mark.parametrize(
    ("original", "a", "b", "expected"),
    [
        ("Gate B2", "Gate B 2.", "Gate B 2.", "Gate B 2."),
        ("Gate B2", "GateB2.", "GateB2.", "Gate B2"),
        ("车站", "车占。", "车占。", "车站"),
        ("车站", "车站。", "车站", "车站"),
    ],
)
def test_line_review_preserves_letters_and_existing_spaces(original, a, b, expected):
    from svsg.recognition.subtitle_ocr import _review_typography

    reads = iter([a, b])

    def engine(*args, **kwargs):
        return [[next(reads), 0.96]], None

    line = dict(text=original, bbox_px=(2, 2, 30, 15), score=0.98)
    assert _review_typography(engine, np.zeros((20, 32, 3), np.uint8), line)["text"] == expected


def test_detached_scene_text_and_connected_multiline_caption():
    from svsg.recognition.subtitle_ocr import split_caption_band

    def line(y):
        return dict(text=str(y), bbox_px=(20, y, 140, y + 12))

    title = line(210)
    block = [line(248), line(264), line(280), line(296)]
    assert split_caption_band([title, *block], 400) == (block, [title])
    assert split_caption_band([title], 400) == ([title], [])


def test_mixed_bank_keeps_style_evidence_and_one_detection_per_icon(tmp_path):
    refs = tmp_path / "refs"
    label = refs / "search"
    label.mkdir(parents=True)
    glyph = Image.new("RGBA", (20, 20))
    ImageDraw.Draw(glyph).ellipse((3, 3, 16, 16), outline="white", width=3)
    for stem in ("dark", "light"):
        glyph.save(label / f"{stem}.png")
    canvas = Image.new("RGB", (72, 40), "black")
    canvas.paste(glyph, (15, 8), glyph)
    path = tmp_path / "scene.png"
    canvas.save(path)
    packet = overlays.measure_overlays(path, only="icons", icon_dir=refs)
    (hit,) = packet["frames"][0]["icons"]
    assert hit["label"] == "search" and hit["template_id"] in packet["icon_templates"]
    assert set(packet["icon_templates"]) == {"search/dark", "search/light"}


def test_soft_toolbar_shape_needs_two_separate_strong_neighbors():
    icons = [i for i in load_icons() if i.label in {"pause", "fullscreen", "mute"}]
    frame = np.full((40, 140, 3), 25, np.uint8)
    for icon, x in zip(icons, (10, 52, 93), strict=True):
        h, w = icon.pixels.shape[:2]
        tw, th = round(16 * w / max(w, h)), round(16 * h / max(w, h))
        glyph = cv2.resize(icon.pixels, (tw, th), interpolation=cv2.INTER_AREA)
        if icon.label == "mute":
            glyph = cv2.dilate(glyph, np.ones((2, 2), np.uint8))
        top = 20 - th // 2
        frame[top : top + th, x : x + tw] = np.uint8(25 + glyph.astype(float) * 0.85)[:, :, None]

    def scan():
        return {
            hit["label"]: hit
            for hit in match_icons(frame, (0, 0, 140, 40), icons, sizes=(16,), review_row=True)
        }

    result = scan()
    assert len(result["mute"]["row_support_px"]) == 2
    frame[:, :32] = 25
    assert "mute" not in scan()
