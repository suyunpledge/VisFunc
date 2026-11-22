"""界面文字读取：区域整理、去重、阅读顺序、像素哈希、HTTP 接口；装了 OCR 时再做真实读取。"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from svsg.api import create_app
from svsg.config import Settings
from svsg.image_functions.raster import RasterFunction
from svsg.recognition.screen_text import clip_regions, dedupe, load_bgr, pixel_sha256, reading_order


def line(text, box, score=0.9):
    return {"text": text, "bbox_px": list(box), "score": score}


def test_regions_are_padded_clipped_and_merged():
    boxes = clip_regions([(10, 10, 50, 20), (45, 12, 90, 22), (500, 500, 600, 590)], 560, 560)
    assert boxes[0] == (4, 0, 96, 33)  # padded by 6, grown to 32 px tall, merged with its neighbour
    assert boxes[1] == (494, 494, 560, 560)  # clipped at the image edge


def test_regions_are_validated():
    with pytest.raises(ValueError):
        clip_regions([(10, 10, 5, 20)], 100, 100)
    with pytest.raises(ValueError):
        clip_regions([(200, 200, 300, 300)], 100, 100)
    with pytest.raises(ValueError):
        clip_regions([(0, 0, 1, 1)] * 65, 100, 100)


def test_dedupe_keeps_one_reading_per_spot():
    lines = [
        line("Display name Morgan Lee", (18, 118, 214, 146), 0.97),
        line("Display name", (21, 119, 119, 140), 0.99),
        line("Morgan Lee", (120, 117, 208, 141), 0.98),
        line("Cancel[", (1638, 1040, 1724, 1076), 0.89),
        line("Cancel", (1647, 1046, 1699, 1068), 0.99),
    ]
    kept = [row["text"] for row in dedupe(lines)]
    assert kept == ["Display name", "Morgan Lee", "Cancel"]


def test_reading_order_groups_rows():
    rows = [
        line("Morgan Lee", (120, 117, 208, 141)),
        line("Display name", (21, 119, 119, 140)),
        line("Email", (20, 149, 121, 172)),
        line("Title", (9, 48, 201, 78)),
    ]
    assert [r["text"] for r in reading_order(rows)] == [
        "Title",
        "Display name",
        "Morgan Lee",
        "Email",
    ]


def png_bytes(mode="RGB") -> bytes:
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)).convert(mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_pixel_hash_matches_raster_functions():
    data = png_bytes()
    assert pixel_sha256(load_bgr(data)) == RasterFunction.from_image(data).pixel_sha256


class FakeReader:
    def __init__(self):
        self.calls = []

    def read(self, data, regions):
        self.calls.append(regions)
        if regions and regions[0][0] < 0:
            raise ValueError("bad region")
        return {
            "version": "svsg.screen-text/1",
            "lines": [],
            "regions_px": regions or [[0, 0, 32, 24]],
            "elapsed_ms": 1.0,
        }


def test_http_endpoint_reads_with_a_resident_reader():
    reader = FakeReader()
    app = create_app(settings=Settings(_env_file=None), screen_reader=reader)
    with TestClient(app) as client:
        ok = client.post("/v1/screen-text", files={"file": ("a.png", png_bytes(), "image/png")})
        assert ok.status_code == 200 and ok.json()["version"] == "svsg.screen-text/1"
        part = client.post(
            "/v1/screen-text",
            files={"file": ("a.png", png_bytes(), "image/png")},
            data={"regions": "[[0, 0, 10, 10]]"},
        )
        assert part.status_code == 200
        assert (
            client.post(
                "/v1/screen-text",
                files={"file": ("a.png", png_bytes(), "image/png")},
                data={"regions": "[[0, 0, 10]]"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/screen-text",
                files={"file": ("a.png", png_bytes(), "image/png")},
                data={"regions": "[[-5, 0, 10, 10]]"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/screen-text", files={"file": ("a.png", b"not an image", "image/png")}
            ).status_code
            == 422
        )
    assert reader.calls[:2] == [None, [(0, 0, 10, 10)]]


def test_http_endpoint_requires_api_key_when_configured():
    app = create_app(settings=Settings(api_key="k", _env_file=None), screen_reader=FakeReader())
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/screen-text", files={"file": ("a.png", png_bytes(), "image/png")}
            ).status_code
            == 401
        )


def test_reads_rendered_text_and_regions():
    pytest.importorskip("rapidocr_onnxruntime")
    import cv2

    from svsg.recognition.screen_text import ScreenTextReader

    image = np.full((400, 900, 3), 240, np.uint8)
    cv2.putText(
        image,
        "Save changes",
        (600, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image, "Account", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2, cv2.LINE_AA
    )
    reader = ScreenTextReader(threads=2)
    full = reader.read(image)
    texts = {row["text"].replace(" ", "").lower(): row for row in full["lines"]}
    assert "account" in texts and "savechanges" in texts
    assert [w["text"] for w in texts["savechanges"].get("words", [])] == ["Save", "changes"]
    x, y = texts["savechanges"]["center_px"]
    assert 600 <= x <= 820 and 320 <= y <= 355
    part = reader.read(image, [(590, 320, 830, 360)])
    assert [row["text"].replace(" ", "").lower() for row in part["lines"]] == ["savechanges"]
    assert abs(part["lines"][0]["center_px"][0] - x) <= 3


def test_split_words_at_spaces_and_wide_gaps():
    from svsg.recognition.screen_text import split_words

    # "#4810 Alvarez": a space between the cells; "显示名称李明": a wide gap, no space.
    quads = [[[x, 0], [x + 7, 0], [x + 7, 20], [x, 20]] for x in (0, 7, 14, 21, 28, 35, 42, 49)]
    words = split_words(list("ab cdefg"), quads, (100, 50, 160, 70), 100)
    assert [w["text"] for w in words] == ["ab", "cdefg"]
    assert words[1]["bbox_px"][0] >= 120 and words[1]["center_px"][1] == 60
    cjk = [[[x, 0], [x + 14, 0], [x + 14, 20], [x, 20]] for x in (0, 15, 29, 44, 66, 81)]
    words = split_words(list("显示名称李明"), cjk, (0, 0, 100, 24), 0)
    assert [w["text"] for w in words] == ["显示名称", "李明"]
    assert split_words(list("one"), quads[:3], (0, 0, 30, 20), 0) == []
    assert split_words(list("ab"), quads[:3], (0, 0, 30, 20), 0) == []  # mismatched lengths


def test_find_text_prefers_whole_lines_and_same_case():
    from svsg.recognition.screen_text import find_text

    result = {
        "lines": [
            {
                "text": "Account settings",
                "bbox_px": [0, 80, 200, 110],
                "score": 0.99,
                "low_confidence": False,
                "words": [
                    {"text": "Account", "bbox_px": [0, 80, 90, 110]},
                    {"text": "settings", "bbox_px": [100, 80, 200, 110]},
                ],
            },
            {
                "text": "Settings",
                "bbox_px": [300, 10, 360, 30],
                "score": 0.95,
                "low_confidence": False,
            },
        ]
    }
    hits = find_text(result, "Settings", exact=True)
    assert [h["text"] for h in hits] == ["Settings", "settings"]
    assert hits[0]["whole_line"] and hits[0]["center_px"] == [330, 20]
    assert find_text(result, "account set")[0]["text"] == "Account settings"
    assert find_text(result, "count", exact=True) == []
    with pytest.raises(ValueError):
        find_text(result, "  ")


def test_widened_regions_read_whole_lines():
    from svsg.recognition.screen_text import lines_touching, widen_regions

    assert widen_regions([(400, 300, 500, 326)], 1000) == [[244, 300, 656, 326]]
    assert widen_regions([(10, 0, 20, 10)], 1000) == [[0, 0, 140, 10]]
    result = {
        "lines": [
            {"text": "Status: done", "bbox_px": [300, 300, 520, 326]},
            {"text": "Other", "bbox_px": [600, 400, 700, 420]},
        ]
    }
    assert [r["text"] for r in lines_touching(result, [(400, 300, 500, 326)])] == ["Status: done"]
    # A checkbox square changed; its label on the same row is reported with it.
    label = {
        "lines": [
            {"text": "Notify team", "bbox_px": [70, 190, 180, 214]},
            {"text": "Confirm", "bbox_px": [60, 280, 140, 300]},
        ]
    }
    assert [r["text"] for r in lines_touching(label, [(40, 192, 62, 214)])] == ["Notify team"]
