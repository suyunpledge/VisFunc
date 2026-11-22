from copy import deepcopy
from types import SimpleNamespace

import pytest

from svsg.recognition.subtitle_sequence import review_caption_sequence


def observation(word, box):
    return dict(text=word, bbox_px=box, score=0.99)


def series():
    word = observation("WHITE SAND", (41, 56, 141, 64))
    parts = [observation("WHITE", (40, 55, 95, 65)), observation("SAND", (94, 55, 142, 65))]
    return [
        dict(
            frame_index=5 * i,
            timestamp_ms=200 * i,
            size_px=[180, 100],
            sequence_id="one",
            pixel_sha256=str(i),
            detached_text_candidates=parts if i < 2 else [],
            subtitles=[] if i < 2 else [word],
        )
        for i in range(5)
    ]


def test_two_direct_witnesses_can_resolve_split_or_joined_words_in_either_direction():
    for frames in [series(), list(reversed(series()))]:
        for i, f in enumerate(frames):
            f["timestamp_ms"] = i * 200 + (5e-13 if i == 4 else 0)
        raw = deepcopy(frames)
        out = review_caption_sequence(frames)
        assert frames == raw
        assert all(not f["subtitles"] for f in out)
        moved = next(f for f in out if f["frame_index"] == 20)["detached_text_candidates"][0]
        assert moved["temporal_review"]["support_frame_indices"] == [0, 5]
        assert review_caption_sequence(frames, 20) == [
            next(f for f in out if f["frame_index"] == 20)
        ]


@pytest.mark.parametrize(
    "change", ["gap", "cut", "word", "duplicate", "lower", "sequence", "connected"]
)
def test_title_review_stops_at_discontinuities_or_ambiguous_evidence(change):
    frames = series()
    if change == "gap":
        frames[-1]["timestamp_ms"] += 800
    elif change == "cut":
        frames[2]["subtitles"] = []
    elif change == "word":
        frames[2]["subtitles"] = [observation("WHITE LAND", (40, 56, 142, 64))]
    elif change == "duplicate":
        frames[0]["pixel_sha256"] = frames[1]["pixel_sha256"]
    elif change == "lower":
        frames[-1]["subtitles"] = [observation("WHITE SAND", (40, 78, 142, 86))]
    elif change == "sequence":
        frames[-1]["sequence_id"] = "two"
    else:
        frames[-1]["subtitles"] = [
            *frames[-1]["subtitles"],
            observation("Second line", (42, 68, 142, 78)),
        ]
    assert review_caption_sequence(frames, 20)[0]["subtitles"] == frames[-1]["subtitles"]


def test_target_frame_reads_only_time_bounded_context(monkeypatch, tmp_path):
    from svsg.recognition.overlays import _frames
    from svsg.video_functions import VideoRuntime

    refs = [
        SimpleNamespace(frame_index=i, timestamp_ms=t)
        for i, t in enumerate([0, 1000, 1200, 1800, 3000])
    ]
    runtime = SimpleNamespace(
        bundle=SimpleNamespace(sequence_id="a", frames=refs), raster=lambda i: i
    )
    monkeypatch.setattr(VideoRuntime, "load", lambda _: runtime)
    source = tmp_path / "functions.json"
    assert [row[1] for row in _frames(source, 2, context_ms=800)] == [1, 2, 3]
    assert [row[1] for row in _frames(source, 2)] == [2]
