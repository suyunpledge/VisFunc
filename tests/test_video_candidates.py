import pytest

from svsg.image_functions.raster import RasterFunction
from svsg.l1_compiler.detector_adapter import RawDetection
from svsg.recognition.video_candidates import frame_candidates


def raster(value):
    return RasterFunction(100, 60, bytes([value]) * (100 * 60 * 4))


class Detector:
    model_name = "fixture"

    def detect_raster(self, image):
        if image.samples[0] == 1:
            return [
                RawDetection("person", (10, 10, 30, 50), 0.28),
                RawDetection("person", (9, 9, 29, 49), 0.29),
                RawDetection("person", (60, 20, 73, 56), 0.9),
                RawDetection("car", (3, 3, 20, 20), 0.99),
            ]
        return [
            RawDetection("person", (10, 10, 30, 50), 0.6),
            RawDetection("person", (0, 0, 8, 18), 0.6),
        ]


def test_temporal_evidence_supports_only_one_current_candidate():
    current, prior = raster(1), raster(2)
    result = frame_candidates(Detector(), current, prior, "person")
    assert [(box["bbox_px"], box["prior_iou"]) for box in result["boxes"]] == [
        ([10, 10, 30, 50], 1),
        ([60, 20, 73, 56], None),
    ]
    assert result["pixel_sha256"] == current.pixel_sha256
    assert result["previous_pixel_sha256"] == prior.pixel_sha256


@pytest.mark.parametrize("previous", [None, raster(1)])
def test_absent_or_repeated_frame_does_not_confirm_low_score_boxes(previous):
    result = frame_candidates(Detector(), raster(1), previous, "person")
    assert [b["bbox_px"] for b in result["boxes"]] == [[60, 20, 73, 56]]


class TileDetector:
    model_name = "tiles-fixture"

    def __init__(self, full_score=None):
        self.full_score = full_score
        self.calls = []

    def detect_raster(self, image):
        # Each pixel encodes the frame number and original x/y coordinates.
        stamp, x, y, _ = image.pixel(0, 0)
        self.calls.append((stamp, x, y, image.width))
        if image.width == 100:
            return (
                []
                if self.full_score is None
                else [RawDetection("person", (48, 27, 54, 33), self.full_score)]
            )
        # A single person appears in all four overlapping crops.
        visible = RawDetection("person", (48 - x, 27 - y, 54 - x, 33 - y), 0.65)
        # A second detection touches the internal edge of the right-hand crop.
        edge = [RawDetection("person", (1, 5, 8, 15), 0.9)] if x else []
        return [visible, *edge]


def marked_raster(stamp):
    return RasterFunction(
        100, 60, bytes(v for y in range(60) for x in range(100) for v in (stamp, x, y, 255))
    )


def test_new_small_box_needs_a_distinct_frame_and_maps_back_only_once():
    now, prior = marked_raster(1), marked_raster(2)
    detector = TileDetector()
    results = frame_candidates(detector, now, prior, "person")
    (box,) = results["boxes"]
    assert box["bbox_px"] == box["prior_bbox_px"] == [48, 27, 54, 33]
    assert box["prior_iou"] == 1 and box["view_px"] != [0, 0, 100, 60]
    for earlier in (None, now):
        assert not frame_candidates(detector, now, earlier, "person")["boxes"]
    assert not frame_candidates(detector, now, prior, "person", detail=False)["boxes"]


def test_spatial_support_retains_original_box_and_skips_unneeded_prior_tiles():
    detector = TileDetector(full_score=0.3)
    result = frame_candidates(detector, marked_raster(1), None, "person")
    (box,) = result["boxes"]
    assert box["bbox_px"] == box["supporting_detail_bbox_px"] == [48, 27, 54, 33]
    assert box["view_px"] == [0, 0, 100, 60] and box["prior_iou"] is None
    detector.calls.clear()
    frame_candidates(detector, marked_raster(1), marked_raster(2), "person", detail=True)
    assert [stamp for stamp, x, y, w in detector.calls if w < 100] == [1] * 4


def test_auto_skips_detail_on_confirmed_larger_box():
    detector = TileDetector(full_score=0.8)
    result = frame_candidates(detector, marked_raster(1), marked_raster(2), "person")
    assert not result["detail_views"] and len(result["boxes"]) == 1
    assert len(detector.calls) == 2


def test_auto_uses_small_hints_and_limits_the_size_of_new_tile_only_boxes():
    class WithSmallHint(TileDetector):
        def detect_raster(self, image):
            if image.width == 100:
                return [RawDetection("person", (1, 1, 3, 6), 0.8)]
            _, x, y, _ = image.pixel(0, 0)
            large = RawDetection("person", (19 - x, 8 - y, 33 - x, 24 - y), 0.7)
            return [*super().detect_raster(image), large]

    now, prior = marked_raster(1), marked_raster(2)
    auto = frame_candidates(WithSmallHint(), now, prior, "person")
    always = frame_candidates(WithSmallHint(), now, prior, "person", detail=True)
    assert auto["detail_views"]
    assert [b["bbox_px"] for b in auto["boxes"]] == [[1, 1, 3, 6], [48, 27, 54, 33]]
    assert [19, 8, 33, 24] in [b["bbox_px"] for b in always["boxes"]]


class CrowdFixture:
    model_name = "body-and-head"
    weights_sha256 = "fixture"

    def __init__(self, now, prior=()):
        self.frames = (now, prior)

    def detect_raster(self, image):
        return self.frames[image.pixel(0, 0)[0] - 1]


def test_crowd_full_body_needs_current_body_and_unique_head_or_distinct_prior():
    bodies = [
        RawDetection("person", (5, 5, 33, 56), 0.78),
        RawDetection("person", (23, 5, 51, 56), 0.8),
    ]
    head = RawDetection("head", (22, 6, 27, 14), 0.65)
    provider = CrowdFixture([*bodies, head], [bodies[1]])
    now, prior = marked_raster(1), marked_raster(2)
    base = TileDetector()
    once = frame_candidates(base, now, now, "person", detail=False, crowd_detector=provider)
    assert len(once["boxes"]) == 1 and once["boxes"][0]["head_bbox_px"] == list(head.bbox)
    assert once["boxes"][0]["prior_bbox_px"] is None
    supported = frame_candidates(base, now, prior, "person", detail=False, crowd_detector=provider)
    assert [b["bbox_px"] for b in supported["boxes"]] == [list(b.bbox) for b in bodies]
    assert supported["boxes"][1]["prior_bbox_px"] == list(bodies[1].bbox)
    assert supported["boxes"][1]["head_bbox_px"] is None
    assert supported["crowd_model"]["name"] == provider.model_name
    for observations in ([head], [RawDetection("person", (23, 5, 51, 56), 0.5), head]):
        provider = CrowdFixture(observations, bodies)
        assert not frame_candidates(
            base, now, prior, "person", detail=False, crowd_detector=provider
        )["boxes"]


def test_crowd_preserves_baseline_box_and_rejects_other_labels():
    detector = TileDetector(full_score=0.8)
    person = RawDetection("person", (48, 27, 54, 33), 0.8)
    crowd = CrowdFixture([person], [person])
    now, prior = marked_raster(1), marked_raster(2)
    result = frame_candidates(detector, now, prior, "person", crowd_detector=crowd)
    (box,) = result["boxes"]
    assert box["bbox_px"] == list(person.bbox) and "source" not in box
    with pytest.raises(ValueError, match="person"):
        frame_candidates(detector, now, prior, "car", crowd_detector=crowd)


def test_weak_body_and_head_need_consistent_prior_position():
    body = RawDetection("person", (10, 5, 30, 50), 0.65)
    head = RawDetection("head", (15, 6, 23, 14), 0.9)
    drifted = RawDetection("person", (18, 5, 38, 50), 0.85)
    now, prior = marked_raster(1), marked_raster(2)
    for before, expected_count in (([drifted], 0), ([body], 1)):
        crowd = CrowdFixture([body, head], before)
        result = frame_candidates(
            TileDetector(), now, prior, "person", detail=False, crowd_detector=crowd
        )
        assert len(result["boxes"]) == expected_count


def test_crowd_decode_unpads_coordinates_and_keeps_overlapping_head_and_body():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from svsg.recognition.crowd_detector import CrowdHumanDetector

    class Session:
        def run(self, names, inputs):
            blob = inputs["image"]
            assert blob.shape == (1, 3, 640, 640)
            assert blob[0, 0, 128, 0] == pytest.approx(1 / 255)
            return [
                np.array(
                    [[[320, 320, 256, 256, 0.9, 0.95, 0.05], [320, 240, 64, 64, 0.9, 0.05, 0.95]]],
                    dtype=np.float32,
                )
            ]

    detector = CrowdHumanDetector.__new__(CrowdHumanDetector)
    detector.session = Session()
    detector.input_name = "image"
    detector.output_name = "output"
    boxes = detector.detect_raster(marked_raster(1))
    assert [(b.label, b.bbox) for b in boxes] == [
        ("person", (30, 10, 70, 50)),
        ("head", (45, 12, 55, 22)),
    ]


@pytest.mark.parametrize("evidence", ["separate", "one_head", "one_prior", "repeat", "disabled"])
def test_overlapping_neighbor_requires_its_own_current_head_and_prior_body(evidence):
    first = RawDetection("person", (10, 5, 42, 55), 0.85)
    second = RawDetection("person", (21, 5, 53, 55), 0.76)
    left_head = RawDetection("head", (17, 6, 26, 15), 0.78)
    right_head = RawDetection("head", (36, 6, 45, 15), 0.72)
    heads = [left_head] if evidence == "one_head" else [left_head, right_head]
    prior = [first] if evidence == "one_prior" else [first, second]
    crowd = CrowdFixture([first, second, *heads], prior)

    class WholeFrame:
        model_name = "whole"

        def detect_raster(self, image):
            return [first]

    now = marked_raster(1)
    result = frame_candidates(
        WholeFrame(),
        now,
        now if evidence == "repeat" else marked_raster(2),
        "person",
        detail=False,
        crowd_detector=crowd,
        overlap_review=evidence != "disabled",
    )
    expected = [list(first.bbox)]
    if evidence == "separate":
        expected.append(list(second.bbox))
        recovered = result["boxes"][-1]
        assert recovered["prior_bbox_px"] == list(second.bbox)
        assert recovered["head_bbox_px"] == list(right_head.bbox)
        assert recovered["separated_from"] == [
            {"bbox_px": list(first.bbox), "head_bbox_px": list(left_head.bbox)}
        ]
    assert [b["bbox_px"] for b in result["boxes"]] == expected


def test_two_new_neighbors_keep_distinct_tracks_and_merge_repeated_crop_observations():
    from svsg.recognition.crowd_pairs import _Pair, _same_observation
    from svsg.recognition.video_evidence import _Box

    full = (0, 0, 100, 60)

    def observation(body, head, view=full):
        box = _Box(body, 0.8, view)
        return _Pair(box, _Box(head, 0.8, view), box, [])

    a = observation((10, 5, 42, 55), (16, 6, 23, 15))
    b = observation((21, 5, 53, 55), (35, 6, 43, 15))
    assert not _same_observation(a, b)
    jittered = observation((11, 6, 43, 54), (18, 7, 26, 16), (5, 0, 55, 60))
    assert _same_observation(a, jittered)
