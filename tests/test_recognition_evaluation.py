from svsg.recognition.evaluation import aggregate, score_objects


def obj(label, box):
    return {"label": label, "bbox_px": box}


def test_scoring_requires_correct_location_and_unique_ground_truth():
    expected = [obj("person", [0, 0, 20, 20])]
    score = score_objects(
        [
            obj("person", [0, 0, 20, 20]),
            obj("person", [0, 0, 20, 20]),
            obj("person", [80, 80, 100, 100]),
            obj("cup", [0, 0, 20, 20]),
        ],
        expected,
    )
    assert (score["tp"], score["fp"], score["fn"]) == (1, 3, 0)
    assert score["precision"] == 0.25
    negative = score_objects([obj("cup", [0, 0, 20, 20])], [])
    assert negative["fp"] == 1
    assert aggregate([score, negative])["precision"] == 0.2


def test_matching_is_not_greedy_and_missing_predictions_lower_recall():
    expected = [obj("cup", [0, 0, 10, 10]), obj("cup", [5, 0, 15, 10])]
    predictions = [obj("cup", [2, 0, 12, 10]), obj("cup", [0, 0, 10, 10])]
    assert score_objects(predictions, expected)["tp"] == 2
    assert score_objects([], expected)["fn"] == 2
