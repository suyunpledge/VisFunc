"""Score decoded recording frames with generic, other-player and same-player icon banks."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from overlay_eval import edits

from svsg.recognition.evaluation import score_objects
from svsg.recognition.icon_matching import load_icons, match_icons
from svsg.recognition.overlays import measure_overlays, toolbar_region


def score(folder, mixed):
    banks = {"other_players": load_icons(mixed)}
    for row in json.loads((folder / "truth.json").read_text(encoding="utf8")):
        path = folder / (row["case"] + ".png")
        encoded = path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        video = path.with_suffix(".webm")
        if sha256(video.read_bytes()).hexdigest() != row["webm_sha256"]:
            raise ValueError(f"{video.name} differs from the labeled recording")
        cap = cv2.VideoCapture(str(video))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_index"])
            ok, decoded = cap.read()
        finally:
            cap.release()
        if not ok or not np.array_equal(bgr, decoded):
            raise ValueError(f"{path.name} differs from the labeled video frame")
        frame = measure_overlays(path, player_toolbar=True)["frames"][0]
        if row["player"] not in banks:
            banks[row["player"]] = load_icons(folder / (row["player"] + "-icons"))
        found = {"generic": frame["icons"]}
        for group, key in (("other_players", "other_players"), ("same_player", row["player"])):
            found[group] = match_icons(
                bgr, toolbar_region(*frame["size_px"]), banks[key], review_row=True
            )
        measurements = {}
        for group, boxes in found.items():
            metrics = score_objects(boxes, row["icons"])
            measurements[group] = {
                **{k: metrics[k] for k in ("tp", "fp", "fn", "matches")},
                "predictions": boxes,
            }
        text = "\n".join(line["text"] for line in frame["subtitles"])
        yield {
            **row,
            "frame_sha256": sha256(encoded).hexdigest(),
            "text_read": text,
            "edits": edits(row["text"], text),
            "subtitles": frame["subtitles"],
            "detached": frame.get("detached_text_candidates", []),
            "icon_scores": measurements,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("other_player_icons", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("w", encoding="utf8") as out:
        for result in score(args.folder, args.other_player_icons):
            print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
