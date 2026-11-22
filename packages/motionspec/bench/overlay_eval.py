"""Score paired native / multiscale overlay recognition on labeled player captures."""

import argparse
import json
import unicodedata
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from svsg.recognition.evaluation import score_objects
from svsg.recognition.icon_matching import load_icons, match_icons
from svsg.recognition.overlays import measure_overlays, toolbar_region
from svsg.recognition.subtitle_ocr import SubtitleReader


def edits(reference, text):
    last = list(range(len(text) + 1))
    for i, char in enumerate(reference, 1):
        curr = [i]
        for j, other in enumerate(text, 1):
            curr.append(min(last[j] + 1, curr[-1] + 1, last[j - 1] + (char != other)))
        last = curr
    return last[-1]


def native(reader, icons, bgr):
    h, w = bgr.shape[:2]
    margin = w // 20
    detected, _ = reader.engine(bgr, use_cls=False)
    lines = []
    for poly, text, score in detected or []:
        x, y = np.array(poly).mean(axis=0)
        if (
            score >= 0.75
            and any(c.isalnum() for c in text)
            and y >= int(h * 0.52)
            and abs(x - w / 2) <= 0.28 * (w - 2 * margin)
        ):
            lines.append(
                dict(
                    text=text,
                    bbox_px=[*np.min(poly, axis=0).tolist(), *np.max(poly, axis=0).tolist()],
                )
            )
    return dict(
        subtitles=sorted(lines, key=lambda a: (a["bbox_px"][1], a["bbox_px"][0])),
        icons=match_icons(bgr, (0, 0, w, h), icons, sizes=(16,)),
    )


def evaluate(folder, mixed_icons=None):
    reader, icons = SubtitleReader(), load_icons()
    calibrated = load_icons(folder / "icons") if (folder / "icons").is_dir() else []
    mixed = load_icons(mixed_icons) if mixed_icons else []
    for row in json.loads((folder / "truth.json").read_text(encoding="utf-8")):
        image = folder / f"case{row['case']}.png"
        raw = image.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        result = measure_overlays(image)["frames"][0]
        base = native(reader, icons, bgr)
        h, w = bgr.shape[:2]
        toolbar = toolbar_region(w, h)
        entry = dict(
            case=row["case"],
            sequence=row["sequence"],
            size=row["size"],
            png_sha256=sha256(raw).hexdigest(),
            reference=row["text"],
            expected_icons=row["icons"],
        )
        for key, data in (("native", base), ("multiscale", result)):
            predicted = "\n".join(x["text"] for x in data["subtitles"])
            ref = unicodedata.normalize("NFC", row["text"])
            pred = unicodedata.normalize("NFC", predicted)
            score = score_objects(data["icons"], row["icons"])
            entry[key] = dict(
                text=predicted,
                edits=edits(ref, pred),
                exact=ref == pred,
                subtitles=data["subtitles"],
                icons=data["icons"],
                tp=score["tp"],
                fp=score["fp"],
                fn=score["fn"],
            )
        if result.get("detached_text_candidates"):
            entry["detached_text_candidates"] = result["detached_text_candidates"]
        for key, bank in (
            ("toolbar", icons),
            ("calibrated_toolbar", calibrated),
            ("mixed_toolbar", mixed),
        ):
            if bank:
                found = match_icons(bgr, toolbar, bank, review_row=True)
                score = score_objects(found, row["icons"])
                entry[key] = dict(icons=found, tp=score["tp"], fp=score["fp"], fn=score["fn"])
        yield entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mixed-icons", type=Path, help="additional multi-style template bank")
    args = parser.parse_args()
    with args.output.open("w", encoding="utf-8") as out:
        for result in evaluate(args.folder, args.mixed_icons):
            print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
