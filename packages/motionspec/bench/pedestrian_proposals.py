"""Generate the detector-assisted ablation from decoded video only."""

import argparse
from hashlib import sha256
from pathlib import Path

import cv2
from pedestrian_clips import SPLITS, read, save

from svsg.image_functions.raster import RasterFunction
from svsg.l1_compiler.detector_adapter import YoloDetector
from svsg.recognition.video_candidates import frame_candidates


def make(
    folder,
    detector,
    weight_digest,
    *,
    detail=None,
    output=None,
    crowd_detector=None,
    overlap_review=True,
    head_focus=True,
):
    output = folder if output is None else output
    cap = cv2.VideoCapture(str(folder / "clip.mp4"))
    samples = len(read(folder / "source.json")["source_frames"])
    target, prior = samples + 1, samples - 2
    rasters = {}
    for index in range(target + 1):
        ok, frame = cap.read()
        if not ok:
            raise ValueError("incomplete clip")
        if index in (prior, target):
            h, w = frame.shape[:2]
            rasters[index] = RasterFunction(w, h, cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).tobytes())
    cap.release()
    assert (
        rasters[target].pixel_sha256 == RasterFunction.from_image(folder / "last.png").pixel_sha256
    )
    data = frame_candidates(
        detector,
        rasters[target],
        rasters[prior],
        "person",
        detail=detail,
        crowd_detector=crowd_detector,
        overlap_review=overlap_review,
        head_focus=head_focus,
    )
    data.update(
        model=Path(detector.model_name.split("::", 1)[-1]).name,
        weights_sha256=weight_digest,
        frame_index=target,
        previous_frame_index=prior,
    )
    save(output / "candidates-px.json", data)
    scale = [w, h, w, h]
    save(
        output / "proposals.json",
        {
            "model": data["model"],
            "status": data["status"],
            "label": "person",
            "coordinates": "normalized 0..1000 [left,top,right,bottom] on last frame",
            "boxes": [
                [round(n * 1000 / d) for n, d in zip(b["bbox_px"], scale, strict=True)]
                for b in data["boxes"]
            ],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("--whole-frame", action="store_true", help="use the whole-frame baseline")
    args = parser.parse_args()
    digest = sha256(args.weights.read_bytes()).hexdigest()
    detector = YoloDetector(str(args.weights), imgsz=1280)
    for sequence in SPLITS[args.split]:
        make(args.root / sequence, detector, digest, detail=False if args.whole_frame else None)
