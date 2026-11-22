"""Export five real recording frames and score the normal manifest recognition path."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from svsg.recognition.evaluation import score_objects
from svsg.recognition.overlays import measure_overlays
from svsg.video_functions import VideoRuntime, export_motion_spec


def samples(recording, row):
    if sha256(recording.read_bytes()).hexdigest() != row["webm_sha256"]:
        raise ValueError(f"recording differs from truth: {recording}")
    cap = cv2.VideoCapture(str(recording))
    try:
        if cap.get(cv2.CAP_PROP_FPS) != row["fps"]:
            raise ValueError("recording and label clocks differ")
        for n in range(row["frame_index"] - 20, row["frame_index"] + 1, 5):
            if n < 0 or not cap.set(cv2.CAP_PROP_POS_FRAMES, n):
                raise ValueError(f"cannot seek to frame {n}")
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"cannot decode frame {n}")
            yield n, cap.get(cv2.CAP_PROP_POS_MSEC), bgr
    finally:
        cap.release()


def prepare_sequence(recording, row, dest):
    refs = []
    with TemporaryDirectory() as tmp:
        for n, stamp, bgr in samples(recording, row):
            h, w = bgr.shape[:2]
            png = cv2.imencode(".png", bgr)[1].tobytes()
            digest = sha256(png).hexdigest()
            filename = f"frame-{n:08d}-{digest}.png"
            (Path(tmp) / filename).write_bytes(png)
            refs.append(dict(frame=n, t_ms=stamp, roles=["overlay"], image=filename, sha256=digest))
        spec = dict(
            schema="motionspec/0.1",
            evidence_frames=refs,
            segments=[],
            source=dict(
                timestamps="container",
                width=w,
                height=h,
                t_start_ms=refs[0]["t_ms"],
                t_end_ms=refs[-1]["t_ms"],
            ),
            time_resolution_ms=1000 / row["fps"],
        )
        export_motion_spec(spec, tmp, dest, sequence_id=row["case"])


def verify_samples(manifest, recording, row):
    bundle = VideoRuntime.load(manifest).bundle
    if bundle.sequence_id != row["case"]:
        raise ValueError("manifest belongs to a different recording")
    for ref, (n, stamp, bgr) in zip(bundle.frames, samples(recording, row), strict=True):
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        if (
            ref.frame_index != n
            or abs(ref.timestamp_ms - stamp) > 1e-6
            or sha256(rgba.tobytes()).hexdigest() != ref.image_functions.pixel_sha256
        ):
            raise ValueError(f"exported frame differs from recording at {n}")


def score_sequence(manifest, row, all_text, fallback=None):
    packet = measure_overlays(
        manifest,
        frame=None if all_text else row["frame_index"],
        only="subtitles" if all_text else "both",
        player_toolbar=True,
        **({"fallback_icon_dir": fallback} if fallback is not None else {}),
    )
    for frame in packet["frames"]:
        result = dict(
            case=row["case"],
            player=row["player"],
            source_sha256=row["webm_sha256"],
            frame=frame["frame_index"],
            timestamp_ms=frame["timestamp_ms"],
            pixel_sha256=frame["pixel_sha256"],
            title=row["title"],
            expected_text=row["text"],
            text="\n".join(x["text"] for x in frame["subtitles"]),
            subtitles=frame["subtitles"],
            detached=frame.get("detached_text_candidates", []),
        )
        if not all_text:
            result.update(
                expected_icons=row["icons"],
                icons=frame["icons"],
                score=score_objects(frame["icons"], row["icons"]),
            )
        yield result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recordings", type=Path, help="directory containing truth.json and WebM files"
    )
    parser.add_argument("manifests", type=Path)
    parser.add_argument("--output", type=Path, help="score prepared manifests into this JSONL")
    parser.add_argument("--fallback-icon-dir", type=Path, help="use prior-player toolbar exemplars")
    parser.add_argument(
        "--all-text", action="store_true", help="score subtitles on all five frames"
    )
    args = parser.parse_args()
    truth = json.loads((args.recordings / "truth.json").read_text(encoding="utf8"))
    if args.output is None:
        for row in truth:
            prepare_sequence(
                args.recordings / (row["case"] + ".webm"), row, args.manifests / row["case"]
            )
    else:
        with args.output.open("w", encoding="utf8") as out:
            for row in truth:
                manifest = args.manifests / row["case"] / "functions.json"
                verify_samples(manifest, args.recordings / (row["case"] + ".webm"), row)
                for entry in score_sequence(manifest, row, args.all_text, args.fallback_icon_dir):
                    print(json.dumps(entry, ensure_ascii=False), file=out, flush=True)
