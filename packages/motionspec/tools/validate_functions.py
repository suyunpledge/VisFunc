"""Local reproducible video/function round-trip checks; no model or network access."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from motionspec import analyze
from motionspec.fit import bezier_ease, spring_response
from motionspec.frames import FrameSource, read_image
from motionspec.synth import pop, slide, write_video
from svsg.video_functions import VideoRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--photo", type=Path, help="optional local photo for a controlled pan, not natural-video ground truth")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    clips = {"compressed_slide": slide(distance=120), "compressed_spring": pop()}
    if args.photo:
        photo = read_image(args.photo)
        if photo is None:
            raise ValueError("cannot decode photo")
        large = cv2.resize(photo, (700, 440))
        frames = [large[0:380, i:i+640].copy() for i in range(0, 61, 2)]
        clips["photo_controlled_pan"] = (frames, {"source_photo": str(args.photo), "not_natural_video": True})
        clips["photo_static"] = ([frames[0]]*12, {"static": True})
    report = {"test_kind": "encoded controlled clips, not object-recognition accuracy", "model_calls": 0, "cases": []}
    for name, (frames, truth) in clips.items():
        video = args.out / (name + ".mp4")
        write_video(video, frames, fps=60)
        start = time.perf_counter()
        spec = analyze(video, args.out / name, functions=True, sequence_id=name,
                       evidence_every_ms=200, sheets=False)
        runtime = VideoRuntime.load(args.out / name / "functions.json")
        expected_ids = {f.frame_index for f in runtime.bundle.frames}
        expected = {i: cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) for i, _, frame in
                    FrameSource(video).selected_raw() if i in expected_ids}
        pixel_errors = []
        for index, array in expected.items():
            restored = runtime.raster(index)
            actual = np.frombuffer(restored.samples, np.uint8).reshape(array.shape)
            pixel_errors.append(int(np.abs(actual.astype(int) - array.astype(int)).max()))
        sample_errors, fit_errors = [], []
        for segment in spec["segments"]:
            for sample in segment["samples"]:
                for key in ("x", "y", "feature_x", "feature_y", "scale", "rot_deg", "opacity", "progress", "luma"):
                    if key in sample:
                        value = runtime.motion(segment["id"], key, sample["t_ms"])["value"]
                        sample_errors.append(abs(value - sample[key]))
            for key, channel in segment["channels"].items():
                curve = channel.get("easing")
                if curve:
                    for tau in np.linspace(0, 1, 11):
                        t = float(channel["t_start_ms"] + tau*(channel["t_end_ms"]-channel["t_start_ms"]))
                        if curve["model"] == "cubic-bezier":
                            value = float(bezier_ease(*curve["params"], tau))
                        else:
                            value = float(spring_response(curve["damping_ratio"], curve["angular_frequency"],
                                                          (t-channel["t_start_ms"])/1000))
                        actual = runtime.motion(segment["id"], key+".progress", t, fitted=True)["value"]
                        fit_errors.append(abs(actual-value))
        for delta in runtime.bundle.differences:
            raw = expected[delta.after_frame].astype(int) - expected[delta.before_frame].astype(int)
            assert int(np.count_nonzero(np.any(raw != 0, axis=2))) == delta.changed_pixels
            assert np.allclose(np.abs(raw).mean(axis=(0, 1)), delta.mean_abs_rgba)
        assert max(pixel_errors) == 0
        assert max(sample_errors, default=0) == 0
        assert max(fit_errors, default=0) < 5e-4
        case = {"name": name, "truth": truth, "elapsed_s": round(time.perf_counter()-start, 3),
                "source_frames": len(frames), "evidence_frames": len(expected),
                "motions": len(spec["segments"]), "max_decoded_pixel_error": max(pixel_errors),
                "sample_values_checked": len(sample_errors), "max_sample_error": max(sample_errors, default=0),
                "fit_values_checked": len(fit_errors), "max_fit_evaluator_difference": max(fit_errors, default=0),
                "difference_pairs_checked": len(runtime.bundle.differences),
                "unique_raster_programs": len(list((args.out/name).glob("raster-*.json"))),
                "manifest_bytes": (args.out/name/"functions.json").stat().st_size,
                "raster_bytes": sum(p.stat().st_size for p in (args.out/name).glob("raster-*.json"))}
        report["cases"].append(case)
        (args.out / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(case, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
