"""命令行入口。

    motionspec analyze clip.mp4 -o out/        分析视频，写出 spec.json / summary.md / 标注图
    motionspec check out/spec.json claims.json 用规格校验一组结构化断言
    motionspec demo -o demo/                   生成合成动效并分析，用来快速看输出长什么样
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze
from .claims import check
from .spec import summarize


def _seconds(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def _reject_constant(value: str):
    raise ValueError(f"invalid JSON numeric constant {value}")


def _load_json(filename: str):
    return json.loads(Path(filename).read_text(encoding="utf-8"), parse_constant=_reject_constant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="motionspec", description="Turn UI motion in a video into a spec an AI can read.")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse a video file or a directory of frames")
    a.add_argument("source")
    a.add_argument("-o", "--out", default="motionspec-out")
    a.add_argument("--start", type=float, help="start time in seconds")
    a.add_argument("--end", type=float, help="end time in seconds")
    a.add_argument("--fps", type=float, help="override frame rate (required for image directories)")
    a.add_argument("--max-dim", type=int, default=960, help="analysis resolution, longest side")
    a.add_argument("--max-segments", type=int, default=12)
    a.add_argument("--min-area", type=float, default=120.0, help="ignore changes smaller than this (analysis px^2)")
    a.add_argument("--threshold", type=int, default=12, help="per-pixel change threshold, 0-255")
    a.add_argument("--no-sheets", action="store_true")
    a.add_argument("--no-raw-frames", action="store_true", help="omit original-resolution evidence PNGs")
    a.add_argument("--functions", action="store_true", help="also export shared SVSG image/time functions")
    a.add_argument("--sequence-id", help="identity used to bind SVSG per-frame recognition")
    a.add_argument("--function-max-frames", type=int, default=64)
    a.add_argument("--evidence-every-ms", type=float, help="also retain periodic source frames")
    a.add_argument("--max-frames", type=int, default=1_000_000, help="maximum frames in the selected range")
    a.add_argument("--max-crop-mb", type=int, default=150, help="shared crop cache budget in decimal MB")
    a.add_argument("--ambient-ms", type=float, default=3000.0,
                   help="treat cells that keep changing for this long as an animated backdrop; 0 disables")

    c = sub.add_parser("check", help="verify structured claims against a spec")
    c.add_argument("spec")
    c.add_argument("claims")

    f = sub.add_parser("functions", help="export shared functions from an existing spec and evidence")
    f.add_argument("spec")
    f.add_argument("-o", "--out", required=True)
    f.add_argument("--sequence-id")
    f.add_argument("--max-frames", type=int, default=64)

    q = sub.add_parser("query", help="evaluate a data-only query against shared functions")
    q.add_argument("manifest")
    q.add_argument("parameters", nargs="?", help="JSON file with one function call or a list of calls")
    q.add_argument("--list", action="store_true", help="print a compact catalogue of segments, functions and frames")

    d = sub.add_parser("demo", help="analyse built-in synthetic clips")
    d.add_argument("-o", "--out", default="motionspec-demo")

    candidates = sub.add_parser("candidates", help="measure object proposals on an exported frame")
    candidates.add_argument("manifest")
    candidates.add_argument("--frame", type=int, required=True)
    candidates.add_argument("--previous-frame", type=int)
    candidates.add_argument("--weights", type=Path, required=True)
    candidates.add_argument("--crowd-weights", type=Path, help="optional CrowdHuman person/head ONNX weights")
    candidates.add_argument("--no-overlap-review", action="store_true", help="skip the second-person review")
    candidates.add_argument(
        "--no-head-focus", action="store_true", help="skip extra close-ups of unassigned heads"
    )
    candidates.add_argument("--label", default="person")
    views = candidates.add_mutually_exclusive_group()
    views.add_argument("--whole-frame", dest="detail", action="store_false", help="skip detail crops")
    views.add_argument("--detail-always", dest="detail", action="store_true", help="always inspect detail crops")
    candidates.set_defaults(detail=None)

    overlays = sub.add_parser("overlays", help="read subtitles and locate UI icons on sampled frames")
    overlays.add_argument("source", type=Path, help="image or exported functions.json")
    overlays.add_argument("--frame", type=int, help="read one exported frame (default: all)")
    overlays.add_argument("--only", choices=("both", "subtitles", "icons"), default="both")
    overlays.add_argument("--subtitle-region", nargs=4, type=int, metavar=("L", "T", "R", "B"))
    icon_area = overlays.add_mutually_exclusive_group()
    icon_area.add_argument("--icon-region", nargs=4, type=int, metavar=("L", "T", "R", "B"))
    icon_area.add_argument(
        "--player-toolbar", action="store_true",
        help="search the bottom 15%% for icons and separate player readouts from captions",
    )
    icon_bank = overlays.add_mutually_exclusive_group()
    icon_bank.add_argument("--icon-dir", type=Path, help="PNG exemplars named by label")
    icon_bank.add_argument("--fallback-icon-dir", type=Path,
                           help="fill missed toolbar icons with row-supported PNG exemplars")

    t = sub.add_parser("trace", help="convert a Qt probe trace (motionspec.qt_probe) into a spec")
    t.add_argument("trace")
    t.add_argument("-o", "--out", default="motionspec-out")
    t.add_argument("--max-segments", type=int, default=40)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.command == "trace":
        from .trace import spec_from_trace

        try:
            spec = spec_from_trace(_load_json(args.trace), args.out, max_segments=args.max_segments)
        except (OSError, ValueError, KeyError) as exc:
            t.exit(2, f"motionspec trace: {exc}\n")
        print(summarize(spec))
        print(f"written to {Path(args.out).resolve()}")
        return 0

    if args.command == "analyze":
        try:
            spec = analyze(
                args.source, args.out, fps=args.fps, max_dim=args.max_dim,
                start_ms=_seconds(args.start), end_ms=_seconds(args.end), max_segments=args.max_segments,
                min_area=args.min_area, threshold=args.threshold, sheets=not args.no_sheets,
                raw_frames=not args.no_raw_frames, max_frames=args.max_frames, max_crop_bytes=args.max_crop_mb * 1_000_000,
                ambient_ms=args.ambient_ms,
                functions=args.functions, sequence_id=args.sequence_id,
                function_max_frames=args.function_max_frames, evidence_every_ms=args.evidence_every_ms,
            )
        except (OSError, ValueError) as exc:
            a.exit(2, f"motionspec analyze: {exc}\n")
        print(summarize(spec))
        print(f"written to {Path(args.out).resolve()}")
        return 0

    if args.command == "functions":
        from .functions import export_functions

        try:
            result = export_functions(_load_json(args.spec), Path(args.spec).parent, args.out,
                                      sequence_id=args.sequence_id, max_frames=args.max_frames)
        except (OSError, ValueError) as exc:
            f.exit(2, f"motionspec functions: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "overlays":
        try:
            from svsg.recognition.overlays import measure_overlays

            result = measure_overlays(
                args.source, frame=args.frame, only=args.only, icon_dir=args.icon_dir,
                subtitle_region=args.subtitle_region, icon_region=args.icon_region,
                player_toolbar=args.player_toolbar, fallback_icon_dir=args.fallback_icon_dir,
            )
        except ImportError as exc:
            overlays.exit(2, f"install svsg[overlays] for screen recognition: {exc}\n")
        except (OSError, ValueError) as exc:
            overlays.exit(2, f"motionspec overlays: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0

    if args.command == "candidates":
        try:
            from svsg.l1_compiler.detector_adapter import YoloDetector
            from svsg.recognition.crowd_detector import CrowdHumanDetector
            from svsg.recognition.video_candidates import frame_candidates
            from svsg.video_functions import VideoRuntime

            rt = VideoRuntime.load(args.manifest)
            if args.previous_frame is not None and args.previous_frame >= args.frame:
                raise ValueError("previous frame must precede target frame")
            if not args.weights.is_file():
                raise ValueError("supply an installed local detector weights file")
            target = rt.raster(args.frame)
            prior = rt.raster(args.previous_frame) if args.previous_frame is not None else None
            detector = YoloDetector(str(args.weights), imgsz=1280)
            crowd_detector = CrowdHumanDetector(args.crowd_weights) if args.crowd_weights else None
            result = frame_candidates(
                detector, target, prior, args.label,
                detail=args.detail, crowd_detector=crowd_detector,
                overlap_review=not args.no_overlap_review, head_focus=not args.no_head_focus,
            )
            result.update(sequence_id=rt.bundle.sequence_id, frame_index=args.frame,
                          previous_frame_index=args.previous_frame)
        except (OSError, ValueError, ImportError) as exc:
            candidates.exit(2, f"motionspec candidates: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0

    if args.command == "query":
        from .functions import catalogue, run_queries

        try:
            if args.list:
                result = catalogue(args.manifest)
            elif args.parameters is None:
                raise ValueError("give a parameters file, or --list to see what can be queried")
            else:
                result = run_queries(args.manifest, _load_json(args.parameters))
        except (OSError, ValueError) as exc:
            q.exit(2, f"motionspec query: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    if args.command == "check":
        try:
            verdict = check(_load_json(args.spec), _load_json(args.claims))
        except (OSError, ValueError) as exc:
            verdict = {"passed": False, "invalid_input": str(exc)}
        print(json.dumps(verdict, ensure_ascii=False, indent=2, allow_nan=False))
        if any(field in verdict for field in ("invalid_spec", "invalid_claims", "invalid_input")):
            return 2
        return {"passed": 0, "failed": 1, "insufficient_evidence": 3}[verdict["verdict"]]

    from .synth import CASES

    for name, make in CASES.items():
        frames, _ = make()
        spec = analyze(frames, Path(args.out) / name, fps=60.0)
        print(f"## {name}\n")
        print(summarize(spec))
    print(f"written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
