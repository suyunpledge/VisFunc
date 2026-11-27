"""命令行入口。

    motionspec analyze clip.mp4 -o out/        分析视频，写出 spec.json / summary.md / 标注图
    motionspec check out/spec.json claims.json 用规格校验一组结构化断言
    motionspec demo -o demo/                   生成合成动效并分析，用来快速看输出长什么样
    motionspec settle --screen --read-text     等屏幕停止变化，报告变化区域和其中的文字
    motionspec screen-text shot.png            读出截图上所有文字，带框和点击点
    motionspec agent-mcp                       以 MCP 服务提供屏幕工具，供 Agent 调用
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
    a.add_argument("--layout-edges", action="store_true", help="track resizing panels and pushed neighbours by their edges")
    a.add_argument("--mark-uncertain", action="store_true",
                   help="mark unsupported fields and explain their uncertainty in the summary")
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

    st = sub.add_parser("settle", help="wait until the screen (or a recording) stops changing")
    st_source = st.add_mutually_exclusive_group(required=True)
    st_source.add_argument("source", nargs="?",
                           help="video file or frame directory recorded after an action")
    st_source.add_argument("--screen", action="store_true", help="capture the live screen")
    st.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                    help="screen area to capture, in physical pixels")
    st.add_argument("--fps", type=float, help="frame rate of a frame directory")
    st.add_argument("--max-fps", type=float, default=30.0, help="screen capture rate limit")
    st.add_argument("--quiet-ms", type=float, default=200.0,
                    help="stillness required after the last change")
    st.add_argument("--expect-ms", type=float, default=1000.0,
                    help="report 'unchanged' if nothing changes within this time")
    st.add_argument("--timeout-ms", type=float, default=10_000.0)
    st.add_argument("--ambient-ms", type=float, default=3000.0,
                    help="ignore areas that keep changing this long (video, caret); "
                         "0 waits for everything")
    st.add_argument("--ignore", nargs=4, type=int, action="append", default=[],
                    metavar=("L", "T", "R", "B"),
                    help="area to exclude, such as a clock; repeatable")
    st.add_argument("--read-text", action="store_true",
                    help="also read the text inside the changed regions")
    st.add_argument("--save-frame", type=Path, help="write the final frame as PNG")

    sc = sub.add_parser("screen-text",
                        help="read all text on a screenshot with boxes and click points")
    sc.add_argument("image", type=Path)
    sc.add_argument("--region", nargs=4, type=int, action="append", metavar=("L", "T", "R", "B"),
                    help="read only this area; repeatable")

    am = sub.add_parser("agent-mcp", help="serve screen tools for agents over MCP (stdio)")
    am.add_argument("--max-fps", type=float, default=60.0, help="screen capture rate limit")
    am.add_argument("--history-seconds", type=float, default=20.0,
                    help="sample window changes between calls for up to 60 s; 0 disables")
    am.add_argument("--http", metavar="HOST:PORT",
                    help="serve Streamable HTTP at http://HOST:PORT/mcp instead of stdio")
    am.add_argument("--allow-window", action="append", default=[], metavar="TITLE",
                    help="enable click/text tools in windows titled with TITLE; "
                         "'TITLE | program.exe' also requires that program; repeatable")
    am.add_argument("--no-images", action="store_true",
                    help="omit the screenshot tool (for clients that pass images as text)")
    am.add_argument("--max-wait-ms", type=float,
                    help="cap waits per call; defaults to 8000 over HTTP for 10 s client timeouts")
    am.add_argument("--log-file", help="append one JSON line per tool call (rotates at 5 MB)")
    am.add_argument("--workers", type=int, default=4,
                    help="HTTP worker threads; calls on one window still run one at a time")
    am.add_argument("--insecure-no-token", action="store_true",
                    help="serve HTTP without VISFUNC_MCP_TOKEN (any local program may then "
                         "read the screen and act on allowed windows)")

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
                layout_edges=args.layout_edges,
                mark_uncertain=args.mark_uncertain,
                functions=args.functions, sequence_id=args.sequence_id,
                function_max_frames=args.function_max_frames, evidence_every_ms=args.evidence_every_ms,
            )
        except (OSError, ValueError) as exc:
            a.exit(2, f"motionspec analyze: {exc}\n")
        print(summarize(spec, mark_uncertain=args.mark_uncertain))
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

    if args.command == "settle":
        from .frames import FrameSource, write_image
        from .settle import capture_screen, watch

        try:
            if args.region and not args.screen:
                raise ValueError("--region applies to --screen; "
                                 "use --ignore to exclude parts of a recording")
            reader = None
            if args.read_text:  # Load OCR before watching so the wait is not delayed afterwards.
                from svsg.recognition.screen_text import ScreenTextReader

                reader = ScreenTextReader()
            if args.screen:
                area = tuple(args.region) if args.region else None
                frames = capture_screen(area, max_fps=args.max_fps)
            else:
                source = FrameSource(args.source, fps=args.fps)
                frames = ((t_ms, frame) for _, t_ms, frame in source.selected_raw())
            result, last = watch(
                frames, quiet_ms=args.quiet_ms, expect_change_ms=args.expect_ms,
                timeout_ms=args.timeout_ms,
                ambient_ms=args.ambient_ms, ignore=[tuple(b) for b in args.ignore],
            )
            if reader is not None:
                from svsg.recognition.screen_text import lines_touching, widen_regions

                regions = [r["bbox_px"] for r in result["changed_regions"]]
                result["text"] = None
                if regions:
                    text = reader.read(last, widen_regions(regions, last.shape[1]))
                    text["lines"] = lines_touching(text, regions)
                    result["text"] = text
            if args.save_frame is not None:
                write_image(args.save_frame, last)
                result["frame_path"] = str(args.save_frame)
        except ImportError as exc:
            st.exit(2, f"install svsg[overlays] to read text: {exc}\n")
        except (OSError, ValueError) as exc:
            st.exit(2, f"motionspec settle: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["state"] in ("settled", "unchanged") else 1

    if args.command == "agent-mcp":
        import os

        from .agent import ScreenAgent, serve_http, serve_mcp

        max_wait = args.max_wait_ms
        if max_wait is None and args.http:
            max_wait = 8000.0
        agent = ScreenAgent(max_fps=args.max_fps, allowed_windows=args.allow_window,
                            max_wait_ms=max_wait, images=not args.no_images,
                            history_seconds=args.history_seconds)
        if args.http:
            host, _, port = args.http.rpartition(":")
            if not host or not port.isdigit():
                am.exit(2, "motionspec agent-mcp: --http needs HOST:PORT\n")
            token = os.environ.get("VISFUNC_MCP_TOKEN", "").strip()
            if not token and not args.insecure_no_token:
                am.exit(2, "motionspec agent-mcp: set VISFUNC_MCP_TOKEN so that only clients "
                           "holding it can read the screen or act; pass --insecure-no-token "
                           "to serve without one\n")
            serve_http(agent, host, int(port), token=token, workers=args.workers,
                       log_file=args.log_file)
        else:
            serve_mcp(agent, log_file=args.log_file)
        return 0

    if args.command == "screen-text":
        try:
            from svsg.recognition.screen_text import ScreenTextReader

            result = ScreenTextReader().read(args.image, args.region)
        except ImportError as exc:
            sc.exit(2, f"install svsg[overlays] to read text: {exc}\n")
        except (OSError, ValueError) as exc:
            sc.exit(2, f"motionspec screen-text: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
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
