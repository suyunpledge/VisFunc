"""规格格式的版本、结构校验，以及给 AI 读的文字摘要。"""

from __future__ import annotations

import math
import re
from itertools import pairwise

SCHEMA = "motionspec/0.1"
KINDS = {"transform", "fade", "nonrigid", "instant"}
REQUIRED_SEGMENT = ("id", "kind", "confidence", "t_start_ms", "t_end_ms", "duration_ms", "region", "channels", "samples")
NUMERIC_CHANNELS = {"translate": ("distance_px",), "scale": ("from", "to", "peak"),
                    "rotate": ("from_deg", "to_deg"), "blend": ("mean_abs_delta", "residual"),
                    "change": ("mean_abs_delta",), "appearance": ("luma_min", "luma_max"),
                    "opacity": ("from", "to")}


def finite(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _number(value, minimum=0) -> bool:
    return finite(value) and value >= minimum


def _vector(value, size: int, predicate=finite) -> bool:
    return isinstance(value, list) and len(value) == size and all(predicate(x) for x in value)


def _interval(row: dict) -> bool:
    start, end, duration = (row.get(k) for k in ("t_start_ms", "t_end_ms", "duration_ms"))
    return all(_number(v) for v in (start, end, duration)) and start <= end and abs(end - start - duration) <= 0.151


def _valid_easing(easing) -> bool:
    if easing is None:
        return True
    if not isinstance(easing, dict) or not all(_number(easing.get(k)) for k in ("rmse", "overshoot")):
        return False
    if easing.get("model") == "cubic-bezier":
        if not isinstance(easing.get("css"), str) or not _vector(easing.get("params"), 4):
            return False
        if any(not 0 <= easing["params"][i] <= 1 for i in (0, 2)):
            return False
    elif easing.get("model") == "spring":
        zeta, omega = easing.get("damping_ratio"), easing.get("angular_frequency")
        if not (finite(zeta) and 0 < zeta < 1 and finite(omega) and omega > 0):
            return False
    else:
        return False
    named = easing.get("named")
    return isinstance(named, list) and bool(named) and all(
        isinstance(row, dict) and isinstance(row.get("name"), str) and bool(row["name"])
        and _number(row.get("rmse")) and _number(row.get("duration_ms")) for row in named
    )


def validate(spec: dict) -> list[str]:
    """返回问题列表，空列表表示结构合法。"""
    problems: list[str] = []
    if not isinstance(spec, dict):
        return ["spec is not an object"]
    if spec.get("schema") != SCHEMA:
        problems.append(f"schema is {spec.get('schema')!r}, expected {SCHEMA!r}")
    source = spec.get("source")
    if not isinstance(source, dict) or any(type(source.get(k)) is not int or source[k] <= 0 for k in ("width", "height")):
        problems.append("source must contain positive integer width and height")
        source = {}
    elif (not _number(source.get("duration_ms")) or not finite(source.get("fps_nominal"))
          or source["fps_nominal"] <= 0 or type(source.get("frames_analysed")) is not int
          or source["frames_analysed"] < 2):
        problems.append("source has invalid duration, fps or frame count")
    if any(k in source for k in ("t_start_ms", "t_end_ms")) and not _interval(source):
        problems.append("source has an invalid interval")
    if not finite(spec.get("time_resolution_ms")) or spec["time_resolution_ms"] <= 0:
        problems.append("time_resolution_ms must be positive")
    warnings = spec.get("warnings", [])
    if not isinstance(warnings, list) or not all(isinstance(warning, str) for warning in warnings):
        problems.append("warnings must be a list of strings")
    segments = spec.get("segments")
    if not isinstance(segments, list):
        return problems + ["segments must be a list"]
    seen: set[str] = set()
    for i, seg in enumerate(segments):
        where = f"segments[{i}]"
        if not isinstance(seg, dict):
            problems.append(f"{where} must be an object")
            continue
        for key in REQUIRED_SEGMENT:
            if key not in seg:
                problems.append(f"{where} missing {key}")
        if not isinstance(seg.get("kind"), str) or seg["kind"] not in KINDS:
            problems.append(f"{where} has unknown kind {seg.get('kind')!r}")
        sid = seg.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            problems.append(f"{where} has an invalid or repeated id")
        else:
            seen.add(sid)
        if seg.get("confidence") not in ("low", "medium", "high") or not _interval(seg):
            problems.append(f"{where} has invalid confidence or interval")
        region = seg.get("region")
        if (not _vector(region, 4, lambda v: type(v) is int and v >= 0)
                or region[0] >= region[2] or region[1] >= region[3]
                or (source and (region[2] > source["width"] or region[3] > source["height"]))):
            problems.append(f"{where} has an invalid region")
        samples = seg.get("samples")
        if not isinstance(samples, list) or not samples or any(
            not isinstance(row, dict) or not _number(row.get("t_ms"))
            or any(not finite(v) for k, v in row.items() if k != "channel_progress")
            or ("channel_progress" in row and (not isinstance(row["channel_progress"], dict)
                                                or not all(finite(v) for v in row["channel_progress"].values())))
            for row in samples
        ):
            problems.append(f"{where} has invalid samples")
        elif any(a["t_ms"] > b["t_ms"] for a, b in pairwise(samples)):
            problems.append(f"{where} has unordered samples")
        channels = seg.get("channels")
        if not isinstance(channels, dict):
            problems.append(f"{where} channels must be an object")
        else:
            for name, channel in channels.items():
                if (name not in NUMERIC_CHANNELS or not isinstance(channel, dict)
                        or any(not finite(channel.get(k)) for k in NUMERIC_CHANNELS[name])
                        or (name == "translate" and not all(_vector(channel.get(k), 2) for k in ("from", "to", "delta")))
                        or (name == "translate" and not _number(channel["distance_px"]))
                        or (name == "scale" and any(channel[k] <= 0 for k in ("from", "to", "peak")))
                        or ("t_start_ms" in channel and not _interval(channel))
                        or not _valid_easing(channel.get("easing"))):
                    problems.append(f"{where} has an invalid {name!r} channel")
        if not _valid_easing(seg.get("easing")):
            problems.append(f"{where} has invalid easing")
        notes = seg.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
            problems.append(f"{where} notes must be a list of strings")
    if "coverage" in spec:
        coverage = spec["coverage"]
        if not isinstance(coverage, dict) or any(type(coverage.get(k)) is not int or coverage[k] < 0
                                                 for k in ("dropped_candidates", "truncated_segments")):
            problems.append("coverage counts must be nonnegative integers")
    if "evidence_frames" in spec:
        evidence = spec["evidence_frames"]
        if not isinstance(evidence, list) or any(not isinstance(row, dict)
                or type(row.get("frame")) is not int or row["frame"] < 0 or not _number(row.get("t_ms"))
                or not isinstance(row.get("roles"), list) or not all(isinstance(role, str) for role in row["roles"])
                or not isinstance(row.get("sha256"), str) or re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is None
                or row.get("image") != f"frame-{row['frame']:08d}-{row['sha256']}.png"
                for row in evidence):
            problems.append("evidence_frames contains an invalid frame reference")
        elif any(a["frame"] >= b["frame"] or a["t_ms"] >= b["t_ms"]
                 for a, b in pairwise(evidence)):
            problems.append("evidence_frames must have strictly increasing frame indices and times")
    if "function_evidence" in spec:
        functions = spec["function_evidence"]
        if (not isinstance(functions, dict) or functions.get("manifest") != "functions.json"
                or functions.get("version") != "svsg.video-functions/1"
                or not isinstance(functions.get("sequence_id"), str)
                or not isinstance(functions.get("sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", functions["sha256"]) is None
                or type(functions.get("frames")) is not int
                or functions["frames"] != len(spec.get("evidence_frames", []))):
            problems.append("invalid function evidence reference")
    return problems


def describe_easing(easing: dict | None) -> str:
    if not easing:
        return "-"
    if easing["model"] == "spring":
        text = f"spring(damping_ratio={easing['damping_ratio']}, omega={easing['angular_frequency']}/s)"
    else:
        text = easing["css"]
    best = easing["named"][0]
    if best["rmse"] > 0.05:  # 没有哪条命名曲线贴合（典型是弹簧），不报以免误导
        return f"{text} rmse={easing['rmse']}; no named curve fits"
    return f"{text} rmse={easing['rmse']}; nearest named: {best['name']} over {best['duration_ms']:.0f} ms (rmse={best['rmse']})"


def describe_change(seg: dict) -> str:
    ch = seg["channels"]
    parts = []
    if "translate" in ch:
        dx, dy = ch["translate"]["delta"]
        move = ch["translate"]
        label = "tracked anchor moves" if move.get("reference") == "tracked_feature_anchor" else "move"
        text = f"{label} ({dx:+g}, {dy:+g}) px"
        if move.get("lower_bound"):
            text += " or more (partly invisible)"
        elif move.get("follows_opacity"):
            text += " (same curve as opacity)"
        elif move.get("extrapolated"):
            text += f" ({move['visible_distance_px']:g} px visible, rest extrapolated)"
        parts.append(text)
    if "scale" in ch and ch["scale"].get("reference") == "fully_shown_frame":
        parts.append(f"scale {ch['scale']['from']:g} -> {ch['scale']['to']:g}")
    elif "scale" in ch:
        parts.append(f"scale x{ch['scale']['to']}" + (f" (peak x{ch['scale']['peak']})" if ch["scale"]["peak"] > ch["scale"]["to"] + 0.02 else ""))
    if "rotate" in ch:
        parts.append(f"rotate {ch['rotate']['to_deg']:+g} deg")
    if "opacity" in ch:
        fade = ch["opacity"]
        text = f"opacity {fade['from']:g} -> {fade['to']:g}"
        if fade.get("easing"):
            text += f" over {fade['duration_ms']:.0f} ms ({fade['easing']['named'][0]['name']})"
        parts.append(text)
    if "blend" in ch:
        parts.append(f"cross-fade, luma {seg['luma']['from']} -> {seg['luma']['to']}")
    if "change" in ch:
        parts.append(f"instant change, luma {seg['luma']['from']} -> {seg['luma']['to']}")
    if "appearance" in ch:
        a = ch["appearance"]
        text = f"non-rigid, luma {a['luma_min']}..{a['luma_max']}"
        if "luma_period_ms" in a:
            text += f", period ~{a['luma_period_ms']} ms"
        parts.append(text)
    if seg.get("pattern"):
        parts.append(f"{seg['pattern']}, peak at +{seg['peak_ms']} ms")
    return "; ".join(parts) or "-"


def summarize(spec: dict) -> str:
    src = spec["source"]
    lines = [
        f"# Motion spec ({spec['schema']})",
        "",
        (f"Source: {src['width']}x{src['height']}, {src['duration_ms']:.0f} ms, {src['frames_analysed']} frames. "
         f"Median frame interval: {spec['time_resolution_ms']:.2f} ms. "
         "Positions are in source pixels."),
        "",
    ]
    if not spec["segments"]:
        lines.append("No change segments were detected at the selected thresholds.")
    else:
        lines += [
            "| id | when (ms) | duration | kind | change | easing | confidence |",
            "|---|---|---|---|---|---|---|",
        ]
        for seg in spec["segments"]:
            lines.append(
                f"| {seg['id']} | {seg['t_start_ms']:.0f}-{seg['t_end_ms']:.0f} | {seg['duration_ms']:.0f} ms | {seg['kind']} "
                f"| {describe_change(seg)} | {describe_easing(seg['easing'])} | {seg['confidence']} |"
            )
        lines.append("")
        for seg in spec["segments"]:
            lines.append(f"- {seg['id']}: region {seg['region']}" + (f", sheet `{seg['sheet']}`" if seg.get("sheet") else ""))
            for note in seg["notes"]:
                lines.append(f"  - {note}")
    if spec.get("ambient"):
        lines.append("\nContinuously changing regions (animated backdrop, video, spinner). They are not analysed "
                     "as motion, and changes drawn on top of them cannot be isolated:")
        for region in spec["ambient"]:
            lines.append(f"- region {region['region']}, {region['t_start_ms']:.0f}-{region['t_end_ms']:.0f} ms")
    for warning in spec.get("warnings", []):
        lines.append(f"\nWarning: {warning}")
    if spec.get("function_evidence"):
        functions = spec["function_evidence"]
        lines.append(f"\nShared SVSG functions: `{functions['manifest']}` "
                     f"({functions['frames']} sampled frames; sequence `{functions['sequence_id']}`). "
                     "Pixel functions are exact at retained frames; motion curves remain estimates.")
    if spec.get("evidence_frames"):
        lines.append("\nSource-resolution evidence frames:")
        for frame in spec["evidence_frames"]:
            lines.append(f"- `{frame['image']}` at {frame['t_ms']:g} ms ({', '.join(frame['roles'])})")
    return "\n".join(lines) + "\n"
