"""规格格式的版本、结构校验，以及给 AI 读的文字摘要。"""

from __future__ import annotations

import math
import re
from itertools import pairwise

from .timing import motion_timing

SCHEMA = "motionspec/0.1"
KINDS = {"transform", "fade", "nonrigid", "instant"}
REQUIRED_SEGMENT = ("id", "kind", "confidence", "t_start_ms", "t_end_ms", "duration_ms", "region", "channels", "samples")
NUMERIC_CHANNELS = {"translate": ("distance_px",), "scale": ("from", "to", "peak"),
                    "rotate": ("from_deg", "to_deg"), "blend": ("mean_abs_delta", "residual"),
                    "change": ("mean_abs_delta",), "appearance": ("luma_min", "luma_max"),
                    "opacity": ("from", "to"),
                    "size": ("width_from", "width_to", "height_from", "height_to"),
                    "color": ("distance",), "property": ("from", "to")}


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
                        or (name == "color" and (
                            not _number(channel["distance"])
                            or not all(_vector(channel.get(k), 4, lambda v: finite(v) and 0 <= v <= 1)
                                       for k in ("from", "to"))))
                        or (name == "property" and (
                            not isinstance(channel.get("name"), str)
                            or not 1 <= len(channel["name"]) <= 64))
                        or (name == "scale" and (
                            channel["peak"] <= 0
                            or any(channel[k] < 0 for k in ("from", "to"))
                            or (channel.get("reference") != "scale_property" and any(channel[k] <= 0 for k in ("from", "to")))
                            or max(channel["from"], channel["to"]) <= 0
                        ))
                        or ("t_start_ms" in channel and not _interval(channel))
                        or not _valid_easing(channel.get("easing"))):
                    problems.append(f"{where} has an invalid {name!r} channel")
                elif "t_start_ms" in channel and _interval(seg) and not (
                    seg["t_start_ms"] <= channel["t_start_ms"] <= channel["t_end_ms"] <= seg["t_end_ms"]
                ):
                    problems.append(f"{where} {name!r} timing exceeds the segment; re-analyse the source")
        if not _valid_easing(seg.get("easing")):
            problems.append(f"{where} has invalid easing")
        if "timing" in seg:
            try:
                if seg["timing"] != motion_timing(seg):
                    problems.append(f"{where} timing disagrees with its measurements")
            except (KeyError, TypeError, ValueError, IndexError):
                problems.append(f"{where} has invalid timing evidence")
        notes = seg.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
            problems.append(f"{where} notes must be a list of strings")
    groups = spec.get("motion_groups", [])
    if not isinstance(groups, list) or len(groups) > len(segments):
        problems.append("invalid motion associations")
    else:
        group_ids, membership = set(), {}
        for group in groups:
            if not isinstance(group, dict):
                problems.append("invalid motion association")
                continue
            gid, ids = group.get("id"), group.get("segment_ids")
            if (not isinstance(gid, str) or not gid or gid in group_ids
                    or group.get("relation") != "equivalent_sampled_motion"
                    or not isinstance(ids, list) or len(ids) < 2
                    or any(not isinstance(sid, str) or sid not in seen or sid in membership for sid in ids)
                    or len(set(ids)) != len(ids)):
                problems.append("invalid or repeated motion association members")
                continue
            group_ids.add(gid)
            membership.update({sid: gid for sid in ids})
        for seg in segments:
            if (isinstance(seg, dict) and isinstance(seg.get("id"), str)
                    and seg.get("motion_group") != membership.get(seg["id"])):
                problems.append("segment motion association disagrees with group membership")
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
                or functions.get("version") != ("svsg.probe-functions/1" if source.get("timestamps") == "probe"
                                                 else "svsg.video-functions/1")
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
    if "scale" in ch and ch["scale"].get("reference") in ("fully_shown_frame", "scale_property"):
        parts.append(f"scale {ch['scale']['from']:g} -> {ch['scale']['to']:g}")
    elif "scale" in ch:
        parts.append(f"scale x{ch['scale']['to']}" + (f" (peak x{ch['scale']['peak']})" if ch["scale"]["peak"] > ch["scale"]["to"] + 0.02 else ""))
    if "rotate" in ch:
        parts.append(f"rotate {ch['rotate']['to_deg']:+g} deg")
    if "size" in ch:
        box = ch["size"]
        parts.append(f"size {box['width_from']:g}x{box['height_from']:g} -> {box['width_to']:g}x{box['height_to']:g}")
    if "color" in ch:
        a, b = ch["color"]["from"], ch["color"]["to"]
        parts.append("colour rgb({},{},{}) -> rgb({},{},{})".format(*(round(v * 255) for v in a[:3] + b[:3])))
        if a[3] != b[3]:
            parts.append(f"colour alpha {a[3]:g} -> {b[3]:g}")
    if "property" in ch:
        parts.append(f"property {ch['property']['name']} {ch['property']['from']:g} -> {ch['property']['to']:g}")
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


def describe_element(element: dict | None) -> str:
    """Qt 探针给出的控件身份：类名、对象名、文字。视觉路径没有这项。"""
    if not element:
        return ""
    label = element.get("class", "")
    if element.get("name"):
        label += "#" + element["name"]
    if element.get("text"):
        label += f" \"{element['text']}\""
    return f", element {label}" if label else ""


def design_timing(seg: dict) -> dict | None:
    """对齐最贴合的命名曲线之后的起点和时长，也就是对"动效是怎么写的"的最好估计。

    实测的起止只覆盖看得出变化的那一段：缓入的头、缓出的尾每帧变化不到一个像素，会被截掉。
    命名曲线匹配时允许起点前移、终点后延，这里把对齐结果换算回视频时间轴。
    没有命名曲线贴合（rmse 大于 0.05）时返回 None。
    """
    return motion_timing(seg)["estimated_design"]


def describe_design(seg: dict, *, timing: dict | None = None) -> str:
    timing = motion_timing(seg) if timing is None else timing
    design = timing["estimated_design"]
    if design is None:
        return f"unavailable ({timing['design_unavailable_reason']})"
    return f"{design['easing']}, {design['t_start_ms']:.0f}-{design['t_end_ms']:.0f} ({design['duration_ms']:.0f} ms)"


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
            "| id | visible change (ms) | estimated design timing: easing, start-end (duration) | kind | change | fitted curve | confidence |",
            "|---|---|---|---|---|---|---|",
        ]
        for seg in spec["segments"]:
            timing = motion_timing(seg)
            visible = timing["visible_change"]
            lines.append(
                f"| {seg['id']} | {visible['t_start_ms']:.0f}-{visible['t_end_ms']:.0f} ({visible['duration_ms']:.0f} ms) "
                f"| {describe_design(seg, timing=timing)} | {seg['kind']} "
                f"| {describe_change(seg)} | {describe_easing(seg['easing'])} | {seg['confidence']} |"
            )
        lines += [
            "",
            "\"Visible change\" is the detected interval; it may omit slow or invisible parts. "
            "\"Estimated design timing\" aligns a named easing curve to those measurements and estimates the "
            "authored start, end and duration. It is not ground truth. Unavailable design timing must remain "
            "unknown; do not substitute the visible interval. JSON timing uses the same calculation.",
            "",
        ]
        for seg in spec["segments"]:
            lines.append(f"- {seg['id']}: region {seg['region']}" + (f", sheet `{seg['sheet']}`" if seg.get("sheet") else "")
                         + describe_element(seg.get("element"))
                         + (f" (+{len(seg['members']) - 1} more moving the same way)" if seg.get("members") else ""))
            for note in seg["notes"]:
                lines.append(f"  - {note}")
    for group in spec.get("motion_groups", []):
        lines.append(f"\nMotion association {group['id']}: {', '.join(group['segment_ids'])}. "
                     "Each member retains its own coordinates and element identity.")
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
