"""Export MotionSpec evidence with the existing SVSG raster and geometry compiler."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import TypeAdapter

from svsg.contracts.recognition import RecognitionResult
from svsg.contracts.video_functions import (
    FrameDifference,
    FrameFunctions,
    FrameSemantics,
    VideoFunctions,
)
from svsg.image_functions.compiler import describe_raster
from svsg.image_functions.raster import RasterFunction
from svsg.recognition.frames import EvidenceFrame, frame_reference
from svsg.recognition.images import canonical_raster

from .temporal import compile_motion

MAX_FILE_BYTES = 96 * 1024 * 1024


def canonical_json(value: dict) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def read_artifact(root: Path, name: str, limit: int = MAX_FILE_BYTES) -> bytes:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes its evidence directory")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("artifact exceeds byte budget")
    return data


def difference(
    a: RasterFunction, b: RasterFunction, before: int, after: int, elapsed: float
) -> FrameDifference:
    from PIL import Image, ImageChops, ImageStat

    if (a.width, a.height) != (b.width, b.height):
        raise ValueError("difference requires identical source dimensions")
    with a.to_image() as first, b.to_image() as second:
        with ImageChops.difference(first, second) as delta:
            red, green, blue, alpha = ImageStat.Stat(delta).mean
            means = (red, green, blue, alpha)
            mask = Image.new("L", delta.size)
            try:
                # RGBA.getbbox() alone can ignore RGB differences when delta-alpha is zero.
                for band in delta.split():
                    with band:
                        combined = ImageChops.lighter(mask, band)
                    mask.close()
                    mask = combined
                changed = a.width * a.height - mask.histogram()[0]
                box = mask.getbbox()
            finally:
                mask.close()
    return FrameDifference(
        before_frame=before,
        after_frame=after,
        elapsed_ms=elapsed,
        changed_pixels=changed,
        changed_bbox=box,
        mean_abs_rgba=means,
    )


def export_motion_spec(
    spec: dict,
    evidence_dir: str | Path,
    out_dir: str | Path,
    *,
    sequence_id: str,
    max_frames: int = 64,
    max_total_pixels: int = 64 * 1024 * 1024,
) -> VideoFunctions:
    """Publish content-addressed rasters, then the summary manifest. No model calls."""
    if type(max_frames) is not int or not 1 <= max_frames <= 256:
        raise ValueError("max_frames must be between 1 and 256")
    if type(max_total_pixels) is not int or max_total_pixels <= 0:
        raise ValueError("max_total_pixels must be positive")
    if not isinstance(spec, dict) or spec.get("schema") != "motionspec/0.1":
        raise ValueError("expected motionspec/0.1")
    source = spec["source"]
    refs = TypeAdapter(list[EvidenceFrame]).validate_python(spec.get("evidence_frames", []))
    if not 1 <= len(refs) <= max_frames:
        raise ValueError("missing evidence frames or frame budget exceeded")
    if source["width"] * source["height"] * len(refs) > max_total_pixels:
        raise ValueError("video function pixel budget exceeded")
    if len(spec["segments"]) > 256:
        raise ValueError("motion segment budget exceeded")
    motions = [compile_motion(row) for row in spec["segments"]]
    root, out = Path(evidence_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    packets: list[FrameFunctions] = []
    deltas: list[FrameDifference] = []
    descriptions = {}
    previous: RasterFunction | None = None
    with TemporaryDirectory(dir=out) as directory:
        stage = Path(directory)
        for ref in refs:
            if ref.image != f"frame-{ref.frame:08d}-{ref.sha256}.png":
                raise ValueError("invalid evidence filename")
            raw = read_artifact(root, ref.image)
            bound = frame_reference(sequence_id, ref.model_dump(), raw)
            raster = canonical_raster(raw)
            if (raster.width, raster.height) != (source["width"], source["height"]):
                raise ValueError("evidence image dimensions differ from source")
            digest = raster.pixel_sha256
            if digest not in descriptions:
                program = canonical_json(raster.to_program())
                program_hash = sha256(program).hexdigest()
                filename = f"raster-{program_hash}.json"
                (stage / filename).write_bytes(program)
                descriptions[digest] = (describe_raster(raster), program_hash)
            functions, program_hash = descriptions[digest]
            filename = f"raster-{program_hash}.json"
            packet = FrameFunctions(
                frame_index=bound.frame_index,
                timestamp_ms=bound.timestamp_ms,
                image_sha256=ref.sha256,
                roles=ref.roles,
                image_functions=functions,
                raster_file=filename,
                raster_sha256=program_hash,
            )
            if previous is not None:
                prev = packets[-1]
                deltas.append(
                    difference(
                        previous,
                        raster,
                        prev.frame_index,
                        packet.frame_index,
                        packet.timestamp_ms - prev.timestamp_ms,
                    )
                )
            packets.append(packet)
            previous = raster
        bundle = VideoFunctions(
            sequence_id=sequence_id,
            source_spec_sha256=sha256(canonical_json(spec)).hexdigest(),
            width_px=source["width"],
            height_px=source["height"],
            t_start_ms=source["t_start_ms"],
            t_end_ms=source["t_end_ms"],
            clock=source["timestamps"],
            time_resolution_ms=spec["time_resolution_ms"],
            frames=packets,
            differences=deltas,
            motions=motions,
            warnings=[
                *spec.get("warnings", []),
                "Pixels are exact at exported frames; intermediate frames are absent.",
                "Motion curves are estimates; segment IDs are not object tracks.",
                "Pixel differences may include cuts, lighting or camera motion.",
            ],
        )
        (stage / "functions.json").write_bytes(canonical_json(bundle.model_dump(mode="json")))
        # Content-addressed filenames prevent an old manifest from seeing overwritten pixels.
        for path in sorted(stage.iterdir(), key=lambda p: p.name == "functions.json"):
            path.replace(out / path.name)
    return bundle


def attach_recognition(bundle: VideoFunctions, result: RecognitionResult) -> VideoFunctions:
    """Attach per-frame multimodal observations only after binding all source identities."""
    if result.frame is None or result.frame.sequence_id != bundle.sequence_id:
        raise ValueError("recognition belongs to another sequence or has no frame reference")
    match = next((f for f in bundle.frames if f.frame_index == result.frame.frame_index), None)
    if match is None:
        raise ValueError("recognition refers to a frame not present in the bundle")
    if (
        result.image_sha256 != match.image_sha256
        or result.pixel_sha256 != match.image_functions.pixel_sha256
        or result.image_functions.pixel_sha256 != match.image_functions.pixel_sha256
        or (result.width, result.height) != (bundle.width_px, bundle.height_px)
        or result.frame.timestamp_ms != match.timestamp_ms
    ):
        raise ValueError("recognition evidence does not match the frame")
    updated = bundle.model_copy(deep=True)
    target = next(f for f in updated.frames if f.frame_index == match.frame_index)
    target.semantics = FrameSemantics(
        status=result.status,
        label_scope=result.label_scope,
        review_mode=result.review_mode,
        observations=result.observations,
        relations=result.relations,
        calls=result.calls,
        warnings=result.warnings,
    )
    return updated
