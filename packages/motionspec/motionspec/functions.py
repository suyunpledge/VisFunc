"""Optional adapter to the shared SVSG image/time function implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .spec import validate


def require_svsg():
    try:
        from svsg.video_functions import export_motion_spec
    except ImportError as exc:
        raise ValueError("function export requires the local SVSG package and Pillow; "
                         "install SVSG with its [functions] extra") from exc
    return export_motion_spec


def export_functions(spec: dict, evidence_dir: str | Path, out_dir: str | Path, *,
                     sequence_id: str | None = None, max_frames: int = 64) -> dict:
    problems = validate(spec)
    if problems:
        raise ValueError("invalid MotionSpec input: " + "; ".join(problems))
    # Artifact references are excluded from the source digest to avoid a hash cycle.
    source = {k: v for k, v in spec.items() if k != "function_evidence"}
    digest = hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    bundle = require_svsg()(source, evidence_dir, out_dir,
                           sequence_id=sequence_id or f"motion-{digest[:24]}",
                           max_frames=max_frames)
    manifest = Path(out_dir) / "functions.json"
    return {"version": bundle.version, "manifest": manifest.name,
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "sequence_id": bundle.sequence_id, "frames": len(bundle.frames)}
