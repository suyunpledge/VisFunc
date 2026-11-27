"""Shared paths and serialization for the frozen motion benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

BENCH = Path(__file__).resolve().parent
TOOLS = BENCH.parent / "packages/motionspec/tools"


def cases(source, split):
    return json.loads((BENCH / "splits.json").read_text(encoding="utf8"))[source][split]


def save(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf8",
    )


def pack(source, work, output):
    # A failed or interrupted capture must not replace a usable frozen corpus.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(".partial")
    with ZipFile(staging, "w", ZIP_DEFLATED) as archive:
        for split in ("dev", "test"):
            for case in cases(source, split):
                for name in ("video.mp4", "truth.json", "trace.json"):
                    path = work / case["id"] / name
                    archive.write(path, f"{case['id']}/{name}")
    staging.replace(output)
