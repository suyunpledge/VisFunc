"""Encode a decoded image to a portable raster program or reconstruct it as PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .raster import RasterFunction


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m svsg.image_functions")
    parser.add_argument("operation", choices=("encode", "decode"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "encode":
            raster = RasterFunction.from_image(args.source)
            args.output.write_text(json.dumps(raster.to_program()), encoding="utf-8")
        else:
            program = json.loads(args.source.read_text(encoding="utf-8"))
            if not isinstance(program, dict):
                raise ValueError("invalid raster program")
            with RasterFunction.from_program(program).to_image() as image:
                image.save(args.output, "PNG")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
