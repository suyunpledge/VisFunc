"""Measure image input token usage on dev only; discard calibration answers."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from motion_corpus import BENCH
from strong_client import MODELS, Ledger, PaidModels
from strong_eval import media, write_result
from strong_inputs import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--providers", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--limit-usd", type=float, required=True)
    p.add_argument("--models", nargs="+", required=True, choices=MODELS)
    args = p.parse_args()
    if not (args.inputs / "q002/media.json").is_file():
        with ZipFile(BENCH / "corpus/qt.zip") as archive:
            prepare(args.inputs, "qt", "q002", archive)
    if not 0 < args.limit_usd <= 10:
        p.error("Limit must be in (0,10]")
    api = PaidModels(args.providers, Ledger(args.ledger, args.limit_usd), args.models)
    for model in args.models:
        measurements = []
        path = args.out / f"{model}.json"
        if path.exists():
            continue
        for case in ("q000", "q002"):
            folder = args.inputs / case
            meta = json.loads((folder / "media.json").read_text(encoding="utf8"))
            for i in (0, 3):
                parts, _ = media(folder, model, "V1", i)
                result = api.call(
                    model,
                    [{"role": "user", "content": parts}],
                    f"cal-{model}-{case}-{i}",
                    max_output=64,
                )
                result.pop("answer")
                measurements.append(
                    {
                        "case": case,
                        "tiles_height": meta["dense"]["tile_px"][1],
                        "sheets": len(meta["dense"]["variants"][i]["images"]),
                        **result,
                    }
                )
        args.out.mkdir(exist_ok=True, parents=True)
        write_result(path, measurements)
        print(model, [r["input_tokens"] for r in measurements], flush=True)


if __name__ == "__main__":
    main()
