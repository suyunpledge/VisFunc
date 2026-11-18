"""Rerun failures into a new experiment; never replace original paid-call evidence."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_hard as R


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--slot")
    parser.add_argument("--providers")
    parser.add_argument("--models", required=True)
    parser.add_argument("--scenes", default="out/qt-hard")
    parser.add_argument("--old-code")
    parser.add_argument("--out", help="new output directory; defaults to folder/retries/<run_id>")
    args = parser.parse_args(argv)
    folder = Path(args.folder)
    source = (folder / "raw.jsonl").read_bytes()
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines() if line.strip()]
    wanted = set(args.models.split(","))
    failed = [(i, row) for i, row in enumerate(rows) if row["model"] in wanted
              and not row.get("delivery", {}).get("accepted", row["score"]["parsed"])]
    if not failed:
        print("No failed tasks selected.")
        return
    models = {m[0]: m for m in R.MODELS}
    if wanted - models.keys():
        raise ValueError("unknown model selection")
    if any(r["condition"] == "video+vision-old" for _, r in failed) and not args.old_code:
        raise ValueError("old-code input is required to rerun the old-code condition")
    slots = ({name: args.slot for name in wanted} if args.slot else
             json.loads(Path(args.providers).read_text(encoding="utf-8")) if args.providers else {})
    for name in wanted:
        R.find(slots.get(name, models[name][2]))
    run_id = uuid.uuid4().hex
    out = Path(args.out) if args.out else folder / "retries" / run_id
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"run_id": run_id, "retry_source": str((folder / "raw.jsonl").resolve()),
                "retry_source_sha256": hashlib.sha256(source).hexdigest(),
                "conditions": sorted({r["condition"] for _, r in failed}), "slots": slots,
                "scope": "Additional calls; original failures and their costs remain in retry_source."}
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = R.prepare(Path(args.scenes), out / "data", Path(args.old_code) if args.old_code else None)

    def run(job):
        index, row = job
        repeat = row.get("repeat", index)
        reply = R.ask(models[row["model"]], row["scene"], scenes[row["scene"]], row["condition"],
                      16000, slots, repeat=repeat, run_id=run_id)
        reply["retry_of"] = row.get("task_id") or {"source_line": index + 1}
        return reply

    with concurrent.futures.ThreadPoolExecutor(6) as pool, (out / "raw.jsonl").open("x", encoding="utf-8") as log:
        for row in pool.map(run, failed):
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
    print(f"Wrote {len(failed)} rerun tasks to {out}; original evidence unchanged.")


if __name__ == "__main__":
    main()
