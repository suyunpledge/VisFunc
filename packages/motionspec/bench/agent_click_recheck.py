"""Exercise click_text when a real Qt canvas repaints during the first full-window OCR.

Run on an interactive Windows desktop with the screen reader and a Qt-capable Python:
    python bench/agent_click_recheck.py QT_PYTHON report.json --repeat 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from motionspec.agent import McpServer, ScreenAgent

from svsg.recognition.screen_text import ScreenTextReader, find_text
from svsg.recognition.ui_elements import find_windows


def wait_until(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("Canvas did not reach the expected state")
        time.sleep(0.03)


def run_case(qt_python, reader, directory, mode):
    title = f"VisFunc repaint {uuid4().hex[:12]}"
    state = directory / f"{title}.json"
    app = Path(__file__).with_name("agent_canvas_app.py")
    proc = subprocess.Popen(
        [qt_python, str(app), str(state), "60", "" if mode == "fixed" else mode],
        env={**os.environ, "VISFUNC_CANVAS_TITLE": title},
    )
    reads = 0
    high_counts = []

    def read_state():
        return json.loads(state.read_text(encoding="utf-8"))

    class ObservedReader:
        def read(self, image, regions=None):
            nonlocal reads
            if regions is None:
                reads += 1
                if reads == 1 and mode != "fixed":
                    # image is already captured. OCR sees the original layout; the next grab
                    # sees the repainted window. Only the fixture receives the signal.
                    state.with_suffix(".reflow").touch()
                    wait_until(lambda: read_state().get("reflowed"))
            result = reader.read(image, regions)
            if regions is None:
                high_counts.append(len(find_text(result, "High", exact=True)))
            return result

    server = McpServer(
        ScreenAgent(
            allowed_windows=[f"{title} | {Path(qt_python).name}"],
            reader_factory=ObservedReader,
            history_seconds=0,
            images=False,
        )
    )
    try:
        wait_until(lambda: state.exists() and bool(read_state()))
        wait_until(lambda: find_windows(title))

        def ready():
            frame = server.agent.grab(server.agent.resolve_region(None, title))
            text = reader.read(frame)
            return all(len(find_text(text, name, exact=True)) == 1 for name in ("Medium", "High"))

        wait_until(ready)
        request = dict(
            jsonrpc="2.0",
            id=1,
            method="tools/call",
            params={
                "name": "click_text",
                "arguments": {"window_title": title, "text": "High"},
            },
        )
        result = server.handle(request)["result"]
        receipt = json.loads(result["content"][0]["text"])
        replay = json.loads(server.handle(request)["result"]["content"][0]["text"])
        assert replay.get("replayed")
        actual = read_state()
        wanted = [] if mode == "duplicate" else ["High"]
        expected_state = "not_started" if mode == "duplicate" else "completed"
        assert high_counts[0] == 1 and (mode == "fixed" or actual["reflowed"])
        passed = actual["clicks"] == wanted and receipt["execution_state"] == expected_state
        return {
            "mode": mode,
            "passed": passed,
            "execution_state": receipt["execution_state"],
            "clicks": actual["clicks"],
            "full_ocr_reads": reads,
            "high_matches_per_read": high_counts,
            "next_step": receipt.get("next_step"),
            "screen_point": receipt.get("operation_marker", {}).get("screen_px"),
            "replayed": True,
        }
    finally:
        server.close()
        state.with_name(state.name + ".stop").touch()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qt_python")
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    reader = ScreenTextReader()
    with tempfile.TemporaryDirectory(prefix="visfunc-repaint-") as tmp:
        runs = [
            run_case(args.qt_python, reader, Path(tmp), mode)
            for _ in range(args.repeat)
            for mode in ("fixed", "reflow", "duplicate")
        ]
    args.output.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    print(f"{sum(row['passed'] for row in runs)}/{len(runs)} passed")


if __name__ == "__main__":
    main()
