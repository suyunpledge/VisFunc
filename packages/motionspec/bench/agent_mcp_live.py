"""在真实桌面上把 `motionspec agent-mcp` 当 MCP 服务调用一遍，对象是 agent_qt_window.py 的窗口。

    python bench/agent_mcp_live.py QT_PYTHON OUT.json

QT_PYTHON 是装了 PySide6 的解释器；本脚本所在的解释器需要 svsg[overlays] 和 comtypes。
窗口不抢焦点、置顶显示约 20 秒，"状态"标签每 1.5 秒变一次，相当于 Agent 操作后界面的变化。
记录每次工具调用的耗时和结果，并和窗口导出的真值比较。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Client:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "motionspec", "agent-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.next_id = 0

    def send(self, method: str, params: dict | None = None, notify: bool = False):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notify:
            self.next_id += 1
            message["id"] = self.next_id
        self.proc.stdin.write(json.dumps(message).encode() + b"\n")
        self.proc.stdin.flush()
        if notify:
            return None
        reply = json.loads(self.proc.stdout.readline())
        assert reply.get("id") == self.next_id, reply
        return reply

    def tool(self, name: str, **arguments):
        start = time.perf_counter()
        reply = self.send("tools/call", {"name": name, "arguments": arguments})
        ms = round((time.perf_counter() - start) * 1000, 1)
        result = reply["result"]
        text = next(c["text"] for c in result["content"] if c["type"] == "text")
        images = [c for c in result["content"] if c["type"] == "image"]
        body = text if result["isError"] else json.loads(text)
        return body, ms, result["isError"], images

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def inside(point, box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def main() -> None:
    qt_python, out = sys.argv[1], Path(sys.argv[2])
    truth_path = Path(tempfile.gettempdir()) / "visfunc-agent-window.json"
    truth_path.unlink(missing_ok=True)
    window = subprocess.Popen([qt_python, str(HERE / "agent_qt_window.py"), str(truth_path), "20"])
    while not truth_path.exists():
        time.sleep(0.1)
    time.sleep(0.3)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    client = Client()
    report: dict = {"steps": []}
    try:
        start = time.perf_counter()
        init = client.send(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "bench", "version": "0"},
            },
        )
        client.send("notifications/initialized", notify=True)
        tools = client.send("tools/list")["result"]["tools"]
        report["initialize_ms"] = round((time.perf_counter() - start) * 1000, 1)
        report["protocol"] = init["result"]["protocolVersion"]
        report["tools"] = [t["name"] for t in tools]
        title = truth["title"]

        base, ms, err, _ = client.tool("wait_for_settle", window_title=title, expect_change_ms=300)
        report["steps"].append(
            {"call": "wait_for_settle (baseline)", "ms": ms, "state": base["state"]}
        )
        for _ in range(2):  # two status changes, 1.5 s apart, standing in for agent actions
            res, ms, err, _ = client.tool(
                "wait_for_settle", window_title=title, expect_change_ms=2500, read_text=True
            )
            report["steps"].append(
                {
                    "call": "wait_for_settle (after change)",
                    "ms": ms,
                    "state": res.get("state"),
                    "first_change_ms": res.get("first_change_ms"),
                    "decided_ms": res.get("elapsed_ms"),
                    "changed": [r["bbox_px"] for r in res.get("changed_regions", [])],
                    "text": [line["text"] for line in res.get("text") or []],
                }
            )
        for element in truth["elements"]:
            hits, ms, err, _ = client.tool(
                "find_on_screen", query=element["name"], window_title=title, exact=True
            )
            first = hits["targets"][0] if not err and hits["targets"] else None
            report["steps"].append(
                {
                    "call": f"find_on_screen {element['name']!r}",
                    "ms": ms,
                    "source": first and first["source"],
                    "enabled": first and first.get("enabled"),
                    "correct": bool(first and inside(first["center_px"], element["bbox_px"])),
                    "icon_only": element["icon_only"],
                }
            )
        hits, ms, err, _ = client.tool(
            "find_on_screen", query="backup-server-07", window_title=title
        )
        report["steps"].append(
            {
                "call": "find_on_screen 'backup-server-07' (field value)",
                "ms": ms,
                "targets": hits["targets"][:2] if not err else hits,
            }
        )
        text, ms, err, _ = client.tool("read_screen_text", window_title=title)
        report["steps"].append(
            {
                "call": "read_screen_text",
                "ms": ms,
                "lines": [line["text"] for line in text["lines"]] if not err else text,
            }
        )
        elems, ms, err, _ = client.tool("list_ui_elements", window_title=title)
        report["steps"].append(
            {
                "call": "list_ui_elements",
                "ms": ms,
                "count": len(elems["elements"]) if not err else elems,
            }
        )
        info, ms, err, images = client.tool("screenshot", window_title=title)
        report["steps"].append(
            {
                "call": "screenshot",
                "ms": ms,
                "info": info,
                "png_bytes": len(images[0]["data"]) * 3 // 4 if images else 0,
            }
        )
    finally:
        client.close()
        window.wait(timeout=30)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
