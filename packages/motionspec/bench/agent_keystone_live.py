"""用 Pledge Keystone 的对话循环 + OpenAI 模型，通过 MCP 实际操作桌面窗口，并核对结果。

    python bench/agent_keystone_live.py KEYSTONE_REPO QT_PYTHON OUT.json
        [--models gpt-6.1-sol gpt-6.1-terra] [--tasks settings recovery web canvas]
        [--repeat 1] [--max-turns 100]

用到的 Keystone 组件都是原样调用：Work 模式的对话循环（qt_frontend.conversation.run）、
桌面端的 MCP HTTP 客户端（qt_frontend.mcp_http.McpHttp）、本机协议网关（run.py gateway，
上游 api.openai.com，GPT-6.1 带工具时由网关转成 Responses 接口）。只有 McpCatalog 把发现的
工具包成对话工具的约 20 行（命名、64 KiB 参数和 128 KiB 结果上限、回执格式）在这里照抄，
因为 McpCatalog 依赖 Qt 事件循环。回合上限由 --max-turns 指定（默认 100，相当于不设上限）；
本机 Keystone 配置里的上限不受影响。

任务（是否完成由应用自己记录的状态判定，不看模型怎么说）：
  settings  Qt 设置窗口：改服务器名、开加密、保存、开始同步并确认，报告结果
  recovery  同上，但第一次同步会因端口错误失败，需按错误框提示改端口后重试
  web       独立的 Chrome 窗口里填注册表单：输入框、下拉框、图标按钮、复选框、提交
  canvas    自绘窗口：按钮是画出来的，只能靠屏幕文字找位置并按坐标点击

运行时桌面上会出现任务窗口（置顶、不抢焦点；Chrome 窗口由 Playwright 单独启动）。OpenAI 密钥
从本机 ~/.forge/secrets.json 读出，只通过环境变量交给网关子进程；网关使用临时 home。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPENAI_SLOT = "custom__78dbc0dd"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, seconds: float = 30) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"port {port} did not open")


def openai_key() -> str:
    sys.path.insert(0, str(HERE))
    import providers

    return providers.find(OPENAI_SLOT).key


class QtTask:
    """A Qt task window that writes its own state file."""

    def __init__(self, qt_python: str, work: Path, script: str, extra: list[str]):
        self.qt_python, self.work, self.script, self.extra = qt_python, work, script, extra

    def start(self, tag: str) -> None:
        self.path = self.work / f"state-{tag}.json"
        for p in (self.path, self.path.with_name(self.path.name + ".stop")):
            p.unlink(missing_ok=True)
        self.proc = subprocess.Popen(
            [self.qt_python, str(HERE / self.script), str(self.path), "600", *self.extra]
        )
        while not self.path.exists():
            time.sleep(0.1)
        time.sleep(0.5)

    def state(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def stop(self) -> None:
        self.path.with_name(self.path.name + ".stop").touch()
        self.proc.wait(timeout=30)


class WebTask:
    """A separate Chrome window opened by Playwright; the page reports its own state."""

    def __init__(self):
        sys.path.insert(0, str(HERE))
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()

    def start(self, tag: str) -> None:
        from agent_web_task import open_page

        self.browser, self.page = open_page(self.playwright, force_accessibility=False)

    def state(self) -> dict:
        from agent_web_task import page_state

        return page_state(self.page)

    def stop(self) -> None:
        self.browser.close()


def settings_ok(s: dict) -> bool:
    return (
        s["synced_to"] == "archive-02"
        and s["synced_encrypted"] is True
        and s["completed"]
        and not s["deleted"]
    )


TASKS = {
    "settings": {
        "window": "Keystone agent task",
        "prompt": "请在「Keystone agent task」窗口里完成：把服务器名称改成 archive-02，"
        "打开 Encrypt transfers，保存设置，然后开始同步。"
        "等同步完成后，告诉我同步了多少个文件、同步到了哪台服务器、是否加密。不要删除任何数据。",
        "make": lambda qt, work: QtTask(qt, work, "agent_task_app.py", ["normal"]),
        "ok": settings_ok,
        "answer": ("48", "archive-02"),
    },
    "recovery": {
        "window": "Keystone agent task",
        "prompt": "请在「Keystone agent task」窗口里完成：把服务器名称改成 archive-02，"
        "打开 Encrypt transfers，保存设置，然后开始同步。"
        "如果同步失败，按窗口里的提示处理后重试，直到同步完成。"
        "最后告诉我同步了多少个文件、同步到了哪台服务器和端口、中间出过什么问题。"
        "不要删除任何数据。",
        "make": lambda qt, work: QtTask(qt, work, "agent_task_app.py", ["flaky"]),
        "ok": lambda s: settings_ok(s) and s["port_saved"] == "8443",
        "answer": ("48", "archive-02", "8443"),
    },
    "web": {
        "window": "Keystone web task",
        "prompt": "请在「Keystone web task」这个 Chrome 窗口的页面里创建账户：姓名 Lin Chen，邮箱 "
        "lin.chen@example.com，套餐选 Pro，座位数改成 3，勾选同意条款，然后提交。告诉我"
        "页面上显示的结果。",
        "make": lambda qt, work: WebTask(),
        "ok": lambda s: (
            s.get("submitted")
            == {
                "name": "Lin Chen",
                "email": "lin.chen@example.com",
                "plan": "Pro",
                "seats": 3,
                "terms": True,
            }
        ),
        "answer": ("Lin Chen", "Pro", "3"),
    },
    "canvas": {
        "window": "Keystone canvas task",
        "prompt": "请在「Keystone canvas task」窗口里：把优先级设为 High，打开 Notify team，然后点 "
        "Confirm。完成后告诉我窗口上显示的状态。",
        "make": lambda qt, work: QtTask(qt, work, "agent_canvas_app.py", []),
        "ok": lambda s: s.get("confirmed_with") == {"priority": "High", "notify": True},
        "answer": ("High",),
    },
}


def run_once(task_name, task, model, mcp_url, gateway, max_turns, tag) -> dict:
    from keystone_client import ChatMessage, KeystoneGatewayClient
    from qt_frontend import conversation
    from qt_frontend.mcp_http import McpHttp

    app = task["app"]
    app.start(tag)
    trace: list[dict] = []
    turns: list[dict] = []
    cancel = threading.Event()
    mcp = McpHttp(mcp_url, "", cancel)
    server_id, server_name = "0" * 32, "VisFunc screen"
    tools = {}
    for tool in mcp.tools():  # same shape as McpCatalog.chat_tools
        key = hashlib.sha256((server_id + ":" + tool["name"]).encode()).hexdigest()[:20]
        name = f"mcp_{re.sub(r'[^a-zA-Z0-9_-]', '_', tool['name'])[:36]}_{key}"
        function = {
            "name": name,
            "description": f"{server_name} · {tool['description']}",
            "parameters": {**tool.get("schema", {}), "type": "object"},
        }
        tools[name] = ({"type": "function", "function": function}, tool["name"])

    def dispatch(target, arguments, cancel_event):  # same checks as McpCatalog.call_chat_tool
        start = time.perf_counter()
        entry = {"tool": target, "arguments": arguments}
        if len(json.dumps(arguments, ensure_ascii=False).encode()) > 64 * 1024:
            response = {
                "ok": False,
                "error": "MCP 工具参数超过 64 KiB。",
                "meta": {"execution_state": "not_started"},
            }
        else:
            result = mcp.invoke(target, arguments)
            size = len(json.dumps(result, ensure_ascii=False).encode())
            entry["result_bytes"] = size
            if size > 128 * 1024:
                response = {
                    "ok": False,
                    "error": "MCP 结果超过对话可接收的大小。",
                    "meta": {"execution_state": "completed"},
                }
            else:
                response = {
                    "ok": not bool(result.get("isError")),
                    "result": result,
                    "meta": {"execution_state": "completed"},
                }
                if not response["ok"]:
                    response["error"] = "MCP 工具返回错误。"
                    entry["error_text"] = result["content"][0]["text"][:300]
        entry["ok"] = response["ok"]
        entry["ms"] = round((time.perf_counter() - start) * 1000)
        trace.append(entry)
        return response

    class Recording(KeystoneGatewayClient):
        def stream_chat(self, messages, **kw):
            start = time.perf_counter()
            result = super().stream_chat(messages, **kw)
            turns.append(
                {
                    "ms": round((time.perf_counter() - start) * 1000),
                    "usage": result.usage,
                    "tool_calls": [c["function"]["name"] for c in result.tool_calls or []],
                }
            )
            return result

    client = Recording(base_url=gateway)
    messages = [ChatMessage("user", task["prompt"])]
    timer = threading.Timer(600, cancel.set)
    timer.start()
    started = time.perf_counter()
    answer, error = "", None
    try:
        result = conversation.run(
            client,
            messages,
            model=model,
            effort=None,
            cancel=cancel,
            authorize=lambda name, arguments, cancel_event: None,  # pre-approved test run
            chunk=lambda text: None,
            checkpoint=lambda snapshot: None,
            status=lambda text: None,
            next_round=lambda: None,
            system_context=conversation.WORK_CONTEXT,
            mcp=(tools, dispatch),
            max_turns=max_turns,
        )
        answer = result.text
    except Exception as exc:  # record failures of the loop itself
        error = f"{type(exc).__name__}: {exc}"
    finally:
        timer.cancel()
    seconds = round(time.perf_counter() - started, 1)
    time.sleep(1.0)
    state = app.state()
    app.stop()
    usage = [t["usage"] or {} for t in turns]
    return {
        "task": task_name,
        "model": model,
        "success": bool(task["ok"](state)),
        "answer_mentions": all(token in answer for token in task["answer"]),
        "seconds": seconds,
        "model_turns": len(turns),
        "max_turns": max_turns,
        "tool_calls": len(trace),
        "tool_errors": sum(not t["ok"] for t in trace),
        "tool_ms_total": sum(t["ms"] for t in trace),
        "model_ms_total": sum(t["ms"] for t in turns),
        "tokens": {
            k: sum(u.get(k, 0) or 0 for u in usage)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "state": state,
        "answer": answer,
        "error": error,
        "trace": trace,
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("qt_python")
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", nargs="+", default=["gpt-6.1-sol", "gpt-6.1-terra"])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args()
    sys.path[:0] = [str(args.repo), str(args.repo / "keystone-gui")]
    work = Path(tempfile.mkdtemp(prefix="keystone-agent-live-"))
    mcp_port, gateway_port = free_port(), free_port()
    allowed = sorted({TASKS[t]["window"] for t in args.tasks})
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "motionspec",
            "agent-mcp",
            "--http",
            f"127.0.0.1:{mcp_port}",
            "--no-images",
            *[a for w in allowed for a in ("--allow-window", w)],
        ],
        stdout=subprocess.DEVNULL,
        stderr=open(work / "mcp.log", "w"),
    )
    env = {**os.environ, "KEYSTONE_GATEWAY_KEY": openai_key()}
    gateway = subprocess.Popen(
        [
            sys.executable,
            str(args.repo / "run.py"),
            "gateway",
            "--home",
            str(work / "home"),
            "--upstream",
            "https://api.openai.com/v1",
            "--upstream-wire",
            "openai",
            "--client-wire",
            "openai",
            "--port",
            str(gateway_port),
        ],
        cwd=str(args.repo),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(work / "gateway.log", "w"),
    )
    del env
    report = {"max_turns": args.max_turns, "runs": []}
    apps = {}
    try:
        wait_port(mcp_port)
        wait_port(gateway_port)
        for round_ in range(args.repeat):
            for task_name in args.tasks:
                task = dict(TASKS[task_name])
                if task_name not in apps:
                    apps[task_name] = task["make"](args.qt_python, work)
                task["app"] = apps[task_name]
                for model in args.models:
                    run = run_once(
                        task_name,
                        task,
                        model,
                        f"http://127.0.0.1:{mcp_port}/mcp",
                        f"http://127.0.0.1:{gateway_port}",
                        args.max_turns,
                        f"{task_name}-{model}-{round_}",
                    )
                    report["runs"].append(run)
                    summary = {
                        k: run[k]
                        for k in (
                            "task",
                            "model",
                            "success",
                            "answer_mentions",
                            "seconds",
                            "model_turns",
                            "tool_calls",
                            "tool_errors",
                        )
                    }
                    summary.update(tokens=run["tokens"]["total_tokens"], error=run["error"])
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                    args.out.write_text(
                        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
                    )
    finally:
        server.terminate()
        gateway.terminate()
        server.wait(timeout=10)
        gateway.wait(timeout=10)
        web = apps.get("web")
        if web is not None:
            web.playwright.stop()


if __name__ == "__main__":
    main()
