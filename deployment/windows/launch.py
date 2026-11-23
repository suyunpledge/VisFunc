"""Start and supervise both services using the portable Windows runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
LOCAL_HTTP = build_opener(ProxyHandler({}))
LOG_LIMIT = 10 * 1024 * 1024  # rotate a service log when it passes 10 MB


def ready(api_url, mcp_url, token):
    with LOCAL_HTTP.open(api_url + "/healthz", timeout=2) as response:
        if json.load(response).get("status") != "ok":
            raise ValueError("unexpected API health response")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    for method in ("initialize", "tools/list"):
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {}}
        request = Request(mcp_url, data=json.dumps(payload).encode(), headers=headers)
        with LOCAL_HTTP.open(request, timeout=2) as response:
            result = json.load(response)["result"]
            if method == "initialize":
                headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
            elif "wait_for_settle" not in {tool["name"] for tool in result["tools"]}:
                raise ValueError("missing desktop MCP tools")


def ensure_token(env_path: Path) -> bool:
    """Write a random MCP token into .env when none is set. Returns True when one was created."""
    if dotenv_values(env_path).get("VISFUNC_MCP_TOKEN") or os.environ.get("VISFUNC_MCP_TOKEN"):
        return False
    text = env_path.read_text(encoding="utf-8-sig")
    line = "VISFUNC_MCP_TOKEN=" + secrets.token_urlsafe(32)
    pattern = re.compile(r"^[ \t]*#?[ \t]*VISFUNC_MCP_TOKEN[ \t]*=.*$", re.M)
    if pattern.search(text):
        text = pattern.sub(lambda _: line, text, count=1)
    else:
        text = text.rstrip() + "\n" + line + "\n"
    env_path.write_text(text, encoding="utf-8")
    return True


def allow_rules(path: Path) -> list[str]:
    rules = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        rules.append(entry)
        if not entry.rpartition("|")[2].strip().casefold().endswith(".exe"):
            print(f"Note: '{entry}' in allowed-windows.txt matches by title only. Add "
                  "' | program.exe' so that other programs cannot borrow the title.", flush=True)
    return rules


def check_desktop() -> None:
    try:
        from svsg.recognition.ui_elements import desktop_state
    except ImportError:
        return
    state = desktop_state()
    if state == "service":
        raise RuntimeError("this runs in session 0 (a Windows service), which has no desktop; "
                           "start it from the signed-in user's session instead")
    if state == "locked":
        print("Warning: the session is locked. Desktop tools will refuse to capture or act "
              "until it is unlocked.", flush=True)


def tee(stream, name: str) -> None:
    """Copy a child's output to the console and to logs/<name>.log, rotating by size."""
    path = LOGS / f"{name}.log"
    out = path.open("ab")
    try:
        for line in iter(stream.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            out.write(line)
            out.flush()
            if out.tell() > LOG_LIMIT:
                out.close()
                path.replace(path.with_suffix(".log.1"))
                out = path.open("ab")
    finally:
        out.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check both services, then stop")
    args = parser.parse_args()
    env_path = ROOT / ".env"
    if not env_path.exists():
        with (ROOT / ".env.example").open("rb") as src, env_path.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    if ensure_token(env_path):
        print("Created an MCP access token in .env (VISFUNC_MCP_TOKEN). Enter the same value "
              "as the Bearer token of the MCP connector in Keystone or another client.",
              flush=True)
    env = {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    env.update(os.environ)
    env["PATH"] = str(ROOT / "bin") + os.pathsep + env.get("PATH", "")
    env.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "ultralytics"))
    env.setdefault("VISFUNC_ACTION_DB", str(ROOT / "data" / "action-receipts.sqlite3"))
    host = env.get("SVSG_HOST", "127.0.0.1")
    api_port = int(env.get("SVSG_PORT", 3002))
    mcp_port = int(env.get("VISFUNC_MCP_PORT", 8765))
    if not all(1 <= p <= 65535 for p in (api_port, mcp_port)) or api_port == mcp_port:
        raise ValueError("configure two distinct TCP ports between 1 and 65535")
    if host not in ("127.0.0.1", "localhost", "::1") and not env.get("SVSG_API_KEY"):
        raise ValueError(f"SVSG_HOST={host} exposes the image API to the network; set "
                         "SVSG_API_KEY in .env first")
    for addr in ((host, api_port), ("127.0.0.1", mcp_port)):
        with socket.socket() as probe:
            probe.bind(addr)
    check_desktop()
    LOGS.mkdir(exist_ok=True)
    mcp = ["motionspec", "agent-mcp", "--http", f"127.0.0.1:{mcp_port}", "--no-images",
           "--log-file", str(LOGS / "agent-mcp-calls.jsonl")]
    for rule in allow_rules(ROOT / "allowed-windows.txt"):
        mcp.extend(["--allow-window", rule])
    api_url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{api_port}"
    mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"
    children = []
    try:
        for name, command in (("api", ["svsg"]), ("agent-mcp", mcp)):
            child = subprocess.Popen([sys.executable, "-m", *command], cwd=ROOT, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            threading.Thread(target=tee, args=(child.stdout, name), daemon=True).start()
            children.append(child)
        deadline = time.monotonic() + 60
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError(
                    f"service exited during startup; see its message above or {LOGS}"
                )
            try:
                ready(api_url, mcp_url, env.get("VISFUNC_MCP_TOKEN", ""))
                break
            except (URLError, TimeoutError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.4)
        print(f"API ready: {api_url}/docs\nDesktop MCP ready: {mcp_url}\nLogs: {LOGS}", flush=True)
        if not args.check:
            print("Press Ctrl+C to stop both services.", flush=True)
            while all(child.poll() is None for child in children):
                time.sleep(0.5)
            raise RuntimeError(f"a service stopped unexpectedly; see {LOGS}")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        sys.exit(f"SVSG startup failed: {exc}")
