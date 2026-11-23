"""Start and supervise both services using the portable Windows runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
LOCAL_HTTP = build_opener(ProxyHandler({}))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check both services, then stop")
    args = parser.parse_args()
    if not (ROOT / ".env").exists():
        with (ROOT / ".env.example").open("rb") as src, (ROOT / ".env").open("xb") as dst:
            shutil.copyfileobj(src, dst)
    env = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None}
    env.update(os.environ)
    env["PATH"] = str(ROOT / "bin") + os.pathsep + env.get("PATH", "")
    env.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "ultralytics"))
    host = env.get("SVSG_HOST", "127.0.0.1")
    api_port = int(env.get("SVSG_PORT", 3002))
    mcp_port = int(env.get("VISFUNC_MCP_PORT", 8765))
    if not all(1 <= p <= 65535 for p in (api_port, mcp_port)) or api_port == mcp_port:
        raise ValueError("configure two distinct TCP ports between 1 and 65535")
    for addr in ((host, api_port), ("127.0.0.1", mcp_port)):
        with socket.socket() as probe:
            probe.bind(addr)
    titles = (ROOT / "allowed-windows.txt").read_text(encoding="utf-8-sig").splitlines()
    mcp = ["motionspec", "agent-mcp", "--http", f"127.0.0.1:{mcp_port}", "--no-images"]
    for line in titles:
        if line.strip() and not line.lstrip().startswith("#"):
            mcp.extend(["--allow-window", line.strip()])
    api_url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{api_port}"
    mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"
    children = []
    try:
        for command in (["svsg"], mcp):
            children.append(subprocess.Popen([sys.executable, "-m", *command], cwd=ROOT, env=env))
        deadline = time.monotonic() + 60
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("service exited during startup; see its message above")
            try:
                ready(api_url, mcp_url, env.get("VISFUNC_MCP_TOKEN", ""))
                break
            except (URLError, TimeoutError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.4)
        print(f"API ready: {api_url}/docs\nDesktop MCP ready: {mcp_url}", flush=True)
        if not args.check:
            print("Press Ctrl+C to stop both services.", flush=True)
            while all(child.poll() is None for child in children):
                time.sleep(0.5)
            raise RuntimeError("a service stopped unexpectedly")
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
