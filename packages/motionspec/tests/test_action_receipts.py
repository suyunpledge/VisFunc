import json
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from motionspec import action_receipts
from motionspec.action_receipts import ActionReceipts
from motionspec.agent import McpServer, ScreenAgent


def request(server, args=None, mid=1, **kw):
    params = {
        "name": "click_point",
        "arguments": {"window_title": "Sample", "x": 1, "y": 2} if args is None else args,
    }
    response = server.handle(
        {"jsonrpc": "2.0", "id": mid, "method": "tools/call", "params": params}, **kw
    )
    body = response["result"]["content"][0]["text"]
    return body if response["result"]["isError"] else json.loads(body)


def server_at(path, dispatch):
    agent = ScreenAgent(allowed_windows=["Sample"])
    agent.click_point = dispatch
    return McpServer(agent, receipts=ActionReceipts(path))


def test_replay_after_lost_reply_and_restart(tmp_path):
    calls = []
    path = tmp_path / "actions.sqlite3"

    def dispatch(**args):
        calls.append(args)
        return {
            "execution_state": "completed",
            "observation_state": "failed",
            "done": "private label",
            "screen_history": {"history_id": "expired"},
        }

    key = str(uuid4())
    first = server_at(path, dispatch)
    assert request(first, operation_id=key)["execution_state"] == "completed"
    first.close()  # Discard the original reply.
    second = server_at(path, dispatch)
    replay = request(second, mid=20, operation_id=key)
    assert replay == {
        "execution_state": "completed",
        "observation_state": "failed",
        "replayed": True,
    }
    assert "different arguments" in request(second, {"x": 3}, operation_id=key)
    assert len(calls) == 1
    assert request(second, mid=20, operation_id=str(uuid4()))["execution_state"] == "completed"
    assert len(calls) == 2
    assert b"private label" not in path.read_bytes() and b"Sample" not in path.read_bytes()
    second.close()


def test_session_fallback_and_stateless_actions(tmp_path):
    calls = []

    def click(**args):
        calls.append(args)
        return {"execution_state": "completed"}

    server = server_at(tmp_path / "actions.db", click)
    assert "initialize" in request(server, session=None)
    assert "UUID" in request(server, session=None, operation_id="click1")
    request(server, session="a")
    assert request(server, session="a")["replayed"]
    request(server, session="b")
    request(server, session="a", mid="1")
    request(server, session="a", mid=2)
    assert len(calls) == 4
    stats = server.handle(
        {"id": 3, "method": "ping", "params": {"_meta": {"visfunc/cacheStats": True}}}
    )
    assert stats["result"]["_meta"]["visfunc/cacheStats"]["actions"] == {
        "reserved": 4,
        "replays": 1,
        "conflicts": 0,
    }
    server.close()


def test_concurrent_reservation_and_uncertain_dispatch(tmp_path):
    path = tmp_path / "actions.sqlite3"
    started, release = threading.Event(), threading.Event()
    calls = []

    def dispatch(**args):
        calls.append(args)
        started.set()
        assert release.wait(3)
        raise OSError("lost window")

    a, b = server_at(path, dispatch), server_at(path, dispatch)
    key = str(uuid4())
    with ThreadPoolExecutor(max_workers=2) as pool:
        original = pool.submit(request, a, operation_id=key)
        assert started.wait(3)
        try:
            concurrent = request(b, operation_id=key)
            assert concurrent["execution_state"] == "unknown" and concurrent["replayed"]
        finally:
            release.set()
        assert "Dispatch unconfirmed" in original.result()
    assert request(b, operation_id=key) == concurrent
    assert len(calls) == 1
    a.close()
    b.close()


def test_unexpired_protection_survives_capacity_and_expired_key_is_reusable(monkeypatch):
    clock = [100]
    monkeypatch.setattr(action_receipts.time, "time", lambda: clock[0])
    monkeypatch.setattr(action_receipts, "MAX_RECEIPTS", 1)
    receipts = ActionReceipts()
    assert receipts.reserve("a", "click", {}) is None
    with pytest.raises(ValueError, match="not dispatched"):
        receipts.reserve("b", "click", {})
    assert receipts.reserve("a", "click", {})["replayed"]
    clock[0] += action_receipts.RETENTION_SECONDS
    assert receipts.reserve("a", "click", {"other": True}) is None
    receipts.close()


def test_storage_failure_prevents_dispatch(tmp_path):
    calls = []
    server = server_at(tmp_path, lambda **kw: calls.append(kw))
    assert isinstance(request(server), str)
    assert not calls
    server.close()


def test_http_reconnect_and_process_restart_do_not_redispatch(tmp_path, monkeypatch):
    from queue import Queue
    from urllib.request import Request, urlopen

    from motionspec.agent import serve_http

    calls = []
    monkeypatch.setenv("VISFUNC_ACTION_DB", str(tmp_path / "receipts.db"))

    def click(**args):
        calls.append(args)
        return {"execution_state": "completed"}

    def start():
        agent = ScreenAgent(allowed_windows=["Sample"])
        agent.click_point = click
        ready = Queue()
        worker = threading.Thread(
            target=serve_http, kwargs={"agent": agent, "port": 0, "ready": ready.put}, daemon=True
        )
        worker.start()
        return ready.get(timeout=5), worker

    def post(server, method, mid, headers):
        args = {"window_title": "Sample", "x": 1, "y": 2}
        params = {"name": "click_point", "arguments": args} if method == "tools/call" else {}
        req = Request(
            f"http://127.0.0.1:{server.server_port}/mcp",
            json.dumps({"method": method, "id": mid, "params": params}).encode(),
            headers,
        )
        with urlopen(req, timeout=3) as response:
            return response.headers, json.load(response)["result"]

    server, worker = start()
    try:
        headers, _ = post(server, "initialize", 1, {})
        session = {"Mcp-Session-Id": headers["Mcp-Session-Id"]}
        post(server, "tools/call", 2, session)  # Caller discards the reply.
        key = {"Idempotency-Key": str(uuid4())}
        post(server, "tools/call", 3, key)
    finally:
        server.shutdown()
        worker.join(4)
    restarted, worker = start()
    try:
        for mid, retry_headers in [(2, session), (99, key)]:
            _, reply = post(restarted, "tools/call", mid, retry_headers)
            assert json.loads(reply["content"][0]["text"])["replayed"]
        assert post(restarted, "tools/call", 2, {})[1]["isError"]
        assert len(calls) == 2
    finally:
        restarted.shutdown()
        worker.join(4)
