"""Agent 屏幕工具和 MCP 协议：用假屏幕、假 OCR 和假控件列表，不碰真实桌面。"""

from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest
from motionspec.agent import TOOLS, McpServer, ScreenAgent, serve_mcp

STEP = 1000 / 60


class FakeScreen:
    """Frames of a scripted scene; `act()` changes it after an unobserved pause."""

    def __init__(self, size=(400, 300)):
        self.w, self.h = size
        self.t = 0.0
        self.saved = False
        self.areas = []

    def frame(self):
        img = np.full((self.h, self.w, 3), 240, np.uint8)
        if self.saved:
            cv2.rectangle(img, (100, 100), (220, 140), (30, 120, 30), -1)
        return img

    def act(self):
        self.t += 1500  # the agent clicks; nobody watches
        self.saved = True

    def capture(self, area, max_fps=60.0):
        self.areas.append(area)
        while True:
            self.t += STEP
            yield self.t, self.frame()


class FakeReader:
    def __init__(self):
        self.calls = []

    def read(self, frame, regions=None):
        self.calls.append(regions)
        return {
            "lines": [
                {
                    "text": "Saved",
                    "bbox_px": [110, 110, 200, 130],
                    "center_px": [155, 120],
                    "score": 0.99,
                    "low_confidence": False,
                }
            ],
            "elapsed_ms": 1.0,
        }


def elements(**kw):
    return {
        "elements": [
            {
                "name": "Refresh",
                "control_type": "button",
                "enabled": True,
                "bbox_px": [10, 10, 40, 40],
                "center_px": [25, 25],
            },
            {
                "name": "Saved",
                "control_type": "button",
                "enabled": False,
                "bbox_px": [0, 0, 5, 5],
                "center_px": [2, 2],
            },
        ]
    }


def make(**kw):
    screen, reader = FakeScreen(), FakeReader()
    agent = ScreenAgent(
        capture=screen.capture,
        reader_factory=lambda: reader,
        elements=kw.pop("elements", elements),
        locate_window=lambda title: (50, 60, 450, 360),
        **kw,
    )
    return agent, screen, reader


def test_wait_for_settle_uses_the_previous_call_as_baseline():
    agent, screen, reader = make()
    first = agent.wait_for_settle(window_title="App", expect_change_ms=200)
    assert first["state"] == "unchanged" and first["baseline"] == "first_frame_of_this_call"
    screen.act()
    after = agent.wait_for_settle(window_title="App", read_text=True)
    assert after["state"] == "settled"
    assert after["baseline"] == "last_frame_of_previous_call"
    box = after["changed_regions"][0]["bbox_px"]  # offset by the window origin (50, 60)
    assert 145 <= box[0] <= 151 and 155 <= box[1] <= 161 and 269 <= box[2] <= 276
    assert after["text"][0]["text"] == "Saved" and after["text"][0]["center_px"] == [205, 180]
    assert after["area_px"] == [50, 60, 450, 360]
    assert screen.areas == [(50, 60, 450, 360)]  # one capture stream per area, kept open


def test_only_a_new_ambient_setting_starts_a_fresh_watcher():
    agent, screen, _ = make()
    agent.wait_for_settle(expect_change_ms=200)
    again = agent.wait_for_settle(expect_change_ms=300, quiet_ms=150)
    assert again["baseline"] == "last_frame_of_previous_call"
    assert again["settings"]["expect_change_ms"] == 300
    fresh = agent.wait_for_settle(expect_change_ms=200, ambient_ms=0)
    assert fresh["baseline"] == "first_frame_of_this_call"


def test_find_prefers_enabled_ui_automation_then_ocr():
    agent, _, reader = make()
    hits = agent.find_on_screen(query="Refresh")
    assert [t["source"] for t in hits["targets"]] == ["ui_automation"]
    assert reader.calls == []  # OCR not needed
    hits = agent.find_on_screen(query="Saved", window_title="App")
    assert [t["source"] for t in hits["targets"]] == ["ui_automation", "ocr"]
    assert hits["targets"][1]["center_px"] == [205, 180]


def test_find_falls_back_to_ocr_without_ui_automation():
    def broken(**kw):
        raise OSError("UI Automation is only available on Windows")

    agent, _, _ = make(elements=broken)
    hits = agent.find_on_screen(query="Saved")
    assert [t["source"] for t in hits["targets"]] == ["ocr"]
    assert "unavailable" in hits["notes"][0]


def test_screenshot_scales_and_reports_mapping():
    agent, _, _ = make()
    info, png = agent.screenshot(region=[0, 0, 400, 300], max_side=200)
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (150, 200)
    assert info["image_scale"] == 0.5 and info["area_px"] == [0, 0, 400, 300]


def test_region_validation():
    agent, _, _ = make()
    with pytest.raises(ValueError):
        agent.resolve_region([10, 10, 5, 20], None)
    with pytest.raises(ValueError):
        agent.resolve_region([0, 0, 10, 10], "App")


def rpc(server, method, params=None, mid=1):
    message = {"jsonrpc": "2.0", "method": method, "id": mid}
    if params is not None:
        message["params"] = params
    return server.handle(message)


def test_mcp_handshake_and_tool_listing():
    server = McpServer(make()[0])
    init = rpc(server, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
    assert init["result"]["protocolVersion"] == "2024-11-05"
    assert init["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert (
        rpc(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"]
        == "2025-06-18"
    )
    names = [t["name"] for t in rpc(server, "tools/list")["result"]["tools"]]
    assert names == [t["name"] for t in server.agent.tool_specs()]
    assert "click_element" not in names  # actions need --allow-window
    assert all(t["inputSchema"]["type"] == "object" for t in TOOLS)
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rpc(server, "ping")["result"] == {}
    assert rpc(server, "resources/list")["error"]["code"] == -32601


def test_mcp_tool_calls_and_errors():
    server = McpServer(make()[0])
    ok = rpc(server, "tools/call", {"name": "find_on_screen", "arguments": {"query": "Refresh"}})
    body = json.loads(ok["result"]["content"][0]["text"])
    assert ok["result"]["isError"] is False and body["targets"][0]["center_px"] == [25, 25]
    shot = rpc(server, "tools/call", {"name": "screenshot", "arguments": {"max_side": 100}})
    assert [c["type"] for c in shot["result"]["content"]] == ["text", "image"]
    bad = rpc(server, "tools/call", {"name": "find_on_screen", "arguments": {"nope": 1}})
    assert bad["result"]["isError"] is True and "TypeError" in bad["result"]["content"][0]["text"]
    assert rpc(server, "tools/call", {"name": "format_disk"})["error"]["code"] == -32602


def test_serve_mcp_over_streams(capsys):
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        "not json",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "",
    ]
    out = io.BytesIO()
    serve_mcp(make()[0], stdin=io.BytesIO("\n".join(lines).encode()), stdout=out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[0] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert replies[1]["error"]["code"] == -32700
    assert len(replies) == 2


def test_action_reports_its_own_effect_and_new_dialog():
    calls = []

    def act(query, action, *, title, **kw):
        calls.append((query, action, title, kw))
        screen.saved = True  # the click changes the window immediately
        return {
            "action": action,
            "performed": "invoke",
            "window": "App",
            "element": {"name": "Save", "control_type": "button"},
            "after": {
                "name": "Save",
                "control_type": "button",
                "enabled": True,
                "center_px": [1, 2],
                "bbox_px": [0, 0, 2, 4],
            },
        }

    def windows(title):
        return {"title": "App - confirm", "controls": [{"name": "Start", "control_type": "button"}]}

    agent, screen, reader = make(act=act, windows=windows, allowed_windows=["App"])
    report = agent.click_element(window_title="App", target="Save")
    assert report["done"] == "invoke 'Save'"
    assert report["element_after"] == {
        "name": "Save",
        "control_type": "button",
        "enabled": True,
        "center_px": [1, 2],
    }
    assert report["screen"]["state"] == "settled"
    assert report["screen"]["text_in_changed_regions"] == [
        {"text": "Saved", "center_px": [205, 180]}
    ]
    assert report["new_top_window"]["title"] == "App - confirm"
    assert calls == [("Save", "click", "App", {"control_type": None})]
    # A later explicit wait compares against the end of the action, not its baseline.
    later = agent.wait_for_settle(window_title="App", expect_change_ms=200)
    assert later["state"] == "unchanged"


def test_actions_need_an_allowed_window():
    agent, _, _ = make()
    with pytest.raises(PermissionError):
        agent.click_element(window_title="App", target="Save")
    server = McpServer(agent)
    reply = rpc(
        server,
        "tools/call",
        {"name": "click_element", "arguments": {"window_title": "App", "target": "Save"}},
    )
    assert reply["error"]["code"] == -32602  # not offered without --allow-window


def test_control_changes_lists_what_an_action_did():
    from motionspec.agent import _control_changes

    before = [
        {"name": "Apply", "control_type": "button", "enabled": False},
        {
            "name": "Encrypt transfers",
            "control_type": "checkbox",
            "enabled": True,
            "toggle_state": "off",
        },
        {"name": "Status: idle", "control_type": "text", "enabled": True},
    ]
    after = [
        {"name": "Apply", "control_type": "button", "enabled": True},
        {
            "name": "Encrypt transfers",
            "control_type": "checkbox",
            "enabled": True,
            "toggle_state": "on",
        },
        {"name": "Status: saved", "control_type": "text", "enabled": True},
    ]
    assert _control_changes(before, after) == [
        {"name": "Apply", "control_type": "button", "enabled": True, "was": False},
        {
            "name": "Encrypt transfers",
            "control_type": "checkbox",
            "toggle_state": "on",
            "was": "off",
        },
        {"name": "Status: saved", "control_type": "text", "appeared": True},
        {"name": "Status: idle", "control_type": "text", "gone": True},
    ]


def test_click_point_reports_effect_and_needs_permission():
    clicks = []

    def click(title, x, y):
        clicks.append((title, x, y))
        screen.saved = True
        return {
            "performed": "click_at",
            "window": "App",
            "element": {"name": f"({x}, {y})"},
            "after": None,
        }

    agent, screen, _ = make(click=click, windows=lambda title: None, allowed_windows=["App"])
    report = agent.click_point(window_title="App", x="210", y=180)
    assert clicks == [("App", 210, 180)]
    assert report["done"] == "click_at '(210, 180)'"
    assert report["screen"]["state"] == "settled"
    assert "click_point" in [t["name"] for t in agent.tool_specs()]
    blocked, _, _ = make(click=click)
    with pytest.raises(PermissionError):
        blocked.click_point(window_title="App", x=1, y=1)


def test_blocked_capture_switches_to_ui_automation():
    snapshots = iter(
        [{"title": "App", "controls": [{"name": "2", "control_type": "text"}]}]  # before 1
        + [{"title": "App", "controls": [{"name": "3", "control_type": "text"}]}]  # after 1
        + [{"title": "App", "controls": [{"name": "3", "control_type": "text"}]}]  # before 2
        + [{"title": "App", "controls": [{"name": "4", "control_type": "text"}]}] * 10
    )

    def act(query, action, *, title, **kw):  # the app changes, the captured pixels do not
        return {
            "performed": "invoke",
            "window": "App",
            "element": {"name": "Add seat", "control_type": "button"},
            "after": None,
        }

    agent, screen, _ = make(act=act, windows=lambda title: next(snapshots), allowed_windows=["App"])
    first = agent.click_element(window_title="App", target="Add seat")
    assert first["screen"]["state"] == "unchanged"
    assert first["screen"]["source"] == "ui_automation" and "unreliable" in first["screen"]["note"]
    assert "app" in agent.blocked
    second = agent.click_element(window_title="App", target="Add seat")
    assert second["screen"]["state"] == "settled"  # judged from the controls, not pixels
    assert {"name": "4", "control_type": "text", "appeared": True} in second["controls_changed"]
    assert "note" in agent.read_screen_text(window_title="App")


def test_caption_buttons_are_dropped():
    from motionspec.agent import _without_caption

    elements = [
        {"name": "关闭", "control_type": "button", "bbox_px": [900, 62, 960, 100]},
        {"name": "系统", "control_type": "menu_item", "bbox_px": [60, 70, 90, 100]},
        {"name": "关闭", "control_type": "button", "bbox_px": [500, 600, 560, 630]},
        {"name": "Start sync", "control_type": "button", "bbox_px": [500, 600, 560, 630]},
    ]
    kept = _without_caption(elements, [60, 60, 1000, 700])
    assert [(e["name"], e["bbox_px"][1]) for e in kept] == [("关闭", 600), ("Start sync", 600)]
