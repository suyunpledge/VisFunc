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
        history_seconds=kw.pop("history_seconds", 0),
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


def test_click_text_exact_match_uses_screen_origin_and_deduplicates():
    posted = []

    def click(title, x, y):
        posted.append((title, x, y))
        return {"performed": "click_at", "window": title, "element": {"name": "Saved"}}

    agent, _, _ = make(allowed_windows=["App"], click=click, windows=lambda _: None)
    server = McpServer(agent)
    args = {"name": "click_text", "arguments": {"window_title": "App", "text": "saved"}}
    report = json.loads(rpc(server, "tools/call", args)["result"]["content"][0]["text"])
    assert posted == [("App", 205, 180)]
    assert report["operation_marker"]["screen_px"] == [205, 180]
    assert report["text_match"]["bbox_px"] == [160, 170, 250, 190]
    replay = json.loads(rpc(server, "tools/call", args)["result"]["content"][0]["text"])
    assert replay["replayed"] and len(posted) == 1


def test_click_text_requires_action_permission_before_reading():
    agent, screen, reader = make()
    assert "click_text" not in {tool["name"] for tool in agent.tool_specs()}
    with pytest.raises(PermissionError):
        agent.click_text(window_title="App", text="Saved")
    assert not screen.areas and not reader.calls


def test_replayed_visual_refusal_stays_not_started():
    agent, _, _ = make(
        allowed_windows=["App"],
        windows=lambda _: None,
        click=lambda *a: pytest.fail("unexpected click"),
    )
    server = McpServer(agent)
    params = {"name": "click_text", "arguments": {"window_title": "App", "text": "Missing"}}
    first = rpc(server, "tools/call", params)["result"]
    second = rpc(server, "tools/call", params)["result"]
    a, b = [json.loads(r["content"][0]["text"]) for r in (first, second)]
    assert first["isError"] and second["isError"]
    assert a["execution_state"] == "not_started" and b == {**a, "replayed": True}


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "partial_label",
        "duplicate",
        "uncertain",
        "moved",
        "changed",
        "ocr_error",
        "blocked",
    ],
)
def test_click_text_with_unverified_evidence_posts_nothing(monkeypatch, case):
    posted = []
    agent, screen, reader = make(
        allowed_windows=["App"], click=lambda *a: posted.append(a), windows=lambda _: None
    )
    lines = reader.read(None)["lines"]
    if case == "missing":
        lines.clear()
    elif case == "partial_label":
        lines[0]["text"] = "Saved drafts"
    elif case == "duplicate":
        lines.append(dict(lines[0], bbox_px=[0, 0, 20, 20]))
    elif case == "uncertain":
        lines[0]["low_confidence"] = True
    elif case == "blocked":
        agent.blocked.add("app")
    elif case == "moved":
        areas = iter([(50, 60, 450, 360), (75, 60, 475, 360)])
        agent.locate_window = lambda _: next(areas)
    elif case == "changed":
        frame = screen.frame

        def repaint():
            image = frame()
            image[0, 0] = round(screen.t / STEP) % 256
            return image

        monkeypatch.setattr(screen, "frame", repaint)

    def read(_):
        if case == "ocr_error":
            raise OSError("reader busy")
        return {"lines": lines}

    monkeypatch.setattr(reader, "read", read)
    result = agent.click_text(window_title="App", text="Saved")
    assert result["execution_state"] == "not_started"
    assert result["next_step"] and posted == []
    if case == "changed":
        assert "kept changing" in result["next_step"]


@pytest.mark.parametrize(
    "initial,final",
    [
        ([20], [20]),  # stable
        ([20], [150]),  # target moves
        ([20], [20, 150]),  # a second label appears elsewhere
        ([], [150]),  # newly visible
        ([20], []),  # disappears
        ([20, 150], [20]),  # ambiguity resolves before dispatch
    ],
)
def test_click_text_uses_last_verified_layout(monkeypatch, initial, final):
    posted = []

    def click(title, x, y):
        posted.append((x, y))
        return {"performed": "click_at", "window": title, "element": {"name": "Run"}}

    agent, screen, reader = make(allowed_windows=["App"], windows=lambda _: None, click=click)
    labels = initial
    ocr_calls = 0

    def frame():
        image = np.full((screen.h, screen.w, 3), 240, np.uint8)
        for x in labels:
            image[100:130, x : x + 60] = 0
        return image

    def read(image, regions=None):
        nonlocal labels, ocr_calls
        # Read from the captured frame; the window redraws while OCR is running.
        lines = [
            {"text": "Run", "bbox_px": [x, 100, x + 60, 130], "low_confidence": False, "score": 1}
            for x in (20, 150)
            if image[100, x, 0] == 0
        ]
        ocr_calls += 1
        labels = final
        return {"lines": lines}

    monkeypatch.setattr(screen, "frame", frame)
    monkeypatch.setattr(reader, "read", read)
    result = agent.click_text(window_title="App", text="Run")
    assert ocr_calls == (1 if initial == final else 2)
    if len(final) == 1:
        assert posted == [(50 + final[0] + 30, 60 + 115)]
        assert result["execution_state"] == "completed"
        # The pre-click repaint must not be reported as an effect of the click.
        assert result["screen"]["state"] == "unchanged"
        assert result["operation_marker"]["screen_px"] == list(posted[0])
    else:
        assert result["execution_state"] == "not_started" and not posted
        assert not agent._armed_areas


def test_click_text_composited_border_preserves_screen_coordinates(monkeypatch):
    posted = []

    def click(title, x, y):
        posted.append((x, y))
        return {"performed": "click_at", "window": title, "element": {"name": "Saved"}}

    agent, screen, reader = make(allowed_windows=["App"], windows=lambda _: None, click=click)
    monkeypatch.setattr("motionspec.agent._text_frame_border", lambda hwnd, area: 2)
    get_frame = screen.frame

    def frame():
        image = get_frame()
        image[:, :2] = round(screen.t / STEP) % 256
        return image

    shapes = []
    original_read = reader.read

    def read(image, regions=None):
        if regions is None:
            shapes.append(image.shape[:2])
        return original_read(image, regions)

    monkeypatch.setattr(screen, "frame", frame)
    monkeypatch.setattr(reader, "read", read)
    result = agent.click_text(window_title="App", text="Saved")
    assert shapes == [(screen.h - 4, screen.w - 4)]
    assert posted == [(50 + 2 + 155, 60 + 2 + 120)]
    assert result["text_match"]["bbox_px"] == [162, 172, 252, 192]
    assert result["operation_marker"]["window_px"] == [157, 122]


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
    assert report["operation_marker"] == {
        "kind": "posted_click",
        "screen_px": [210, 180],
        "window_px": [160, 120],
    }
    assert report["screen"]["state"] == "settled"
    assert "click_point" in [t["name"] for t in agent.tool_specs()]
    blocked, _, _ = make(click=click)
    with pytest.raises(PermissionError):
        blocked.click_point(window_title="App", x=1, y=1)


def test_marker_crop_is_pre_action_and_does_not_contaminate_pixels():
    agent, screen, _ = make(allowed_windows=["App"], windows=lambda _: None)

    def invoke(*args, **kwargs):
        screen.saved = True
        return {
            "performed": "invoke",
            "window": "App",
            "element": {"name": "Save", "center_px": [195, 180]},
        }

    agent._act = invoke
    content = McpServer(agent).call(
        "click_element", {"window_title": "App", "target": "Save", "marker_image": True}
    )
    import base64

    marker = json.loads(content[0]["text"])["operation_marker"]
    center = marker["image_point_px"]
    assert marker["kind"] == "uia_target_center"
    assert [a + b for a, b in zip(center, marker["crop_origin_px"], strict=True)] == [195, 180]
    png = cv2.imdecode(np.frombuffer(base64.b64decode(content[1]["data"]), np.uint8), 1)
    assert png[center[1], center[0]].tolist() == [20, 220, 255]
    assert png[center[1] + 15, center[0]].tolist() == [240] * 3
    assert screen.frame()[120, 145].tolist() == [30, 120, 30]


def test_partial_mouse_dispatch_returns_observe_before_retry():
    agent, _, _ = make(
        allowed_windows=["App"],
        windows=lambda _: None,
        click=lambda *args: {
            "performed": "mouse_down_at",
            "window": "App",
            "element": {"name": "button"},
            "dispatch_state": "partial",
        },
    )
    report = agent.click_point(window_title="App", x=150, y=100)
    assert report["execution_state"] == "partial"
    assert report["operation_marker"]["kind"] == "posted_mouse_down"
    assert "release failed" in report["next_step"]


def test_unavailable_history_keeps_successful_action_and_point(monkeypatch):
    calls = []

    def click(title, x, y):
        calls.append((x, y))
        return {"performed": "click_at", "element": {"name": "button"}, "window": title}

    agent, _, _ = make(
        click=click,
        windows=lambda _: None,
        allowed_windows=["App"],
        history_seconds=3,
        images=False,
    )

    def vanished(*args):
        raise ValueError("window closed")

    monkeypatch.setattr(agent, "_start_history", vanished)
    parts = McpServer(agent).call(
        "click_point", {"window_title": "App", "x": 180, "y": 150, "marker_image": True}
    )
    assert len(parts) == 1 and calls == [(180, 150)]
    report = json.loads(parts[0]["text"])
    assert report["execution_state"] == "completed"
    assert report["operation_marker"]["screen_px"] == [180, 150]
    assert report["screen_history"]["state"] == "capture_unavailable"


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


@pytest.mark.parametrize(
    "tool,args",
    [
        ("click_element", {"target": "Add"}),
        ("set_field_text", {"field": "Seats", "text": "7"}),
        ("click_point", {"x": 200, "y": 180}),
        ("click_text", {"text": "Saved"}),
    ],
)
def test_mcp_returns_completed_receipt_if_post_action_ocr_fails(tool, args):
    calls = []

    def mutate(*a, **kw):
        calls.append(1)
        screen.act()
        return {"performed": "test", "window": "App", "element": {"name": "target"}}

    agent, screen, reader = make(
        act=mutate, click=mutate, windows=lambda _: None, allowed_windows=["App"]
    )

    original_read = reader.read

    def fail(*args):
        if not screen.saved:
            return original_read(*args)
        raise RuntimeError("lost OCR")

    reader.read = fail
    rpc = McpServer(agent).handle(
        {
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": {"window_title": "App", **args}},
        }
    )["result"]
    assert screen.saved and calls == [1] and not rpc["isError"]
    report = json.loads(rpc["content"][0]["text"])
    assert report["execution_state"] == "completed" and report["observation_state"] == "failed"
    assert "lost OCR" in report["observation_error"] and "Do not repeat" in report["next_step"]


def test_blocked_explicit_wait_uses_saved_controls_and_honors_waits():
    line = {"name": "Seats", "control_type": "edit", "center_px": [10, 20], "value": "2"}
    other = {**line, "center_px": [70, 20]}
    before = {"title": "App", "controls": [line, other]}
    after = {"title": "App", "controls": [dict(line, value="7"), other]}
    agent, screen, _ = make(windows=lambda _: after)
    agent.blocked.add("app")
    agent.last_controls["app"] = before
    report = agent.wait_for_settle(window_title="App", quiet_ms=5, read_text=True)
    assert not screen.areas and report["source"] == "ui_automation"
    assert report["state"] == "settled" and report["window"] == after
    assert report["controls_changed"] == [
        {"name": "Seats", "control_type": "edit", "center_px": [10, 20], "was": "2", "value": "7"}
    ]
    again = agent.wait_for_settle(window_title="App", expect_change_ms=0)
    assert again["state"] == "unchanged" and not again["controls_changed"]
    with pytest.raises(ValueError):
        agent.wait_for_settle(window_title="App", timeout_ms=-1)
    agent._windows = lambda _: None
    assert agent.wait_for_settle(window_title="App")["state"] == "unavailable"


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


# -- production guards -------------------------------------------------------------------------


def test_allow_rules_bind_titles_to_programs():
    from motionspec.agent import parse_allow_rule

    assert parse_allow_rule("Keystone agent task | Python.EXE") == (
        "keystone agent task",
        "python.exe",
    )
    assert parse_allow_rule("Report | draft") == ("report | draft", None)  # not a program name
    assert parse_allow_rule("| notepad.exe") == ("", "notepad.exe")
    agent, _, _ = make(
        allowed_windows=["Sync settings | sync.exe", "Plain title"],
        process_name=lambda hwnd: {1: "sync.exe", 2: "evil.exe"}[hwnd],
    )
    agent._permit(1, "Sync settings - main")
    agent._permit(2, "Plain title")  # a title-only rule still matches any program
    with pytest.raises(PermissionError, match="evil.exe"):
        agent._permit(2, "Sync settings - fake")
    with pytest.raises(PermissionError):
        agent._permit(1, "Something else")


def test_locked_desktop_refuses_before_reserving_an_action():
    calls = []
    agent, _, _ = make(
        allowed_windows=["App"], desktop=lambda: "locked", act=lambda *a, **k: calls.append(a)
    )
    server = McpServer(agent)
    reply = rpc(
        server,
        "tools/call",
        {
            "name": "click_element",
            "arguments": {"window_title": "App", "target": "Save"},
            "_meta": {"visfunc/operationId": "11111111-2222-3333-4444-555555555555"},
        },
    )
    assert reply["result"]["isError"] and "locked" in reply["result"]["content"][0]["text"]
    assert calls == [] and server.receipts.stats()["reserved"] == 0
    reply = rpc(server, "tools/call", {"name": "read_screen_text", "arguments": {}}, mid=2)
    assert "DesktopUnavailable" in reply["result"]["content"][0]["text"]
    with pytest.raises(Exception, match="service"):
        make(desktop=lambda: "service")[0].require_desktop()


class SlowAgent:
    """Records how many calls overlap, per window and in total."""

    allowed_windows = []

    def __init__(self):
        import threading

        self.lock, self.active, self.peak, self.peak_by_window = threading.Lock(), {}, 0, {}

    def tool_specs(self):
        return [{"name": "list_ui_elements", "inputSchema": {"type": "object"}}]

    def require_desktop(self):
        pass

    def __getattr__(self, name):  # the other tools are looked up but never called here
        return lambda **kw: {}

    def list_ui_elements(self, window_title=None):
        import time

        with self.lock:
            self.active[window_title] = self.active.get(window_title, 0) + 1
            self.peak = max(self.peak, sum(self.active.values()))
            self.peak_by_window[window_title] = max(
                self.peak_by_window.get(window_title, 0), self.active[window_title]
            )
        time.sleep(0.2)
        with self.lock:
            self.active[window_title] -= 1
        return {"elements": []}


def test_calls_on_different_windows_overlap_and_one_window_queues():
    import threading

    agent = SlowAgent()
    server = McpServer(agent)
    threads = [
        threading.Thread(
            target=rpc,
            args=(
                server,
                "tools/call",
                {"name": "list_ui_elements", "arguments": {"window_title": title}},
                i,
            ),
        )
        for i, title in enumerate(["A", "B", "A", "B"])
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert agent.peak == 2  # A and B ran side by side
    assert agent.peak_by_window == {"A": 1, "B": 1}  # never two calls on one window at once


def test_tool_calls_are_logged_without_arguments(tmp_path):
    import json

    from motionspec.agent import json_log

    path = tmp_path / "logs" / "calls.jsonl"
    server = McpServer(
        make(
            allowed_windows=["App"],
            act=lambda *a, **k: {
                "performed": "set_value",
                "window": "App",
                "element": {"name": "Note"},
                "after": None,
            },
            windows=lambda title: None,
        )[0],
        log=json_log(str(path)),
    )
    rpc(
        server,
        "tools/call",
        {
            "name": "set_field_text",
            "arguments": {"window_title": "App", "field": "Note", "text": "secret value 123"},
            "_meta": {"visfunc/operationId": "21111111-2222-3333-4444-555555555555"},
        },
    )
    rpc(
        server,
        "tools/call",
        {"name": "click_element", "arguments": {"window_title": "Nope"}},
        mid=2,
    )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(e["tool"], e["ok"]) for e in lines] == [
        ("set_field_text", True),
        ("click_element", False),
    ]
    assert lines[0]["window"] == "App" and lines[0]["execution_state"] == "completed"
    assert "secret value" not in path.read_text(encoding="utf-8")


def test_http_requires_a_token_unless_explicitly_waived(capsys, monkeypatch):
    from motionspec.cli import main

    monkeypatch.delenv("VISFUNC_MCP_TOKEN", raising=False)

    with pytest.raises(SystemExit) as stop:
        main(["agent-mcp", "--http", "127.0.0.1:1"])
    assert stop.value.code == 2
    assert "VISFUNC_MCP_TOKEN" in capsys.readouterr().err


def test_http_refusals_reach_the_client_as_status_codes():
    import json
    import threading
    import urllib.error
    import urllib.request

    from motionspec.agent import serve_http

    started = threading.Event()
    holder = {}

    def ready(httpd):
        holder["httpd"] = httpd
        started.set()

    thread = threading.Thread(
        target=serve_http,
        kwargs=dict(agent=make()[0], port=0, token="right-token", ready=ready),
        daemon=True,
    )
    thread.start()
    started.wait(5)
    url = f"http://127.0.0.1:{holder['httpd'].server_port}/mcp"
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    try:
        for headers, expected in (
            ({}, 401),
            ({"Authorization": "Bearer wrong"}, 401),
            ({"Authorization": "Bearer right-token", "Origin": "https://evil.example"}, 403),
        ):
            request = urllib.request.Request(
                url, body, {"Content-Type": "application/json", **headers}
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=5)
            assert error.value.code == expected  # a status, not a reset connection
        request = urllib.request.Request(
            url, body, {"Content-Type": "application/json", "Authorization": "Bearer right-token"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.load(response)["result"]["tools"]
    finally:
        holder["httpd"].shutdown()
        thread.join(5)


def test_window_outside_the_rules_is_refused_before_reserving(monkeypatch):
    from svsg.recognition import ui_elements

    monkeypatch.setattr(ui_elements, "find_window", lambda title: 7)
    monkeypatch.setattr(ui_elements, "window_title", lambda hwnd: "Sync settings")
    screen, reader = FakeScreen(), FakeReader()
    agent = ScreenAgent(
        capture=screen.capture,
        reader_factory=lambda: reader,
        locate_window=lambda title: (50, 60, 450, 360),
        history_seconds=0,
        allowed_windows=["Sync settings | sync.exe"],
        desktop=lambda: "interactive",
        process_name=lambda hwnd: "impostor.exe",
    )
    server = McpServer(agent)
    meta = {"visfunc/operationId": "31111111-2222-3333-4444-555555555555"}
    reply = rpc(
        server,
        "tools/call",
        {
            "name": "click_element",
            "arguments": {"window_title": "Sync settings", "target": "Save"},
            "_meta": meta,
        },
    )
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] and "impostor.exe" in text and "unconfirmed" not in text
    assert server.receipts.stats()["reserved"] == 0  # nothing reserved, a retry is not "unknown"
