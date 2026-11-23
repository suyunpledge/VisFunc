"""UI Automation 元素查找：排序规则（纯函数）；Windows 上再检查找不到窗口时的报错。"""

from __future__ import annotations

import sys

import pytest

from svsg.recognition.ui_elements import find_element


def element(name, kind="button", enabled=True, **extra):
    return {
        "name": name,
        "control_type": kind,
        "enabled": enabled,
        "bbox_px": [0, 0, 10, 10],
        "center_px": [5, 5],
        **extra,
    }


RESULT = {
    "elements": [
        element("Refresh page", kind="text"),
        element("Refresh", enabled=False),
        element("Refresh"),
        element("Server name", kind="edit", value="backup-server-07"),
        element("", kind="button", help_text="Delete selected"),
    ]
}


def test_exact_enabled_interactive_first():
    hits = find_element(RESULT, "refresh")
    assert [(h["name"], h["enabled"]) for h in hits] == [
        ("Refresh", True),
        ("Refresh", False),
        ("Refresh page", True),
    ]
    assert [h["name"] for h in find_element(RESULT, "Refresh", exact=True)] == [
        "Refresh",
        "Refresh",
    ]


def test_matches_values_help_text_and_type():
    assert find_element(RESULT, "backup-server-07")[0]["control_type"] == "edit"
    assert find_element(RESULT, "delete")[0]["help_text"] == "Delete selected"
    assert find_element(RESULT, "refresh", control_type="text")[0]["name"] == "Refresh page"
    with pytest.raises(ValueError):
        find_element(RESULT, " ")


@pytest.mark.parametrize("action", ["click", "set_text"])
def test_same_name_needs_live_unambiguous_reference(action):
    from svsg.recognition.ui_elements import _action_target

    west = element("Seats", kind="edit", element_ref="uia:123.1")
    east = element("Seats", kind="edit", element_ref="uia:123.2")
    with pytest.raises(ValueError, match="ambiguous.*123.1.*123.2"):
        _action_target([west, east], "Seats", action, None, None)
    assert _action_target([east, west], "Seats", action, None, east["element_ref"]) is east
    with pytest.raises(ValueError, match="no current element"):
        _action_target([west], "Seats", action, None, east["element_ref"])
    assert _action_target([east, dict(west, enabled=False)], "Seats", action, None, None) is east


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_missing_window_is_reported():
    pytest.importorskip("comtypes")
    from svsg.recognition.ui_elements import find_window

    with pytest.raises(ValueError, match="no visible window"):
        find_window("no window has this title 7d1c0e")


def test_password_field_value_is_omitted_from_ui_event_input():
    from svsg.recognition.ui_elements import _PROPERTIES, _describe

    props = {
        "rect": [0, 0, 20, 10],
        "control_type": 50004,
        "name": "PIN",
        "password": True,
        "value": "sensitive input",
    }

    class Element:
        def GetCachedPropertyValue(self, prop):
            return next((props[key] for key in props if _PROPERTIES[key] == prop), None)

    assert "value" not in _describe(Element())
    props["password"] = False
    assert _describe(Element())["value"] == "sensitive input"
