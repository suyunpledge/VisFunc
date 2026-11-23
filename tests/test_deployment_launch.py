"""Windows launcher helpers: the MCP token is created once, allow rules are read as written."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("dotenv")
SPEC = importlib.util.spec_from_file_location(
    "launch", Path(__file__).resolve().parents[1] / "deployment" / "windows" / "launch.py"
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


def test_token_is_created_once_and_replaces_the_placeholder(tmp_path, monkeypatch):
    monkeypatch.delenv("VISFUNC_MCP_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "SVSG_PORT=3002\n# VISFUNC_MCP_TOKEN=\nVISFUNC_MCP_PORT=8765\n", encoding="utf-8"
    )
    assert launch.ensure_token(env) is True
    first = env.read_text(encoding="utf-8")
    token = next(
        line.split("=", 1)[1]
        for line in first.splitlines()
        if line.startswith("VISFUNC_MCP_TOKEN=")
    )
    assert len(token) >= 40 and "# VISFUNC_MCP_TOKEN" not in first
    assert launch.ensure_token(env) is False and env.read_text(encoding="utf-8") == first
    monkeypatch.setenv("VISFUNC_MCP_TOKEN", "from-environment")
    other = tmp_path / "other.env"
    other.write_text("SVSG_PORT=3002\n", encoding="utf-8")
    assert launch.ensure_token(other) is False  # a process setting wins; nothing written


def test_allow_rules_skip_comments_and_note_title_only_entries(tmp_path, capsys):
    rules = tmp_path / "allowed-windows.txt"
    rules.write_text("# comment\n\nSync settings | sync.exe\nPlain title\n", encoding="utf-8")
    assert launch.allow_rules(rules) == ["Sync settings | sync.exe", "Plain title"]
    out = capsys.readouterr().out
    assert "Plain title" in out and "Sync settings" not in out
