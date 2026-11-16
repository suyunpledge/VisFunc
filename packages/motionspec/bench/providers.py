"""从本机 Forge 配置里读取模型提供方（地址、模型名、密钥），供实测脚本使用。

密钥只在进程内使用：不打印、不写入任何输出文件。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

FORGE = Path.home() / ".forge"


@dataclass
class Provider:
    slot: str  # Forge 里的 custom__xxxx 标识
    base_url: str
    model: str
    label: str
    key: str

    def describe(self) -> str:
        return f"{self.label:<16} {self.model:<34} {self.base_url}  key:{'yes' if self.key else 'MISSING'}"


def load() -> list[Provider]:
    patch = json.loads((FORGE / "forge.patch.json").read_text(encoding="utf-8"))
    secrets = json.loads((FORGE / "secrets.json").read_text(encoding="utf-8"))
    out = []
    for entry in patch:
        config = entry.get("config") or {}
        name = entry.get("name", "")
        if not name.startswith("provider:") or "baseURL" not in config:
            continue
        slot = name.split(":", 1)[1]
        out.append(
            Provider(slot, config["baseURL"].rstrip("/"), config.get("model", ""), config.get("modelLabel", ""), secrets.get(slot, ""))
        )
    return out


def find(pattern: str) -> Provider:
    """按模型名或地址的正则找一个提供方。"""
    for provider in load():
        if re.search(pattern, provider.model, re.I) or re.search(pattern, provider.base_url, re.I):
            return provider
    raise KeyError(pattern)


if __name__ == "__main__":
    for provider in load():
        print(provider.describe())
