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
        slot = entry.get("id") or name.split(":", 1)[1]
        out.append(
            Provider(slot, config["baseURL"].rstrip("/"), config.get("model", ""), config.get("modelLabel", ""), secrets.get(slot, ""))
        )
    return out


def find(pattern: str) -> Provider:
    """Select a unique configured slot; never silently choose a billing endpoint."""
    candidates = [p for p in load() if any(re.search(pattern, value, re.I)
                  for value in (p.slot, p.model, p.base_url))]
    if not candidates:
        raise KeyError(pattern)
    if len(candidates) != 1:
        raise ValueError("ambiguous provider selection; use an exact slot via --providers: "
                         + ", ".join(p.slot for p in candidates))
    if not candidates[0].key:
        raise ValueError("selected provider has no credential")
    return candidates[0]


if __name__ == "__main__":
    for provider in load():
        print(provider.describe())
