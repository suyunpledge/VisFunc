"""对三家模型的最小调用封装：文本、图片、视频内容块，统一返回文本和用量。

只用标准库。各家视频内容块的写法依据其官方文档：
- 火山方舟：{"type": "video_url", "video_url": {"url": <data URL>, "fps": ≤5}}
- 小米 MiMo：{"type": "video_url", "video_url": {"url": <data URL>}, "fps": ≤10, "media_resolution": ...}
- OpenAI：官方文档没有视频输入，这里的几种写法只用于探测。
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def text(value: str) -> dict:
    return {"type": "text", "text": value}


def image(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(path, "image/png")}}


def video(path: Path, vendor: str, fps: float | None = None) -> dict:
    url = data_url(path, "video/mp4")
    if vendor == "ark":
        inner = {"url": url}
        if fps is not None:
            inner["fps"] = fps
        return {"type": "video_url", "video_url": inner}
    part = {"type": "video_url", "video_url": {"url": url}}
    if vendor == "mimo" and fps is not None:
        part["fps"] = fps
    return part


def vendor_of(base_url: str) -> str:
    if "volces" in base_url:
        return "ark"
    if "xiaomimimo" in base_url:
        return "mimo"
    if "openai.com" in base_url:
        return "openai"
    return "other"


def post(url: str, key: str, body: dict, timeout: float = 300.0) -> tuple[int, dict | str, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    began = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8")), time.time() - began
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:600], time.time() - began
    except Exception as error:  # 超时、连接中断
        return 0, f"{type(error).__name__}: {error}"[:300], time.time() - began


def chat(base_url: str, key: str, model: str, parts: list[dict], *, max_tokens: int = 2000, extra: dict | None = None) -> dict:
    """Chat Completions。返回 {"ok", "status", "text", "usage", "seconds", "error"}。"""
    body = {"model": model, "messages": [{"role": "user", "content": parts}]}
    # OpenAI 的新模型只认 max_completion_tokens，其余两家用 max_tokens
    body["max_completion_tokens" if "openai.com" in base_url else "max_tokens"] = max_tokens
    body.update(extra or {})
    status, payload, seconds = post(base_url + "/chat/completions", key, body)
    if status == 200 and isinstance(payload, dict):
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        return {"ok": True, "status": status, "text": message.get("content") or "", "usage": payload.get("usage") or {},
                "seconds": round(seconds, 1), "error": None}
    return {"ok": False, "status": status, "text": "", "usage": {}, "seconds": round(seconds, 1), "error": payload}
