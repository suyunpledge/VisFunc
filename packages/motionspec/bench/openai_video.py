"""OpenAI 的视频输入：Responses 接口的 input_video 需要先经 Files API 上传拿到 file_id。

这条路是实测探出来的（官方文档里没有找到视频输入的说明）：
- Chat Completions 明确拒绝 video_url；
- Responses 的 input_video 不接受 data URL，报错要求 file_id。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402


def upload(base_url: str, key: str, path: Path, purpose: str = "user_data") -> tuple[int, dict | str]:
    boundary = "----motionspec" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n{purpose}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        "Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8")
    body = head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        base_url + "/files", data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:500]


def delete(base_url: str, key: str, file_id: str) -> None:
    request = urllib.request.Request(base_url + "/files/" + file_id, method="DELETE", headers={"Authorization": "Bearer " + key})
    try:
        urllib.request.urlopen(request, timeout=60).read()
    except Exception:
        pass


def respond(base_url: str, key: str, model: str, content: list[dict], *, max_tokens: int = 4000, extra: dict | None = None) -> dict:
    """Responses 接口。content 用 input_text / input_image / input_video 内容块。"""
    body = {"model": model, "input": [{"role": "user", "content": content}], "max_output_tokens": max_tokens}
    body.update(extra or {})
    status, payload, seconds = client.post(base_url + "/responses", key, body)
    if status == 200 and isinstance(payload, dict):
        texts = [part.get("text", "") for item in payload.get("output", []) for part in (item.get("content") or [])]
        return {"ok": True, "status": status, "text": "".join(texts), "usage": payload.get("usage") or {},
                "seconds": round(seconds, 1), "error": None}
    return {"ok": False, "status": status, "text": "", "usage": {}, "seconds": round(seconds, 1), "error": payload}


if __name__ == "__main__":
    import tempfile

    from canary import QUESTION, make_video
    from providers import find

    provider = find(r"openai\.com")
    with tempfile.TemporaryDirectory() as tmp:
        path, code = make_video(Path(tmp))
        print("code in video:", code)
        status, uploaded = upload(provider.base_url, provider.key, path)
        print("upload:", status, uploaded if status != 200 else {k: uploaded.get(k) for k in ("id", "purpose", "bytes", "status")})
        if status == 200:
            time.sleep(2)
            for model in ("gpt-6.1-sol", "gpt-6.1-terra"):
                result = respond(provider.base_url, provider.key, model,
                                 [{"type": "input_video", "file_id": uploaded["id"]}, {"type": "input_text", "text": QUESTION}])
                shown = result["text"].strip()[:100] if result["ok"] else f"HTTP {result['status']}: {str(result['error'])[:300]}"
                print(f"{model:<16} {'SEES VIDEO' if code in result['text'].upper() else 'no':<11} {result['seconds']}s  {shown}  usage={result['usage']}")
            delete(provider.base_url, provider.key, uploaded["id"])
