"""暗号测试：确认每个模型是否真的收到了视频。

视频里只有画面上才有的一串随机字符；模型答得出来才算收到。
有的接口会悄悄丢掉不认识的内容块、照样返回 200，这时模型会瞎猜或说没看到视频。

    python bench/canary.py
"""

from __future__ import annotations

import random
import string
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402
from providers import find  # noqa: E402

MODELS = [
    ("mimo-v2.6-flash", r"mimo-v2\.6-flash$"),
    ("mimo-v2.6-pro", r"mimo-v2\.6-pro$"),
    ("doubao-seed-2-1-turbo-260628", r"volces"),
    ("doubao-seed-2-1-pro-260915", r"volces"),
    ("gpt-6.1-sol", r"openai\.com"),
    ("gpt-6.1-terra", r"openai\.com"),
]
QUESTION = (
    "A short code made of letters and digits is displayed in this video. "
    "Reply with only that code. If you did not receive any video, reply exactly NOVIDEO."
)


def make_video(folder: Path) -> tuple[Path, str]:
    code = "".join(random.choice(string.ascii_uppercase + "23456789") for _ in range(5))
    frames = folder / "frames"
    frames.mkdir()
    for i in range(60):  # 2 秒，30 帧/秒，字符串缓慢右移
        img = np.full((360, 640, 3), 30, np.uint8)
        cv2.putText(img, code, (80 + i * 3, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (240, 240, 240), 6, cv2.LINE_AA)
        cv2.imwrite(str(frames / f"{i:03d}.png"), img)
    out = folder / "canary.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(frames / "%03d.png"),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out, code


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, code = make_video(Path(tmp))
        print("code in video:", code, " size:", path.stat().st_size, "bytes")
        for model, pattern in MODELS:
            provider = find(pattern)
            vendor = client.vendor_of(provider.base_url)
            result = client.chat(provider.base_url, provider.key, model, [client.video(path, vendor, fps=2), client.text(QUESTION)], max_tokens=600)
            answer = result["text"].strip().replace("\n", " ")[:80] if result["ok"] else f"HTTP {result['status']}: {str(result['error'])[:220]}"
            verdict = "SEES VIDEO" if result["ok"] and code in result["text"].upper() else "no"
            print(f"{model:<32} {verdict:<11} {result['seconds']:>5}s  {answer}   usage={ {k: v for k, v in result['usage'].items() if isinstance(v, int)} }")
        if "--openai-probe" in sys.argv:
            provider = find(r"openai\.com")
            url = client.data_url(path, "video/mp4")
            for label, body in (
                ("responses input_video", {"model": "gpt-6.1-sol", "input": [{"role": "user", "content": [
                    {"type": "input_video", "video_url": url}, {"type": "input_text", "text": QUESTION}]}]}),
                ("responses input_file", {"model": "gpt-6.1-sol", "input": [{"role": "user", "content": [
                    {"type": "input_file", "filename": "canary.mp4", "file_data": url}, {"type": "input_text", "text": QUESTION}]}]}),
            ):
                status, payload, seconds = client.post(provider.base_url + "/responses", provider.key, body)
                if status == 200 and isinstance(payload, dict):
                    texts = [c.get("text", "") for item in payload.get("output", []) for c in item.get("content", []) or []]
                    out = " ".join(texts).strip()[:120]
                    print(f"{label:<26} 200 {'SEES VIDEO' if code in out.upper() else 'no':<11} {out}")
                else:
                    print(f"{label:<26} HTTP {status} {str(payload)[:260]}")


if __name__ == "__main__":
    main()
