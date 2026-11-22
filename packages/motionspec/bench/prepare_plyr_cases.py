"""Prepare hard-subtitled H.264 clips and capture the actual Plyr toolbar on each."""

import argparse
import json
import shutil
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

# width, font pixels, cue, outline-only, picture quality (CRF)
CASES = [
    (640, 17, "列车将从右侧进站。", True, 29),
    (800, 20, "Follow the signs to exit C.", False, 25),
    (560, 14, "请在此等候。\nWait here, please.", True, 32),
    (720, 16, "Service resumes at 07:35.", False, 29),
    (600, 15, "下一班车还有五分钟。", True, 34),
    (896, 19, "Exit through the north gate.", True, 30),
    (480, 13, "出口在左侧。", False, 34),
    (768, 18, "注意站台间隙。\nMind the gap.", True, 30),
    (700, 16, "Transfer at Central Station.", False, 28),
    (624, 15, "请携带随身物品。", True, 32),
    (576, 14, "", False, 32),
    (820, 17, "", False, 28),
    (640, 16, "", True, 30),
    (720, 15, "", False, 32),
    (692, 16, "Next train in 12 minutes.", False, 31),
    (536, 15, "请到 6 号窗口", True, 32),
    (864, 18, "Keep Clear", True, 29),
    (600, 14, "GateB12", False, 31),
    (728, 17, "雨后道路湿滑。\nWatch your step.", True, 33),
    (640, 15, "下一站北门", False, 26),
    (780, 16, "Please wait until 18:40.", True, 32),
    (580, 15, "AB12345", False, 34),
    (912, 19, "广场向右。\nTurn right here.", True, 30),
    (516, 13, "请勿倚靠车门。", True, 35),
    (614, 14, "", False, 30),
    (818, 16, "", True, 32),
    (746, 18, "请按顺序入场。\nEnter in order.", True, 31),
    (568, 15, "Board at gate 4.", False, 33),
    (822, 17, "Central Park", False, 28),
    (658, 15, "东门开放", True, 32),
    (626, 15, "AB 204", False, 29),
    (752, 16, "不要拥挤。\nPlease stay in line.", True, 34),
    (606, 16, "", False, 30),
    (792, 17, "", False, 33),
]
HTML = """<!doctype html><meta charset=utf-8><link rel=stylesheet href=plyr.css>
<style>*{box-sizing:border-box}html,body,.plyr,.plyr__video-wrapper{margin:0;width:100%;height:100%}
body{background:black}.plyr__controls{opacity:1!important;transform:none!important}</style>
<video id=v playsinline></video><script src=plyr.js></script>"""
HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, \
Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,{fs},{color},&H000000FF,&H00151515,&H00151515,0,0,0,0,100,100,\
0,0,{style},1,0,2,12,12,{margin},1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def render(sprite, dest):
    dest.mkdir(exist_ok=True, parents=True)
    ns = "{http://www.w3.org/2000/svg}"
    ET.register_namespace("", ns[1:-1])
    symbols = {s.get("id"): s for s in ET.parse(sprite).iter(ns + "symbol")}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 96, "height": 96})
        for label, key in [
            ("play", "play"),
            ("pause", "pause"),
            ("fullscreen", "enter-fullscreen"),
            ("mute", "muted"),
            ("volume", "volume"),
        ]:
            sym = symbols["plyr-" + key]
            root = ET.Element(
                ns + "svg",
                {
                    "viewBox": sym.get("viewBox", "0 0 18 18"),
                    "width": "96",
                    "height": "96",
                    "fill": "white",
                },
            )
            root.extend(list(sym))
            page.set_content(
                "<style>body{margin:0;color:white}</style>" + ET.tostring(root, encoding="unicode")
            )
            page.screenshot(path=dest / (label + ".png"), omit_background=True)
        browser.close()


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def prepare(assets, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("plyr.js", "plyr.css", "plyr.svg"):
        shutil.copyfile(assets / name, folder / name)
    (folder / "index.html").write_text(HTML, encoding="utf8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(folder)))
    Thread(target=server.serve_forever, daemon=True).start()
    truth = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chrome", args=["--autoplay-policy=no-user-gesture-required"]
        )
        for i, (w, fs, text, outline, crf) in enumerate(CASES):
            h = 2 * round(w * 9 / 32)
            source = "flower.mp4" if i % 2 == 0 else "trailer.mp4"
            seconds = (0.25 + (i % 5) * 0.4) if i % 2 == 0 else (6 + 2 * i)
            if i >= 14:
                source = "bunny.mp4"
                seconds = 1 + (i - 14) * 1.3
            if i >= 26:
                source = ("bunny.mp4", "flower.mp4", "trailer.mp4")[(i - 26) % 3]
                seconds = (1.8, 2.4, 19, 2.8, 0.7, 22, 2.1, 1.6)[i - 26]
            subtitle = HEADER.format(
                w=w,
                h=h,
                fs=fs,
                color="&H007CEAFF" if outline else "&H00FFFFFF",
                style=1 if outline else 3,
                margin=round(h * 0.19),
            )
            cue = text.replace("\n", r"\N")
            (folder / "caption.ass").write_text(
                subtitle + "Dialogue: 0,0:00:00.00,0:00:08.00,Default,,0,0,0,," + cue + "\n",
                encoding="utf-8-sig",
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    str(seconds),
                    "-i",
                    str((assets / source).resolve()),
                    "-t",
                    "2",
                    "-an",
                    "-vf",
                    f"scale={w}:{h},ass=caption.ass",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(crf),
                    f"clip{i}.mp4",
                ],
                cwd=folder,
                check=True,
            )
            page = browser.new_page(viewport=dict(width=w, height=h), device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            page.evaluate(
                """async i => {
               const v=document.querySelector('video');
               v.src='clip'+i+'.mp4';
               const controls=['play','progress','current-time','mute','fullscreen'];
               const player=window.player=new Plyr(v,{controls,
                 iconUrl:'plyr.svg',storage:{enabled:false},hideControls:false,autopause:false});
               await new Promise(r=>player.on('ready',r));
               v.muted=i%3!==1;
               await v.play();if(i%2===0)v.pause();
               v.currentTime=.5;await new Promise(r=>v.addEventListener('seeked',r,{once:true}));
            }""",
                i,
            )
            if i in (12, 13, 25, 33):
                page.add_style_tag(content=".plyr__controls{display:none!important}")
            page.wait_for_timeout(100)
            page.screenshot(path=folder / f"case{i}.png")
            # Isolate visible glyphs, while retaining their CSS layout and state.
            page.add_style_tag(
                content=".plyr__controls{background:black!important}"
                ".plyr__controls button{color:white!important;background:black!important}"
            )
            mask = np.asarray(Image.open(BytesIO(page.screenshot())).convert("L"))
            icons = []
            for name in ("play", "mute", "fullscreen"):
                button = page.locator(f'.plyr__controls button[data-plyr="{name}"]')
                if not button.is_visible():
                    continue
                glyph = button.locator("svg:visible")
                bounds = glyph.bounding_box()
                kind = glyph.locator("use").evaluate("u=>u.href.baseVal.split('#').pop()")
                label = {
                    "plyr-play": "play",
                    "plyr-pause": "pause",
                    "plyr-muted": "mute",
                    "plyr-volume": "volume",
                    "plyr-enter-fullscreen": "fullscreen",
                }[kind]
                x, y = int(bounds["x"]), int(bounds["y"])
                right, bottom = (
                    int(np.ceil(bounds["x"] + bounds["width"])),
                    int(np.ceil(bounds["y"] + bounds["height"])),
                )
                ys, xs = np.where(mask[y:bottom, x:right] > 160)
                icons.append(
                    dict(
                        label=label,
                        bbox_px=[
                            x + int(xs.min()),
                            y + int(ys.min()),
                            x + int(xs.max()) + 1,
                            y + int(ys.max()) + 1,
                        ],
                    )
                )
            truth.append(
                dict(case=i, size=[w, h], sequence=source, text=text, icons=icons, crf=crf)
            )
            page.close()
        browser.close()
    render(assets / "plyr.svg", folder / "icons")
    server.shutdown()
    server.server_close()
    (folder / "truth.json").write_text(json.dumps(truth, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    prepare(args.assets.resolve(), args.folder.resolve())
