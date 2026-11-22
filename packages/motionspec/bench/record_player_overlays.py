"""Record player UIs and label a decoded frame using authored cues and UI vectors."""

import argparse
import json
import shutil
import subprocess
from functools import partial
from hashlib import sha256
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from prepare_plyr_cases import HEADER, Quiet

# source, seek, width, opening title, caption, playing, muted, toolbar visible
CASES = {
    "dplayer": [
        ("park", 18, 640, "FIELD NOTES", "", True, False, True),
        ("sintel", 35, 752, "", "Line R 42", False, True, True),
        ("park", 76, 600, "", "Route B12", True, True, True),
        ("sintel", 21, 720, "NORTH COAST", "EXIT NOW", False, False, True),
        ("park", 108, 680, "", "", True, False, False),
        ("sintel", 41, 800, "", "Meet at K 72.", True, False, True),
    ],
    "artplayer": [
        ("sintel", 24, 704, "WINTER LIGHT", "", False, False, True),
        ("park", 51, 600, "", "RX 318", True, True, True),
        ("sintel", 37, 768, "", "RX318", False, True, True),
        ("park", 128, 720, "OPEN WATER", "HOLD HERE", True, False, True),
        ("sintel", 29, 640, "", "", True, False, False),
        ("park", 93, 816, "", "Gate T 6", False, False, True),
    ],
    "mejs": [
        ("oceans", 3, 704, "DEEP BLUE", "", True, False, True),
        ("oceans", 11, 632, "", "Pier N 24", False, True, True),
        ("oceans", 17, 768, "", "N24", True, True, True),
        ("oceans", 24, 720, "COASTAL LOG", "TURN BACK", False, False, True),
        ("oceans", 30, 656, "", "", True, False, False),
        ("oceans", 36, 816, "", "Deck Q 7", True, False, True),
    ],
}
CONTROLS = {
    "dplayer": [".dplayer-play-icon", ".dplayer-volume-icon", ".dplayer-full-icon"],
    "artplayer": [".art-control-playAndPause", ".art-control-volume", ".art-control-fullscreen"],
    "mejs": [".mejs__playpause-button", ".mejs__volume-button", ".mejs__fullscreen-button"],
}


def prepare_clip(folder, assets, case):
    source, seconds, w, title, text, *_ = case
    h = 2 * round(w * 9 / 32)
    ass = HEADER.format(
        w=w, h=h, fs=round(h * 0.05), color="&H00FFFFFF", style=3, margin=round(h * 0.2)
    )
    if title:
        style = (
            r"{\an5\pos("
            + f"{w // 2},{round(h * 0.575)}"
            + r")\fsp2\fs"
            + str(round(h * 0.075))
            + "}"
        )
        ass += f"Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{style}{title}\n"
    if text:
        ass += f"Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{text}\n"
    (folder / "cue.ass").write_text(ass, encoding="utf-8-sig")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(seconds),
            "-i",
            str(assets / f"{source}.mp4"),
            "-t",
            "8",
            "-an",
            "-vf",
            f"scale={w}:{h},ass=cue.ass",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "clip.mp4",
        ],
        cwd=folder,
        check=True,
    )
    return dict(width=w, height=h)


def label_glyph(renderer, element, label, templates):
    """Rasterize the player's vector on transparency, independent of recorded RGB."""
    bounds = element.bounding_box()
    markup = element.evaluate("""async e=>{
        const clone=e.cloneNode(true), use=clone.querySelector('use');
        function copyPaint(source, target) {
            const style=getComputedStyle(source);
            for (const key of ['color','fill','stroke','stroke-width','stroke-linecap',
                'stroke-linejoin','fill-rule','fill-opacity','stroke-opacity','opacity']) {
                target.style.setProperty(key, style.getPropertyValue(key));
            }
        }
        copyPaint(e, clone);
        if(use){
            const url=new URL(use.href.baseVal,location.href);
            const xml=await (await fetch(url)).text();
            const sym=new DOMParser().parseFromString(xml,'image/svg+xml')
                .getElementById(url.hash.slice(1));
            clone.setAttribute('viewBox',sym.getAttribute('viewBox'));
            clone.innerHTML=sym.innerHTML;
            copyPaint(e.querySelector('use'), clone);
        } else {
            const copies=clone.querySelectorAll('*');
            e.querySelectorAll('*').forEach((node,i)=>copyPaint(node,copies[i]));
        }
        return clone.outerHTML;
    }""")
    x, y = bounds["x"], bounds["y"]
    w, h = bounds["width"], bounds["height"]
    renderer.set_content(
        "<style>*{margin:0} svg{display:block;color:white;fill:white;position:absolute;"
        f"width:{w}px;height:{h}px;left:{x % 1}px;top:{y % 1}px}}</style>" + markup
    )
    alpha = Image.open(BytesIO(renderer.screenshot(omit_background=True))).convert("RGBA")
    ys, xs = np.where(np.asarray(alpha.getchannel("A")) >= 128)
    renderer.locator("svg").evaluate("""e=>{
        const r=e.getBoundingClientRect(), scale=96/Math.max(r.width,r.height);
        Object.assign(e.style,{left:0,top:0,width:r.width*scale+'px',height:r.height*scale+'px'});
    }""")
    ref = renderer.locator("svg").screenshot(omit_background=True)
    templates.setdefault(label, ref)
    return dict(
        label=label,
        bbox_px=[
            int(x) + int(xs.min()),
            int(y) + int(ys.min()),
            int(x) + int(xs.max()) + 1,
            int(y) + int(ys.max()) + 1,
        ],
    )


def record(browser, renderer, folder, url, player, size, case, name, templates):
    source, seconds, _, title, text, playing, muted, visible = case
    context = browser.new_context(viewport=size, record_video_dir=folder, record_video_size=size)
    page = context.new_page()
    page.goto(url)
    init = {
        "dplayer": "new DPlayer({container:app,video:{url:'clip.mp4'},volume:.8})",
        "artplayer": "new Artplayer({container:app,url:'clip.mp4',fullscreen:true,volume:.8})",
        "mejs": "new MediaElementPlayer(document.querySelector('video'),{alwaysShowControls:true,"
        "videoWidth:innerWidth,videoHeight:innerHeight,"
        "iconSprite:'mejs-controls.svg',features:['playpause','current','progress',"
        "'volume','fullscreen']})",
    }[player]
    page.evaluate("window.player=" + init + "; true")
    if player != "mejs":
        page.wait_for_function("document.querySelector('video')?.readyState >= 2")
    page.evaluate(
        """async ([playing, muted, skin]) => {
        const v=document.querySelector('video');
        if(skin==='dplayer') player.volume(muted ? 0 : .8, true, true);
        else if(skin==='artplayer') player.muted=muted;
        else v.muted=muted;
        await v.play();
        if (!playing) v.pause();
    }""",
        [playing, muted, player],
    )
    page.mouse.move(size["width"] // 2, size["height"] - 45)
    if not visible:
        page.add_style_tag(
            content=".dplayer-controller,.art-bottom,.mejs__controls{display:none!important}"
        )
    page.wait_for_timeout(1100)
    state = page.locator("video").evaluate("v=>[!v.paused, v.muted || v.volume===0]")
    assert state == [playing, muted], (name, state)
    labels = ["pause" if playing else "play", "mute" if muted else "volume", "fullscreen"]
    icons = []
    for selector, label in zip(CONTROLS[player], labels, strict=True):
        glyph = page.locator(selector + " svg:visible")
        if visible:
            assert glyph.count() == 1, (player, selector, glyph.count())
            item = label_glyph(renderer, glyph, label, templates)
            x, y, right, bottom = item["bbox_px"]
            assert 0 <= x < right <= size["width"]
            assert size["height"] * 0.85 <= y < bottom <= size["height"], (name, item)
            icons.append(item)
        else:
            assert glyph.count() == 0
    recorded = page.video
    context.close()
    path = folder / f"{name}.webm"
    recorded.save_as(path)
    recorded.delete()
    cap = cv2.VideoCapture(str(path))
    index = 0
    last = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        last = img
        index += 1
    cap.release()
    assert last is not None and last.shape[:2] == (size["height"], size["width"])
    cv2.imencode(".png", last)[1].tofile(str(folder / f"{name}.png"))
    return dict(
        case=name,
        player=player,
        source=source,
        source_seek_s=seconds,
        title=title,
        text=text,
        icons=icons,
        frame_index=index - 1,
        fps=fps,
        webm_sha256=sha256(path.read_bytes()).hexdigest(),
    )


def prepare(assets, folder, players):
    folder.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(folder)))
    Thread(target=server.serve_forever, daemon=True).start()
    truth = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                channel="chrome",
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                ],
            )
            renderer = browser.new_page(viewport=dict(width=96, height=96))
            for player in players:
                cases = CASES[player]
                shutil.copyfile(assets / f"{player}.js", folder / f"{player}.js")
                html = (
                    "<!doctype html><meta charset=utf-8><style>body{margin:0;background:black}"
                    "#app{width:100vw;height:100vh}</style><div id=app></div>"
                    f"<script src='{player}.js'></script>"
                )
                if player == "mejs":
                    for asset in ("mediaelementplayer.css", "mejs-controls.svg"):
                        shutil.copyfile(assets / asset, folder / asset)
                    html = html.replace(
                        "<div id=app></div>",
                        "<link rel=stylesheet "
                        "href=mediaelementplayer.css><video src=clip.mp4 style='width:100vw;"
                        "height:100vh'></video>",
                    )
                (folder / "player.html").write_text(html, encoding="utf8")
                bank = folder / f"{player}-icons"
                bank.mkdir(exist_ok=True)
                templates = {}
                for i, case in enumerate(cases):
                    size = prepare_clip(folder, assets, case)
                    row = record(
                        browser,
                        renderer,
                        folder,
                        f"http://127.0.0.1:{server.server_port}/player.html",
                        player,
                        size,
                        case,
                        f"{player}-{i}",
                        templates,
                    )
                    truth.append(row)
                    print(row["case"], row["frame_index"], flush=True)
                for label, ref in templates.items():
                    (bank / f"{label}.png").write_bytes(ref)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    (folder / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--players", nargs="+", choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    prepare(args.assets.resolve(), args.folder.resolve(), args.players)
