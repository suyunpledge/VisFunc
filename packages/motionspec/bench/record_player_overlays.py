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
    "xg": [
        ("sintel", 14, 680, "NIGHT WATCH", "", True, True, True),
        ("oceans", 7, 736, "", "OPEN GATE", False, True, True),
        ("sintel", 30, 800, "", "AIRLOCK", True, False, True),
        ("oceans", 20, 640, "", "STAND CLEAR", True, True, True),
        ("sintel", 46, 768, "EAST WIND", "Hold here.", False, False, True),
        ("oceans", 27, 704, "", "", True, False, True),
        ("sintel", 9, 656, "", "", True, False, False),
    ],
    "clappr": [
        ("oceans", 1, 768, "SILVER TIDE", "", False, True, True),
        ("sintel", 17, 640, "", "KEEP LEFT", True, False, True),
        ("oceans", 14, 704, "", "SKYLINE", False, True, True),
        ("sintel", 32, 816, "", "LAST CALL", True, True, True),
        ("oceans", 22, 720, "WEST BAY", "Wait at exit B.", False, False, True),
        ("sintel", 47, 680, "", "", True, False, True),
        ("oceans", 32, 752, "", "", True, False, False),
    ],
    "chrome-ui": [
        ("bunny", 7, 704, "OVER THE HILL", "", True, True, True),
        ("oceans", 9, 768, "", "GO WEST", False, False, True),
        ("bunny", 16, 672, "", "LIFETIME", False, True, True),
        ("oceans", 19, 800, "", "FALL BACK", True, True, True),
        ("bunny", 23, 720, "RIVER ROAD", "Walk with me.", True, False, True),
        ("oceans", 35, 736, "", "", False, False, True),
        ("bunny", 11, 656, "", "", True, False, False),
    ],
    "op": [
        ("sintel", 12, 720, "CITY LIMITS", "", True, True, True),
        ("bunny", 18, 688, "", "WAIT HERE", False, True, True),
        ("oceans", 13, 784, "", "FOOTPATH", True, False, True),
        ("bunny", 25, 752, "", "LOOK UP", True, True, True),
        ("oceans", 29, 800, "TIDAL POOL", "Keep going.", False, False, True),
        ("sintel", 38, 736, "", "", True, False, True),
        ("bunny", 20, 696, "", "", True, False, False),
    ],
    "fluid": [
        ("oceans", 4, 688, "OPEN SEAS", "", True, True, True),
        ("sintel", 27, 768, "", "COME HOME", False, True, True),
        ("bunny", 12, 736, "", "RAILWAY", True, False, True),
        ("oceans", 16, 704, "", "STEP OUT", True, True, True),
        ("sintel", 39, 800, "HIGH PASS", "We have time.", False, False, True),
        ("bunny", 10, 720, "", "", True, False, True),
        ("oceans", 31, 656, "", "", True, False, False),
    ],
}
CONTROLS = {
    "dplayer": [".dplayer-play-icon", ".dplayer-volume-icon", ".dplayer-full-icon"],
    "artplayer": [".art-control-playAndPause", ".art-control-volume", ".art-control-fullscreen"],
    "mejs": [".mejs__playpause-button", ".mejs__volume-button", ".mejs__fullscreen-button"],
    "xg": [".xgplayer-play", ".xgplayer-volume", ".xgplayer-fullscreen"],
    "clappr": ["button[data-playpause]", ".drawer-icon[data-volume]", "button[data-fullscreen]"],
    "chrome-ui": ["media-play-button", "media-mute-button", "media-fullscreen-button"],
    "op": [".op-controls__playpause", ".op-controls__mute", ".op-controls__fullscreen"],
    "fluid": [".fluid_control_playpause", ".fluid_control_mute", ".fluid_control_fullscreen"],
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


def label_css_glyph(element, selector, label, templates):
    """Difference the actual CSS pseudo-element against the same button with its ink hidden."""
    box = element.bounding_box()
    x, y = box["x"], box["y"]
    backdrop = selector + "{background:black!important}"
    raw = element.screenshot(style=backdrop)
    blank = element.screenshot(style=backdrop + selector + "::before{visibility:hidden!important}")
    lit = np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.int16)
    dark = np.asarray(Image.open(BytesIO(blank)).convert("RGB"), dtype=np.int16)
    contrast = np.max(np.abs(lit - dark), axis=2)
    if contrast.max() < 32:
        raise ValueError(f"no measurable glyph in {selector}")
    alpha = np.uint8(contrast / contrast.max() * 255)
    ys, xs = np.where(alpha >= 128)
    ref = Image.new("RGBA", (alpha.shape[1], alpha.shape[0]), "white")
    ref.putalpha(Image.fromarray(alpha))
    encoded = BytesIO()
    ref.save(encoded, format="PNG")
    templates.setdefault(label, encoded.getvalue())
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
    if player == "fluid":
        sprite = "583850a426583410ea8e.svg"
        context.route(
            "https://cdn.fluidplayer.com/3.45.0/" + sprite,
            lambda route: route.fulfill(path=folder / sprite, content_type="image/svg+xml"),
        )
    page = context.new_page()
    page.goto(url)
    init = {
        "dplayer": "new DPlayer({container:app,video:{url:'clip.mp4'},volume:.8})",
        "artplayer": "new Artplayer({container:app,url:'clip.mp4',fullscreen:true,volume:.8})",
        "mejs": "new MediaElementPlayer(document.querySelector('video'),{alwaysShowControls:true,"
        "videoWidth:innerWidth,videoHeight:innerHeight,"
        "iconSprite:'mejs-controls.svg',features:['playpause','current','progress',"
        "'volume','fullscreen']})",
        "xg": "new Player({id:'app',url:'clip.mp4',width:innerWidth,height:innerHeight,"
        "videoInit:true,lang:'en',ignores:['cssFullscreen']})",
        "clappr": "new Clappr.Player({parentId:'#app',source:'clip.mp4',width:'100%',height:'100%',"
        "autoPlay:false,hideMediaControl:false})",
        "chrome-ui": "document.querySelector('media-controller')",
        "op": "new OpenPlayerJS('app',{alwaysVisible:true})",
        "fluid": "fluidPlayer('app',{layoutControls:{fillToContainer:true,"
        "controlBar:{autoHide:false}}})",
    }[player]
    page.evaluate("window.player=" + init + "; true")
    if player == "op":
        page.evaluate("player.init()")
    if player not in ("mejs", "clappr"):
        page.wait_for_function("document.querySelector('video')?.readyState >= 2")
    page.evaluate(
        """async ([playing, muted, skin]) => {
        const v=document.querySelector('video');
        if(skin==='dplayer') player.volume(muted ? 0 : .8, true, true);
        else if(skin==='artplayer') player.muted=muted;
        else if(skin==='clappr') {player.play();player.setVolume(muted?0:80);v.muted=muted;}
        else if(skin==='op') {
            if ((v.muted || v.volume===0)!==muted)
                document.querySelector('.op-controls__mute').click();
        }
        else v.muted=muted;
        await v.play();
        if (!playing) {if(skin==='clappr') player.pause();v.pause();}
    }""",
        [playing, muted, player],
    )
    page.mouse.move(size["width"] // 2, size["height"] - 45)
    if not visible:
        page.add_style_tag(
            content=".dplayer-controller,.art-bottom,.mejs__controls,.xgplayer-controls,"
            ".media-control,media-control-bar,.op-controls,.fluid_controls_container{display:none!important}"
        )
    page.wait_for_timeout(1100)
    state = page.locator("video").evaluate("v=>[!v.paused, v.muted || v.volume===0]")
    assert state == [playing, muted], (name, state)
    labels = ["pause" if playing else "play", "mute" if muted else "volume", "fullscreen"]
    icons = []
    for selector, label in zip(CONTROLS[player], labels, strict=True):
        css_icon = player in ("op", "fluid")
        glyph = page.locator(selector + (":visible" if css_icon else " svg:visible"))
        if visible:
            assert glyph.count() == 1, (player, selector, glyph.count())
            if css_icon:
                item = label_css_glyph(glyph, selector, label, templates)
            else:
                item = label_glyph(renderer, glyph, label, templates)
            x, y, right, bottom = item["bbox_px"]
            assert 0 <= x < right <= size["width"], (name, item, glyph.bounding_box())
            assert size["height"] * 0.85 <= y < bottom <= size["height"], (name, item)
            icons.append(item)
        else:
            assert glyph.count() == 0
    if player in ("op", "fluid") and visible:
        # All five scored frames follow the temporary CSS used for ground truth.
        page.mouse.move(size["width"] // 2 + 1, size["height"] - 45)
        page.wait_for_timeout(1150)
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
                elif player in ("xg", "op"):
                    css = player + ".css"
                    shutil.copyfile(assets / css, folder / css)
                    html = html.replace("<style>", f"<link rel=stylesheet href={css}><style>")
                    if player == "op":
                        html = html.replace(
                            "<div id=app></div>",
                            "<video id=app class=op-player__media "
                            "src=clip.mp4 playsinline></video>",
                        )
                        html = html.replace("#app{", ".op-player,#app{")
                elif player == "fluid":
                    sprite = "583850a426583410ea8e.svg"
                    shutil.copyfile(assets / sprite, folder / sprite)
                    html = html.replace(
                        "<div id=app></div>",
                        "<video id=app playsinline><source src=clip.mp4 type=video/mp4></video>",
                    )
                    html = html.replace("body{margin:0;", "html,body{height:100%;margin:0;")
                elif player == "chrome-ui":
                    html = html.replace(
                        "<div id=app></div>",
                        "<media-controller autohide=-1 style='width:100vw;height:100vh'>"
                        "<video src=clip.mp4 slot=media style='width:100%;height:100%'></video>"
                        "<media-control-bar><media-play-button></media-play-button>"
                        "<media-mute-button></media-mute-button><media-time-display></media-time-display>"
                        "<media-time-range></media-time-range><media-fullscreen-button></media-fullscreen-button>"
                        "</media-control-bar></media-controller>",
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
    parser.add_argument(
        "--players", nargs="+", choices=list(CASES), default=["dplayer", "artplayer", "mejs"]
    )
    args = parser.parse_args()
    prepare(args.assets.resolve(), args.folder.resolve(), args.players)
