"""Render bilingual player captures and independent glyph references from local Video.js assets."""

import argparse
import base64
import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

# sequence, width, height, subtitle pixels, cue, displayed playback control, UI scale
CASES = [
    ("01", 640, 360, 15, "下一班列车即将到站。", "play", 1),
    ("02", 800, 450, 17, "Keep the platform clear.", "pause", 1.15),
    ("03", 540, 304, 12, "请在黄线外等候", "pause", 0.85),
    ("05", 960, 540, 19, "下一站是人民广场。\nNext stop: People’s Square.", "play", 1.4),
    ("02", 720, 405, 14, "Please use exit B2.", "play", 1.25),
    ("01", 600, 338, 13, "请保管好随身物品。", "pause", 0.95),
    ("05", 880, 495, 16, "Doors close in 10 seconds.", "pause", 1.5),
    ("03", 480, 270, 12, "换乘 2 号线", "play", 0.78),
    ("01", 768, 432, 18, "", "pause", 1.3),
    ("05", 580, 326, 15, "", "play", 0.9),
    ("03", 512, 288, 10, "车门即将关闭。", "play", 1.1),
    ("02", 720, 405, 11, "Mind the gap at platform 3.", "pause", 1.2),
    ("05", 640, 360, 10, "请前往 3 号站台。", "play", 0.82),
    ("01", 960, 540, 13, "The last train leaves at 21:45.", "pause", 1.35),
    ("02", 640, 360, 16, "", "play", 1),
    ("03", 640, 360, 16, "", "play", 1),
    ("05", 734, 414, 17, "出口在左侧。", "play", 1.16),
    ("02", 846, 476, 15, "Keep both entrances open.", "pause", 1.36),
    ("01", 620, 350, 14, "下一站：中央广场。\nNext stop: Central Plaza.", "play", 1.06),
    ("03", 700, 394, 13, "", "pause", 1.22),
]

HTML = """<!doctype html><meta charset=utf-8><link rel=stylesheet href="video-js.min.css">
<style>*{box-sizing:border-box}body{margin:0}.video-js .vjs-control-bar{
display:flex!important;opacity:1!important;visibility:visible!important}
.vjs-big-play-button{display:none!important}.video-js .vjs-text-track-display{
bottom:3em!important}</style><video id=v class="video-js" playsinline muted></video>
<script src="video.min.js"></script>"""

START = """async ([seq, width, height, fontSize, caption, action, scale]) => {
    const p = window.player = videojs('v', {controls:true,width,height,
        html5:{nativeTextTracks:false},
        controlBar:{pictureInPictureToggle:false,remainingTimeDisplay:false}});
    p.el().style.fontSize = (10 * scale) + 'px';
    p.src({src:seq+'.mp4',type:'video/mp4'});
    await new Promise(r => p.one('loadeddata', r));
    p.currentTime(.8);
    await new Promise(r => p.one('seeked',r));
    await p.play();
    if (action === 'play') p.pause();
    if (caption) {
        const track = p.addTextTrack('captions','test','zh');
        track.mode = 'showing';
        track.addCue(new VTTCue(0,30,caption));
    }
    const st = document.createElement('style');
    st.textContent = '.vjs-text-track-cue {font-size:'+fontSize+
        'px!important;font-family:Arial,Microsoft YaHei,sans-serif!important}';
    document.head.append(st);
}"""

GLYPH = """async name => {
    const span = document.createElement('span');
    span.className = 'vjs-icon-' + name;
    document.body.append(span);
    const ps = getComputedStyle(span, ':before');
    const font = ps.fontFamily, glyph = ps.content.slice(1,-1);
    await document.fonts.load('64px '+font, glyph);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.font = '64px '+font;
    ctx.fillStyle = 'white';
    ctx.fillText(glyph,24,88);
    return canvas.toDataURL().split(',')[1];
}"""


def capture(browser, folder, index, case):
    seq, w, h, _, caption, action, _ = case
    page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
    page.goto((folder / "capture.html").as_uri())
    page.evaluate(START, case)
    if index % 3 == 2:
        page.evaluate("player.muted(false)")
    if index in (14, 15):
        page.add_style_tag(content=".video-js .vjs-control-bar{display:none!important}")
    if index >= 16:
        page.add_style_tag(content="""
          .video-js .vjs-control-bar {background:#f3f3f3ee!important}
          .video-js .vjs-control {color:#17212a!important}
          .vjs-text-track-cue > div {color:#ffef98!important;background:transparent!important;
              text-shadow:1px 1px 2px black,-1px -1px 2px black}
        """)
    page.wait_for_timeout(180)
    page.screenshot(path=folder / f"case{index}.png")
    # Turn the existing controls white on black for a separate foreground mask.
    # The measured screenshot above still has its original composited toolbar.
    page.add_style_tag(
        content=".video-js .vjs-control-bar{background:#000!important}"
        ".video-js .vjs-control{color:white!important}"
    )
    mask = np.asarray(Image.open(BytesIO(page.screenshot())).convert("L"))
    mute = page.locator(".vjs-mute-control")
    volume = "mute" if "vjs-vol-0" in mute.get_attribute("class").split() else "volume"
    controls = [
        (action, ".vjs-play-control"),
        (volume, ".vjs-mute-control"),
        ("fullscreen", ".vjs-fullscreen-control"),
        ("captions", "button.vjs-subs-caps-button"),
    ]
    icons = []
    for label, selector in controls:
        box = page.locator(selector).bounding_box()
        if box is None:
            continue
        x, y = round(box["x"]), round(box["y"])
        right, bottom = round(box["x"] + box["width"]), round(box["y"] + box["height"])
        ys, xs = np.where(mask[y:bottom, x:right] > 160)
        if not xs.size:
            raise ValueError(f"empty reference glyph for {label}, case {index}")
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
    page.close()
    return dict(case=index, sequence=seq, size=[w, h], text=caption, icons=icons)


def prepare(clips, dist, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("video.min.js", "video-js.min.css"):
        shutil.copyfile(dist / name, folder / name)
    (folder / "capture.html").write_text(HTML, encoding="utf-8")
    for seq in ("01", "02", "03", "05"):
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(clips / seq / "clip.mp4"),
                "-vf",
                "scale=800:-2",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "24",
                str(folder / f"{seq}.mp4"),
            ],
            check=True,
        )
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chrome",
            args=["--allow-file-access-from-files", "--autoplay-policy=no-user-gesture-required"],
        )
        rows = [capture(browser, folder, index, case) for index, case in enumerate(CASES)]
        page = browser.new_page()
        page.goto((folder / "capture.html").as_uri())
        (folder / "icons").mkdir(exist_ok=True)
        for label, name in (
            ("play", "play"),
            ("pause", "pause"),
            ("mute", "volume-mute"),
            ("volume", "volume-high"),
            ("fullscreen", "fullscreen-enter"),
            ("captions", "subtitles"),
        ):
            png = page.evaluate(GLYPH, name)
            (folder / "icons" / f"{label}.png").write_bytes(base64.b64decode(png))
        browser.close()
    (folder / "truth.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path, help="MOT20 3-second sample root with 01,02,03,05")
    parser.add_argument("dist", type=Path, help="local video.js 8.23.4 dist directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.clips, args.dist, args.output.resolve())
