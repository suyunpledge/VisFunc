"""第二批 Qt 场景：比 qt_scenes.py 更接近真实界面的动效，用来找 motionspec 的短板。

    python tools/qt_hard_scenes.py out/qt-hard      （需要 PySide6）
    python tools/qt_hard_compare.py out/qt-hard

覆盖第一批没有的情况：带遮罩的模态框、展开折叠、侧栏收起、裁剪区域内滚动、细长指示条、
两个相邻元素对调、回弹、纯颜色变化、进度条。真值是逻辑像素；rect 是元素在整个动效里扫过的范围。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPoint
from PySide6.QtWidgets import QApplication, QFrame, QGraphicsOpacityEffect, QLabel, QPushButton, QWidget

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qt_scenes import FPS, LEAD, STYLE, H, W, ease, render  # noqa: E402

START = LEAD * 1000.0 / FPS
EXTRA = """
QFrame#backdrop { background: black; }
QFrame#dialog { background: #262c3d; border: 1px solid #3b4460; border-radius: 12px; }
QFrame#panel { background: #1c2030; border: 1px solid #2c3246; }
QFrame#header { background: #232838; border: 1px solid #343b52; border-radius: 6px; }
QFrame#viewport { background: #171a22; border: 1px solid #2c3246; }
QFrame#indicator { background: #2b6ff2; }
QFrame#track { background: #232838; border-radius: 4px; }
QFrame#fill { background: #2b6ff2; border-radius: 4px; }
"""


def root_widget() -> QWidget:
    root = QWidget()
    root.setObjectName("root")
    root.setFixedSize(W, H)
    root.setStyleSheet(STYLE + EXTRA)
    side = QFrame(root)
    side.setObjectName("side")
    side.setGeometry(0, 0, 180, H)
    for i, text in enumerate(["会话", "模型配置", "本地工具", "圆桌", "设置"]):
        QLabel(text, side).move(24, 70 + i * 36)
    title = QLabel("欢迎使用 Forge", root)
    title.setObjectName("title")
    title.move(210, 24)
    return root


def frame(parent, name: str, x: int, y: int, w: int, h: int, text: str | None = None) -> QFrame:
    box = QFrame(parent)
    box.setObjectName(name)
    box.setGeometry(x, y, w, h)
    if text:
        QLabel(text, box).move(14, max(4, h // 2 - 10))
    return box


def faded(widget: QWidget, opacity: float) -> QGraphicsOpacityEffect:
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(opacity)
    widget.setGraphicsEffect(effect)
    return effect


def scene(truth: dict, out: Path, name: str, total_ms: float, root: QWidget, apply, segments: list[dict]) -> None:
    from PySide6.QtCore import Qt

    root.setAttribute(Qt.WA_DontShowOnScreen, True)
    root.show()
    apply(-1000.0)
    render({"root": root}, out, name, total_ms, apply)
    truth["scenes"][name] = {"segments": segments}
    root.close()


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/qt-hard")
    app = QApplication(sys.argv[:1])
    truth: dict = {"fps": FPS, "size": [W, H], "scenes": {}}
    E = QEasingCurve

    # 1 模态框：全窗口遮罩淡入到 50%，对话框上移 16px 并淡入
    root = root_widget()
    frame(root, "card", 210, 110, 300, 150, "快速配置")
    backdrop = frame(root, "backdrop", 0, 0, W, H)
    dim = faded(backdrop, 0.0)
    dialog = frame(root, "dialog", 270, 170, 360, 200, "确认删除这个会话？")
    QPushButton("删除", dialog).move(270, 150)
    shown = faded(dialog, 0.0)

    def modal(t):
        dim.setOpacity(0.5 * ease(E.Linear, t, 0, 200))
        v = ease(E.OutCubic, t, 0, 250)
        dialog.move(270, 170 + round(16 * (1 - v)))
        shown.setOpacity(v)

    scene(truth, out, "modal-open", 250, root, modal, [
        {"element": "backdrop", "kind": "fade", "start_ms": START, "duration_ms": 200, "easing": "linear", "rect": [0, 0, W, H]},
        {"element": "dialog", "kind": "transform", "start_ms": START, "duration_ms": 250, "easing": "out-cubic",
         "delta": [0, -16], "opacity": [0, 1], "rect": [270, 170, 630, 386]},
    ])

    # 2 展开：内容面板高度 0→180，下面两行随之下移 180
    root = root_widget()
    frame(root, "header", 210, 100, 400, 40, "高级设置 ▾")
    panel = frame(root, "panel", 210, 140, 400, 0)
    for i, text in enumerate(["温度  0.7", "最大输出  4096", "流式输出  开"]):
        QLabel(text, panel).move(16, 14 + i * 52)
    below = [frame(root, "row", 210, 150 + i * 50, 400, 40, text) for i, text in enumerate(["工具权限", "网络代理"])]

    def expand(t):
        h = round(180 * ease(E.OutCubic, t, 0, 300))
        panel.resize(400, h)
        panel.setVisible(h > 0)
        for i, row in enumerate(below):
            row.move(210, 150 + h + i * 50)

    scene(truth, out, "accordion", 300, root, expand, [
        {"element": "panel", "kind": "size", "start_ms": START, "duration_ms": 300, "easing": "out-cubic",
         "size": {"height": [0, 180]}, "rect": [210, 140, 610, 320]},
        {"element": "rows below", "kind": "transform", "start_ms": START, "duration_ms": 300, "easing": "out-cubic",
         "delta": [0, 180], "rect": [210, 150, 610, 420]},
    ])

    # 3 侧栏收起：宽 220→64，右侧内容整体左移 156
    root = QWidget()
    root.setObjectName("root")
    root.setFixedSize(W, H)
    root.setStyleSheet(STYLE + EXTRA)
    side = frame(root, "side", 0, 0, 220, H)
    for i, text in enumerate(["会话", "模型配置", "本地工具", "圆桌"]):
        QLabel(text, side).move(24, 70 + i * 36)
    content = QFrame(root)
    content.setGeometry(250, 0, 600, H)
    frame(content, "card", 0, 100, 300, 150, "快速配置")
    frame(content, "row", 330, 100, 260, 40, "阅读文件")

    def collapse(t):
        w = 220 - round(156 * ease(E.InOutCubic, t, 0, 300))
        side.resize(w, H)
        content.move(w + 30, 0)

    scene(truth, out, "sidebar-collapse", 300, root, collapse, [
        {"element": "sidebar", "kind": "size", "start_ms": START, "duration_ms": 300, "easing": "in-out-cubic",
         "size": {"width": [220, 64]}, "rect": [0, 0, 220, H]},
        {"element": "content", "kind": "transform", "start_ms": START, "duration_ms": 300, "easing": "in-out-cubic",
         "delta": [-156, 0], "rect": [94, 100, 840, 250]},
    ])

    # 4 滚动：裁剪视口里列表上移 120
    root = root_widget()
    viewport = frame(root, "viewport", 560, 110, 300, 240)
    listing = QFrame(viewport)
    listing.setGeometry(0, 0, 300, 640)
    for i in range(14):
        frame(listing, "row", 10, 10 + i * 45, 280, 38, f"第 {i + 1} 条记录  local.read")

    def scroll(t):
        listing.move(0, -round(120 * ease(E.OutQuad, t, 0, 250)))

    scene(truth, out, "scroll", 250, root, scroll, [
        {"element": "list", "kind": "transform", "start_ms": START, "duration_ms": 250, "easing": "out-quad",
         "delta": [0, -120], "rect": [560, 110, 860, 350]},
    ])

    # 5 标签指示条：3px 高，右移 120 并变宽 80→110
    root = root_widget()
    for i, text in enumerate(["概览", "模型", "工具", "日志"]):
        QLabel(text, root).move(230 + i * 120, 80)
    bar = frame(root, "indicator", 210, 108, 80, 3)

    def tab(t):
        v = ease(E.OutCubic, t, 0, 250)
        bar.setGeometry(210 + round(120 * v), 108, 80 + round(30 * v), 3)

    scene(truth, out, "tab-indicator", 250, root, tab, [
        {"element": "indicator", "kind": "transform", "start_ms": START, "duration_ms": 250, "easing": "out-cubic",
         "delta": [135, 0], "size": {"width": [80, 110]}, "rect": [210, 108, 440, 111]},
    ])

    # 6 两行对调：A 下移 50，B 上移 50
    root = root_widget()
    a = frame(root, "row", 560, 110, 300, 40, "阅读文件  local.read")
    b = frame(root, "row", 560, 160, 300, 40, "运行命令  local.exec")

    def swap(t):
        v = ease(E.InOutCubic, t, 0, 300)
        a.move(560, 110 + round(50 * v))
        b.move(560, 160 - round(50 * v))

    scene(truth, out, "reorder-swap", 300, root, swap, [
        {"element": "row A", "kind": "transform", "start_ms": START, "duration_ms": 300, "easing": "in-out-cubic",
         "delta": [0, 50], "rect": [560, 110, 860, 200]},
        {"element": "row B", "kind": "transform", "start_ms": START, "duration_ms": 300, "easing": "in-out-cubic",
         "delta": [0, -50], "rect": [560, 110, 860, 200]},
    ])

    # 7 回弹：卡片上移 60，OutBack
    root = root_widget()
    card = frame(root, "card", 300, 220, 300, 150, "快速配置")

    def bounce(t):
        card.move(300, 220 - round(60 * ease(E.OutBack, t, 0, 400)))

    scene(truth, out, "overshoot", 400, root, bounce, [
        {"element": "card", "kind": "transform", "start_ms": START, "duration_ms": 400, "easing": "out-back",
         "delta": [0, -60], "rect": [300, 150, 600, 370]},
    ])

    # 8 按钮变色：蓝→绿，不动
    root = root_widget()
    button = QPushButton("一键配置", root)
    button.setGeometry(300, 200, 160, 44)

    def recolor(t):
        v = ease(E.Linear, t, 0, 200)
        r, g, b_ = (round(x + (y - x) * v) for x, y in zip((0x2B, 0x6F, 0xF2), (0x1E, 0x9E, 0x5A)))
        button.setStyleSheet(f"background: rgb({r},{g},{b_}); color: white; border-radius: 6px;")

    scene(truth, out, "button-color", 200, root, recolor, [
        {"element": "button", "kind": "fade", "start_ms": START, "duration_ms": 200, "easing": "linear", "rect": [300, 200, 460, 244]},
    ])

    # 9 进度条：宽 0→400，线性 600ms
    root = root_widget()
    frame(root, "track", 250, 300, 400, 8)
    fill = frame(root, "fill", 250, 300, 0, 8)

    def progress(t):
        w = round(400 * ease(E.Linear, t, 0, 600))
        fill.resize(w, 8)
        fill.setVisible(w > 0)

    scene(truth, out, "progress", 600, root, progress, [
        {"element": "fill", "kind": "size", "start_ms": START, "duration_ms": 600, "easing": "linear",
         "size": {"width": [0, 400]}, "rect": [250, 300, 650, 308]},
    ])

    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out, "scenes:", ", ".join(truth["scenes"]))
    del app


if __name__ == "__main__":
    main()
