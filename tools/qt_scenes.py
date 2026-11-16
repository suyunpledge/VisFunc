"""用真实的 Qt 控件渲染几段已知真值的动效，逐帧存成 PNG，用来检验 motionspec。

需要 PySide6，本身不依赖 motionspec：

    python tools/qt_scenes.py out/qt

每个场景写到 out/qt/<名字>/frames/*.png，真值写到 out/qt/truth.json。
窗口不会显示在屏幕上（WA_DontShowOnScreen），逐帧用 QEasingCurve 算出属性值再截图，
所以帧间隔严格均匀，位置是 Qt 实际会用的整数像素。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPoint, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

FPS = 60
LEAD, TAIL = 10, 14
W, H = 900, 560

STYLE = """
QWidget#root { background: #14161c; }
QFrame#side { background: #1b1e27; }
QLabel { color: #d7dae3; font-size: 14px; }
QLabel#title { font-size: 20px; font-weight: 600; color: #f2f3f7; }
QLabel#dim { color: #7d8496; font-size: 12px; }
QFrame#card { background: #232838; border: 1px solid #343b52; border-radius: 10px; }
QFrame#toast { background: #2b6ff2; border-radius: 8px; }
QFrame#toast QLabel { color: white; }
QFrame#row { background: #1f2330; border: 1px solid #2c3246; border-radius: 6px; }
QPushButton { background: #2b6ff2; color: white; border: none; border-radius: 6px; padding: 6px 14px; font-size: 13px; }
"""


def build() -> dict:
    root = QWidget()
    root.setObjectName("root")
    root.setFixedSize(W, H)
    root.setStyleSheet(STYLE)

    side = QFrame(root)
    side.setObjectName("side")
    side.setGeometry(0, 0, 180, H)
    for i, text in enumerate(["会话", "模型配置", "本地工具", "圆桌", "设置"]):
        label = QLabel(text, side)
        label.move(24, 70 + i * 36)
    brand = QLabel("Forge", side)
    brand.setObjectName("title")
    brand.move(24, 22)

    title = QLabel("欢迎使用 Forge", root)
    title.setObjectName("title")
    title.move(210, 24)
    hint = QLabel("选择一个模型开始，或查看快速配置指引。", root)
    hint.setObjectName("dim")
    hint.move(210, 56)

    def card(text: str, sub: str, button: str | None = None) -> QFrame:
        frame = QFrame(root)
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 14, 18, 14)
        head = QLabel(text)
        head.setObjectName("title")
        layout.addWidget(head)
        body = QLabel(sub)
        body.setWordWrap(True)
        layout.addWidget(body)
        if button:
            layout.addWidget(QPushButton(button), 0, Qt.AlignLeft)
        frame.resize(300, 150)
        return frame

    main = card("快速配置", "连接一个模型提供方，填入密钥后即可开始对话。", "一键配置")
    main.move(210, 110)

    toast = QFrame(root)
    toast.setObjectName("toast")
    toast_layout = QVBoxLayout(toast)
    toast_layout.setContentsMargins(16, 10, 16, 10)
    toast_layout.addWidget(QLabel("配置已保存，模型连接正常"))
    toast.resize(260, 44)
    toast.move(600, 480)

    rows = []
    for i, text in enumerate(["阅读文件  local.read", "搜索代码  local.grep", "运行命令  local.exec"]):
        row = QFrame(root)
        row.setObjectName("row")
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(14, 8, 14, 8)
        row_layout.addWidget(QLabel(text))
        row.resize(300, 40)
        row.move(560, 110 + i * 50)
        rows.append(row)

    for widget in (toast, *rows):
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(1.0)
        widget.setGraphicsEffect(effect)

    root.setAttribute(Qt.WA_DontShowOnScreen, True)
    root.show()
    return {"root": root, "card": main, "toast": toast, "rows": rows}


def ease(curve: QEasingCurve.Type, t_ms: float, start_ms: float, duration_ms: float) -> float:
    p = min(1.0, max(0.0, (t_ms - start_ms) / duration_ms))
    return QEasingCurve(curve).valueForProgress(p)


def render(ui: dict, out: Path, name: str, total_ms: float, apply) -> None:
    folder = out / name / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    frames = LEAD + int(round(total_ms * FPS / 1000)) + TAIL
    for i in range(frames):
        apply((i - LEAD) * 1000.0 / FPS)
        QApplication.processEvents()
        ui["root"].grab().save(str(folder / f"{i:04d}.png"))


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/qt")
    app = QApplication(sys.argv[:1])
    ui = build()
    card, toast, rows = ui["card"], ui["toast"], ui["rows"]
    card_home, toast_home = card.pos(), toast.pos()
    row_home = [r.pos() for r in rows]
    start_ms = LEAD * 1000.0 / FPS
    truth: dict = {"fps": FPS, "size": [W, H], "scenes": {}}

    def reset() -> None:
        card.move(card_home)
        toast.move(toast_home)
        toast.graphicsEffect().setOpacity(1.0)
        toast.show()
        for r, home in zip(rows, row_home):
            r.move(home)
            r.graphicsEffect().setOpacity(1.0)

    # 1 已有卡片平移：纯位移，Qt 的 OutCubic
    reset()
    render(ui, out, "card-slide", 300, lambda t: card.move(card_home + QPoint(round(200 * ease(QEasingCurve.OutCubic, t, 0, 300)), 0)))
    truth["scenes"]["card-slide"] = {"segments": [{"kind": "transform", "start_ms": start_ms, "duration_ms": 300, "delta": [200, 0], "easing": "out-cubic"}]}

    # 2 提示条进入：上移 24px + 淡入，同一条 OutCubic
    def toast_enter(t: float) -> None:
        v = ease(QEasingCurve.OutCubic, t, 0, 250)
        toast.move(toast_home + QPoint(0, round(24 * (1 - v))))
        toast.graphicsEffect().setOpacity(v)

    reset()
    render(ui, out, "toast-enter", 250, toast_enter)
    truth["scenes"]["toast-enter"] = {"segments": [{"kind": "transform", "start_ms": start_ms, "duration_ms": 250, "delta": [0, -24], "easing": "out-cubic", "opacity": [0, 1], "opacity_easing": "out-cubic"}]}

    # 3 提示条退出：上移 16px + 淡出，InCubic
    def toast_exit(t: float) -> None:
        v = ease(QEasingCurve.InCubic, t, 0, 200)
        toast.move(toast_home + QPoint(0, -round(16 * v)))
        toast.graphicsEffect().setOpacity(1 - v)

    reset()
    render(ui, out, "toast-exit", 200, toast_exit)
    truth["scenes"]["toast-exit"] = {"segments": [{"kind": "transform", "start_ms": start_ms, "duration_ms": 200, "delta": [0, -16], "easing": "in-cubic", "opacity": [1, 0], "opacity_easing": "in-cubic"}]}

    # 4 纯淡入：位置不动，InOutQuad
    reset()
    render(ui, out, "toast-fade", 300, lambda t: toast.graphicsEffect().setOpacity(ease(QEasingCurve.InOutQuad, t, 0, 300)))
    truth["scenes"]["toast-fade"] = {"segments": [{"kind": "fade", "start_ms": start_ms, "duration_ms": 300, "easing": "in-out-quad"}]}

    # 5 列表项依次进入：各上移 16px + 淡入，200ms OutCubic，间隔 50ms
    def stagger(t: float) -> None:
        for i, (r, home) in enumerate(zip(rows, row_home)):
            v = ease(QEasingCurve.OutCubic, t, 50 * i, 200)
            r.move(home + QPoint(0, round(16 * (1 - v))))
            r.graphicsEffect().setOpacity(v)

    reset()
    render(ui, out, "list-stagger", 300, stagger)
    truth["scenes"]["list-stagger"] = {"segments": [
        {"kind": "transform", "start_ms": start_ms + 50 * i, "duration_ms": 200, "delta": [0, -16], "easing": "out-cubic", "opacity": [0, 1]}
        for i in range(3)
    ]}

    # 6 卡片平移 + 提示条淡入同时发生：两个互不相干的元素
    def both(t: float) -> None:
        card.move(card_home + QPoint(0, round(60 * ease(QEasingCurve.InOutCubic, t, 0, 400))))
        toast.graphicsEffect().setOpacity(ease(QEasingCurve.Linear, t, 100, 200))

    reset()
    render(ui, out, "two-at-once", 400, both)
    truth["scenes"]["two-at-once"] = {"segments": [
        {"kind": "transform", "start_ms": start_ms, "duration_ms": 400, "delta": [0, 60], "easing": "in-out-cubic"},
        {"kind": "fade", "start_ms": start_ms + 100, "duration_ms": 200, "easing": "linear"},
    ]}

    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out, "scenes:", ", ".join(truth["scenes"]))
    del app


if __name__ == "__main__":
    main()
