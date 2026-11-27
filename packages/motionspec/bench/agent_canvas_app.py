"""自绘界面的测试窗口：按钮都是画出来的，UI Automation 看不到名字，只能靠屏幕文字和坐标点击。

    python bench/agent_canvas_app.py STATE.json [最长秒数]      （需要 PySide6）

窗口标题 "Keystone canvas task"，置顶、不抢焦点。优先级（Low / Medium / High）、
"Notify team" 开关和 Confirm 按钮都由 paintEvent 画出，点击靠 mousePressEvent 命中测试。
每次状态变化重写 STATE.json；存在 STATE.json.stop 时退出。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QWidget

TITLE = "Keystone canvas task"


class Canvas(QWidget):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.setWindowTitle(os.environ.get("VISFUNC_CANVAS_TITLE", TITLE))
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.resize(640, 360)
        self.move(80, 80)
        self.state = {
            "priority": None,
            "notify": False,
            "confirmed": False,
            "clicks": [],
            "misses": 0,
            "confirmed_with": None,
        }
        self.items = {
            "Low": QRectF(40, 110, 120, 44),
            "Medium": QRectF(180, 110, 120, 44),
            "High": QRectF(320, 110, 120, 44),
            "Notify team": QRectF(40, 190, 200, 40),
            "Confirm": QRectF(460, 280, 140, 48),
        }
        self.write()

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def status(self) -> str:
        if self.state["confirmed"]:
            notify = "on" if self.state["notify"] else "off"
            return f"Confirmed: priority {self.state['priority']}, notify {notify}"
        return f"Priority: {self.state['priority'] or 'none'}"

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#f4f5f7"))
        font = QFont(self.font())
        font.setPointSizeF(15)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor("#1f2328"))
        p.drawText(QRectF(40, 30, 560, 40), Qt.AlignVCenter, "Release checklist")
        font.setPointSizeF(11)
        font.setBold(False)
        p.setFont(font)
        for name, rect in self.items.items():
            selected = name == self.state["priority"] or (
                name == "Notify team" and self.state["notify"]
            )
            if name == "Confirm":
                p.setBrush(QColor("#1f883d"))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(rect, 8, 8)
                p.setPen(QColor("white"))
                p.drawText(rect, Qt.AlignCenter, name)
                continue
            if name == "Notify team":
                box = QRectF(rect.left(), rect.top() + 10, 20, 20)
                p.setPen(QColor("#57606a"))
                p.setBrush(QColor("#0969da") if selected else QColor("white"))
                p.drawRect(box)
                p.setPen(QColor("#1f2328"))
                p.drawText(
                    QRectF(rect.left() + 30, rect.top(), rect.width() - 30, rect.height()),
                    Qt.AlignVCenter,
                    name,
                )
                continue
            p.setBrush(QColor("#0969da") if selected else QColor("white"))
            p.setPen(QColor("#57606a"))
            p.drawRoundedRect(rect, 22, 22)
            p.setPen(QColor("white") if selected else QColor("#1f2328"))
            label = "High" if name == "Low" and self.state.get("duplicate_high") else name
            p.drawText(rect, Qt.AlignCenter, label)
        p.setPen(QColor("#1f2328"))
        p.drawText(QRectF(40, 280, 400, 48), Qt.AlignVCenter, self.status())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        point = event.position()
        hit = next((name for name, rect in self.items.items() if rect.contains(point)), None)
        self.state["clicks"].append(hit or f"miss@{int(point.x())},{int(point.y())}")
        if hit is None:
            self.state["misses"] += 1
        elif hit == "Notify team":
            self.state["notify"] = not self.state["notify"]
        elif hit == "Confirm":
            if self.state["priority"]:
                self.state["confirmed"] = True
                self.state["confirmed_with"] = {
                    "priority": self.state["priority"],
                    "notify": self.state["notify"],
                }
        else:
            self.state["priority"] = hit
            self.state["confirmed"] = False
        self.write()
        self.update()


def main() -> None:
    path = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    canvas = Canvas(path)
    canvas.show()
    stop = path.with_name(path.name + ".stop")
    poll = QTimer()
    mode = sys.argv[3] if len(sys.argv) > 3 else ""

    def tick():
        if stop.exists():
            app.quit()
        elif mode and not canvas.state.get("reflowed") and path.with_suffix(".reflow").exists():
            if mode == "duplicate":
                canvas.state["duplicate_high"] = True
            elif mode == "reflow":
                canvas.items["High"], canvas.items["Medium"] = (
                    canvas.items["Medium"],
                    canvas.items["High"],
                )
            canvas.repaint()
            canvas.state["reflowed"] = True
            canvas.write()

    poll.timeout.connect(tick)
    poll.start(40 if mode else 300)
    QTimer.singleShot(int(seconds * 1000), app.quit)
    app.exec()


if __name__ == "__main__":
    main()
