"""在桌面上短暂显示一个真实的 Qt 窗口，用来测 UI Automation 和实时截屏读字。

    python bench/agent_qt_window.py truth.json [秒数]      （需要 PySide6）

窗口不抢焦点，显示指定秒数（默认 12）后自行关闭。工具栏是只有图标、没有文字的按钮，靠
tooltip 和无障碍名称标注；另有勾选的复选框、填了内容的输入框和禁用的按钮。真值写成 JSON，
坐标是屏幕物理像素 [l, t, r, b]。窗口出现后，"状态"标签每 1.5 秒改一次文字，用来测等稳定。
"""

from __future__ import annotations

import json
import sys

from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

TITLE = "VisFunc agent test"
ICONS = [
    ("Back", QStyle.SP_ArrowBack),
    ("Forward", QStyle.SP_ArrowForward),
    ("Refresh", QStyle.SP_BrowserReload),
    ("Save", QStyle.SP_DialogSaveButton),
    ("Delete", QStyle.SP_TrashIcon),
    ("Help", QStyle.SP_DialogHelpButton),
]


def main() -> None:
    out, seconds = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    root = QWidget()
    root.setWindowTitle(TITLE)
    root.setAttribute(Qt.WA_ShowWithoutActivating)
    root.setWindowFlag(Qt.WindowDoesNotAcceptFocus)
    root.setWindowFlag(Qt.WindowStaysOnTopHint)  # stay visible over the user's windows
    root.resize(760, 420)
    root.move(40, 40)
    v = QVBoxLayout(root)
    bar = QHBoxLayout()
    widgets = []
    for name, icon in ICONS:
        b = QToolButton(root)
        b.setIcon(root.style().standardIcon(icon))
        b.setIconSize(QSize(24, 24))
        b.setToolTip(name)
        b.setAccessibleName(name)
        bar.addWidget(b)
        widgets.append(("button", name, b))
    bar.addStretch(1)
    v.addLayout(bar)
    title = QLabel("Sync settings", root)
    title.setStyleSheet("font-size: 20px; font-weight: 600;")
    v.addWidget(title)
    field = QLineEdit("backup-server-07", root)
    field.setAccessibleName("Server name")
    v.addWidget(field)
    widgets.append(("edit", "Server name", field))
    check = QCheckBox("Sync on startup", root)
    check.setChecked(True)
    v.addWidget(check)
    widgets.append(("checkbox", "Sync on startup", check))
    status = QLabel("Status: idle", root)
    v.addWidget(status)
    v.addStretch(1)
    row = QHBoxLayout()
    row.addStretch(1)
    disabled = QPushButton("Apply", root)
    disabled.setEnabled(False)
    row.addWidget(disabled)
    widgets.append(("button", "Apply", disabled))
    ok = QPushButton("Start sync", root)
    row.addWidget(ok)
    widgets.append(("button", "Start sync", ok))
    v.addLayout(row)
    root.show()

    def dump() -> None:
        ratio = root.devicePixelRatioF()
        truth = []
        for kind, name, w in widgets:
            tl = w.mapToGlobal(QPoint(0, 0))
            box = [
                round(tl.x() * ratio),
                round(tl.y() * ratio),
                round((tl.x() + w.width()) * ratio),
                round((tl.y() + w.height()) * ratio),
            ]
            truth.append(
                {
                    "control_type": kind,
                    "name": name,
                    "bbox_px": box,
                    "enabled": w.isEnabled(),
                    "icon_only": kind == "button" and not w.text(),
                }
            )
        frame = root.frameGeometry()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "title": TITLE,
                    "ratio": ratio,
                    "elements": truth,
                    "window_px": [
                        round(frame.left() * ratio),
                        round(frame.top() * ratio),
                        round((frame.right() + 1) * ratio),
                        round((frame.bottom() + 1) * ratio),
                    ],
                },
                f,
                indent=1,
            )

    states = iter(["Status: connecting", "Status: syncing 40%", "Status: done"])

    def tick() -> None:
        status.setText(next(states, status.text()))

    QTimer.singleShot(500, dump)
    timer = QTimer(root)
    timer.timeout.connect(tick)
    timer.start(1500)
    QTimer.singleShot(int(seconds * 1000), app.quit)
    app.exec()


if __name__ == "__main__":
    main()
