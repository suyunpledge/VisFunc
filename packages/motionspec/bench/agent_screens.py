"""渲染几张接近 Agent 实际会看到的桌面界面截图，并导出每个控件的文字和像素框作真值。

    python bench/agent_screens.py OUT_DIR      （需要 PySide6）

每个场景一张 PNG 和一个 truth JSON：texts 是带文字的控件（按钮、标签、输入框、表格单元），
icons 是工具栏按钮（标准 Qt 图标，带名字），坐标是截图的设备像素 [l, t, r, b]。
缩放场景通过 setDevicePixelRatio 等价的放大渲染实现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

W, H = 1920, 1080

ICONS = {
    "play": QStyle.SP_MediaPlay,
    "pause": QStyle.SP_MediaPause,
    "volume": QStyle.SP_MediaVolume,
    "mute": QStyle.SP_MediaVolumeMuted,
    "save": QStyle.SP_DialogSaveButton,
    "open": QStyle.SP_DialogOpenButton,
    "trash": QStyle.SP_TrashIcon,
    "refresh": QStyle.SP_BrowserReload,
    "back": QStyle.SP_ArrowBack,
    "forward": QStyle.SP_ArrowForward,
    "maximize": QStyle.SP_TitleBarMaxButton,
    "help": QStyle.SP_DialogHelpButton,
}


def dark(app: QApplication) -> None:
    p = QPalette()
    for role, c in [
        (QPalette.Window, "#1f2228"),
        (QPalette.WindowText, "#e6e6e6"),
        (QPalette.Base, "#16181d"),
        (QPalette.AlternateBase, "#24272e"),
        (QPalette.Text, "#e6e6e6"),
        (QPalette.Button, "#2c3038"),
        (QPalette.ButtonText, "#e6e6e6"),
        (QPalette.Highlight, "#2b6ff2"),
        (QPalette.HighlightedText, "#ffffff"),
    ]:
        p.setColor(role, QColor(c))
    app.setPalette(p)


def toolbar(root: QWidget, names: list[str]) -> QHBoxLayout:
    bar = QHBoxLayout()
    for name in names:
        b = QToolButton(root)
        b.setIcon(root.style().standardIcon(ICONS[name]))
        b.setIconSize(QSize(24, 24))
        b.setObjectName("icon:" + name)
        bar.addWidget(b)
    bar.addStretch(1)
    return bar


def settings(root: QWidget, zh: bool = False) -> None:
    t = (lambda en, cn: cn) if zh else (lambda en, cn: en)
    v = QVBoxLayout(root)
    v.addLayout(toolbar(root, ["back", "forward", "refresh", "save", "open", "trash", "help"]))
    title = QLabel(t("Account settings", "账户设置"), root)
    title.setStyleSheet("font-size: 22px; font-weight: 600;")
    v.addWidget(title)
    box = QGroupBox(t("Profile", "个人资料"), root)
    form = QFormLayout(box)
    form.addRow(QLabel(t("Display name", "显示名称")), QLineEdit(t("Morgan Lee", "李明")))
    form.addRow(QLabel(t("Email address", "电子邮箱")), QLineEdit("morgan.lee@example.com"))
    combo = QComboBox()
    combo.addItems([t("English (United States)", "简体中文"), "Deutsch"])
    form.addRow(QLabel(t("Language", "语言")), combo)
    v.addWidget(box)
    v.addWidget(QCheckBox(t("Sync files automatically", "自动同步文件"), root))
    v.addWidget(QCheckBox(t("Send crash reports", "发送崩溃报告"), root))
    lst = QListWidget(root)
    lst.addItems(
        [t(f"Device {i}: Laptop-{100 + i}", f"设备 {i}：笔记本-{100 + i}") for i in range(1, 6)]
    )
    lst.setFixedHeight(170)
    v.addWidget(lst)
    v.addStretch(1)
    row = QHBoxLayout()
    row.addStretch(1)
    for label in (t("Cancel", "取消"), t("Apply", "应用"), t("Save changes", "保存更改")):
        row.addWidget(QPushButton(label, root))
    v.addLayout(row)


def table(root: QWidget) -> None:
    v = QVBoxLayout(root)
    v.addLayout(toolbar(root, ["play", "pause", "volume", "mute", "maximize", "refresh"]))
    search = QLineEdit(root)
    search.setPlaceholderText("Search orders")
    search.setText("status:open")
    v.addWidget(search)
    tab = QTableWidget(18, 5, root)
    tab.setHorizontalHeaderLabels(["Order", "Customer", "Amount", "Status", "Updated"])
    names = ["Alvarez", "Brooks", "Chen", "Dubois", "Evans", "Fischer", "Garcia", "Haddad", "Ito"]
    for r in range(18):
        for c, val in enumerate(
            [
                f"#{4810 + r}",
                names[r % 9],
                f"${(r * 37) % 900 + 12}.50",
                ["Open", "Shipped", "Refunded"][r % 3],
                f"2026-11-{(r % 20) + 1:02d}",
            ]
        ):
            tab.setItem(r, c, QTableWidgetItem(val))
    tab.resizeColumnsToContents()
    v.addWidget(tab, 1)
    row = QHBoxLayout()
    row.addWidget(QLabel("Showing 18 of 240 orders", root))
    row.addStretch(1)
    for label in ("Previous", "Next", "Export CSV"):
        row.addWidget(QPushButton(label, root))
    v.addLayout(row)


def rect_of(w: QWidget, root: QWidget, scale: float) -> list[int]:
    tl = w.mapTo(root, QPoint(0, 0))
    r = QRect(tl, w.size())
    return [
        round(r.left() * scale),
        round(r.top() * scale),
        round((r.right() + 1) * scale),
        round((r.bottom() + 1) * scale),
    ]


def truth(root: QWidget, scale: float) -> dict:
    texts, icons = [], []
    for w in root.findChildren(QWidget):
        if not w.isVisible():
            continue
        name = w.objectName()
        if name.startswith("icon:"):
            icons.append({"label": name[5:], "bbox_px": rect_of(w, root, scale)})
            continue
        if isinstance(w, QTableWidget):
            hh = w.horizontalHeader()
            for c in range(w.columnCount()):
                x = hh.sectionViewportPosition(c)
                tl = hh.mapTo(root, QPoint(x, 0))
                texts.append(
                    {
                        "text": w.horizontalHeaderItem(c).text(),
                        "kind": "header",
                        "bbox_px": [
                            round(v * scale)
                            for v in (
                                tl.x(),
                                tl.y(),
                                tl.x() + hh.sectionSize(c),
                                tl.y() + hh.height(),
                            )
                        ],
                    }
                )
            vp = w.viewport()
            for r in range(w.rowCount()):
                for c in range(w.columnCount()):
                    vr = w.visualItemRect(w.item(r, c))
                    if not vp.rect().contains(vr):
                        continue
                    tl = vp.mapTo(root, vr.topLeft())
                    texts.append(
                        {
                            "text": w.item(r, c).text(),
                            "kind": "cell",
                            "bbox_px": [
                                round(v * scale)
                                for v in (tl.x(), tl.y(), tl.x() + vr.width(), tl.y() + vr.height())
                            ],
                        }
                    )
            continue
        if isinstance(w, QListWidget):
            vp = w.viewport()
            for i in range(w.count()):
                vr = w.visualItemRect(w.item(i))
                tl = vp.mapTo(root, vr.topLeft())
                texts.append(
                    {
                        "text": w.item(i).text(),
                        "kind": "list",
                        "bbox_px": [
                            round(v * scale)
                            for v in (tl.x(), tl.y(), tl.x() + vr.width(), tl.y() + vr.height())
                        ],
                    }
                )
            continue
        text = None
        if isinstance(w, (QPushButton, QCheckBox, QLabel)):
            text = w.text()
        elif isinstance(w, QLineEdit):
            text = w.text()
        elif isinstance(w, QComboBox):
            text = w.currentText()
        elif isinstance(w, QGroupBox):
            text = w.title()
        if text:
            texts.append(
                {"text": text, "kind": type(w).__name__, "bbox_px": rect_of(w, root, scale)}
            )
    return {
        "size_px": [round(W * scale), round(H * scale)],
        "scale": scale,
        "texts": texts,
        "icons": icons,
    }


SCENES = {
    "settings-light": (settings, {}, False, 1.0),
    "settings-dark": (settings, {}, True, 1.0),
    "settings-zh": (settings, {"zh": True}, False, 1.0),
    "orders-table": (table, {}, False, 1.0),
    "settings-150pct": (settings, {}, False, 1.5),
}


def settings_dialog(root: QWidget) -> None:
    """设置页上弹出一个确认框，用来做端到端的等稳定 + 读字测试。"""
    from PySide6.QtWidgets import QFrame

    settings(root)
    dim = QFrame(root)
    dim.setGeometry(0, 0, W, H)
    dim.setStyleSheet("background: rgba(0, 0, 0, 90);")
    box = QFrame(root)
    box.setObjectName("dialog")
    box.setGeometry(W // 2 - 260, H // 2 - 110, 520, 220)
    box.setStyleSheet(
        "QFrame#dialog { background: #ffffff; border: 1px solid #9aa0aa; border-radius: 8px; }"
    )
    v = QVBoxLayout(box)
    title = QLabel("Remove 3 devices?", box)
    title.setStyleSheet("font-size: 20px; font-weight: 600;")
    v.addWidget(title)
    v.addWidget(QLabel("Laptop-101, Laptop-102 and Laptop-104 will be signed out.", box))
    v.addStretch(1)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(QPushButton("Keep devices", box))
    row.addWidget(QPushButton("Remove", box))
    v.addLayout(row)


SCENES_DIALOG = {"dialog-after": (settings_dialog, {}, False, 1.0)}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    base = app.palette()
    for name, (build, kw, is_dark, scale) in {**SCENES, **SCENES_DIALOG}.items():
        app.setPalette(base)
        if is_dark:
            dark(app)
        root = QWidget()
        root.setFixedSize(W, H)
        f = root.font()
        f.setPointSizeF(10.5)
        root.setFont(f)
        build(root, **kw)
        root.show()
        app.processEvents()
        # 按指定倍率渲染（不用 grab，它会跟随本机显示缩放）；放大时字形重新栅格化，不是图片插值
        from PySide6.QtGui import QPixmap

        pm = QPixmap(round(W * scale), round(H * scale))
        pm.setDevicePixelRatio(scale)
        pm.fill(Qt.transparent)
        root.render(pm)
        pm.setDevicePixelRatio(1.0)
        pm.save(str(out / f"{name}.png"))
        (out / f"{name}.truth.json").write_text(
            json.dumps(truth(root, scale), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        root.close()
        print(name, pm.width(), pm.height())


if __name__ == "__main__":
    main()
