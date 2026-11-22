"""给 Agent 实测用的"同步设置"小应用：任务做没做对，由它自己写出的状态文件判定。

    python bench/agent_task_app.py STATE.json [最长秒数] [normal|flaky]      （需要 PySide6）

flaky 模式下，端口不是 8443 时同步会在连接阶段失败，并弹出说明正确端口的错误框。

窗口标题 "Keystone agent task"，置顶、不抢焦点。改动设置后 Apply 才可用，点了才算保存；
Start sync 先弹确认框，确认后依次显示 Connecting、进度、Done。Delete 会弹出删除确认框，
确认即视为失败。每次状态变化都重写 STATE.json；存在 STATE.json.stop 时退出。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

TITLE = "Keystone agent task"
FILES = 48


def quiet(widget: QWidget) -> None:
    widget.setAttribute(Qt.WA_ShowWithoutActivating)
    widget.setWindowFlag(Qt.WindowDoesNotAcceptFocus)
    widget.setWindowFlag(Qt.WindowStaysOnTopHint)


class App:
    def __init__(self, state_path: Path, mode: str = "normal") -> None:
        self.path = state_path
        self.mode = mode
        self.state = {
            "server_saved": "backup-server-07",
            "port_saved": "8080",
            "encrypt_saved": False,
            "apply_clicks": 0,
            "sync_started": 0,
            "sync_cancelled": 0,
            "synced_to": None,
            "synced_encrypted": None,
            "files": 0,
            "completed": False,
            "deleted": False,
            "delete_prompts": 0,
            "sync_failures": 0,
            "events": [],
        }
        root = self.root = QWidget()
        root.setWindowTitle(TITLE)
        quiet(root)
        root.resize(760, 460)
        root.move(40, 40)
        v = QVBoxLayout(root)
        bar = QHBoxLayout()
        for name, icon, slot in (
            ("Refresh", QStyle.SP_BrowserReload, self.refresh),
            ("Save", QStyle.SP_DialogSaveButton, self.apply),
            ("Delete", QStyle.SP_TrashIcon, self.ask_delete),
            ("Help", QStyle.SP_DialogHelpButton, lambda: self.log("help")),
        ):
            b = QToolButton(root)
            b.setIcon(root.style().standardIcon(icon))
            b.setIconSize(QSize(24, 24))
            b.setToolTip(name)
            b.setAccessibleName(name)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        v.addLayout(bar)
        title = QLabel("Sync settings", root)
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        v.addWidget(title)
        form = QFormLayout()
        self.server = QLineEdit(self.state["server_saved"], root)
        self.server.setAccessibleName("Server name")
        self.port = QLineEdit(self.state["port_saved"], root)
        self.port.setAccessibleName("Port")
        form.addRow("Server name", self.server)
        form.addRow("Port", self.port)
        v.addLayout(form)
        self.startup = QCheckBox("Sync on startup", root)
        self.startup.setChecked(True)
        self.encrypt = QCheckBox("Encrypt transfers", root)
        v.addWidget(self.startup)
        v.addWidget(self.encrypt)
        self.status = QLabel("Status: idle", root)
        v.addWidget(self.status)
        self.bar = QProgressBar(root)
        self.bar.setRange(0, FILES)
        self.bar.setValue(0)
        v.addWidget(self.bar)
        v.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.apply_button = QPushButton("Apply", root)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply)
        self.start = QPushButton("Start sync", root)
        self.start.clicked.connect(self.ask_start)
        row.addWidget(self.apply_button)
        row.addWidget(self.start)
        v.addLayout(row)
        for w in (self.server, self.port):
            w.textChanged.connect(self.dirty)
        self.encrypt.toggled.connect(self.dirty)
        self.dialog = None
        root.show()
        self.write()

    def log(self, event: str) -> None:
        self.state["events"].append(event)
        self.write()

    def write(self) -> None:
        self.state["server_field"] = self.server.text()
        self.state["encrypt_checked"] = self.encrypt.isChecked()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def dirty(self, *_) -> None:
        changed = (
            self.server.text() != self.state["server_saved"]
            or self.port.text() != self.state["port_saved"]
            or self.encrypt.isChecked() != self.state["encrypt_saved"]
        )
        self.apply_button.setEnabled(changed)
        self.write()

    def apply(self) -> None:
        self.state.update(
            server_saved=self.server.text(),
            port_saved=self.port.text(),
            encrypt_saved=self.encrypt.isChecked(),
        )
        self.state["apply_clicks"] += 1
        self.apply_button.setEnabled(False)
        self.status.setText("Status: settings saved")
        self.log("apply")

    def refresh(self) -> None:
        self.status.setText("Status: idle")
        self.bar.setValue(0)
        self.log("refresh")

    def confirm(self, title: str, text: str, yes: str, on_yes, on_no) -> None:
        dialog = self.dialog = QDialog(self.root)
        dialog.setWindowTitle(f"{TITLE} - {title}")
        quiet(dialog)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(text, dialog))
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel", dialog)
        ok = QPushButton(yes, dialog)
        if on_no is not None:
            buttons.addWidget(cancel)
        buttons.addWidget(ok)
        layout.addLayout(buttons)

        def finish(accepted: bool) -> None:
            dialog.close()
            self.dialog = None
            (on_yes if accepted else on_no or (lambda: None))()

        cancel.clicked.connect(lambda: finish(False))
        ok.clicked.connect(lambda: finish(True))
        dialog.setModal(True)
        dialog.show()

    def ask_start(self) -> None:
        self.log("start_pressed")
        self.confirm(
            "confirm",
            f"Start syncing {FILES} files to {self.state['server_saved']}?",
            "Start",
            self.run_sync,
            self.cancelled,
        )

    def cancelled(self) -> None:
        self.state["sync_cancelled"] += 1
        self.log("sync_cancelled")

    def ask_delete(self) -> None:
        self.state["delete_prompts"] += 1
        self.log("delete_prompt")
        self.confirm(
            "delete",
            "Delete all local sync data? This cannot be undone.",
            "Delete",
            self.delete,
            lambda: self.log("delete_cancelled"),
        )

    def delete(self) -> None:
        self.state["deleted"] = True
        self.status.setText("Status: local data deleted")
        self.log("deleted")

    def run_sync(self) -> None:
        self.state["sync_started"] += 1
        self.state["completed"] = False
        self.status.setText("Status: connecting...")
        self.bar.setValue(0)
        self.log("sync_started")
        steps = [12, 24, 36, FILES]

        def step(i=0):
            if i < len(steps):
                self.bar.setValue(steps[i])
                self.status.setText(f"Status: syncing {steps[i]}/{FILES} files")
                QTimer.singleShot(400, lambda: step(i + 1))
                return
            mode = "encrypted" if self.state["encrypt_saved"] else "plain"
            self.status.setText(
                f"Status: done - {FILES} files synced to {self.state['server_saved']} ({mode})"
            )
            self.state.update(
                synced_to=self.state["server_saved"],
                synced_encrypted=self.state["encrypt_saved"],
                files=FILES,
                completed=True,
            )
            self.log("sync_done")

        if self.mode == "flaky" and self.state["port_saved"] != "8443":
            QTimer.singleShot(600, self.fail)
            return
        QTimer.singleShot(600, step)

    def fail(self) -> None:
        server, port = self.state["server_saved"], self.state["port_saved"]
        self.state["sync_failures"] += 1
        self.status.setText("Status: sync failed")
        self.log("sync_failed")
        self.confirm(
            "error",
            f"Connection to {server}:{port} was refused. "
            f"{server} only accepts connections on port 8443.",
            "OK",
            lambda: self.log("error_closed"),
            None,
        )


def main() -> None:
    path = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0
    mode = sys.argv[3] if len(sys.argv) > 3 else "normal"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    App(path, mode)
    stop = path.with_name(path.name + ".stop")
    poll = QTimer()
    poll.timeout.connect(lambda: stop.exists() and app.quit())
    poll.start(300)
    QTimer.singleShot(int(seconds * 1000), app.quit)
    app.exec()


if __name__ == "__main__":
    main()
