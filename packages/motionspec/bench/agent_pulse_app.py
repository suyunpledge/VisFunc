"""Qt ground truth for the Keystone transient-code benchmark."""

import json
import secrets
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget


def main():
    path = Path(sys.argv[1])
    armed = path.with_suffix(".armed")
    armed.unlink(missing_ok=True)
    state = {"clicks": 0, "code": f"PULSE {10000 + secrets.randbelow(90000)}", "hidden": False}
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("VisFunc pulse benchmark")
    window.setWindowFlag(Qt.WindowStaysOnTopHint)
    window.resize(520, 230)
    layout = QVBoxLayout(window)
    label = QLabel("Ready")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("font: bold 26px Arial; background: white; color: black; padding: 8px")
    button = QPushButton("Start check")
    button.setMinimumHeight(48)
    layout.addWidget(label)
    layout.addWidget(button)

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)

    def click():
        state["clicks"] += 1
        label.setText("Waiting")
        save()

    def hide():
        label.setText("Waiting")
        state["hidden"] = True
        save()

    def poll():
        if path.with_name(path.name + ".stop").exists():
            app.quit()
        elif armed.exists() and state["clicks"] and not state.get("shown"):
            label.setText(state["code"])
            state["shown"] = True
            save()
            QTimer.singleShot(625, hide)

    button.clicked.connect(click)
    window.move(200, 200)
    button.winId()
    window.show()
    window.raise_()
    save()
    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(40)
    QTimer.singleShot(180000, app.quit)
    app.exec()


if __name__ == "__main__":
    main()
