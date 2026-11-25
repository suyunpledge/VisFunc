"""Exercise the Windows common file dialog against a fresh directory of disposable files."""

import ctypes
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel

TITLE = "VisFunc file chooser"


def main():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    state_file = Path(sys.argv[1])
    files = state_file.with_suffix(".files")
    files.mkdir(exist_ok=True)
    expected = files / "report-final.txt"
    for name in ("report-draft.txt", expected.name):
        (files / name).write_text("VisFunc desktop benchmark", encoding="utf-8")
    state = {"native_dialog": False, "opened_expected_file": False, "accepts": 0, "file": None}

    def save():
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(state_file)

    dialog = QFileDialog(None, TITLE, str(files), "Text files (*.txt)")
    dialog.setFileMode(QFileDialog.ExistingFile)
    result = QLabel()
    result.setWindowTitle(TITLE + " result")
    result.resize(360, 100)

    def accepted():
        selected = Path(dialog.selectedFiles()[0])
        state.update(
            accepts=state["accepts"] + 1,
            file=selected.name,
            opened_expected_file=selected.resolve() == expected.resolve(),
        )
        result.setText("Selected: " + selected.name)
        result.show()
        save()

    dialog.accepted.connect(accepted)
    dialog.open()
    user32 = ctypes.windll.user32
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))

    def poll():
        if state_file.with_name(state_file.name + ".stop").exists():
            app.quit()
        elif not state["native_dialog"]:
            hwnd = user32.FindWindowW("#32770", TITLE)
            pid = wintypes.DWORD()
            if (
                hwnd
                and user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                and pid.value == os.getpid()
            ):
                state["native_dialog"] = True
                save()

    save()
    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(50)
    QTimer.singleShot(int(float(sys.argv[2]) * 1000), app.quit)
    app.exec()


if __name__ == "__main__":
    main()
