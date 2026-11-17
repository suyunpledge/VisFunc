"""在一个真实运行的 QML 窗口上试 Qt 探针：动画由 Qt 的事件循环实时驱动，探针按渲染帧取样。

    QT_QPA_PLATFORM=offscreen QT_QUICK_BACKEND=software python tools/qml_probe_demo.py out/qml/trace.json
    python -m motionspec trace out/qml/trace.json -o out/qml

需要 PySide6。场景里的真值：
- card：300ms 起，x 从 40 到 340（300 逻辑像素），300ms，Easing.OutCubic
- toast：900ms 起，opacity 0→1、scale 0.9→1，250ms，Easing.InOutQuad
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

QML = b"""
import QtQuick
import QtQuick.Window

Window {
    width: 640; height: 400; visible: true; title: "probe-demo"; color: "#14161c"

    Rectangle {
        id: card; objectName: "card"
        x: 40; y: 60; width: 200; height: 100; radius: 10; color: "#232838"
        Text { objectName: "cardTitle"; x: 16; y: 14; text: "Quick setup"; color: "white" }
        NumberAnimation on x { id: slide; running: false; from: 40; to: 340; duration: 300; easing.type: Easing.OutCubic }
    }
    Rectangle {
        id: toast; objectName: "toast"
        x: 300; y: 300; width: 220; height: 44; radius: 8; color: "#2b6ff2"; opacity: 0; scale: 0.9
        ParallelAnimation {
            id: pop
            NumberAnimation { target: toast; property: "opacity"; to: 1; duration: 250; easing.type: Easing.InOutQuad }
            NumberAnimation { target: toast; property: "scale"; to: 1; duration: 250; easing.type: Easing.InOutQuad }
        }
    }
    Timer { interval: 300; running: true; onTriggered: slide.start() }
    Timer { interval: 900; running: true; onTriggered: pop.start() }
    Timer { interval: 1500; running: true; onTriggered: Qt.quit() }
}
"""


def load_probe():
    path = Path(__file__).resolve().parent.parent / "motionspec" / "qt_probe.py"
    spec = importlib.util.spec_from_file_location("motionspec_qt_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/qml/trace.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    app = QGuiApplication(sys.argv[:1])
    engine = QQmlApplicationEngine()
    engine.loadData(QML)
    if not engine.rootObjects():
        raise SystemExit("QML failed to load")
    recorder = load_probe().Recorder()
    QTimer.singleShot(120, recorder.start)  # 等窗口建好、根元素定好尺寸再开始记
    app.exec()
    recorder.stop()
    recorder.dump(out)
    trace = recorder.trace()
    print(f"samples {len(trace['ticks_ms'])}, elements seen {trace['elements_seen']}, changed {len(trace['elements'])}, wrote {out}")
    del engine


if __name__ == "__main__":
    main()
