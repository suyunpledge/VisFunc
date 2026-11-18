"""Qt 探针（应用内的那一半）：按帧记录界面元素的几何与透明度，写成轨迹文件。

这个文件只依赖 Qt 绑定（PySide6 / PyQt6），不依赖 numpy、OpenCV，也不引用包内其他模块，
可以整个文件拷进被测应用，或按路径加载。轨迹转规格在离线那一半做（motionspec.trace）。

同时支持 QWidget 和 QML（QQuickItem）界面。记录的是控件自身的属性，不是像素：
    x, y, width, height, opacity, scale, rotation, visible，以及它在窗口里的绝对位置。

用法一，跟着应用的帧走（实时）：

    from motionspec.qt_probe import Recorder
    recorder = Recorder()
    recorder.start()
    ...                       # 操作界面
    recorder.stop()
    recorder.dump("trace.json")

用法二，自己控制时间（逐帧渲染、测试）：每推进一帧调用一次 recorder.sample(t_ms)。

用法三，不改代码逻辑，只加一行并用环境变量开关：

    from motionspec.qt_probe import attach_from_env
    attach_from_env()         # 设了 MOTIONSPEC_TRACE=路径 才生效，应用退出时写出轨迹
"""

from __future__ import annotations

import json
import math
import os
import re
import time

SCHEMA = "motionspec-trace/0.1"
FIELDS = ("x", "y", "width", "height", "opacity", "scale", "rotation", "visible", "scene_x", "scene_y")
LOCAL = 8  # 前 8 个是元素自己的属性，后两个是它在窗口里的绝对位置


def _pointer(obj) -> int:
    """同一个 C++ 对象每次取到的 Python 包装可能不同，用底层指针做身份。"""
    for module, call in (("shiboken6", "getCppPointer"), ("PyQt6.sip", "unwrapinstance"), ("sip", "unwrapinstance")):
        try:
            found = getattr(__import__(module, fromlist=[call]), call)(obj)
            return int(found[0] if isinstance(found, (tuple, list)) else found)
        except Exception:
            continue
    return id(obj)


def _is_item(obj) -> bool:
    return hasattr(obj, "childItems")


def _children(obj) -> list:
    if _is_item(obj):
        return list(obj.childItems())
    return [c for c in obj.children() if c.isWidgetType()] if hasattr(obj, "isWidgetType") else []


def _snapshot(obj) -> tuple | None:
    """(x, y, w, h, opacity, scale, rotation, visible, scene_x, scene_y)，单位是逻辑像素。"""
    try:
        if _is_item(obj):
            origin = obj.mapToScene(type(obj.position())(0.0, 0.0))
            return (obj.x(), obj.y(), obj.width(), obj.height(), obj.opacity(), obj.scale(), obj.rotation(),
                    1.0 if obj.isVisible() else 0.0, origin.x(), origin.y())
        window = obj.window()
        origin = obj.mapTo(window, type(obj.pos())(0, 0)) if window is not obj else type(obj.pos())(0, 0)
        effect = obj.graphicsEffect()
        opacity = effect.opacity() if effect is not None and hasattr(effect, "opacity") else 1.0
        if window is obj:
            opacity *= obj.windowOpacity()
        # 只看元素自己的可见性：父级隐藏导致的"不可见"属于父级，不应在每个子元素上各报一次
        parent = obj.parentWidget()
        shown = obj.isVisibleTo(parent) if parent is not None else obj.isVisible()
        return (float(obj.x()), float(obj.y()), float(obj.width()), float(obj.height()), float(opacity), 1.0, 0.0,
                1.0 if shown else 0.0, float(origin.x()), float(origin.y()))
    except RuntimeError:  # 对象已被销毁
        return None


_BACKGROUND = re.compile(r"background(?:-color)?\s*:\s*(#[0-9a-fA-F]{6}|rgba?\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*(?:,\s*[\d.]+\s*)?\))")


def _colour_of(obj) -> tuple | None:
    """元素自身的颜色（0~1 的 r, g, b, a）。QML 读 color 属性；控件读样式表里的背景色。"""
    try:
        if _is_item(obj):
            value = obj.property("color")
            if value is not None and hasattr(value, "redF") and value.isValid():
                return (round(value.redF(), 4), round(value.greenF(), 4), round(value.blueF(), 4), round(value.alphaF(), 4))
            return None
        sheet = obj.styleSheet() if hasattr(obj, "styleSheet") else ""
        match = _BACKGROUND.search(sheet or "")
        if not match:
            return None
        text = match.group(1)
        if text.startswith("#"):
            r, g, b = (int(text[i : i + 2], 16) / 255 for i in (1, 3, 5))
            return (round(r, 4), round(g, 4), round(b, 4), 1.0)
        parts = [float(v) for v in re.findall(r"[\d.]+", text)]
        alpha = parts[3] if len(parts) > 3 else 1.0
        return (round(parts[0] / 255, 4), round(parts[1] / 255, 4), round(parts[2] / 255, 4), round(alpha if alpha <= 1 else alpha / 255, 4))
    except (RuntimeError, ValueError, IndexError):
        return None


def _extras(obj, names: tuple[str, ...]) -> dict:
    """颜色和指定的数值属性（例如 QML 里的 hoverAmount）。读不到的跳过。"""
    out = {}
    colour = _colour_of(obj)
    if colour is not None:
        out["color"] = colour
    for name in names:
        try:
            value = obj.property(name)
        except RuntimeError:
            continue
        if type(value) in (int, float) and math.isfinite(value):
            out[name] = (float(value),)
    return out


def _describe(obj) -> dict:
    info = {"class": obj.metaObject().className(), "name": obj.objectName() or ""}
    try:
        text = obj.property("text")
        if isinstance(text, str) and text:
            info["text"] = text[:60]
    except Exception:
        pass
    chain, node = [], obj
    for _ in range(6):
        node = node.parentItem() if _is_item(node) else node.parent()
        if node is None:
            break
        chain.append(node.objectName() or node.metaObject().className())
    info["path"] = "/".join(reversed(chain))
    return info


class Recorder:
    """记录一个或多个根对象之下所有元素的属性变化。

    只在值变化时存一行，外加变化前最后一帧和变化后第一帧，所以静止的界面几乎不占空间。
    """

    def __init__(self, roots=None, *, max_elements: int = 20000, frame_ms: float = 1000.0 / 60,
                 properties: tuple[str, ...] | list[str] = ()):
        if type(max_elements) is not int or max_elements < 1 or not math.isfinite(frame_ms) or frame_ms <= 0:
            raise ValueError("invalid recorder element budget or frame interval")
        if not isinstance(properties, (tuple, list)):
            raise ValueError("properties must be a list or tuple of names")
        names = tuple(properties)
        if (len(names) > 15 or any(type(n) is not str or not n or len(n) > 64 or n == "color" for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("properties must be up to 15 property names (color is recorded automatically)")
        self._properties = names
        self._roots = list(roots) if roots is not None else None
        self._max = max_elements
        self.frame_ms = frame_ms  # 一帧的名义时长，用来识别"隔了很久才来的一帧"
        self._ticks: list[float] = []
        self._elements: dict[int, dict] = {}
        self._active: dict[int, int] = {}  # Native pointer -> stable recording identity.
        self._clock = None
        self._timer = None
        self._windows: list = []
        self._quick_windows: list = []
        self._size = (0, 0)
        self._ratio = 1.0
        self._title = ""
        self.truncated = False

    # ---- 取样 ----
    def _find_roots(self) -> list:
        if self._roots is not None:
            return self._roots
        roots = []
        try:
            from PySide6.QtGui import QGuiApplication as Gui
        except ImportError:
            from PyQt6.QtGui import QGuiApplication as Gui
        # 不先导入 QtQuick，topLevelWindows() 给回的是基类 QWindow 的包装，拿不到 contentItem
        for binding in ("PySide6.QtQuick", "PyQt6.QtQuick"):
            try:
                __import__(binding)
                break
            except ImportError:
                continue
        # 窗口的包装要一直留着：它一旦被回收，从它取到的 contentItem 包装会跟着失效
        if not self._quick_windows:
            self._quick_windows = [w for w in Gui.topLevelWindows() if hasattr(w, "contentItem")]
        for window in self._quick_windows:
            try:
                roots.append(window.contentItem())
            except RuntimeError:
                continue
        if not roots:
            try:
                from PySide6.QtWidgets import QApplication as Widgets
            except ImportError:
                from PyQt6.QtWidgets import QApplication as Widgets
            app = Widgets.instance()
            if app is not None and hasattr(app, "topLevelWidgets"):
                roots = list(app.topLevelWidgets())
        return roots

    def _note_window(self, root) -> None:
        try:
            if _is_item(root):
                window = root.window()
                if window is not None:
                    self._size = (window.width(), window.height())
                    self._ratio = float(window.devicePixelRatio())
                    self._title = window.title()
            else:
                self._size = (root.window().width(), root.window().height())
                self._ratio = float(root.devicePixelRatioF())
                self._title = root.window().windowTitle()
        except Exception:
            pass

    def sample(self, t_ms: float | None = None) -> None:
        """取一帧。不给时间就用墙上时钟（从第一次取样算起）。"""
        if t_ms is None:
            now = time.perf_counter()
            if self._clock is None:
                self._clock = now
            t_ms = (now - self._clock) * 1000.0
        if type(t_ms) not in (int, float) or not math.isfinite(t_ms) or t_ms < 0 or (self._ticks and t_ms <= self._ticks[-1]):
            raise ValueError("sample times must be finite, nonnegative and strictly increasing")
        # 界面空闲时 QML 不渲染帧，也就没有取样点。隔了很久才来的这一帧之前，属性其实一直没变，
        # 所以在它前面补一个时间点，"变化前最后一帧"的时间才是对的
        if self._ticks and t_ms - self._ticks[-1] > 2.5 * self.frame_ms:
            self._ticks.append(float(t_ms) - self.frame_ms)
        tick = len(self._ticks)
        self._ticks.append(float(t_ms))
        roots = self._find_roots()
        if roots and tick == 0:
            self._note_window(roots[0])
        stack = list(roots)
        seen = set()
        while stack:
            obj = stack.pop()
            values = _snapshot(obj)
            if values is None:
                continue
            stack.extend(_children(obj))
            key = _pointer(obj)
            if key in seen:
                continue
            seen.add(key)
            identity = self._active.get(key)
            entry = self._elements.get(identity)
            if entry is None:
                if len(self._elements) >= self._max:
                    self.truncated = True
                    continue
                identity = len(self._elements)
                self._active[key] = identity
                entry = self._elements[identity] = {"info": _describe(obj), "last": values, "last_tick": tick,
                                                   "stored": tick if tick else -1, "settle": False,
                                                   "rows": [[tick, *values]] if tick else [],
                                                   "events": [{"tick": tick, "kind": "appeared"}] if tick else [],
                                                   "props": {name: {"last": value, "stored": -1, "settle": False, "rows": []}
                                                             for name, value in _extras(obj, self._properties).items()}}
                continue
            for name, value in _extras(obj, self._properties).items():  # 颜色和自定义属性，记法和几何属性一样
                prop = entry["props"].setdefault(name, {"last": value, "stored": -1, "settle": False, "rows": []})
                if value != prop["last"]:
                    if prop["stored"] != tick - 1:
                        prop["rows"].append([tick - 1, *prop["last"]])
                    prop["rows"].append([tick, *value])
                    prop["stored"], prop["settle"] = tick, True
                elif prop["settle"]:
                    prop["rows"].append([tick, *value])
                    prop["stored"], prop["settle"] = tick, False
                prop["last"] = value
            # 只看元素自己的属性。窗口内的绝对位置会随父级一起变，那是父级的动效，不是它的
            if values[:LOCAL] != entry["last"][:LOCAL]:
                if entry["info"] is None:
                    entry["info"] = _describe(obj)
                if entry["stored"] != tick - 1:  # 补上变化前的最后一帧
                    entry["rows"].append([tick - 1, *entry["last"]])
                entry["rows"].append([tick, *values])
                entry["stored"], entry["settle"] = tick, True
            elif entry["settle"]:  # 变化后的第一帧静止
                entry["rows"].append([tick, *values])
                entry["stored"], entry["settle"] = tick, False
            entry["last"], entry["last_tick"] = values, tick
        for pointer in set(self._active) - seen:
            entry = self._elements[self._active.pop(pointer)]
            if not entry["rows"]:
                entry["rows"].append([entry["last_tick"], *entry["last"]])
            entry["events"].append({"tick": tick, "kind": "disappeared"})

    # ---- 实时驱动 ----
    def start(self, interval_ms: int = 8) -> None:
        """跟着应用运行。QML 窗口按渲染帧取样，其余用定时器。"""
        self.sample()
        hooked = False
        for window in self._quick_windows:
            window.afterAnimating.connect(self._on_frame)
            self._windows.append(window)
            hooked = True
        if not hooked:
            try:
                from PySide6.QtCore import Qt, QTimer
            except ImportError:
                from PyQt6.QtCore import Qt, QTimer
            self._timer = QTimer()
            self._timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._timer.setInterval(interval_ms)
            self._timer.timeout.connect(self._on_frame)
            self._timer.start()

    def _on_frame(self) -> None:
        self.sample()

    def stop(self) -> None:
        for window in self._windows:
            try:
                window.afterAnimating.disconnect(self._on_frame)
            except Exception:
                pass
        self._windows = []
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    # ---- 输出 ----
    def trace(self) -> dict:
        elements = []
        for key, entry in self._elements.items():
            props = {name: prop["rows"] for name, prop in entry.get("props", {}).items() if prop["rows"]}
            rows = entry["rows"] or ([[entry["last_tick"], *entry["last"]]] if props else [])
            if rows:
                element = {"id": key, **(entry["info"] or {}), "fields": ["tick", *FIELDS], "rows": rows,
                           "events": entry["events"]}
                if props:
                    element["props"] = props
                elements.append(element)
        return {
            "schema": SCHEMA,
            "title": self._title,
            "size": [self._size[0], self._size[1]],
            "device_pixel_ratio": self._ratio,
            "ticks_ms": self._ticks,
            "elements_seen": len(self._elements),
            "truncated": self.truncated,
            "elements": elements,
        }

    def dump(self, path) -> None:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self.trace(), stream, ensure_ascii=False)


def attach_from_env(variable: str = "MOTIONSPEC_TRACE") -> Recorder | None:
    """设了环境变量才启用：开始记录，并在应用退出时把轨迹写到变量指向的路径。"""
    path = os.environ.get(variable)
    if not path:
        return None
    try:
        from PySide6.QtCore import QCoreApplication, QTimer
    except ImportError:
        from PyQt6.QtCore import QCoreApplication, QTimer
    names = tuple(n for n in os.environ.get("MOTIONSPEC_PROPS", "").replace(" ", "").split(",") if n)
    recorder = Recorder(properties=names)

    def finish() -> None:
        recorder.stop()
        recorder.dump(path)

    QCoreApplication.instance().aboutToQuit.connect(finish)
    QTimer.singleShot(0, recorder.start)  # 等事件循环起来、窗口建好再开始
    return recorder
