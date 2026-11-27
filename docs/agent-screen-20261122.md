# Agent 屏幕操作：等界面稳定、找点击位置、读界面文字

Agent 每做一步都要回答三个问题：界面停下来没有，变了什么，下一步该点哪里。原有的 `motionspec analyze` 和 `motionspec overlays` 都不是为这个设计的。前者为了描述动效要拟合曲线，一段 0.6 秒的片段要算 0.8～3.6 秒。后者按视频字幕和播放器底栏调参，用在普通应用界面上会把前进箭头报成播放键，还会对同一处文字给出几条互相矛盾的读数。

这次加的部分：

| 入口 | 作用 |
|---|---|
| `motionspec agent-mcp` | 常驻的 MCP 服务（stdio），把下面的能力作为工具交给 Agent，模型和学到的状态在步骤之间保留。[实现](../packages/motionspec/motionspec/agent.py) |
| `wait_for_settle`，命令行 `motionspec settle` | 边截屏边判断，界面静止后返回，并列出相对上一步变化的区域和其中的文字。[实现](../packages/motionspec/motionspec/settle.py) |
| `find_on_screen` | 按名字或可见文字找点击位置：先查 Windows UI Automation，再查 OCR。 |
| `list_ui_elements` | 通过 UI Automation 列出窗口里的控件：名字、类型、框、是否可用、勾选状态、输入框内容。[实现](../svsg/recognition/ui_elements.py) |
| `read_screen_text`，命令行 `motionspec screen-text`，`POST /v1/screen-text` | 读出整块区域的文字，每行一个框和点击点，长行再按词切开。[实现](../svsg/recognition/screen_text.py) |
| `screenshot` | 截取区域或窗口，附带图片像素到屏幕像素的换算。 |
| `click_point` | 按屏幕坐标点击允许窗口内的位置，给没有名字的自绘控件用；发送窗口消息，不移动指针。 |
| `click_element`、`set_field_text` | 通过 UI Automation 点按钮、切换复选框、选标签页、填输入框，不移动鼠标；只对 `--allow-window` 允许的窗口生效。每次操作自己等界面稳定，返回控件状态变化、变化区域里的文字和新弹出的窗口。 |

所有坐标都是屏幕物理像素 `[left, top, right, bottom]`，也就是点击时用的坐标；在缩放 150% 的显示器上也一样。

## 接入

```bash
python -m pip install -e ".[overlays,uia]" -e "./packages/motionspec[capture]"
motionspec agent-mcp
```

也可以用 `--http 127.0.0.1:8765` 以 Streamable HTTP 提供（地址 `/mcp`，可用环境变量 `VISFUNC_MCP_TOKEN` 要求令牌）。在 Pledge Keystone 里的接入和实测见 [Keystone 实测](agent-keystone-20261122.md)。在支持 stdio 的 Agent 客户端里，也可以登记为 stdio 服务，例如：

```json
{"mcpServers": {"visfunc-screen": {"command": "python", "args": ["-m", "motionspec", "agent-mcp"]}}}
```

每一步的用法：

1. 第一次操作之前调用一次 `wait_for_settle`，留下基准帧；
2. 用 Agent 自己的输入工具点击或输入；
3. 再调用 `wait_for_settle`（可加 `read_text`）：它和上一次调用的最后一帧比较，等界面静止，报告变化的区域和文字；
4. 用 `find_on_screen` 找下一个目标的点击点。

`window_title` 参数可以把截屏和查找限定在一个窗口内。直接盯整个屏幕时，屏幕上总有别的东西在动（比如正在刷新的对话窗口），结论会变成 `timeout`。

## 工作方式

等稳定：结论有四种，`settled`（变过并已静止）、`unchanged`（`expect_change_ms` 内没有变化，动作可能没生效）、`timeout`、`ended`（录像先放完）。"静止"同时看相邻两帧和 `quiet_ms` 之前的那一帧，所以缓出动画尾段每帧低于阈值的变化也会继续等；一段 800 ms、每帧不到 6 个灰度级的淡入变暗，只看相邻帧会被当成没有变化。持续变化满 `ambient_ms` 的区域（视频、动画背景）之后不再算作变化，服务里每个区域一个观察器，所以只需认识一次。两帧之间超过 250 ms 的空档视为 Agent 在操作，不计入等待，也不会冲掉学到的区域；空档里发生的变化会体现在空档前后两帧的差别上。加载圈和这类区域分不开，需要等加载完成时设 `ambient_ms` 为 0。

找点击位置：UI Automation 由应用自己提供控件信息，只有图标的按钮也有名字，禁用、勾选和输入内容都有。找到可用的控件时不再跑 OCR；找不到或应用不支持 UI Automation 时，用 OCR 的结果。OCR 常把相邻的两格或"标签 + 输入框"读成一行，所以每行再按空格和明显的间隙切成词，查找时匹配整行或连续的几个词。排序依次看：文字完全相同、整行优先于行内片段、大小写一致、更短、分数更高。同样文字出现多次时返回全部，并标 `ambiguous`。

读变化区域的文字时，区域会沿水平方向放宽再读，只保留和原区域相交的行。这样状态栏从 `Status: idle` 变成 `Status: syncing 40%` 时，读到的是整行，而不是只有变化的那半截。

截屏优先用 mss（`motionspec[capture]`），没有时退回 Pillow。

## 测试

机器为 32 个逻辑核心的 Windows 11，显示器 2560×1600、缩放 150%，Python 3.12，rapidocr-onnxruntime 1.4.4，Chrome 157。原始数据见 [结果文件](../bench/results/20261122-agent-screen/agent-screen-20261122.json)。

### 真实桌面上的 MCP 调用

用 [MCP 客户端脚本](../packages/motionspec/bench/agent_mcp_live.py) 启动 `motionspec agent-mcp`，对象是一个真实运行、置顶但不抢焦点的 Qt 窗口：6 个只有图标的工具栏按钮（tooltip 和无障碍名称）、一个输入框、一个勾选的复选框、一个禁用按钮，状态文字每 1.5 秒变一次。

| 调用 | 结果 | 耗时 |
|---|---|---|
| 握手与工具列表 | 协议 2025-06-18，5 个工具 | 337 ms |
| `find_on_screen` 10 个控件（6 个只有图标） | 10/10 点击点落在控件上，全部来自 UI Automation；禁用按钮标为不可用 | 多数 28～62 ms |
| `find_on_screen` 按输入框内容 `backup-server-07` | 找到那个输入框（UI Automation 的值） | 49 ms |
| `wait_for_settle` + `read_text`，状态文字变化后 | `settled`，变化后约 210 ms 判定，读出整行 `Status: syncing 40%`、`Status: done` | |
| `read_screen_text` 整个窗口 | 7 行全部读对 | 256 ms |
| `list_ui_elements` | 18 个元素 | 71 ms |
| `screenshot` | 1144×677 PNG | 20 ms |

找禁用按钮那一次要 761 ms，因为可用的控件没找到，又跑了一遍 OCR。

### Chrome 渲染的网页

用 [网页评测脚本](../packages/motionspec/bench/agent_web_eval.py) 在本机 Chrome（无头）里渲染四类本地页面，真值取自 DOM：每个文字节点的文字和框、输入框的值。"能点中"指某个精确匹配的点击点落在该文字的框内；"唯一文字第一个就对"只看屏幕上文字不重复的目标。

| 页面 | 读字耗时 | 能点中 | 唯一文字第一个就对 | 只有图标的按钮（OCR） |
|---|---:|---:|---:|---:|
| 后台设置页 | 0.80 s | 19/19 | 19/19 | 0/3 |
| 订单表格，25 行 6 列 | 2.42 s | 164/164 | 90/90 | 0/3 |
| 中文设置页 | 0.74 s | 19/19 | 19/19 | 0/3 |
| 深色仪表盘 | 0.89 s | 18/19 | 18/19 | 0/3 |
| 设置页，150% 缩放（2880×1620） | 0.88 s | 19/19 | 19/19 | 0/3 |

漏掉的一条是深色页上一整句事件描述，OCR 读出的文字和 DOM 不完全相同。只有图标的按钮 OCR 找不到，需要 UI Automation；无头浏览器没有窗口，这里没测，Chrome 窗口的无障碍树还要单独验证。

### Qt 渲染的桌面截图

六张 1920×1080 截图（150% 缩放的一张为 2880×1620），三种读法都先预热。

| 截图 | screen-text | 能点中 | 唯一文字第一个就对 | 直接调 RapidOCR | 字幕读取 overlays |
|---|---|---:|---:|---|---|
| 设置页 | 1.31 s，矛盾 0 | 18/18 | 18/18 | 1.43 s | 4.31 s，矛盾 4 |
| 深色主题 | 1.19 s，矛盾 0 | 18/18 | 18/18 | 1.52 s | 3.78 s |
| 中文界面 | 1.00 s，矛盾 0 | 17/18 | 17/18 | 1.21 s | 3.44 s，矛盾 1 |
| 订单表格 | 1.90 s，矛盾 0 | 97/100 | 62/64 | 2.03 s | 8.31 s，矛盾 6 |
| 150% 缩放 | 1.73 s，矛盾 0 | 18/18 | 18/18 | 1.91 s | 3.96 s，矛盾 5 |
| 确认框 | 0.90 s，矛盾 0 | 22/22 | 22/22 | 1.01 s | 5.11 s，矛盾 3 |

"矛盾"指互相重叠、文字不同的两行。按词切分之前，表格里只有 51/100 个单元格有自己的框；切分后 97/100 能点中。剩下的漏项是两个订单号读错一位，以及中文复选框前的方框被读成"口"（精确查找找不到，包含查找可以）。

### 等稳定

9 个 Qt 动效场景（1350×840，60 fps，`quiet_ms` 100）全部在动效真正结束之后才下结论，每帧 6.5～7.5 ms：

| 场景 | 模态框 | 展开折叠 | 侧栏收起 | 滚动 | 指示条 | 对调 | 回弹 | 变色 | 进度条 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 真实结束 ms | 416.7 | 466.7 | 466.7 | 416.7 | 416.7 | 466.7 | 566.7 | 366.7 | 766.7 |
| 判定 ms | 450.0 | 533.3 | 550.0 | 533.3 | 483.3 | 533.3 | 650.0 | 466.7 | 866.7 |

1080p 模拟点击（250 ms 缓出淡入的确认框，`quiet_ms` 200）：

| 情形 | 结束后多等 | 读字 |
|---|---:|---|
| 带遮罩的确认框 | 383 ms | 整屏，4/4 |
| 同上，屏幕上有视频，每次新建观察器 | 2850 ms | 整屏，4/4 |
| 同上，同一个观察器、操作前已看过视频 | 383 ms | 整屏，4/4 |
| 不带遮罩的小弹窗 | 383 ms | 只读弹窗区域，4/4 |

每帧平均 5～6 ms，最长约 11 ms。多等的 383 ms 来自 200 ms 的静止要求和缓出尾巴的累积漂移，调小 `quiet_ms` 可以缩短。

### 截屏

在这台显示器上，mss 截 2560×1600 整屏约 30 fps、1280×800 窗口约 58 fps（含复制为连续数组）；Pillow 分别约 15 和 18 fps。

## 目前做不到的

- 浏览器里只有图标的按钮：OCR 找不到，Chrome 窗口通过 UI Automation 暴露的名字还没验证。
- UI Automation 只在 Windows 上可用；不实现它的应用（部分游戏、自绘界面、远程桌面画面）只能靠 OCR，只有图标的按钮仍然找不到。
- 加载圈和背景动画分不开，只能靠 `ambient_ms` 或区域区分。
- 文字重复时只能返回全部候选，由 Agent 结合上下文选择。
- 测试用的界面是本地写的页面和 Qt 窗口，还没有在真实的第三方应用上做系统评测。
- `motionspec overlays` 仍是给视频播放器用的，不要用在普通应用界面上。

## 复现

```bash
python packages/motionspec/bench/agent_screens.py screens                  # 需要 PySide6
python packages/motionspec/bench/agent_text_eval.py screens > text.json    # 需要 svsg[overlays]
python packages/motionspec/bench/agent_settle_eval.py screens > agent.json
python packages/motionspec/bench/agent_web_eval.py web > web.json          # 需要 Playwright 和本机 Chrome
python packages/motionspec/bench/agent_mcp_live.py <PySide6 的 python> mcp-live.json   # 会在桌面上显示约 20 秒的测试窗口
```
