# VisFunc

**在本地测量图片和界面动效，为 AI 提供可查询、可核对的数值证据。**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[English](README.md) · 中文

Windows x64 用户可解压部署包并双击 `Start.cmd`，同时启动图片 API 和桌面 Agent MCP。[部署说明](deployment/windows/README.zh.md) · [生产加固与实测](docs/agent-production-20261123.md)

多模态模型能说出画面里有什么，但问到"这张卡片移动了多少像素""这个动画用了多久""这两块区域有没有挨着"，
回答经常不对。VisFunc 在本地把这些量测出来，整理成可以按帧、按坐标、按时间查询的函数，模型读数字作答。
模型给出的结论也可以拿同一份数据逐条核对。

仓库分两个子模块：

| 子模块 | 代码位置 | 做什么 |
|---|---|---|
| [VisFunc-Image](svsg/README.zh.md) | `svsg/`，包名 `svsg` | 把图片编译成像素、区域、轮廓和空间关系；根据这些参数回答问题并核对模型的结论；多视图物体识别；视频函数的统一格式与查询 |
| [VisFunc-Motion](packages/motionspec/README.zh.md) | `packages/motionspec/`，包名 `motionspec` | 从界面录屏或 Qt 应用的实时属性里测出动画：哪个元素、何时开始、移动多远、尺寸和透明度怎么变、什么缓动曲线 |

两个子模块可以分开用。VisFunc-Motion 导出的函数使用 VisFunc-Image 的格式和查询实现。

## 一个例子

一段 0.7 秒的录屏，右下角的提示条在 250 毫秒内上移 36 像素并淡入。我们把它交给六个多模态模型，问它怎么动。

| 模型 | 只给视频 | 同时给 VisFunc-Motion 的测量结果 |
|---|---|---|
| MiMo v2.6 Flash | 没有移动 | 上移 36.5 像素，167 ms 开始 |
| MiMo v2.6 Pro | 没有移动，400 ms | 上移 36.5 像素，167 ms 开始 |
| 豆包 Seed 2.1 Turbo | 没有移动，100 ms | 上移 36.5 像素，167 ms 开始 |
| 豆包 Seed 2.1 Pro | 向左 30 像素；另一次回答向左 300 像素 | 上移 36.5 像素，167 ms 开始 |
| GPT 6.1 Sol / Terra | 上移 36 / 40 像素 | 上移 36.5 像素，167 ms 开始 |

GPT 两个模型收到的是带时间戳的截图，因为这把密钥所属的组织没有开通视频输入。

在 9 个更接近真实界面的场景（模态框、展开面板、侧栏收起、滚动、标签指示条、两行对调、回弹、按钮变色、进度条）上，
六个模型逐项判分的正确率：只给视频 58%，加上视觉测量 82%，加上从 Qt 应用读到的控件属性 100%。
共 216 次调用，没有返回答案的调用按 0 分计。

## 快速上手

需要 Python 3.11 或更高版本。下面的步骤全部离线完成，不需要模型密钥。

```bash
python -m pip install -e ".[functions]" -e "./packages/motionspec"
```

生成一段卡片右移的示例视频，测量它，并导出可查询的函数：

```bash
python -c "from motionspec.synth import slide, write_video; frames, _ = slide(); write_video('slide.mp4', frames)"
```

```bash
python -m motionspec analyze slide.mp4 -o slide-out --functions --evidence-every-ms 250
```

`slide-out/summary.md` 里写着：卡片右移约 299 像素，能看出变化的时间约 350 毫秒，按曲线对齐估计的设计参数是 out-cubic、400 毫秒。
示例的设定是 300 像素、out-cubic、400 毫秒。缓出的最后一段每帧移动不到一个像素，所以能看出变化的时间比设计时长短。

列出可以查询的帧和函数：

```bash
python -m motionspec query slide-out/functions.json --list
```

把下面的内容存为 `query.json`，查询第 0 帧左上角像素的颜色和 250 毫秒时卡片的横坐标：

```json
[
  {"id": "color", "op": "pixel_rgba", "frame_index": 0, "x": 0, "y": 0},
  {"id": "position", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 250}
]
```

```bash
python -m motionspec query slide-out/functions.json query.json
```

返回 `[41, 45, 49, 255]`，附帧号、时间和像素哈希；以及 `261.0`，单位 `source_px`，来源 `motion_estimate`。

### 读取字幕和界面图标

安装本地 OCR 后，可从截图或已导出的源像素帧读取中英字幕和已知图标。视频结果带帧号、时间、像素哈希和源画面坐标。播放器底栏可用 `--player-toolbar` 限定搜索；自定义图标可用 `--icon-dir` 提供按类别命名的 PNG 样本。[评测与用法](docs/subtitle-icon-evaluation-20261122.md)

```bash
python -m pip install -e ".[overlays]" -e "./packages/motionspec"
motionspec overlays screenshot.png
motionspec overlays slide-out/functions.json --player-toolbar
```

这一套规则是按视频字幕和播放器底栏调出来的。用在普通应用界面上，会把前进箭头报成播放键，也会对同一处文字给出几条不同的读数，请改用下面的 `screen-text`。

### 给 Agent 用：等界面稳定、找点击位置、读界面文字

`motionspec agent-mcp` 是一个常驻的 MCP 服务，Agent 客户端登记后就能调用这些工具：

- `wait_for_settle` 用于等待较长的操作或其他工具触发的界面变化，默认静止区间为 200 ms。内置动作会自行等待并返回变化。
- `find_on_screen`：按名字或可见文字找点击位置，先查 Windows UI Automation（只有图标的按钮也有名字，禁用和勾选状态都知道），再查 OCR；
- `click_element`、`set_field_text`：通过 UI Automation 点按钮、填输入框，只对启动时 `--allow-window` 允许的窗口生效，每次操作直接带回效果；
- `read_screen_text`、`list_ui_elements`、`screenshot`。

`click_point` 回执的 `operation_marker.screen_px` 是发送鼠标消息的屏幕物理坐标；按名字操作的 `uia_target_center` 标记控件中心。支持图像的 MCP 客户端可以传 `marker_image=true`，取得标记在操作前截图上的位置。

自绘按钮可以直接用 `click_text(window_title=…, text=…)` 操作。若识别期间界面重排，工具会在新画面再查一次，按最新位置点击。派发前会复核整个识别区域，连同别处新出现的同名标签一起检查。标签缺失、重名、置信度低或画面持续变化时会返回 `execution_state=not_started`。成功回执包含文字、边框和屏幕点击坐标。[Keystone 桌面操作实测](docs/agent-screen-targets-20261125.md) · [识别期间重绘对照](docs/agent-click-recheck-20261126.md)

动作回执会给出 `screen_history.history_id`。用 `read_screen_history` 并设置 `compact=true`，可以查到提示文字、字段状态、对话框和局部画面变化。后续传入 `after=next_after` 增量读取，需要识别裁图时加 `read_text=true`。默认每次记录 20 秒，像素采样最高 8 Hz，控件最高 4 Hz。`watch_screen_changes` 可单独开始记录，`--history-seconds 0` 可关闭。[界面事件缓存与 token 实测](docs/agent-ui-events-20261123.md)

客户端为每次有意操作生成一个 UUID，跨连接重试时沿用同一个 HTTP `Idempotency-Key` 或 MCP `params._meta["visfunc/operationId"]`。已协商的同一会话也可用原 JSON-RPC 请求编号重试。动作回执保留 24 小时，遇到 `unknown` 时先只读核对窗口状态。[回执与恢复方式](docs/agent-ui-events-20261123.md)

坐标都是屏幕物理像素，即点击用的坐标。视频、动画背景这类一直在变的区域，服务只需认识一次，之后不再拖慢等待。不用 MCP 时，也可以直接用命令行 `motionspec settle --screen --read-text`、`motionspec screen-text`，或 HTTP 接口 `POST /v1/screen-text`。

```bash
python -m pip install -e ".[overlays,uia]" -e "./packages/motionspec[capture]"
motionspec agent-mcp
```

在真实桌面的 Qt 窗口上，10 个控件（其中 6 个只有图标）全部按名字找到，单次查找多数在 60 ms 以内；状态文字变化后约 210 ms 判定稳定，并读出完整的一行。在 Chrome 渲染的四类网页上，240 段文字里 239 段能按文字点中，包括 25 行表格的全部单元格。浏览器里只有图标的按钮 OCR 找不到，Chrome 窗口的 UI Automation 还没验证。[用法与评测](docs/agent-screen-20261122.md)

接入 Pledge Keystone（`--http`）后，GPT-6.1 sol 和 terra 完成了四类桌面任务：Qt 设置窗口（带确认弹窗）、同步失败后按错误提示改端口重试、Chrome 里的注册表单、只能按坐标点击的自绘界面。不设回合上限的最近 16 次全部完成，单个任务 20～65 秒；出错重试的任务要 13～15 次工具调用，Keystone 原先默认 12 次就会停下，现在默认不限步数，需要时在其高级配置里设置上限。Keystone 自带的窗口驱动在 Qt 窗口里看不到任何控件。本机的 ESET 浏览器屏幕保护会挡住浏览器截屏，此时服务改用 UI Automation 判断操作效果。[Keystone 实测](docs/agent-keystone-20261122.md)

## 结果汇总

独立留出集上，分析器在 30 个 Qt 场景中有 13 个全项正确，12 段 Chromium 录屏中有 6 个全项正确。浏览器的 6 个单独平移均通过，6 个布局重排均有缺测或误差。[分析器评测、逐场景结果和置信区间](docs/analyzer-evaluation-20261127.md)

| 测试 | 结果 |
|---|---|
| 6 个模型读 6 个基础界面动画（216 次调用） | 只给视频 55%，加视觉测量 92%，加 Qt 属性 94% |
| 6 个模型读 9 个较难界面动画（216 次调用） | 只给视频 58%，加视觉测量 82%，加 Qt 属性 100% |
| 同上，只换视觉测量的版本，每个组合两遍 | 改进前 68%，当前 86% |
| 只给参数、不给画面（两段动画，2 个模型） | 读时间参数 145/147 项正确，读文字摘要 147/147 项 |
| 视觉测量本身，9 个较难场景 | 逐项正确率 85%；Qt 属性 100% |
| 导出函数逐帧求值，对比 Qt 记录的真实属性 | 基础场景位置误差 ≤ 1.0 像素，进入/退出元素宽高误差 ≤ 1.0 像素，透明度误差 0 |
| COCO8 物体识别（8 张图，30 个目标） | 多视图复核后误报 9 → 2，漏检 10 → 16 |

这些界面场景是用 Qt 逐帧渲染的，帧间隔严格均匀，真实录屏的丢帧和录制延迟验证得很少。较难的 9 个场景在开发中反复使用过，
不算独立验证。测试数据、判分规则和原始回答见 [模型实测记录](bench/results/README.md)、
[纯参数复测](docs/timing-protocol-retest-20261117.md)、[采样率消融](docs/evidence-ablation-20261116.md)、
[识别评测](docs/recognition-evaluation-20261116.md)。
在四段 MOT20 实拍录像的八个新取样时刻，本地行人检测多找回了 3 个重遮挡行人（514 → 517，共 740 个标注），
两版误报均为 36 个。小目标命中和复核耗时详见[配对评测](docs/pedestrian-focused-review-20261122.md)。

## 目前做不到的

- 3D 画面和游戏画面只能给出统计量，没有可用的运动曲线。
- 布局重排测不准。面板展开推动下方元素、侧栏收起拉动内容时，变尺寸的那块会被测成位移，位置可能偏差一百多像素。
- 压缩视频里的颜色过渡会被 H.264 的色度抽样抹平。
- 不跨帧跟踪物体身份。片段编号只代表一段变化。
- 物体识别不保证找全。

## 安装与测试

```bash
python -m pip install -e ".[functions]" -e "./packages/motionspec"   # 测量与函数导出，离线
python -m pip install -e ".[api,llm]"                                  # HTTP 服务与模型调用
python -m pip install -r requirements-dev.txt                          # 开发与测试
python -m pytest
```

Qt 相关的场景工具和探针演示另外需要 PySide6。单仓库的安装、测试与代码边界见 [docs/monorepo.md](docs/monorepo.md)。

## 目录

```
svsg/                  VisFunc-Image
packages/motionspec/   VisFunc-Motion（独立的包和命令）
docs/                  函数协议、识别、单仓库说明、评测记录
tests/                 VisFunc-Image 的测试；VisFunc-Motion 的测试在 packages/motionspec/tests
scripts/               服务启动与评测脚本
```

## 许可

[MIT](LICENSE)
