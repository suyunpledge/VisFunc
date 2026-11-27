# VisFunc

[English](README.md) · 中文

**把界面录屏和测量摘要一起交给多模态模型。** VisFunc 在本地测出元素移动了多少像素、变化持续多久、尺寸和透明度如何改变，并在 `summary.md` 中区分可见变化时间和估计设计时序。模型可以对着画面辨认控件，用摘要里的数值回答它是怎么动的。

| 输入或测量方式 | Qt 动画（留出 12 段） | Chrome 录屏（留出 12 段） | 有效答卷数（共 72 份） |
| --- | ---: | ---: | ---: |
| 只给视频或带时间戳的稀疏帧 | 48.1% | 52.3% | 60 |
| 带时间戳的密集帧 | 37.7% | 42.1% | 49 |
| 模型在沙盒内自行编写 OpenCV 代码测量 | 39.0% | 51.2% | 39 |
| **视频或稀疏帧 + VisFunc 测量摘要** | **82.7%** | **78.2%** | **69** |

三款模型在相同素材上每种条件各答一次。MiMo 和豆包接收视频，GPT 接收带时间戳的帧图。每段按适用的动效字段和动效数量计分，各场景、各模型等权。缺测字段不得分，交付失败的整份答卷计零。密集帧组与摘要组的实际输入 token 数随素材而异。Qt 动画为合成渲染，Chrome 素材录自本地测试页面。[强基线报告、置信区间与原始答卷](bench/results/20261127-strong-baselines/report.md)

**MiMo 和豆包的收益最大。** GPT 在 Chrome 录屏上只看稀疏帧已有 91.7%，看密集帧达到 95.8%。让 GPT 用 OpenCV 自测时，成功交付的 9 段 Chrome 答卷也全部答对。MiMo 则有 21 个密集帧任务触及输出上限。程序消费模型答案时，能否交出有效答卷也会影响最终得分。

| 模型 | Qt，只看画面 → 加摘要 | Chrome，只看画面 → 加摘要 |
| --- | ---: | ---: |
| MiMo V2.6 Flash | 32.2% → 80.9% | 27.8% → 63.9% |
| 豆包 Seed 2.1 Pro | 39.0% → 71.2% | 37.5% → 72.2% |
| GPT-6.1 Sol | 73.1% → 95.8% | 91.7% → 98.6% |

## 快速上手

使用 Python 3.11 或更高版本，从仓库根目录安装动效分析器。安装后，测量在本地运行，无需模型密钥。

```bash
python -m pip install -e "./packages/motionspec"
```

生成一段卡片右移的示例录屏并测量。测自己的录屏时，在最后一条命令中换掉 `slide.mp4` 即可。

```bash
python -c "from motionspec.synth import slide, write_video; frames, _ = slide(); write_video('slide.mp4', frames)"
python -m motionspec analyze slide.mp4 -o slide-out
```

打开 `slide-out/summary.md`。卡片上的跟踪点**向右移动 299.4 像素**。可见变化从 **167 到 517 ms，持续 350 ms**。按 `out-cubic` 曲线拟合，估计设计区间为 **167 到 567 ms，持续 400 ms**。示例设定为在 400 ms 内移动 300 像素，缓出末段每帧位移已不足一个像素。

将原录屏或带时间戳的帧图连同 `summary.md` 的内容发给模型，可以问「哪个元素移动了，移动多远？两个时长的依据分别是什么？」同目录的 `spec.json` 保存结构化测量，证据帧可用来核对检测区域。[动效测量指南](packages/motionspec/README.zh.md)

### 分析器本身测得多准

在最初的独立留出集上，分析器与录制真值直接比对，30 段 Qt 动画的场景均分为 **69.8%**，12 段 Chrome 录屏为 **72.2%**。所有适用字段和动效数量都正确的场景分别有 13/30 和 6/12。[分析器独立评测](docs/analyzer-evaluation-20261127.md)

面板展开并推动邻近控件时，可加 `--layout-edges`，分别追踪面板四边和相邻控件的位移。在同类场景重新采集的新留出集上，开关前后 Qt 场景均分从 70.6% 升至 91.1%，Chrome 从 70.8% 升至 100%，其中 16 个重排场景全部测准。旧 Chrome 录屏中仍有一段多报了片段，部分 Qt 淡入也缺少透明度端点。[布局重排对照与剩余误差](bench/results/20261127-layout-edges/report.md)

### 把测不准的字段标出来，模型会答得更好吗？

`--mark-uncertain` 可按需开启。例如，同样的像素混合既可能来自变色，也可能来自淡入。标记版摘要会解释透明度端点为何不确定，并保留观察到的变化。在 27 段开发集录屏上，分析器测错或缺测的 30 个对象字段里有 **5 个**被标出，另有 11 个被标字段原本就在容差内。

| 模型 | 场景均分，普通摘要 → 标记摘要 | 5 个确有缺口且被标出的字段，答对数 |
| --- | ---: | ---: |
| GPT-6.1 Sol | 95.7% → 96.0% | 5/5 → 5/5 |
| MiMo V2.6 Flash | 85.0% → 72.0% | 4/5 → 3/5 |
| 豆包 Seed 2.1 Pro | 89.0% → 87.2%（网络补测后） | 4/5 → 3/5 |

两组使用相同画面、独立会话。GPT 拿到普通摘要就补对了这 5 项。MiMo 在一段淡入中原本答对两个透明度端点，加标记后把两项都留空了，整轮格式失败也从 3 次增至 6 次。豆包普通摘要组有两次网络失败，计零时为 81.5%，独立补测后为 89.0%。默认仍使用普通摘要。[配对实测、豆包两种计分与逐字段误差](bench/results/20261127-uncertainty/report.md)

## 高级用法

### 按帧、坐标和时间查询函数

需要在特定帧读一个像素，或在给定时刻复算位置时，可以导出函数。在首页同一批 24 段留出素材上，给模型画面、函数目录和查询工具，Qt、Chrome 的端到端得分分别为 **46.9%、66.0%**，直接给画面和摘要为 **82.7%、78.2%**。函数组交付了 50/72 份有效答卷，摘要组为 69/72。额外的交互轮次和答卷交付失败影响了函数组得分。[函数查询对照](bench/results/20261127-strong-baselines/report.md)

从仓库根目录安装图像函数运行库，沿用快速上手生成的测量和证据帧导出函数。

```bash
python -m pip install -e ".[functions]"
python -m motionspec functions slide-out/spec.json -o slide-functions
python -m motionspec query slide-functions/functions.json --list
```

将下面的内容保存为 `query.json`，读取第 0 帧左上角像素和 250 ms 时跟踪点的横坐标。

```json
[
  {"id": "color", "op": "pixel_rgba", "frame_index": 0, "x": 0, "y": 0},
  {"id": "position", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 250}
]
```

```bash
python -m motionspec query slide-functions/functions.json query.json
```

像素结果为 `[41, 45, 49, 255]`，附帧时间和像素哈希。位置结果为 `261.0 source_px`，来源为 `motion_estimate`。同一套运行库也可以表达静态图片的像素、区域、轮廓和空间关系。[视频函数协议](docs/video-functions.md) · [静态图片指南](svsg/README.zh.md)

## 桌面与图像工具

Windows x64 用户可解压部署包并双击 `Start.cmd`，启动图片 API 和桌面 MCP 服务。[部署说明](deployment/windows/README.zh.md) · [生产加固与实测](docs/agent-production-20261123.md)

### 读取字幕和界面图标

安装本地 OCR 后，可从截图或已导出的源像素帧读取中英字幕和已知图标。视频结果带帧号、时间、像素哈希和源画面坐标。播放器底栏可用 `--player-toolbar` 限定搜索；自定义图标可用 `--icon-dir` 提供按类别命名的 PNG 样本。[评测与用法](docs/subtitle-icon-evaluation-20261122.md)

```bash
python -m pip install -e ".[overlays]" -e "./packages/motionspec"
motionspec overlays screenshot.png
motionspec overlays slide-functions/functions.json --player-toolbar
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

## 历史结果

早期测试使用六种模型配置，以只看画面为基线。6 个基础 Qt 动画的逐项得分为只看画面 55%、加视觉测量 92%、加 Qt 控件属性 94%。在 9 个较难场景上，三组分别为 58%、82%、100%。这些场景在开发中反复使用，分析器自身在这 9 段上的逐项正确率为 85%。在同一批场景上比较两个历史版本的视觉测量时，得分为 68% 和 86%。首页的留出集对照使用场景均分，并纳入密集帧和 OpenCV。[历史答卷与评分](bench/results/README.md) · [采样率消融](docs/evidence-ablation-20261116.md)

另一次无画面对照用两段动效测试两款模型，在都有有效答卷的 9 组配对中，只读参数答对 145/147 项，只读文字摘要答对 147/147 项。在基础 Qt 场景中，对照同帧记录的控件属性，导出函数的位置误差及进入、退出元素的尺寸误差均不超过 1 像素，透明度误差为零。[纯参数复测](docs/timing-protocol-retest-20261117.md) · [函数证据](docs/video-functions.md)

在 COCO8 的 8 张图片、30 个目标上，多视图复核使误报从 9 降至 2，漏检从 10 增至 16。在 4 段 MOT20 实拍录像的 8 个新取样时刻，本地行人检测多找回 3 个重遮挡行人（514 → 517，共 740 个标注），两版误报均为 36 个。[图像识别评测](docs/recognition-evaluation-20261116.md) · [行人配对评测](docs/pedestrian-focused-review-20261122.md)

## 目前做不到的

- 3D 画面和游戏画面只能给出统计量，没有可用的运动曲线。
- 元素相互覆盖时仍可能被合并成一个片段。四边跟踪在面板边缘持续可见时效果最好。
- 压缩视频里的颜色过渡会被 H.264 的色度抽样抹平。
- 不跨帧跟踪物体身份。片段编号只代表一段变化。
- 物体识别不保证找全。

## 安装与测试

```bash
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
