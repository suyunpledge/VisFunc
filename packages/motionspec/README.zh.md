# VisFunc-Motion

[English](README.md) · 中文 · [返回 VisFunc](../../README.zh.md)

VisFunc-Motion 从界面录屏里测出动画：哪个元素在动、什么时候开始、移动了多远、尺寸和透明度怎么变、用了什么缓动曲线。
如果被测的是你自己的 Qt 应用，也可以不录屏，直接逐帧读控件属性。结果是一份数值规格、一页摘要和几张标注图，
可以连同原始画面一起交给模型，也可以导出成能按时间查询的函数。

代码在 `packages/motionspec/`，Python 包名和命令是 `motionspec`。只依赖 numpy 和 opencv-python；
导出函数时还需要同仓库的 [VisFunc-Image](../../svsg/README.zh.md)。

```
录屏 ─▶ 哪里在动、何时动 ─▶ 怎么动（位移、缩放、旋转、宽高、透明度） ─▶ 缓动曲线 ─▶ spec.json、summary.md、标注图、函数
```

## 安装与使用

在本目录：

```bash
pip install -e .
```

```bash
motionspec analyze 录屏.mp4 -o out/
```

```bash
motionspec demo -o demo/
```

在仓库根目录同时装上 VisFunc-Image，就可以导出函数并查询：

```bash
python -m pip install -e ".[functions]" -e "./packages/motionspec"
motionspec analyze clip.mp4 -o out --functions --evidence-every-ms 500
motionspec query out/functions.json --list
```

Python 里：

```python
from motionspec import analyze, check

spec = analyze("clip.mp4", "out/")   # 也接受图片目录或内存里的帧列表
verdict = check(spec, [{"segment": "s1", "field": "easing", "value": "ease-out"}])
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--start`、`--end` | 只分析这段时间（秒） |
| `--fps` | 输入是图片目录时必须给；帧按文件名自然排序，`1.png`、`2.png`、`10.png` |
| `--max-dim` | 分析时的最长边，默认 960 |
| `--min-area`、`--max-segments` | 忽略更小的变化；最多保留几个片段 |
| `--ambient-ms` | 连续变化多久算"持续变化的区域"，默认 3000，设 0 关闭 |
| `--max-frames`、`--max-crop-mb` | 所选区间的帧数上限；裁剪帧缓存上限，默认 150 MB |
| `--no-sheets`、`--no-raw-frames` | 不输出标注图；不输出原分辨率证据帧 |
| `--functions`、`--evidence-every-ms`、`--function-max-frames` | 导出函数；按时间间隔补充证据帧；函数包的帧数上限 |

### 等界面稳定（给 Agent 用）

`analyze` 要拟合曲线，一段 0.6 秒的片段要算 1～3 秒，不适合 Agent 每次点击后等待。`settle` 只回答"停下来没有、哪里变了"：边收帧边判断，静止 `--quiet-ms`（默认 200）后返回 `settled`，`--expect-ms` 内没变化返回 `unchanged`，并列出相对第一帧净变化的区域。每帧判断约 6 ms。

```bash
motionspec settle --screen --region 0 0 1920 1080 --read-text   # 点击后运行，读出变化区域里的文字
motionspec settle recording.mp4                                  # 也接受录好的视频或图片目录
```

一直在变的区域（视频、动画背景）满 `--ambient-ms` 后不再算作变化。加载圈和它们分不开，需要等加载完成时设 `--ambient-ms 0`；时钟之类用 `--ignore` 排除。

给 Agent 用时，更方便的是常驻的 MCP 服务 `motionspec agent-mcp`：它保留截屏流、OCR 模型和学到的背景区域，提供 `wait_for_settle`、`find_on_screen`（先查 Windows UI Automation，再查 OCR）、`read_screen_text`、`list_ui_elements` 和 `screenshot`；加 `--allow-window 标题` 后还有 `click_element`、`set_field_text`。默认走 stdio，`--http 主机:端口` 改为 HTTP，接 Pledge Keystone 时再加 `--no-images`。截屏装了 mss（`pip install -e ".[capture]"`）时更快。[用法与评测](../../docs/agent-screen-20261122.md)

## 输出

| 文件 | 内容 |
|---|---|
| `summary.md` | 每个片段一行：可见变化区间、估计的设计时间、类型、变化量、缓动、置信度。先读这个 |
| `spec.json` | 完整规格 `motionspec/0.1`，每个片段最多 40 个采样点，记录被过滤或截断的部分 |
| `s1.png` … | 每个片段一张：关键帧并排、轨迹线、实测点与拟合曲线 |
| `overview.png` | 首帧上标出各片段的位置 |
| `frame-<帧号>-<SHA-256>.png` | 原分辨率解码帧：所选范围首尾、各片段关键帧。`evidence_frames` 记录帧号、时间、用途和哈希 |
| `functions.json`、`raster-*.json` | 用 `--functions` 时生成的函数包和像素数据 |

没有检测到任何变化时，也会保留首尾两帧。

### 片段类型

| 类型 | 含义 | 给出的数值 |
|---|---|---|
| `transform` | 一个元素在移动、缩放、旋转或改变宽高，可以同时改变透明度 | 每个通道各自的时间和拟合 |
| `fade` | 画面接近首尾两帧的混合：淡入淡出、颜色过渡、遮罩变暗 | 时长和混合进度曲线 |
| `instant` | 一帧之内完成的突变 | 发生时刻 |
| `nonrigid` | 以上都解释不了：辉光、模糊、粒子、3D 画面 | 只有亮度范围、周期等统计量，置信度一律为 low |

### 怎么测出来的

第一遍找"哪里在动、何时动"。每帧和参考帧比较颜色差，落到粗网格上；某个格子连续几帧没变才刷新它的参考，所以很慢的淡入也能检出。
连续变化超过 3 秒的格子（动态背景、视频、转圈、闪烁的光标）记入 `ambient`，不参与片段检测，否则它会把附近所有变化连成一个贯穿全片的片段。
剩下的变化按时空连通域分成若干窗口。

第二遍逐个窗口解释它，按下表顺序尝试。结果里的 `model` 字段说明用了哪一种：

| 路径 | `model` | 适用情况 | 做法 |
|---|---|---|---|
| 遮罩分离 | 无（拆成两个片段） | 整个窗口按同一条曲线变暗或变色，上面还有元素在动（模态框） | 先把遮罩报成一个 `fade`，再从每个像素里扣掉遮罩的变化，测上面的元素 |
| 纯色块 | `solid-box` | 只沿一个方向改变尺寸的色块：进度条、标签指示条 | 逐帧找出那块颜色的外接框；`size` 给出宽高起止，`anchor` 说明从哪条边长出来，两条边都动时另给 `translate` |
| 图层 | `layer` | 进入或退出：元素一边移动或缩放，一边淡入淡出 | 把首尾帧中没有该元素的一帧当背景，每帧解成"背景 + 透明度 × 变换后的元素"，同时得到位移、缩放、透明度和亚像素宽高 |
| 跟踪 | 无 | 首尾两帧里都有的元素换了位置或大小 | 特征点光流加相似变换 |
| 多体 | `multibody` | 一块区域里几个元素朝不同方向平移：列表重排、两项对调 | 特征点按整条轨迹分组，每组用模板前后双向跟踪，穿过互相遮挡的帧 |
| 混合、突变、非刚性 | 无 | 其余情况 | 见上面的片段类型 |

本来就在、只是挪了位置的元素不走图层路径，包括纹理很少的纯色卡片，因为它也能在数值上被硬解成"淡入"。

图层路径有一个光流没有的问题：透明度为 0 的那一端，元素的位置看不见。通道里的字段说明怎么补的：

| 字段 | 含义 |
|---|---|
| `follows_opacity: true` | 位移和 (1 − 透明度) 成正比，两者用同一条曲线，看不见那一端的位置由比例算出 |
| `extrapolated: true` | 两者曲线不同，看不见的部分不超过全程三成，由拟合曲线外推 |
| `lower_bound: true` | 看不见的部分太多，只报看得见的距离，真实值不小于它 |
| `visible_distance_px` | 实际量到的那一段距离 |

图层的透明度是相对"完整显示那一帧"的比例。真实不透明度从 0 到 0.8 的元素，读出来是 0 到 1。

### 缓动与时间

每个通道分别拟合三种曲线：自由的 `cubic-bezier(...)`；欠阻尼弹簧（有回弹时）；以及最接近的命名曲线（CSS、Material、easings.net 常用曲线）。
命名曲线对齐时允许起点提前、终点推后，因为缓入缓出的头尾每帧移动不到一个像素，量不出来。

所以每个片段有两套时间，格式是 `svsg.motion-timing/1`：

- `visible_change`：实际看得出变化的区间，片段顶层的 `t_start_ms`、`t_end_ms`、`duration_ms` 也是这个意思；
- `estimated_design`：对齐最贴合的命名曲线后估计的设计起点和时长，附 RMSE 和对齐依据。拟合不可靠（RMSE 超过 0.05）、
  片段被截断或估计起点早于 0 时为 `null`，并在 `design_unavailable_reason` 里写明原因。

读数时注意：

- 时间有大约一帧的误差，60 帧录屏约 17 毫秒，30 帧约 33 毫秒。
- `duration_ms` 通常比设计时长短。要还原设计参数，看 `estimated_design`。
- 样本点少时曲线不唯一。150 毫秒的动画在 60 帧下只有 9 个点，好几条曲线都贴得上，这种片段会带 "few samples" 备注。
- 可变帧率的视频用解码器给的时间戳；时间戳重复、倒序或非法时，改用帧号和名义帧率重新扫描。
- 跟踪路径的位移是特征锚点的位移，不一定是元素中心。`transform_reference` 说明参考点是什么。

## 函数与查询

`--functions`（或对已有结果运行 `motionspec functions out/spec.json -o functions`）会生成 `functions.json`，
格式是 `svsg.video-functions/1`，由 VisFunc-Image 编译。每个保留帧有可还原的像素函数和区域、轮廓描述；相邻保留帧之间有差分统计；
每个运动片段有采样函数（点间线性插值）和拟合函数（贝塞尔或弹簧）。

先看目录。目录只列出能查什么，比完整清单小得多（一个 132 KB 的清单，目录约 4 KB）：

```bash
motionspec query out/functions.json --list
```

再查询。参数文件可以是一个调用，也可以是最多 1000 个调用的列表；某个调用出错只在它自己的结果里报错：

```json
[
  {"id": "p", "op": "pixel_rgba", "frame_index": 0, "x": 100, "y": 80},
  {"id": "x", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 300},
  {"id": "e", "op": "motion_value", "segment_id": "s1", "name": "translate.progress", "t_ms": 300, "mode": "fitted"}
]
```

```bash
motionspec query out/functions.json query.json
```

还支持 `mean_rgba` 和 `delta_rgba`。帧号、时间和函数名都要来自本次的目录；时间超出函数的支持范围会报错，不外推。

各条路径导出的采样函数：

| 路径 | `x`、`y` | `width`、`height` | `opacity` | `*.progress` |
|---|---|---|---|---|
| 跟踪 | 有（特征锚点） | 无 | 无 | 有 |
| 图层 | 有（元素中心） | 有（亚像素；外接框碰到分析窗口边时不给） | 有 | 有 |
| 多体 | 有（点组中心） | 无 | 无 | 有 |
| 纯色块 | 有（框中心） | 有 | 无 | 有 |
| Qt 探针 | 有 | 有 | 有 | 有，另有 `color.r/g/b/a`、`property.value` |

导出默认最多 64 帧，可用 `--function-max-frames` 调整。缺帧、哈希不符、尺寸不符、时间倒序和超出预算都会报错。
无网络的复算检查：`python tools/validate_functions.py --out checks`，加 `--photo` 用一张真实照片生成受控的平移视频。
它只验证编码与复算，不代表自然视频上的识别准确率。协议细节见 [视频函数说明](../../docs/video-functions.md)。

## Qt 探针

被测的是自己的 Qt 应用时，可以直接读控件属性，不从像素反推。QWidget 和 QML（QQuickItem）都支持。分两部分：

| | 文件 | 依赖 | 做什么 |
|---|---|---|---|
| 应用内 | `motionspec/qt_probe.py` | 只要 PySide6 或 PyQt6 | 逐帧记录每个元素的位置、宽高、透明度、缩放、旋转、可见性、颜色，以及点名的自定义数值属性 |
| 离线 | `motionspec/trace.py` | numpy、OpenCV | 把轨迹切成片段、拟合缓动，输出同样的 `spec.json` 和 `summary.md` |

应用内那个文件不引用包里的其他模块，可以直接拷进被测应用，或者按路径加载（包的 `__init__` 会导入 numpy，被测应用的环境里不一定有）：

```python
from qt_probe import attach_from_env   # 或按路径加载 motionspec/qt_probe.py
attach_from_env()                      # 设了 MOTIONSPEC_TRACE=轨迹路径 才生效，应用退出时写出
```

```bash
motionspec trace trace.json -o probe
motionspec functions probe/spec.json -o probe/functions
```

需要自己控制时机时用 `Recorder`：`start()`、`stop()` 跟着应用的帧走（QML 窗口按渲染帧，其余用定时器），或者每推进一帧调用一次 `sample(t_ms)`。

颜色取 QML 的 `color` 属性，或控件样式表里写的 `background` / `background-color`。自定义属性要点名：
`Recorder(properties=("hoverAmount", "emphasis"))`，或环境变量 `MOTIONSPEC_PROPS=hoverAmount,emphasis`。
它们各自成为 `fade` 片段，通道是 `color`（起止 RGBA，0 到 1）或 `property`（属性名和起止值）。

规格里 `model` 为 `qt-probe`，每个片段带 `element`（类名、对象名、文字）和录制内的 `element_id`。同时同样移动的几个元素各自保留片段，
由 `motion_groups[].segment_ids` 关联。从 0 尺寸长出来的元素（展开的面板、进度条）报尺寸变化，不会把"0 尺寸时隐藏"当成透明度变化。
坐标是设备像素（逻辑像素乘屏幕缩放），和同一窗口的截图、录屏对得上。导出的函数格式是 `svsg.probe-functions/1`，没有像素帧。

| | 视觉测量 | Qt 探针 |
|---|---|---|
| 需要什么 | 一段录屏 | 能改被测应用的启动代码 |
| 位移、透明度 | 测量值，误差在亚像素到 1 像素左右 | 属性的真实值 |
| 知道是哪个控件 | 只有区域坐标 | 类名、对象名、文字 |
| 透明度为 0 时的位置 | 靠推算 | 照常读到 |
| 能看到什么 | 画面上的一切，包括着色器和别人的应用 | 只有记录的属性 |
| 缓动曲线 | 拟合 | 也是拟合（看不到动画对象本身），但用的是精确的进度值 |

探针每帧要遍历整棵界面树，元素很多时会有可感知的开销，适合调试时开。还没有在大型应用上长时间挂过。

## 断言校验

模型读完规格给出的结论，可以拿规格核对：

```bash
motionspec check out/spec.json claims.json
```

```json
[
  {"field": "count", "value": 2},
  {"segment": "s1", "field": "direction", "value": "right"},
  {"segment": "s1", "field": "duration_ms", "value": 400, "easing": "out-cubic"},
  {"segment": "s1", "field": "easing", "value": "out-cubic"}
]
```

支持的字段：`count`、`kind`、`duration_ms`、`easing`、`overshoot`、`distance_px`、`direction`、`scale_to`。
`duration_ms`、`easing`、`overshoot` 可以指定 `channel`（`translate`、`scale`、`rotate`、`blend`、`opacity`、`size`）。
容差由校验器决定，请求里带 `tol` 会被拒绝。`count` 只代表当前检测条件下找到的片段数。
退出码：0 通过，1 不符，2 输入非法，3 证据不足。

## 精度

合成动画（已知真值，60 帧）：

| 样本 | 真值 | 测得 |
|---|---|---|
| 滑动 | 300 px，out-cubic，400 ms | 299.8 px，out-cubic，400 ms |
| 淡入 | 线性，250 ms | linear，250 ms |
| 弹簧缩放 | 阻尼比 0.35，角频率 28，放大 1.667 倍 | 0.355，28.04，1.68 倍 |
| 上移 + 淡入 | 24 px，decelerate，300 ms | 23.9 px，material-decelerate，300 ms |
| 三个相同的列表项依次进入 | 起点 167 / 233 / 300 ms，各 20 px | 3 个片段，起点相同，各 19.9 到 20.0 px |

用真实 Qt 控件逐帧渲染的场景（带中文文字、圆角、`QGraphicsOpacityEffect`，屏幕缩放 1.5 倍，下表为设备像素）：

| 场景 | 真值 | 测得 |
|---|---|---|
| 卡片平移 | 300 px，OutCubic，300 ms | 300.0 px，out-cubic，300 ms |
| 提示条进入 | 上移 36 px 并淡入，OutCubic，250 ms | 36.1 px，out-cubic，250 ms |
| 提示条退出 | 上移 24 px 并淡出，InCubic，200 ms | 23.1 px，in-cubic，200 ms |
| 纯淡入 | InOutQuad，300 ms | in-out-quad，300 ms |
| 卡片平移，同时另一处淡入 | 90 px InOutCubic 400 ms；Linear 200 ms | 89.9 px in-out-cubic 400 ms；linear 200 ms |

另外 9 个更难的 Qt 场景（模态框、展开、侧栏、滚动、指示条、对调、回弹、变色、进度条）用 `tools/qt_hard_compare.py` 自动判分：
视觉测量逐项正确率 85%，探针 100%。导出的函数逐帧求值，和探针记录的真实属性比较（`tools/function_accuracy.py`）：
基础场景位置误差不超过 1.0 像素，进入/退出元素宽高误差不超过 1.0 像素，透明度误差为 0。

速度：1350 × 840、不到一秒的片段，分析加函数导出大约 2 到 4 秒；上面 9 个场景合计约 24 秒（同一台开发机）。

这些素材都是逐帧渲染的，帧间隔严格均匀；真实录屏的丢帧和录制延迟验证得很少。9 个较难场景在开发中反复使用过，不算独立验证。

## 模型实测

`bench/` 是对比脚本：同一段动画只给模型视频，或者同时给出测量结果，各自答得多准。密钥从本机 Forge 配置读取，只在进程内使用。
`bench/canary.py` 先确认模型真的收到了视频。

| 场景 | 只给视频 | 加视觉测量 | 加 Qt 探针 |
|---|---|---|---|
| 6 个基础场景，6 个模型 | 55% | 92% | 94% |
| 9 个较难场景，6 个模型 | 58% | 82% | 100% |

GPT 两个模型收到的是带时间戳的截图，因为所用密钥的组织没有开通视频输入。完整结果、判分规则和原始回答见
[实测记录](bench/results/README.md)。`bench/pedestrian_*.py` 是在 MOT17 街景视频上做行人检测的实验，还在进行中。

## 已知限制

- 面向 2D 界面动画。复杂画面可能被标为 `nonrigid`，也可能碰巧符合某个 2D 模型。不分析音频、字幕和物体语义。
- 一个片段描述一个整体运动。贴在一起同时同向移动的几个元素会合成一个片段。方向不同时走多体路径，只处理平移；
  被完全遮挡的那几帧是插值，那个元素的时长和曲线不可靠。同一位置前后相隔不到约 8 帧的两段动画会被合并。
- 布局重排测不准。面板展开推动下方几行、侧栏收起拉动内容时，变尺寸的那块和被推动的那块落在同一个窗口里，
  前者会被测成位移，或者两者合成一个片段。纯色块路径只在那块颜色单独成片、没被窗口边裁到时生效。
- 图层路径只解平移和等比缩放，不解旋转；元素必须在首尾某一帧里完全不存在；只在不超过 3 秒、180 帧的窗口上尝试。
- 画在持续变化区域上的动画（动态背景上的半透明面板）分离不出来，会标为低置信。持续超过 `--ambient-ms` 的慢动画（很长的进度条）会归入 `ambient`。
- 往返运动（抖动、脉冲）只报峰值和峰值时刻，不拟合曲线。
- H.264 的色度抽样会把颜色过渡的曲线抹平。
- 鼠标指针移动会被当成一个小元素的位移。
- 单个窗口默认最多 300 帧。超过的片段只测量取到的部分，标为低置信并记录 `truncated`。

## 代码结构

```
motionspec/
  frames.py       帧来源：视频、图片目录、内存帧；统一时间戳
  segment.py      第一遍：颜色帧差 → 粗网格 → 剔除持续变化区域 → 时空连通域
  motion.py       第二遍：遮罩分离、纯色块、图层、跟踪、多体、分类
  layer.py        图层模型：进入/退出时的位移、缩放、透明度联合求解
  fit.py          缓动拟合：贝塞尔、命名曲线、弹簧（只用 numpy）
  timing.py       两套时间的统一计算
  annotate.py     标注图
  spec.py         规格格式、校验、文字摘要
  claims.py       断言校验
  functions.py    函数导出、查询目录、批量查询（调用 VisFunc-Image）
  settle.py       等界面稳定：边收帧边判断，列出变化区域；实时截屏
  agent.py        给 Agent 的屏幕工具和 MCP 服务
  qt_probe.py     Qt 探针的应用内部分
  trace.py        Qt 探针的离线部分；trace_contract.py 校验轨迹格式
  synth.py        合成动画，用于测试和演示
  cli.py          命令行
tools/
  qt_scenes.py、qt_hard_scenes.py     用真实 Qt 控件渲染已知真值的场景，同时让探针取样（需要 PySide6）
  qt_compare.py、qt_hard_compare.py   视觉测量、探针和真值对比，后者自动判分
  function_accuracy.py                导出函数后逐帧求值，和探针的真实属性比较，并计时
  qml_probe_demo.py                   在实时运行的 QML 窗口上试探针
  validate_functions.py               无网络的函数编码与复算检查
bench/
  run.py、run_hard.py、report*.py      模型实测与汇总
  providers.py、client.py、canary.py   读取本机模型配置、调用封装、确认收到视频
  pedestrian_*.py                     MOT17 行人实验（进行中）
  agent_*.py                          Agent 场景：桌面截图与读字、Chrome 网页、真实窗口上的 MCP 调用
```

测试在 `tests/`，在本目录运行 `python -m pytest`。

## 还没做

- 分析过程中边分析边发布结果，目前要等整段分析完才写出规格
- 布局重排的视觉测量
- 两份规格的对比，例如参考动画和自己的实现、视觉测量和探针

## 许可

[MIT](../../LICENSE)
