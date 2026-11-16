# motionspec

从录屏测量 2D 界面动效，输出时间、坐标、曲线和可追溯的原始关键帧。

数值规格可以和原始画面一起交给模型，用于回答与时间和运动有关的问题。

```
视频 ──> 哪里在动、何时动 ──> 怎么动（位移/缩放/旋转/淡入淡出）──> 缓动曲线 ──> spec.json + summary.md + 标注图
```

## 安装与使用

```bash
pip install -e .
```

```bash
motionspec analyze 录屏.mp4 -o out/
```

```bash
motionspec demo -o demo/
```

Python 里：

```python
from motionspec import analyze, check

spec = analyze("clip.mp4", "out/")          # 也接受图片目录或内存里的帧列表
verdict = check(spec, [{"segment": "s1", "field": "easing", "value": "ease-out"}])
```

常用参数：`--start` / `--end`（秒）截取时间段，`--max-dim` 分析分辨率（默认最长边 960），
`--min-area` 忽略更小的变化，`--max-segments` 最多保留几个片段。
输入图片目录时必须指定 `--fps`，帧按文件名自然排序，例如 `1.png`、`2.png`、`10.png`。
`--max-frames` 限制所选区间的帧数，`--max-crop-mb` 限制同一次任务并行缓存的裁剪帧（默认 150 MB）。
窗口重叠且无法共用预算时，会分批重读输入。裁剪缓存限额不包含解码器、拟合工作区和 Python 内存。

## 输出

| 文件 | 给谁看 | 内容 |
|---|---|---|
| `summary.md` | AI，首先读这个 | 每个片段一行：时间、时长、类型、变化量、缓动、置信度 |
| `spec.json` | 程序 | 格式版本 `motionspec/0.1`，每个片段最多 40 个采样点，记录被过滤或截短的范围 |
| `s1.png` … | AI，需要看图时 | 关键帧并排 + 轨迹线 + 实测点与拟合曲线对比 |
| `overview.png` | AI | 首帧上标出各片段所在区域 |
| `frame-<帧号>-<SHA-256>.png` | AI 或其他视觉系统 | 原分辨率的解码帧，含所选范围首尾及片段关键帧。`evidence_frames` 提供源帧号、毫秒时间戳、用途和 SHA-256 |

即使没有检测到片段，也会保留所选范围的首尾帧。用 `--no-sheets` 关闭标注图，用 `--no-raw-frames` 关闭原分辨率帧。自动处理时以本次 `spec.json` 中引用的文件为准。

片段有四种类型：

| 类型 | 含义 | 给出的数值 |
|---|---|---|
| `transform` | 一个元素在移动、缩放或旋转，可同时改变透明度 | 位移、均匀缩放、旋转、透明度，各通道有独立的时段和拟合 |
| `fade` | 画面近似首尾帧的混合 | 时长和混合进度曲线，可对应透明度或颜色过渡 |
| `instant` | 一帧内完成的突变 | 发生时刻 |
| `nonrigid` | 以上都解释不了：辉光、模糊、粒子、3D 画面 | 只有亮度范围、周期等统计量，置信度恒为 low |

缓动给三种说法：自由拟合的 `cubic-bezier(...)`；有回弹时的弹簧参数（阻尼比、角频率）；
以及最接近的命名曲线和它对应的时长，例如 "out-cubic over 400 ms"。

`transform` 内部有两条测量路径，结果里用 `model` 字段区分：

- **跟踪**（无 `model` 字段）：元素在首尾两帧里都存在，只是换了位置或大小。用特征点光流测量。
- **图层**（`"model": "layer"`）：进入或退出，即元素一边移动或缩放、一边淡入淡出。
  把首尾两帧中没有该元素的一帧当背景，每一帧解成"背景 + 透明度 × 变换后的元素"，
  同时得到位移、缩放、透明度三条曲线。同一区域里依次进入的多个元素会拆成各自的片段。

图层路径要处理一个光流处理不了的问题：透明度为 0 的那一端，元素的位置看不见。有两种补法，
通道里的字段会说明用了哪一种：

| 字段 | 含义 |
|---|---|
| `follows_opacity: true` | 位移与 (1 − 透明度) 成正比，两者共用一条曲线，看不见那端的位置由这个比例直接得出 |
| `extrapolated: true` | 两者曲线不同，看不见的部分不超过全程三成，由拟合曲线外推 |
| `lower_bound: true` | 看不见的部分太多，只报看得见的距离，真实值不小于它 |
| `visible_distance_px` | 实际量到的那一段距离 |

透明度是相对"完整显示那一帧"的比例：真实不透明度从 0 到 0.8 的元素，这里读出来是 0 到 1。

`spec.json` 里的 `ambient` 列出**持续变化的区域**（动态背景、视频、转圈、闪烁的光标）：
连续活动超过 3 秒（中间停顿不超过 0.6 秒）的格子不参与片段检测。否则它们会把时空上相邻的变化
全连成一个贯穿全片的候选。可用 `--ambient-ms` 调整时长，设为 0 关闭。

## 读数时要知道的事

- **时长有一帧左右的不确定度。** 60 帧录屏约 17ms，30 帧约 33ms。
- **`duration_ms` 是"看得见在动"的时长，通常比设计值短。** 缓出曲线最后 10%~15% 的时间里，
  物体总共只移动不到 1 像素，量不出来。要还原设计参数，看命名曲线那一项给的时长。
- **样本点少时曲线不唯一。** 150ms 的动效在 60 帧下只有 9 个点，好几条曲线都能贴合，
  这种片段会带一条 "few samples" 的备注。
- **可变帧率使用解码器报告的时间戳。** 遇到重复、倒序或非法值时，以源帧号和名义帧率重新扫描，然后按同一时间轴截取。
- **位移是特征锚点的像素位移。** `origin` 是拟合出的固定点，画面可能同时允许其他等价变换分解。用 `transform_reference` 查看拟合参考和特征锚点位移。片段级 `easing` 属于 `dominant` 通道，各通道的时间和曲线在 `channels` 内。

## 断言校验

AI 读完规格后给出的结论，可以再拿规格核对一遍，不符时返回实测值作为修正依据：

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
`duration_ms`、`easing`、`overshoot` 可指定 `channel`（`translate`、`scale`、`rotate`、`blend`、`opacity`）。`count` 只表示在当前检测条件下找到的片段数。请求中的 `tol` 会被拒绝，容差由校验器决定。
退出码 0 表示通过，1 表示断言错误，2 表示输入非法，3 表示证据不足。

## 实测精度

合成动效（已知真值，60 帧），测试见 `tests/test_synthetic.py`：

| 样本 | 真值 | 实测 |
|---|---|---|
| 滑动 | 300px，out-cubic，400ms | 299.8px，out-cubic，400ms |
| 淡入 | 线性，250ms | linear，250ms |
| 弹簧缩放 | 阻尼比 0.35，角频率 28，放大 1.667 倍 | 0.355，28.04，1.68 倍 |
| 先后两个元素 | 位移 300ms，淡入 200ms | 300ms，200ms，分成两个片段 |

本机复测滑动样本。经 H.264 压缩（CRF 23 和 32）后仍识别为 `out-cubic`、400 ms。再丢掉约五分之一帧且保留其余帧时间戳，仍识别为 `out-cubic`，拟合时长为 383.3 ms。

进入 / 退出类合成动效（60 帧），测试见 `tests/test_layer.py`：

| 样本 | 真值 | 实测 |
|---|---|---|
| 上移 + 淡入，同一条曲线 | 24px，decelerate，300ms | 23.9px，material-decelerate，300ms |
| 上移 + 淡入，曲线不同 | 位移 decelerate，透明度线性 | 23.7px material-decelerate，透明度 linear 300ms |
| 弹出层缩放 + 淡入 | 0.9 → 1，300ms | 0.901 → 1，300ms |
| 下移 + 淡出 | 24px，ease-in，250ms | 23.9px，ease-in，250ms |
| 三个相同的列表项依次进入 | 起点 167/233/300ms，各 20px | 3 个片段，起点 167/233/300ms，各 19.9~20.0px |

真实 Qt 控件渲染（PySide6，带中文文字、圆角、`QGraphicsOpacityEffect`，位置为整数像素，屏幕缩放 1.5 倍，
下表为设备像素）。用 `tools/qt_scenes.py` 生成、`tools/qt_compare.py` 对比，素材不入库：

| 场景 | 真值 | 实测 |
|---|---|---|
| 卡片平移 | 300px，OutCubic，300ms | 300.0px，out-cubic，300ms |
| 提示条进入 | 上移 36px + 淡入，OutCubic，250ms | 36.1px，out-cubic，250ms |
| 提示条退出 | 上移 24px + 淡出，InCubic，200ms | 23.1px，in-cubic，200ms |
| 纯淡入 | InOutQuad，300ms | in-out-quad，300ms |
| 三行依次进入 | 各 24px，OutCubic，200ms，起点 167/217/267ms | 24.7px，out-cubic，183~200ms，起点 167/217/267ms |
| 卡片平移与另一处淡入同时发生 | 90px InOutCubic 400ms；淡入 Linear 200ms | 89.9px in-out-cubic 400ms；linear 200ms |

同样的 Qt 场景压成 H.264：CRF 23 时结论不变；CRF 30 时"三行依次进入"里有两行的位移读成 21.4px。

这些素材都是逐帧渲染的，帧间隔严格均匀。真实录屏还会有丢帧和录制软件自身的延迟，这部分没有用界面类录屏验证过。

## 已知限制

- 测量面向 2D 界面动效。复杂画面可能被标为 `nonrigid`，也可能碰巧符合 2D 运动模型。项目未分析音频、字幕和对象语义。
- 一个片段只描述一个整体运动。已经存在的多个元素贴在一起同时动，会被合成一个片段，
  方向不同时会被判成 `nonrigid`（进入 / 退出的多个元素能拆开）；
  同一位置前后相隔不到约 8 帧的两段动效也会被合并。
- 图层路径只解平移和等比缩放，不解旋转；元素须在首尾某一帧里完全不存在。
  从半透明变到不透明、带遮罩变暗的模态框、进入时背景本身也在变，这几种情况没有验证过。
- 图层路径只在不超过 3 秒、180 帧的窗口上尝试。
- 画在持续变化区域上面的动效（动态背景上的半透明面板）分离不出来，这类片段会标为低置信。
- 持续超过 `--ambient-ms` 的慢动效（长进度条）会被归入 `ambient` 而不是片段。
- 往返运动（抖动、脉冲）只报峰值和峰值时刻，不拟合曲线。
- 鼠标指针移动会被当成一个小元素的位移。
- 默认单窗口最多 300 帧，受裁剪预算约束。超过上限的片段只测量取到的窗口，标为低置信并记录 `truncated`。

## 结构

```
motionspec/
  frames.py    帧来源：视频 / 图片目录 / 内存帧，统一时间戳
  segment.py   第一遍：颜色帧差 → 粗网格 → 剔除持续变化的格子 → 增量时空连通域
  motion.py    第二遍：先试图层模型，再走特征点跟踪、首尾混合度、分类
  layer.py     图层模型：进入 / 退出时的平移、缩放、透明度联合求解
  fit.py       缓动拟合：贝塞尔、命名曲线、弹簧（只用 numpy）
  annotate.py  标注图
  spec.py      规格版本、结构校验、文字摘要
  claims.py    断言校验
  synth.py     合成动效（测试素材和演示）
  cli.py       命令行
tools/
  qt_scenes.py   用真实 Qt 控件渲染已知真值的动效（需要 PySide6）
  qt_compare.py  把分析结果和真值并排打印
```

依赖只有 numpy 和 opencv-python。

## 与 SVSG 共用图像和时间函数

可选安装本机 SVSG 源码的 `[functions]` 扩展后，启用统一函数导出：

```bash
python -m pip install -e "<svsg目录>[functions]" -e "<motionspec目录>"
motionspec analyze clip.mp4 -o out --functions --sequence-id clip-001 --evidence-every-ms 500
motionspec functions out/spec.json -o functions --sequence-id clip-001
```

`functions.json` 使用 `svsg.video-functions/1`，内部图像保持 `svsg.functions/1`，精确像素
编码仍为 `svsg.raster/1`。每个保留帧复用 SVSG 的像素、轮廓和空间计算；相邻保留帧输出
RGBA 差分统计，运动采样输出分段线性函数，拟合结果输出可求值的贝塞尔/弹簧参数。
实现只维护在 SVSG，MotionSpec 的 `functions.py` 负责适配和引用绑定。

`--evidence-every-ms` 添加周期证据，默认仅保留原有关键帧。函数导出默认上限 64 帧、
64×1024² 总像素（单帧 ≤ 16×1024²），帧数可用 `--function-max-frames` 调整。
缺帧、错误哈希、尺寸不符、倒序时间和预算超限会报错，不静默丢失证据。
完整像素载荷保存在 `raster-<文件SHA-256>.json`，另保留像素哈希，相同像素复用文件，
不需把载荷发给模型。

模型可以只提交 JSON 函数参数，例如保存以下内容为 `query.json`：

```json
{"id":"p","op":"pixel_rgba","frame_index":0,"x":100,"y":80}
```

```bash
motionspec query out/functions.json query.json
```

另支持 `mean_rgba`、`delta_rgba` 和 `motion_value`。帧索引必须在本次清单内。
Python 入口为 `svsg.video_functions.VideoRuntime`；现有识别结果可用 `attach_recognition`
在校验序列、帧号、时间、文件/像素哈希和尺寸后接入。它不会自动调用模型。

像素函数精确复现保留帧的解码图像；采样点之间的运动是插值，拟合值带 RMSE，不外推到
片段范围之外。运动片段不是物体轨迹，帧差不代表物体身份或动作。自然视频的跟踪和切镜
语义仍需后续验证。详细协议说明维护在 SVSG 的 `docs/video-functions.md`。

本地无网络验证可运行 `python tools/validate_functions.py --out checks`；可选 `--photo`
添加真实照片生成的受控平移/静止视频。测试只验证编码与函数复算，不代表自然视频识别准确率。

## 还没做

- 增量发布与模型编排，当前 CLI 等分析完成后写规格
- Qt 探针：直接采样控件属性，跳过视觉反推，输出同一种规格
- 两份规格的对比（参考动效 vs 自己的实现）
