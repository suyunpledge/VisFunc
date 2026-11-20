# VisFunc

[English](README.md) · 中文

VisFunc 把图片和视频里能测量的东西先用程序测出来，再交给 AI。

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

## 结果汇总

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
不算独立验证。测试数据、判分规则和原始回答见 [模型实测记录](packages/motionspec/bench/results/README.md)、
[纯参数复测](docs/timing-protocol-retest-20261117.md)、[采样率消融](docs/evidence-ablation-20261116.md)、
[识别评测](docs/recognition-evaluation-20261116.md)。
另有一组在 MOT17 真实街景视频上做行人检测的实验正在进行。

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

MIT
