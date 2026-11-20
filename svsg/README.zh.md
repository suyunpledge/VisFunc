# VisFunc-Image

[English](README.md) · 中文 · [返回 VisFunc](../README.zh.md)

VisFunc-Image 把一张图片编译成一组可以计算的参数：每个像素的颜色、同色区域的范围和轮廓、区域之间的位置关系。
模型可以只读这些参数回答问题，服务再把模型给出的每条结论和参数核对一遍。它还负责多视图物体识别，
以及 VisFunc 里视频函数的统一格式与查询求值。

代码在仓库的 `svsg/` 目录，Python 包名和命令是 `svsg`（这是项目原来的名字，结构化视觉语义网关）。

## 一个例子

一张 8 × 4 像素的图，左半红、右半蓝。编译后得到两个区域，各 16 像素，包围框分别是 `[0, 0, 4, 4]` 和 `[4, 0, 8, 4]`，
两者的包围框 IoU 是 0，共享边长 4 像素。只看这些参数就能回答"红色和蓝色区域沿一条边相接，没有重叠"，
而且任何人拿到同一张图都能算出同样的结果。

照片里的人或物体不能这样直接算出来。那需要识别，见下文"物体识别"。

## 能做什么

| 用途 | 入口 |
|---|---|
| 把单张图片保存为可无损还原的像素函数 | `python -m svsg.image_functions encode input.png raster.json`，还原用 `decode` |
| 编译图片的区域、轮廓和空间关系 | `POST /v1/compile-functions`，或 Python 里的 `svsg.image_functions` |
| 让模型根据参数回答问题，并核对它的结论 | `POST /v1/analyze-image`、`POST /v1/analyze` |
| 识别图片里的物体，多个视图交叉确认 | `POST /v1/recognize-image` |
| 查询视频的保留帧和运动函数 | `svsg.video_functions.VideoRuntime`，命令行经由 `motionspec query` |

## 图像函数

图片按 EXIF 方向转正后解码为 RGBA8，编码成 `svsg.raster/1` 光栅函数，可以逐字节还原。在此基础上计算：

- 同色连通区域：面积、包围框、重心、周长；
- 闭合轮廓：外环和孔洞；
- 区域之间的关系：`left_of`、`right_of`、`above`、`below`、`bbox_contains`、`bbox_iou`、`center_distance_px`、
  `shared_edge_px`、`touching`。

颜色太多的图（照片、渐变）另外在量化网格上分析，网格尺寸和量化方法（`box_median_cut_16` 或 `box_octree_rgba_16`）写在结果里，
网格上的长度单位是格，不是像素。整份描述的格式是 `svsg.functions/1`。

图像查询只有两种：`pixel_rgba`（一个像素的颜色）和 `mean_rgba`（一个矩形的平均颜色）。都在本地求值，不执行任何代码字符串。

## 问答与核对

`/v1/analyze-image` 按以下顺序处理一次提问：

1. L1 把图片编译成中间表示（IR）。`SVSG_DETECTOR=functions` 时 IR 就是上面的图像函数；也可以换成 YOLO 检测器（需要 `.[ml]`）。
2. L1.5（可选）对标为需要验证的实例调用视觉模型复核。它有独立的超时；超时的实例按最低置信度处理并记录下来，其余实例照常继续。
3. L3 让文本模型只根据 IR 回答，并以结构化断言的形式给出结论，例如"类别是什么""有几个""谁在谁左边"。
4. L2 把每条断言和 IR 核对：类别要和检测一致；数量差一个也不通过；方位关系用同一套几何代码重算。
5. 不通过时，带着具体的不符之处让模型重答一次；还不通过就转人工复核。

状态和错误码：

| 状态 | 错误码 | 处理 |
|---|---|---|
| S2 输入非法 | E1001–E1003 | 直接返回，不调用模型 |
| S3 依赖缺失 | E2001–E2003 | 软重试一次 |
| S4 冲突或核对不通过 | E3001、E3002 | 带修正反馈重答 |
| S5 超时 | E4001、E4003 | L1 或 L3 超时，请求中止 |
| L1.5 超时 | E4002 | 受影响实例降为最低置信度，其余继续 |
| S6 人工复核 | E5001 | 原样返回，不吞掉 |

## 物体识别

`/v1/recognize-image` 每次只给模型一张图，默认最多调用 5 次：

1. 主模型看全图；
2. 复核模型看水平镜像，框映射回原图；
3. 两个模型分别看两个重叠的半图；如果出现面积小于全图 2% 的候选，换成该候选附近的放大图；
4. 还有预算时，优先把最小的未确认候选再放大确认一次。

同一类别、IoU ≥ 0.5 的框视为同一个物体。确认需要两个不同视图的支持；配置了两个模型时，两个模型都要出现在支持里。
同一模型对同一张编码图重复给出的结果不算独立证据。结果分两类：`corroborated`（已确认）和 `needs_review`（证据不足，保留待查），
并始终标 `coverage=not_exhaustive`，因为仍可能漏检。识别结果可以绑定到 VisFunc-Motion 导出的视频证据帧，
绑定时会核对序列、帧号、时间、尺寸和哈希。

COCO8 上（8 张图，30 个目标）复核把误报从 9 个降到 2 个，漏检从 10 个增到 16 个。
详见 [识别协议](../docs/recognition.md) 和 [评测记录](../docs/recognition-evaluation-20261116.md)。

## 视频函数

VisFunc-Motion 的测量结果在这里变成统一的函数包 `functions.json`：

| 格式 | 内容 |
|---|---|
| `svsg.video-functions/1` | 保留帧的图像函数（每帧复用上面的计算）、相邻保留帧的 RGBA 差分统计、运动片段的采样函数和拟合函数 |
| `svsg.probe-functions/1` | 来自 Qt 属性轨迹，没有像素帧，坐标是窗口设备像素 |
| `svsg.motion-timing/1` | 每个片段的两套时间：实际看得到变化的区间，以及对齐命名曲线后估计的设计区间 |

运动函数有三种：采样点之间线性插值、三次贝塞尔、欠阻尼弹簧。每个函数都带单位和时间支持范围，范围外求值会报错，不外推。
查询操作：`pixel_rgba`、`mean_rgba`、`delta_rgba`（两个保留帧之间的像素差）和 `motion_value`（采样值或拟合值）。

导出时默认最多 64 帧、总像素不超过 64 × 1024²（单帧不超过 16 × 1024²）。缺帧、哈希不符、尺寸不符、时间倒序或超出预算都会直接报错。
完整像素保存在按内容哈希命名的 `raster-*.json` 里，不需要发给模型。协议细节和精度记录见 [视频函数说明](../docs/video-functions.md)。

```python
from svsg.video_functions import VideoRuntime

runtime = VideoRuntime.load("out/functions.json")
runtime.call({"id": "x", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 250})
```

## HTTP 服务

```bash
python -m pip install -e ".[api,llm]"
python -m svsg
```

默认监听 `127.0.0.1:3002`。

| 接口 | 作用 |
|---|---|
| `POST /v1/compile-functions` | 上传 `file`，可带 `queries`；返回图像函数，不调用模型 |
| `POST /v1/analyze-image` | 上传 `file` 和问题 `query`；按上面的流程作答 |
| `POST /v1/analyze` | 提交已有的 `ir` 和新问题 |
| `POST /v1/recognize-image` | 上传 `file`，可带类别列表 `labels` 和视频帧引用 `frame` |
| `GET /healthz` | 健康检查，不需要鉴权 |

配置写在 `.env`，模板见 [.env.example](../.env.example)。常用项：

| 配置 | 用途 |
|---|---|
| `SVSG_DETECTOR` | L1 编译方式：`disabled`、`functions`、`demo`、`yolo` |
| `SVSG_LLM_PROVIDER`、`SVSG_LLM_BASE_URL`、`SVSG_LLM_API_KEY`、`SVSG_LLM_MODEL` | L3 文本模型，`provider` 设为 `openai` 时使用兼容接口 |
| `SVSG_L15_*` | L1.5 视觉复核；超时必须小于 L3 超时的三分之一 |
| `SVSG_VISION_*` | 物体识别的模型、复核模型、调用次数上限和视图数 |
| `SVSG_API_KEY` | 设置后 `/v1/*` 需要 `X-API-Key` 请求头 |
| `SVSG_AUTH_ENABLED=1` | 改用登录签发的 Bearer 令牌 |

Windows 下可以用 [启动脚本](../scripts/start-svsg.cmd) 从单独的运行目录加载配置。

## 代码结构

```
svsg/
  image_functions/   光栅编码、区域、轮廓、空间关系、函数断言
  video_functions/   视频函数的编译、导出与查询求值
  recognition/       多视图识别、坐标映射、一致性判定、视频帧绑定、评测
  contracts/         IR、图像函数、视频函数、识别结果的数据格式
  l1_compiler/       检测器适配与 IR 编译
  l15_verifier/      视觉复核与降级
  l2_runtime/        状态机、意图路由、冲突处理、断言核对
  l3_orchestrator/   模型调用与重答循环
  api.py             HTTP 服务
```

测试在仓库根目录的 `tests/`，运行 `python -m pytest tests`。

## 许可

MIT
