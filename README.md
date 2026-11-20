# SVSG · 可复算的视觉证据

SVSG 把图片和视频中的视觉信息写成可以求值的函数。它的目的是让 AI 即使只读参数，也能回答有测量依据的问题，例如「这两个区域是否相接」「卡片向右移动了多少像素」「画面里的变化持续了多久」。程序保留测量所用的画面、坐标或时间范围，读者可以拿同一份数据复算答案。

一张图是起点。给定帧号 `i` 和坐标 `(x, y)`，`F_i(x, y)` 返回那个像素的颜色。把区域的边界也记下来，就能计算形状和相对位置。到了视频，仍然用这个函数读每一张保留帧，再引入时间 `t` 描述帧之间的变化。遇到「这是人还是桌椅」这样的语义问题，可以另行请求视觉模型辨认，并把复核证据关联到具体帧。

## 从画面到答案

| 步骤 | 程序做什么 | 下游能读到什么 |
|---|---|---|
| 1. 取样 | 图片按 EXIF 方向转正，解码为 RGBA8。视频扫描所选时段，在变化的关键位置保留源分辨率帧，也可按时间间隔补采。 | 图像尺寸、源帧号、时间戳和证据哈希。 |
| 2. 描述画面 | 把像素编码成可无损还原的函数，提取同色连通区域，用闭合轮廓表示边界和孔洞。复杂画面还可以在单独标明尺寸的量化网格上分析。 | 像素值、矩形均色、区域面积和重心。对有轮廓的区域，还能复算周长、形状特征、共享边长和相接关系。 |
| 3. 测量变化 | 对保留帧计算逐通道色差，在运动片段内测量可用的位移、宽高、缩放或透明度，并拟合缓动或弹簧曲线。可接入的 Qt 应用也能直接采样控件属性。 | 每个时间函数的名称、单位和有效区间。求值时可区分采样点、点间插值和曲线估计。 |
| 4. 按问题取证 | 模型或调用方读取函数目录，选择帧、坐标、片段和时间，提交 JSON 查询。求值在本地完成。需要识别物体时，视觉模型再读全图、镜像或局部放大图。 | 查询返回数值与来源。多视图识别另行返回支持框、已复核候选和待复核候选，可绑定回原始证据帧。 |
| 5. 核对结论 | 图片问答服务把模型给出的结构化断言与测量结果核对，用通过的断言生成答案。遇到冲突时可反馈重答，仍无法解决时进入人工复核。视频测量也可用 `motionspec check` 核对结构化断言。 | 可交付的答案，或失败、证据不足和需要复核的原因。 |

例如，一张 8 × 4 像素的图，左半红、右半蓝。编译后可以读出两个区域各占 16 像素，包围框的交并比（IoU）为 0，共享边长为 4 像素。因此，单凭参数就能回答「红蓝区域沿一条边相接，面积没有重叠」。对于照片中的人物，这类几何信息只能描述区域，辨认人物还需要识别证据。

视频有类似的区别。保留帧上的像素可逐点还原，相邻运动采样点之间的位置来自插值。协议分别记录实际检测到的变化区间 `timing.visible_change` 和从曲线推断的设计区间 `timing.estimated_design`。函数参数和文字摘要使用同一次时间计算，依据不足时设计区间为 `null`。一个运动片段可能覆盖多个元素，其 ID 仅标识这段变化。

## 只用参数读一段滑动动画

需要 Python 3.11 或更高版本。先在仓库根目录安装依赖，随后的素材生成、分析和查询均可离线运行，无需模型密钥。

```bash
python -m pip install -e ".[functions]" -e "./packages/motionspec"
python -c "from motionspec.synth import slide, write_video; frames, _ = slide(); write_video('slide.mp4', frames)"
python -m motionspec analyze slide.mp4 -o slide-out --functions --evidence-every-ms 250
python -m motionspec query slide-out/functions.json --list
```

例子生成一张向右滑动的卡片。打开 `slide-out/summary.md`，可以直接回答「卡片移动了多远、用了多久」。在本次运行中，跟踪点向右移动约 **299 像素**，可见变化约 **350 毫秒**，拟合出的设计时长约 **400 毫秒**。素材生成器设定的是 300 像素和 400 毫秒。缓出末段每帧移动很小，所以可见区间比设计区间短。

要查看某一刻，`--list` 会列出实际保留的帧号，以及每个片段可查的函数。把下面的内容保存为 `query.json`，它会一次查询像素颜色和跟踪点位置。

```json
[
  {"id": "color", "op": "pixel_rgba", "frame_index": 0, "x": 0, "y": 0},
  {"id": "position", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 250}
]
```

```bash
python -m motionspec query slide-out/functions.json query.json
```

第一项给出第 0 帧左上角的 RGBA 值，并附帧号、时间和像素哈希。第二项给出 250 毫秒时跟踪点的横坐标，本次约为 **261 像素**。`source_px` 表示源画面像素，`motion_estimate` 说明该位置来自运动跟踪。换用自己的视频时，先读它的目录，再选择片段和有效时间。最小的模型输入可以只包含问题、相关参数和查询结果，完整像素数据留在本地。

### 在自己的素材上使用

| 我想做什么 | 入口和结果 |
|---|---|
| 保存单张图片的完整像素函数 | `python -m svsg.image_functions encode input.png raster.json` 编码，再用 `python -m svsg.image_functions decode raster.json restored.png` 还原。可核对解码后的 RGBA8 像素。 |
| 分析视频或按序排列的帧 | 将示例的 `slide.mp4` 换成自己的视频。输入为帧目录时加 `--fps`。输出的 `spec.json` 是运动规格，`summary.md` 供人阅读，`functions.json` 是查询入口，旁边的证据 PNG 和光栅文件供复算。 |
| 查询界面控件的真实属性采样 | 对 Qt Widgets 或 Qt Quick 应用采集探针轨迹，用 `python -m motionspec trace trace.json -o probe` 生成规格，再用 `python -m motionspec functions probe/spec.json -o probe/functions` 导出。位置、颜色或数值属性取决于采集项，只有录到画面时才有像素证据。[探针与函数查询](docs/video-functions.md) |

## 接入 AI 或调用 HTTP 服务

在仓库根目录安装服务依赖并启动。默认地址为 `http://127.0.0.1:3002`，`GET /healthz` 可检查服务状态。可把 `.env.example` 复制为 `.env`，按使用的端点填写模型配置。[配置示例](.env.example)

```bash
python -m pip install -e ".[api,llm]"
python -m svsg
```

| 任务 | 请求 | 输入与返回 |
|---|---|---|
| 只取图片参数 | `POST /v1/compile-functions` | 上传 `file`，可带 JSON 表单字段 `queries`（如 `pixel_rgba`、`mean_rgba`）。返回含 `image_functions` 的中间表示（IR），无需模型。 |
| 让文本模型根据参数回答 | `POST /v1/analyze-image` | 上传 `file`、问题 `query` 和可选 `queries`。设 `SVSG_DETECTOR=functions`，配置 `SVSG_LLM_PROVIDER=openai`、`SVSG_LLM_BASE_URL`、`SVSG_LLM_API_KEY`、`SVSG_LLM_MODEL`。模型只读结构化证据，服务核对断言。 |
| 用已有证据继续提问 | `POST /v1/analyze` | 以 JSON 提交 `ir` 和新的 `query`，复用已编译的数据。 |
| 识别真实物体 | `POST /v1/recognize-image` | 上传 `file`，可带类别列表 `labels` 和帧引用 `frame`。配置 `SVSG_VISION_PROVIDER=openai` 及 `SVSG_VISION_BASE_URL`、`SVSG_VISION_API_KEY`、`SVSG_VISION_MODEL`。模型在多个视图里独立识别，服务把候选框映射回原图再复核。 |

识别结果用 `corroborated` 标记得到多视图支持的候选，用 `needs_review` 保留证据不足的候选。`coverage=not_exhaustive` 表示画面中仍可能有漏检。给视频帧附加识别结果时，程序会核对序列、帧号、时间、尺寸和哈希，避免把对象标到错误的画面上。[识别与复核流程](docs/recognition.md)

设置 `SVSG_API_KEY` 后，`/v1/*` 请求须携带 `X-API-Key`。启用 `SVSG_AUTH_ENABLED=1` 时使用登录签发的 Bearer 令牌。Windows 下可用启动脚本从独立运行目录加载配置。[启动脚本](scripts/start-svsg.cmd)

## 验证和开发

- **检查像素和运动。** 可以用保留帧逐像素对照还原结果，并把运动函数的采样值与原轨迹比较。布局重排、多个元素互相遮挡时，视觉估计的位置仍可能偏离控件位置。[函数协议与精度记录](docs/video-functions.md) · [视频分析、支持场景与限制](packages/motionspec/README.md)
- **检查模型能读懂多少。** 在两段 UI 视频的纯参数复测中，只计三个条件均成功的配对，模型读取新时间参数答对 145/147 项，读取同源文字摘要答对 147/147 项。分母按答题项加权，只覆盖这两段动效。另有一组实验固定模型收到的 5 fps 帧，分别加入从 5 fps 和 40 fps 测得的参数，用来区分参数表达与提高采样率的影响。[纯参数复测](docs/timing-protocol-retest-20261117.md) · [采样率消融](docs/evidence-ablation-20261116.md)
- **检查物体识别。** 对有标注的真实图片分别统计误报和漏检。例如，在新增 COCO8 图像组中，复核让误报从 9 个降至 2 个，同时漏检从 10 个增至 16 个。这个取舍决定了为什么接口保留待复核候选。[真实图片识别评测](docs/recognition-evaluation-20261116.md)

源码可沿数据流阅读，从[图像函数编译](svsg/image_functions/compiler.py)和[视频测量](packages/motionspec/motionspec/analyze.py)，到[函数求值](svsg/video_functions/runtime.py)、[识别复核](svsg/recognition/)和[问答校验](svsg/l3_orchestrator/agent_loop.py)。以下命令在仓库根目录运行回归测试。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

许可证为 MIT。
