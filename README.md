# SVSG — 给 LLM 视觉问答加一道证据校验闸门

SVSG 的函数路径把输入图像编译为可求值的像素函数和可核验的几何参数。
后端负责测量，模型可以只读取结构化证据来作答。

## 多模态物体识别与视频帧准备

需要识别真实物体时，使用新增的 `POST /v1/recognize-image`。配置
`SVSG_VISION_PROVIDER=openai`、`SVSG_VISION_BASE_URL`、`SVSG_VISION_API_KEY`、
`SVSG_VISION_MODEL`，安装 `.[api,llm]` 后即可调用，不依赖 YOLO 或 L3 问答后端。
该路径向所配置的视觉服务发送全图和局部图像；原有 functions 路径仍只发送参数。

请求包含 multipart `file`，可选 `labels` 为类别列表 JSON（如 `["person","cup"]`，
默认空列表表示开放类别），可选 `frame` 为以下 JSON：

```json
{"sequence_id":"video-001","frame_index":17,"timestamp_ms":413.725}
```

默认流程为 **全图识别 → 独立镜像识别 → 重叠局部扫描 → 小目标定向确认 → 函数证据**。
每次请求只有一张图，复核不接收第一次的类别判断。局部扫描能发现首次未检出的目标；
出现面积不足全图 2% 的候选时，其中一个视图改为该候选及邻域的放大图，改善极小目标的输入尺度。
额外确认优先处理小候选。每个视图默认最多 12 个候选、每图最多 5 次外部调用、总预算
90 秒，不自动重试。设置 `SVSG_VISION_STRATEGY=crop` 可使用原来的两次调用流程。
可用 `SVSG_VISION_REVIEW_MODEL` 指定同一提供商下的另一复核模型；省略时复用识别模型。
可用 `SVSG_VISION_MAX_CALLS`（2–8）和 `SVSG_VISION_DISCOVERY_VIEWS`（0–2）控制开销。

- `observations[].status=corroborated` 要求不同视图类别一致、框 IoU ≥ 0.5、用于确认的框未被内部视图边缘截断。配置不同模型时，还必须包含两个模型的支持；同一个检测器重复看到不等于双模型确认。未通过的候选保留为 `needs_review`，附原因。
- `pipeline=multiview` 的 `views` 记录每次视图的源图范围、镜像变换、实际输入尺寸、编码图像哈希及检测结果。`supports` 记录对象的各份来源，`origin` 标识首次从全图还是局部发现；调用失败和已知用量也保留。
- 小尺寸模型视图采用普通插值，最长边至少 768；这不会改变原图函数参数，也不能恢复丢失的视觉细节。最终框取一致视图坐标的均值，再从原图重算测量。
- `corroborated_counts` 只计复核一致的候选。`coverage=not_exhaustive` 始终明确：没有检出不能证明不存在，计数也不能当作完整物体总数。两次调用同一模型并非独立真值验证。
- `bbox_px` 是模型估计的可见范围，坐标基准为 **EXIF 转正后的 RGBA8**。`crop_mean_rgba` 和裁剪像素哈希属于框内所有像素（含背景），不是物体分割；颜色连通区域轮廓同样不能直接当成物体轮廓。
- `image_sha256` 对应原上传字节，`pixel_sha256` 对应转正像素。原图透明区域在发送模型时用白色衬底合成，原始 RGBA 函数仍保留透明度。
- 帧时间由调用方明确提供，不用处理时间或推测 FPS 代替。`observation_id` 仅标识单图观察，**不是跨帧 track ID**；`relations` 提供复核对象间可重算的归一化中心差、像素距离及框 IoU。

MotionSpec 的 `evidence_frames` 可通过 `svsg.recognition.frames.frame_reference()` 接入：
先核对文件名、帧号、毫秒时间和实际图片 SHA-256，再将其返回的 `FrameReference`
传给 `RecognitionService.recognize()` 或序列化到 HTTP `frame` 字段。
跨帧身份匹配、遮挡恢复、动作和因果分析仍需后续实现，不能由两个框的变化直接断言。

MotionSpec 现可通过 `--functions` 导出同一套 SVSG 图像函数，以及可求值的帧间差分、
采样插值、贝塞尔和弹簧时间函数。`svsg.video_functions.VideoRuntime` 同时接受图像操作
与帧/时间参数查询。详见 [视频统一函数表达与接入](docs/video-functions.md)。

可复现评测脚本为 `scripts/evaluate_recognition.py`：在调用模型之前冻结图片哈希和标注框，
按“同类别 + 一对一框匹配”统计首次候选与复核结果的 precision/recall，分别报告
IoU 0.5 和 0.75。变换图片、负样本、待复核和失败请求都要保留在报告中。
详见 [识别与评测说明](docs/recognition.md) 与
[多视图真实图片评测记录](docs/recognition-evaluation-20261116.md)。

## 函数解参

- **完整像素表示。** 将单帧图像解码成 RGBA8，定义 `f(x, y)` 为该像素的四通道值。结构简单的行使用常值区间 `[x_start, x_end)`，纹理密集的图像使用无损压缩的样本。两种形式可逐像素还原同一份 RGBA8。
- **区域与轮廓。** 合并上下行中颜色相同且重叠的区间，得到同色四连通区域。较大的区域保留有向闭环 `contour.rings`，外环为顺时针，孔洞为逆时针，顶点落在像素单元的边界。面积、周长和重心都能从环复算，并用于校验编译结果。
- **形状和空间。** 轮廓可导出孔洞、凸包面积比、等周紧致度、主轴角和主次轴比例。两个区域之间可以复算包围框方向和包含、IoU、重心距离、共享轮廓边以及是否相接。包围框包含和像素区域相交具有不同含义。
- **参数求值。** HTTP 可用 `pixel_rgba` 和 `mean_rgba` 查询任意有效坐标或矩形。坐标为从左上角开始的零基索引，矩形右、下边界不包含在内。均值按 R、G、B、A 顺序各自计算，重心以像素中心为采样位置。
- **分析覆盖。** 颜色表最多列 16 项，区域最多列 64 项，轮廓只计算前 16 个区域，超出点数预算时轮廓为空。不同颜色超过 4,096 或区间超过 100,000 时，相应原图计数为 `null`。色彩丰富、区域过多或超预算时，另提供最长边 160、最多 16 色的量化网格。网格参数可以精确复算，坐标、面积和距离按网格计，不能当成原图分割结果。完整 RGBA8 和指定原图坐标查询仍可用。

在安装 `.[api]` 后，可以离线导出和还原像素函数：

```bash
python -m svsg.image_functions encode input.png raster.json
python -m svsg.image_functions decode raster.json reconstructed.png
```

`raster.json` 包含完整像素表示。像素一致性按解码后的 RGBA8 检查，原文件容器、元数据和高位深数据不在此协议的还原范围内。

`POST /v1/compile-functions` 接收 multipart 字段 `file`，可带 JSON 字符串字段 `queries`。
此接口直接返回带 `image_functions` 的 IR，无需配置任何模型。以下是 `queries` 的内容示例：

```json
[
  {"id": "sample", "op": "pixel_rgba", "x": 20, "y": 10},
  {"id": "area_mean", "op": "mean_rgba", "box": [0, 0, 32, 32]}
]
```

如需让文本模型作答，设 `SVSG_DETECTOR=functions` 后调用 `POST /v1/analyze-image`，
同时传入问题 `query`。服务只向 L3 发送测得的参数及查询结果，并校验 `function:*` 断言。
模型上下文只包含前 12 个区域的摘要，以及部分外环采样点和区域对。完整轮廓保留在 IR。
例如，`{"field":"function:spatial:shared_edge_px","region_basis":"quantized","region_id":1,"target_region_id":2,"value":0}`
核验量化网格中区域 1 和 2 的共享边长度。`region_basis` 省略时指原图像素区域。
现有 YOLO 路径仍可用于获取类别。函数路径当前覆盖可计算的颜色和几何，文字、物体类别和场景含义需要额外的识别证据。

**LLM 说：图里有 3 颗螺丝**
**锚定校验：count 断言与 IR 不符 → 否决，打回重生成**

---

SVSG 把视觉问题转换成结构化断言，对照检测器和二次验证结果逐条校验。网关使用通过校验的断言生成交付文本。断言不符时可以重生成，持续失败时转人工。

> **核心机制**：LLM 声称"图中有 3 颗螺丝"→ 锚定校验发现 count 断言与结构化 IR 不符 → 系统直接否决该断言，LLM 收到修正反馈后重生成。

---

## 为什么需要它

| 问题 | 后果 |
|---|---|
| LLM 视觉问答经常"编造"不存在的物体或数量 | 下游任务（质检、安防、医疗）直接出安全事故 |
| 没有结构化中间层 | 无法审计、无法重放、无法定位幻觉来源 |
| prompt engineering 治标不治本 | 下次换模型、换场景又要重新调 |

SVSG 不是另一个 VLM。它是在 VLM 前面**再加一层强制校验**，把"我觉得图里有 X"变成"IR 里检测到 X，并且验证器确认了 X，所以答案可以说 X"。

## 核心机制

1. **S0** 校验 IR 输入合法性
2. **S3** 软检测依赖间隔参
3. **S4a** 调用 L1.5 验证器对 `verification_required=true` 的实例执行双确
4. **L3** LLM 生成答案（受总超时限制）
5. **强制后置锚定**：逐条校验 claims，拒绝空断言、未知字段、缺少实例 ID 和非法类型。
6. 通过后从断言生成交付文本。校验失败时带反馈重生成，重试耗尽后转 S6 人工复审。

**关键设计决策：**

- **class 断言**：必须与 IR 检测结果严格相等；验证报告判定 `ambiguous` 时一律否决
- **count 断言**：按有效检测结果计数，排除二次验证判定不存在的实例。类别未确定时不交付精确分类计数。
- **exists 断言**：与验证报告互锁，双向冲突都否决
- **relation 断言**：方向关系由框坐标重算。距离和最近邻使用有效实例的框中心，在响应中提供 `distance_px`。
- 所有违规带修正反馈回传 LLM，不直接返回错误码——**可重试**

## 快速开始

```bash
pip install -e ".[api,llm]"        # 基础服务
pip install -e ".[ml]"             # 需要 YOLO 真实检测时
cp .env.example .env               # 按需修改（默认 127.0.0.1:3002）
python -m svsg
```

健康检查：`GET /healthz`（无鉴权）

实际答题需配置 `SVSG_LLM_PROVIDER=openai` 及相应的地址、密钥和模型。图片入口默认关闭。
安装 `.[ml]` 并设置 `SVSG_DETECTOR=yolo` 后可使用真实检测器。
设置 `SVSG_DETECTOR=demo` 可联调合成检测数据，响应标为 `status=demo`。
旧配置 `SVSG_DETECTOR=stub` 仍可作为 `demo` 的别名。

需要二次验证时，设置 `SVSG_L15_PROVIDER=openai`、`SVSG_L15_BASE_URL`、
`SVSG_L15_API_KEY` 和支持图片输入的 `SVSG_L15_MODEL`。
服务把请求图像中的实例裁剪发送给该后端，每批默认最多 8 个。
外部模型超时或缺少裁剪证据时，对应实例进入降级。L1.5 超时须小于 L3 总时限的三分之一，
例如可设 `SVSG_L15_TIMEOUT_S=8`、`SVSG_LLM_TIMEOUT_S=30`。L3 首答与重生成共享时限。

### Windows / Trae 运行目录

如果 `.env` 和日志放在独立运行目录，不要从该目录直接执行未安装的
`python -m svsg`；Python 会找不到仓库源码。首次部署时在仓库目录执行：

```powershell
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m pip install -e ".[api]"
```

看门狗可调用仓库内的 `scripts\start-svsg.cmd`。它会把仓库加入
`PYTHONPATH`，再切换到运行目录启动服务，因此运行目录仍可提供 `.env`、日志和模型权重。
也可以显式指定路径：

```powershell
$env:SVSG_RUNTIME_DIR = 'C:\path\to\trae\work-mode-project'
$env:SVSG_PYTHON = "$env:LocalAppData\Programs\Python\Python312\python.exe"
& .\scripts\start-svsg.cmd
```

迁移 Trae 项目后，只需更新 `SVSG_RUNTIME_DIR`；不需要重新复制源码到运行目录。
如果旧目录里存在 `TRAE_PROJECT_MOVED_*.md`，以文件中记录的目标目录为准；
旧目录的 watchdog 日志可能只反映迁移前的失败启动。

## 一个对比 demo

```bash
python -m examples.demo
```

输出格式（示意）：

```
=== 无锚定（直接问 LLM）===
Q: 图中有几颗螺丝？
A: 图中有 3 颗螺丝，分布在左上、中部和右下角。
   [无证据，LLM 自行编造]

=== 有锚定（SVSG）===
Q: 图中有几颗螺丝？
→ IR: instance 1=screw, instance 2=screw（共 2 个实例）
→ 锚定校验: count 断言值=3 ≠ IR 实例总数=2 → EVIDENCE_MISMATCH
→ LLM 重生成: "图中有 2 颗螺丝"（与 IR 一致）
→ 锚定校验: 通过 ✓
交付: 2 颗螺丝（附 IR 实例锚定）
```

## API 端点

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| POST | `/v1/analyze-image` | 是 | 上传图片+问题，完整闭环 |
| POST | `/v1/compile-functions` | 是 | 上传图片及可选查询，返回确定性函数 IR |
| POST | `/v1/recognize-image` | 是 | 多模态识别、盲裁剪复核、函数证据及可选帧元数据 |
| POST | `/v1/analyze` | 是 | 传入 L1 已产出的 IR 载荷 |
| GET | `/healthz` | 否 | 健康检查 |

响应的 `verified_instances` 列出实际收到二次验证证据的实例，
`degraded_instances` 列出验证缺失的实例。`evidence_mode` 标记上传图片、提交的 IR 或演示数据。
`/v1/analyze-image` 返回上传文件的 `image_sha256`，可与 MotionSpec 原始帧清单中的哈希核对。
仅提交 IR 的请求没有图片字节供视觉后端裁剪。

## 架构分层

| 层 | 模块 | 职责 |
|---|---|---|
| **contracts** | `contracts/` | 跨层契约（IR Schema、错误码、枚举）唯一来源 |
| **多模态识别** | `recognition/` | 全图候选、上下文复核、坐标变换、观察账本、帧证据绑定与评测 |
| **L1 检测编译** | `l1_compiler/` | 检测器适配（stub/demo/YOLO）→ 几何引擎 → 不确定性估计 → IR |
| **L1.5 声明验证** | `l15_verifier/` | 视觉验证服务（超时隔离 + 静默降级） |
| **L2 运行时** | `l2_runtime/` | FSM（S0~S6）、意图路由、冲突解决、证据锚定 |
| **L3 编排** | `l3_orchestrator/` | LLM 编排循环、双通道答案、锚定否决重生成 |

## 错误码与状态机

| 错误码 | 含义 | 处置 |
|---|---|---|
| E1001–E1003 | IR 硬拒绝（S2） | 直接返回，不入 LLM |
| E2001–E2003 | 依赖缺失（S3） | 软重试最多 1 次 |
| E3001–E3002 | 语义冲突 / 锚定否决（S4） | 修正反馈重生成 |
| E4001–E4003 | 超时（S5） | 静默降级或中止 |
| E5001 | 人工复审（S6） | 不吞，不沉默 |

完整状态流转见 `docs/fsm-states.md`。

## 测试与静态检查

```bash
python -m pytest -q
python -m ruff check svsg
python -m mypy --python-version 3.12
```

## 适用场景

- 工业质检：外观缺陷计数 + 位置描述
- 安防监控：异常行为视觉确认
- 医疗影像：病灶描述与结构化报告
- 任何需要"LLM 说了什么必须能追到证据"的场景

## 不适用场景

- 纯文本 / 多模态不含视觉验证的场景（用普通 LLM 就行）
- 需要实时 <100ms 响应的流水线（L1.5 有独立超时，端到端延迟天然 >1s）
- 无结构化 IR 输入的纯对话 VQA（SVSG 以 IR 为输入契约）

## 贡献

详见 `CONTRIBUTING.md`。Issue 和 PR 欢迎。

## 许可

MIT


---

## 项目结构图谱 / Project Tree Map

- 中文：[项目结构图谱](docs/project-tree.zh.png)
- English: [Project tree map](docs/project-tree.en.png)
- 分层拆解与中英对照：[docs/PROJECT-TREE.md](docs/PROJECT-TREE.md)
