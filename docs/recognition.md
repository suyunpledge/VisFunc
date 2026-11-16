# 多模态识别与准确率验证

`svsg.recognition/1` 是独立于 L3 自由问答的物体观察协议。物体类别和位置由视觉模型提出，
函数模块计算颜色和几何，两者的来源严格分开。模型的自报置信度不参与通过判定。
识别结果没有被自动塞入旧检测器的 `detection_score`，避免把“模型两次说了一样的话”
伪装成已校准的检测概率。

## 模块边界

| 模块 | 责任 |
|---|---|
| `contracts/recognition.py` | 输出版本、对象框、复核结果、调用审计与帧引用 |
| `recognition/images.py` | 单帧限制、EXIF 转正、透明合成、模型视图编码 |
| `recognition/provider.py` | HTTP 调用、严格解析、明确支持的裸数组封装适配、失败计量 |
| `recognition/service.py` | 两阶段流程、总网络时限、局部复核与最终观察集 |
| `recognition/adaptive.py` | 默认多视图流程、发现漏检、小目标优先确认、调用预算 |
| `recognition/views.py` | 单图视图编码、镜像/裁剪坐标逆变换、图像证据哈希 |
| `recognition/consensus.py` | 视图间候选关联、双模型一致性、重复候选抑制 |
| `recognition/geometry.py` | 0..1000 → 原图坐标、裁剪偏移、IoU、保守类别同义词 |
| `recognition/frames.py` | MotionSpec 原始证据帧的文件名/哈希/时间绑定 |
| `recognition/evaluation.py` | 一对一类别和框匹配，避免重复框虚增成绩 |

## 默认多视图流程

`SVSG_VISION_STRATEGY=multiview` 为默认配置，Python 入口是 `AdaptiveRecognitionService`。
每次模型请求只含一张图片，并请求全新的识别结果，避免旧批裁剪协议发生 ID 错配。
默认依次执行：

1. 首模型识别全图。
2. 复核模型独立识别水平镜像，网关把框准确映射回源图。
3. 首模型与复核模型分别扫描两个重叠半图；若出现面积不足全图 2% 的候选，第一张改为
   该候选周边的高倍率视图（边长至少 128 源像素，受原图短边限制），第二张保留较广扫描。
   局部也能发现邻近漏检。
4. 若还有预算，优先扩展最小的未确认候选做一次独立识别；默认每图最多 5 次请求。

请求仍包含用户指定的类别范围，但不传递先前候选的类别、框或标准答案。
小视图通过 Lanczos 普通插值放大到最长边至少 768，
最长边限制 1536；这不是生成式超分，也不会为原图补造细节。模型框始终映射到原图尺寸。
使用聚焦视图时，局部扫描不再覆盖全图；响应有对应提示，整体覆盖始终标为 `not_exhaustive`。

同类别候选按 IoU ≥ 0.5 关联，一个视图中的检测不能重复支持同一对象。
确认需要两份不同视图中未被内部裁剪边缘截断的证据；配置不同模型时必须包含两种模型，
同一模型反复返回一致结果不能替代另一模型的复核。相同模型、相同编码图像不能因换 ID
重复计为独立证据。关联仍是单图内近似框关联，不是跨帧身份识别。

通过的框由两份一致证据坐标取均值，框内颜色和像素哈希重新从源图计算。
原始支持框保留在 `supports`；`views` 包括源图范围、镜像标记、实际模型输入大小、JPEG
字节哈希和全部视图对象。类别范围外的对象留在视图审计中，不作为范围内确认结果。
`origin` 可区分首次全图、镜像以及局部新增候选。

通过设置 `SVSG_VISION_MAX_CALLS=2`、`SVSG_VISION_DISCOVERY_VIEWS=0` 可只做两次全图识别；
提高发现能力需要多一次或多次调用。每视图上限由 `SVSG_VISION_MAX_OBJECTS` 控制，
所有视图共享 `SVSG_VISION_TIMEOUT_S`。超时或格式失败也消耗调用预算，不会隐式重试。
多视图结果只有全部外部调用不可用才为 `unavailable`；部分成功仍返回已有证据与失败记录。

## 兼容的裁剪流程

设置 `SVSG_VISION_STRATEGY=crop` 使用下面的旧策略，Python 入口仍是 `RecognitionService`。

先发送最长边不超过 1536 的全图。模型返回有限个可见物体候选，其坐标为整数 0..1000。
第二次请求发送每个候选最长边不超过 768 的上下文裁剪；既不提供首次类别，也不提供自报
置信度。只给裁剪提供 ID，输出框须相对于该裁剪。没有在同一批次混入全图参照，避免部分
提供商把全图误当成另一个待复核对象而错配 ID。

对每个候选，按框重合度选择最接近的复核对象；必须类别一致、IoU ≥ 0.5，且复核对象
没有贴住原图内部的裁剪边缘。位于原图边缘的真实截断不会因这一条单独否决。
这会牺牲部分召回率；候选及原因仍可供人检查，或由上层用另一个后端复核。
Python 服务支持注入单独 `reviewer`；HTTP 可通过 `SVSG_VISION_REVIEW_MODEL` 选择同一 API
提供商的另一模型。默认使用同一个模型；换模型同样不存在独立真值保证。

超时、残缺 JSON、越界/退化框、重复或未知裁剪 ID 不会产生确认结果。
提供商的裸数组、单元素外层封装、`bbox_2d` 别名可做无损适配，并记录 `normalizations`。
坐标范围和类型继续严格检查，不猜测坐标单位，不接纳任意拼错的字段。
默认 `SVSG_VISION_JSON_MODE=true` 发送 `response_format={"type":"json_object"}`，
不支持该参数的提供商可显式关闭。JSON 语法合法仍不代表字段齐全，所有输出继续过 Schema。
此开关符合 [MiMo 官方结构化输出说明](https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/structured-output)。
已完成 HTTP 响应但结构校验失败时仍记录使用量和字段路径错误；网络层未收到用量时为 null，
不能当作没有费用。主图失败为 `unavailable`，复核失败保留候选为 `needs_review`。
总预算包括预处理所耗时间；本地函数计算受像素/区间上限限制，但已启动的线程不会被强杀。

## 从 MotionSpec 关键帧接入

```python
from svsg.recognition.frames import frame_reference

# row 来自 MotionSpec 的 evidence_frames；image 为对应文件的原始字节。
frame = frame_reference("video-001", row, image)
result = await service.recognize(image, frame=frame)
assert result.image_sha256 == row["sha256"]
```

`frame.frame_index` 保留原帧号，`timestamp_ms` 保留 `t_ms`，包括非均匀帧间隔。
所有位置以 EXIF 转正后的源图为基准，另有归一化坐标供不同尺寸的帧比较。
`observation_id` 由源图哈希、类别和框决定，只能用于单图幂等引用。
后续跟踪需要明确分配 track ID，并对低重合、遮挡、切镜、类别变化执行重新识别。
当前没有把相邻帧近似框自动标成同一个物体，也没有声称识别了连续动作。

## 评测规则

准备如下清单；坐标须来自独立标注，不能用待测模型的输出当答案。

```json
{
  "cases": [{
    "id": "sample-01", "file": "image.png", "sha256": "原文件的64位小写哈希",
    "labels": ["person", "cup"],
    "expected": [{"label": "person", "bbox_px": [10, 20, 100, 200]}]
  }]
}
```

```bash
python scripts/evaluate_recognition.py --manifest cases.json --output results.json
```

先设置 `SVSG_VISION_*`。脚本不读取 Forge，也不会写出密钥。图片路径相对于清单目录。
每次报告包括清单哈希、模型、预算、原始候选、通过/待复核对象、调用耗时/用量及误报和漏检。
默认顺序执行；可用 `--concurrency 2` 做有界并发。

主要指标为同类别、IoU ≥ 0.5 的一对一匹配 precision/recall，同时保留 IoU ≥ 0.75 结果。
负样本误报计入 FP，失败样本仍计入漏检；不把“正确读回函数数值”算作物体识别成功。
类别范围外的候选不参与限定类别成绩，但在响应中保留并标记。空候选不是有效的零计数证明。
评测的颜色区域、语义物体和跨帧轨迹必须分别评估。

少量公开样例只适合回归，尤其镜像和降采样都不是独立场景。应用部署前应冻结有人工复核的
目标领域验证集，另留不参与调参的测试集，并按小物体、遮挡、低清、夜景、切镜分别报告。
任何仅针对少量回归图得到的 100% precision，都不能被解释为实际场景永不误认。
