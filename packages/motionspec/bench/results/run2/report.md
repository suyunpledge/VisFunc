# 模型实测：接入 motionspec 前后

144 次调用，6 个模型 × 6 个场景 × 3 种条件。每个数字是逐项判分的正确率（动效数量、起点、时长、方向、距离、透明度变化、缓动类型）。

## 按模型

| 模型 | 画面输入 | video | video+vision | video+probe |
|---|---|---|---|---|
| mimo-v2.6-flash | video@10fps | - | 100% | 100% |
| mimo-v2.6-pro | video@10fps | - | 100% | 100% |
| seed-2.1-turbo | video@5fps | - | 100% | 100% |
| seed-2.1-pro | video@5fps | - | 100% | 100% |
| gpt-6.1-sol | 8 stills@10fps | - | 100% | 100% |
| gpt-6.1-terra | 8 stills@10fps | - | 100% | 100% |
| **全部** | | **-** | **100%** | **100%** |

## 按测量项（所有模型合计）

| 测量项 | video | video+vision | video+probe |
|---|---|---|---|
| count | - | 100% | 100% |
| start_ms | - | 100% | 100% |
| duration_ms | - | 100% | 100% |
| direction | - | 100% | 100% |
| distance_px | - | 100% | 100% |
| opacity_change | - | 100% | 100% |
| easing | - | 100% | 100% |

## 按场景（所有模型合计）

| 场景 | video | video+vision | video+probe |
|---|---|---|---|
| card-slide | - | 100% | 100% |
| toast-enter | - | 100% | 100% |
| toast-exit | - | 100% | 100% |
| toast-fade | - | 100% | 100% |
| list-stagger | - | 100% | 100% |
| two-at-once | - | 100% | 100% |

## 开销（每次调用的平均值）

| 条件 | 总 token | 耗时 |
|---|---|---|
| video+vision | 8196 | 17.4s |
| video+probe | 6807 | 17.0s |
