# 模型实测：接入 motionspec 前后

216 次调用，6 个模型 × 6 个场景 × 3 种条件。每个数字是逐项判分的正确率（动效数量、起点、时长、方向、距离、透明度变化、缓动类型）。

## 按模型

| 模型 | 画面输入 | video | video+vision | video+probe |
|---|---|---|---|---|
| mimo-v2.6-flash | video@10fps | 28% | 90% | 97% |
| mimo-v2.6-pro | video@10fps | 38% | 90% | 81% |
| seed-2.1-turbo | video@5fps | 58% | 90% | 97% |
| seed-2.1-pro | video@5fps | 48% | 90% | 97% |
| gpt-6.1-sol | 8 stills@10fps | 82% | 97% | 97% |
| gpt-6.1-terra | 8 stills@10fps | 78% | 97% | 97% |
| **全部** | | **55%** | **92%** | **94%** |

## 按测量项（所有模型合计）

| 测量项 | video | video+vision | video+probe |
|---|---|---|---|
| count | 90% | 100% | 99% |
| start_ms | 44% | 89% | 86% |
| duration_ms | 30% | 59% | 86% |
| direction | 59% | 100% | 97% |
| distance_px | 32% | 100% | 97% |
| opacity_change | 86% | 100% | 97% |
| easing | 56% | 100% | 97% |

## 按场景（所有模型合计）

| 场景 | video | video+vision | video+probe |
|---|---|---|---|
| card-slide | 57% | 100% | 100% |
| toast-enter | 63% | 100% | 100% |
| toast-exit | 44% | 90% | 100% |
| toast-fade | 76% | 100% | 100% |
| list-stagger | 50% | 89% | 92% |
| two-at-once | 52% | 85% | 85% |

## 开销（每次调用的平均值）

| 条件 | 总 token | 耗时 |
|---|---|---|
| video | 7792 | 63.0s |
| video+vision | 8060 | 20.4s |
| video+probe | 6656 | 16.8s |

## 没有得到可解析回答的调用

- mimo-v2.6-pro，video+probe：1 次
- seed-2.1-pro，video：1 次
- seed-2.1-turbo，video：1 次
