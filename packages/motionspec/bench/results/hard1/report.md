# 第二批场景模型实测

216 次调用。数字是逐项判分的正确率。

## 按模型

| | video | video+vision-old | video+vision | video+probe |
|---|---|---|---|---|
| mimo-v2.6-flash | 44% | 64% | 68% | 98% |
| mimo-v2.6-pro | 43% | 77% | 79% | 100% |
| seed-2.1-turbo | 45% | 59% | 81% | 100% |
| gpt-6.1-sol | 92% | 91% | 92% | 100% |
| gpt-6.1-terra | 81% | 87% | 95% | 100% |
| seed-2.1-pro | 45% | 56% | 78% | 100% |
| **全部** | **58%** | **72%** | **82%** | **100%** |

失败的调用（没有可解析回答）在上表按 0 分计。下表只比较所有条件都拿到回答的模型×场景：

| | video | video+vision-old | video+vision | video+probe |
|---|---|---|---|---|
| mimo-v2.6-flash | 44% | 64% | 68% | 98% |
| mimo-v2.6-pro | 43% | 77% | 79% | 100% |
| seed-2.1-turbo | 65% | 86% | 92% | 100% |
| gpt-6.1-sol | 92% | 91% | 92% | 100% |
| gpt-6.1-terra | 81% | 87% | 95% | 100% |
| seed-2.1-pro | 62% | 81% | 87% | 100% |
| **全部** | **65%** | **81%** | **85%** | **100%** |

## 按测量项

| | video | video+vision-old | video+vision | video+probe |
|---|---|---|---|---|
| start | 44% | 82% | 94% | 100% |
| duration | 22% | 55% | 74% | 100% |
| move | 54% | 72% | 77% | 100% |
| size | 65% | 74% | 78% | 100% |
| opacity | 85% | 83% | 90% | 100% |
| color | 83% | 86% | 94% | 100% |
| easing | 56% | 54% | 69% | 97% |

## 按场景

| | video | video+vision-old | video+vision | video+probe |
|---|---|---|---|---|
| modal-open | 58% | 67% | 67% | 100% |
| accordion | 51% | 46% | 75% | 100% |
| sidebar-collapse | 44% | 60% | 71% | 98% |
| scroll | 55% | 100% | 95% | 100% |
| tab-indicator | 60% | 76% | 100% | 100% |
| reorder-swap | 63% | 74% | 90% | 100% |
| overshoot | 48% | 100% | 100% | 100% |
| button-color | 88% | 71% | 74% | 100% |
| progress | 76% | 100% | 93% | 100% |

## 多报的动效（回答里配不上真值的条目）与开销

| 条件 | 多报条目 | 平均 token | 平均耗时 |
|---|---|---|---|
| video | 11 | 8625 | 75.4s |
| video+vision-old | 7 | 9055 | 48.3s |
| video+vision | 5 | 8984 | 44.0s |
| video+probe | 5 | 7145 | 20.2s |

## 没有得到可解析回答的调用

- seed-2.1-pro，video：1 次
- seed-2.1-pro，video+vision-old：2 次
- seed-2.1-turbo，video：2 次
- seed-2.1-turbo，video+vision-old：2 次
