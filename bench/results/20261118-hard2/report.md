# 第二批场景模型实测

216 次调用。数字是逐项判分的正确率。

## 按模型

| | video+vision-old | video+vision |
|---|---|---|
| mimo-v2.6-flash | 66% | 85% |
| mimo-v2.6-pro | 60% | 82% |
| seed-2.1-turbo | 55% | 84% |
| seed-2.1-pro | 62% | 82% |
| gpt-6.1-sol | 82% | 94% |
| gpt-6.1-terra | 81% | 89% |
| **全部** | **68%** | **86%** |

失败的调用（没有可解析回答）在上表按 0 分计。下表只比较所有条件都拿到回答的模型×场景：

| | video+vision-old | video+vision |
|---|---|---|
| mimo-v2.6-flash | 66% | 85% |
| mimo-v2.6-pro | 60% | 82% |
| seed-2.1-turbo | 80% | 93% |
| seed-2.1-pro | 73% | 90% |
| gpt-6.1-sol | 82% | 94% |
| gpt-6.1-terra | 81% | 89% |
| **全部** | **74%** | **89%** |

## 按测量项

| | video+vision-old | video+vision |
|---|---|---|
| start | 76% | 95% |
| duration | 51% | 79% |
| move | 66% | 87% |
| size | 72% | 80% |
| opacity | 78% | 91% |
| color | 81% | 95% |
| easing | 50% | 76% |

## 按场景

| | video+vision-old | video+vision |
|---|---|---|
| modal-open | 66% | 100% |
| accordion | 56% | 57% |
| sidebar-collapse | 29% | 78% |
| scroll | 100% | 99% |
| tab-indicator | 68% | 100% |
| reorder-swap | 75% | 93% |
| overshoot | 100% | 100% |
| button-color | 71% | 71% |
| progress | 90% | 93% |

## 多报的动效（回答里配不上真值的条目）与开销

| 条件 | 多报条目 | 平均 token | 平均耗时 |
|---|---|---|---|
| video+vision-old | 13 | 8688 | 37.5s |
| video+vision | 10 | 8979 | 37.2s |

## 没有得到可解析回答的调用

- gpt-6.1-sol，video+vision-old：1 次
- gpt-6.1-terra，video+vision：1 次
- gpt-6.1-terra，video+vision-old：1 次
- mimo-v2.6-flash，video+vision-old：1 次
- mimo-v2.6-pro，video+vision-old：1 次
- seed-2.1-pro，video+vision-old：4 次
- seed-2.1-turbo，video+vision-old：5 次
