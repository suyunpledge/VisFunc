# 片头、音量键与全屏键复测

以本地提交 `64f1ddc` 为基线，回放上一轮的 35 段录屏，再对新录的 25 段打分。原先漏掉的 5 个有声音量键、1 个全屏键全部找回。新录屏启用同一套旧模板补检时命中从 50/60 提高到 54/60，两版均观测到 0 个图标误报。新录屏的 6 个漏检中有 5 个来自首次测试的 Shaka Player。[逐录屏的像素哈希、转写、预测框与匹配结果](../bench/results/20261122-volume-title-player/volume-title-player-20261122.jsonl)

## 片头与真正的字幕

旧版在五个连续采样时刻都读出了 `CITY LIMITS`，也都把它当成字幕，原有投票缺少片头观测。现在会测量靠近画面中部的大写字行。在两档亮度阈值下，要求可分离的字形数与转写一致，再排除词间大空格，测量字母间距与字形高度之比。目标帧和至少两个不同像素哈希的邻帧都提供同位置、同文字的排版证据，且横跨至少 400 ms，才据此将文字移到 `detached_text_candidates`。显式指定字幕框仍可直接取字。下接另一行的字幕沿用原有保护。[排版测量和相邻帧复核](../svsg/recognition/subtitle_sequence.py)

`CITY LIMITS` 在第 74、79、84、89、94 帧均测得 10 个字形，字母间距中位数约为字形高度的 0.294。片头误收由 5/5 降至 0/5，证据帧号随输出返回。先前的 `DEEP BLUE` 则保持 0/10。两个片头的相邻帧重复测试单独计数。[CITY LIMITS 逐帧对照](../bench/results/20261122-city-limits-review/city-limits-review-20261122.jsonl)

| 字幕指标 | 基线 | 本轮 |
|---|---:|---:|
| 新录屏中无字幕却输出文字 | 2/10 | 0/10 |
| 新录屏中有字幕，整行含标点逐字正确 | 14/15 | 14/15 |
| 其中抬到片头高度的真字幕仍保留为字幕 | 5/5 | 5/5 |

新录屏里，`RIVER VALLEY` 和 `STILL WATERS` 两处片头得到纠正。五条抬高的真字幕中，OpenPlayerJS 的 `WAIT HERE` 在两版中均被转写为 `WAIT H ERE`。旧录屏仍有一处漏句点。该组覆盖英语大写展示字和常规字幕，无法据此判断排版与片头完全相同的对白。

## 有声、静音和全屏图标

Clappr 静音时显示带叉扬声器，有声时显示无声波的空心扬声器。本轮增加闭合轮廓的几何模板，检查右侧是否还有笔画，并要求同排控件支持。Fluid Player 的底栏让背景亮度随高度变化。对音量键与全屏键的弱匹配，本轮比较图标左右边缘的亮度变化，只有两侧一致才逐行扣除背景。校正后同时检查原形状、模糊后的形状和附近的控件位置。输出可回看校正分数和支持框。[匹配实现](../svsg/recognition/icon_matching.py)

| 样本 | 标注 | 默认命中，基线 → 本轮 | 旧模板补检命中，基线 → 本轮 | 补检误报，基线 → 本轮 |
|---|---:|---:|---:|---:|
| 上轮录屏回放，35 段 | 90 | 75 → 81 | 84 → 90 | 0 → 0 |
| 已测播放器的新录屏，20 段 | 48 | 38 → 42 | 43 → 47 | 0 → 0 |
| 首次测试的 Shaka Player，5 段 | 12 | 5 → 5 | 7 → 7 | 0 → 0 |
| **仅新录屏合计，25 段** | **60** | **43 → 47** | **50 → 54** | **0 → 0** |

新录屏的默认模式同样为 0 个误报。旧模板补检下，有声音量键从 10/15 升到 **13/15**，全屏从 16/20 升到 **17/20**，静音保持 4/5，播放和暂停合计 20/20。剩余实例可以直接在录屏中定位：

| 录屏 | 本轮旧模板补检仍漏掉的图标 |
|---|---|
| Clappr 第 0 段 | 有声音量键 |
| Shaka 第 0 段 | 有声音量键、全屏 |
| Shaka 第 1 段 | 静音、全屏 |
| Shaka 第 2 段 | 全屏 |

新录的 Clappr 第 0 段中，在真值框附近测得空心扬声器模板的最佳原始相关度约为 0.84，低于该模板进入弱匹配复核的 0.88 门槛。Shaka 三次漏掉的全屏标注均约 15 × 14 像素。它在识别规则定稿后才引入，使用原生 Material Icons Round 字体控件。该皮肤上尚未观察到图标收益。

## 录制、评分与复现

每款播放器录制 5 段，4 段显示底栏，1 段隐藏。播放器为 Clappr、Fluid Player、OpenPlayerJS、Media Chrome 和 Shaka Player。使用 Chrome 播放预告片与 Oceans 片段，改变片段起点、视口、文字和控件状态。四款已测皮肤的新录屏与上一轮源文件哈希全部不同；Shaka 单列为留出皮肤。显示底栏的 20 个评分时刻各标注 3 个控件，隐藏底栏的 5 个时刻用于查误报。每段取末尾 0.8 秒的 5 帧，仅最后一帧计入主表。前四帧用于字幕复核，图标识别只看目标帧。两版共用同一批解码像素及源文件哈希。[四款播放器的录制场景](../packages/motionspec/bench/title_volume_cases.json) · [录制器与 Shaka 场景](../packages/motionspec/bench/record_player_overlays.py)

每个标注通过实际播放器状态及控件字形取框。Shaka 录制时额外核对按钮当前显示的字体连字，并在同一布局下切换字形颜色测得边界。录制使用可寻址的本地视频 Blob，避免本地 HTTP 服务使播放器错误显示重播。此前的试录已排除。程序绑定 Shaka 4.12.6 和随其样式表指定的字体。本轮补检复用上一轮那五款旧皮肤的参考库，未加入任何本轮录屏的模板。同标签、IoU ≥ 0.5、一对一匹配计命中，重复框计误报。[Shaka 固定版本](https://registry.npmjs.org/shaka-player/-/shaka-player-4.12.6.tgz) · [Material Icons Round 字体](https://fonts.gstatic.com/s/materialiconsround/v108/LDItaoyNOAY6Uewc665JcIzCKsKc_M9flwmM.otf)

从本轮录屏证据包解压 `new-recordings` 和 `preceding-player-refs` 后，可在安装了项目及 OCR 依赖的环境中运行：

```shell
python packages/motionspec/bench/score_overlay_sequences.py new-recordings sampled
python packages/motionspec/bench/score_overlay_sequences.py new-recordings sampled --output default.jsonl
python packages/motionspec/bench/score_overlay_sequences.py new-recordings sampled --fallback-icon-dir preceding-player-refs --output supplemented.jsonl
```

每种皮肤只有 4 个显示底栏的评分时刻，片头与对白也共享字体和源素材。按实例汇总用于定位错误，不能用来估计任意播放器的命中率。
