# 相邻帧字幕复核与陌生播放器图标复测

在五款此前未参与项目评测的播放器上新录制 35 段 WebM。基线为 `022e462`。默认图标识别命中从 66/90 提高到 75/90，加入此前五款播放器的参考图标补检后为 84/90，三种配置均观测到 0 个误报。补检的 9 个新增命中全部来自 OpenPlayerJS。最终留出的 Fluid Player 仍然命中 15/18。字幕和图标分别按整行和实例计分。[35 段的真值、转写、预测框和匹配记录](../bench/results/20261122-temporal-player-overlays/temporal-player-overlays-20261122.jsonl)

## 片头与字幕

同一个 `DEEP BLUE` 片头有两次旧录屏。每次取 5 帧，在完全相同的 10 张画面上比较三种读法。基线误收 3 帧，本轮代码逐张读取 PNG 仍误收相同的 3 帧，通过视频函数清单读取后为 **0/10**。误收集中在第二段的第 60、65、70 帧，三帧均可追溯至第 50 和 55 帧的独立观测。这个消融只验证该片头，上述 10 帧单独计数。[逐帧消融与证据位置](../bench/results/20261122-temporal-title-ablation/temporal-title-ablation-20261122.jsonl)

自动字幕识别现在复核目标帧前后 800 ms 内的文字。如果至少两个像素哈希互异、且与目标帧不同的邻帧，都把同位置的相同文字标为独立画面文字，目标行会移到 `detached_text_candidates` 并保留证据帧号。相邻 OCR 框可沿同一基线合并，因此也能核对分成 `DEEP` 与 `BLUE` 的读数。文字消失、位置变化超过容差或相邻采样间隔超过 400 ms 即停止向该方向取证。复核保护与下方字幕相连的多行文本，使用各帧原始判断投票。显式指定字幕框时直接转写框内文字。[相邻帧复核](../svsg/recognition/subtitle_sequence.py)

英文空格由两档笔画阈值下都可见的间隙提出。间隙两侧独立转写与原字符和大小写一致，才允许插入空格。紧排的字母、数字以及触及裁框边沿的字母笔画也参与排除误切。旧录屏中的 `EXIT NOW` 与 `TURN BACK` 均已读对。[字距复核](../svsg/recognition/subtitle_ocr.py)

| 播放器 | 有字幕逐字正确，基线 | 有字幕逐字正确，本轮 | 无字幕画面误收，基线 → 本轮 |
|---|---:|---:|---:|
| xgplayer | 3/4 | 4/4 | 0 → 0 / 3 |
| Clappr | 1/4 | 3/4 | 0 → 0 / 3 |
| Media Chrome | 2/4 | 4/4 | 0 → 0 / 3 |
| OpenPlayerJS | 2/4 | 4/4 | 1 → 1 / 3 |
| Fluid Player（最终留出） | 2/4 | 4/4 | 0 → 0 / 3 |
| **合计** | **10/20** | **19/20** | **1 → 1 / 15** |

剩余逐字错误为 Clappr 的 `Wait at exit B.`，词间空格已恢复，末尾句点仍漏读。无字幕误收为 OpenPlayerJS 的 `CITY LIMITS`，邻帧没有提供足够的独立片头证据。这两个错误分别影响转写和文字角色判定。

另行回放上一轮 18 段录屏的评分帧，字幕整帧逐字正确从 16/18 提高到 18/18。更早的 54 张截图仍为 47/54，`Keep the platform clear.` 中紧排的 `platform` 保留完整。两批历史回放的默认图标命中分别从 27/45 升到 37/45、从 111/159 升到 130/159，本轮均为零误报。[上一轮录屏与历史样本口径](player-recordings-20261122.md)

## 图标命中与误报

下表每格为命中数 / 误报数。每款播放器标注 18 个可见图标；漏检数等于标注数减去命中数。同标签且框的 IoU ≥ 0.5 方可一对一匹配，重复预测计入误报。每款另有一段隐藏工具栏录屏参加误报计数。

| 播放器 | 标注数 | 基线默认 | 本轮默认 | 本轮默认加旧模板补检 |
|---|---:|---:|---:|---:|
| xgplayer | 18 | 18 / 0 | 18 / 0 | 18 / 0 |
| Clappr | 18 | 6 / 0 | 15 / 0 | 15 / 0 |
| Media Chrome | 18 | 18 / 0 | 18 / 0 | 18 / 0 |
| OpenPlayerJS | 18 | 9 / 0 | 9 / 0 | 18 / 0 |
| Fluid Player（最终留出） | 18 | 15 / 0 | 15 / 0 | 15 / 0 |
| **合计** | **90** | **66 / 0** | **75 / 0** | **84 / 0** |

| 控件 | 标注数 | 基线命中 | 本轮默认命中 | 加旧模板补检命中 |
|---|---:|---:|---:|---:|
| 静音 | 15 | 9 | 12 | 15 |
| 全屏 | 30 | 21 | 27 | 29 |
| 有声音量 | 15 | 10 | 10 | 10 |
| 暂停 | 18 | 14 | 14 | 18 |
| 播放 | 12 | 12 | 12 | 12 |

默认底栏识别增加了细线角框、对角箭头和带叉或斜线的扬声器形状，并检查无声波扬声器右侧的余量。可选的 `--fallback-icon-dir` 使用较高匹配分数及同排其他控件的支持，在空位补框。已有图标框覆盖的扬声器局部不能再次报成静音。补上的框携带 `reference/` 模板标识，输出列出模板哈希。本轮参考库仅含此前使用过的 Video.js、Plyr、DPlayer、ArtPlayer、MediaElement.js。[图标匹配和补检规则](../svsg/recognition/icon_matching.py)

启用补检后仍漏掉 6 个实例。Clappr 的第 1、4、5 号样本各漏一个有声音量键。Fluid Player 的第 2 号漏音量，第 5 号同时漏音量与全屏。因此，默认模式和补检模式在最终留出播放器上均漏 3 个，误报 0 个。

## 录制与复现

每款播放器在 Chrome 中录制 7 段，覆盖显示工具栏时的 3 次静音、3 次有声，以及 1 次隐藏。背景使用 Sintel、Big Buck Bunny 预告片和 Oceans 素材，已知片头与字幕经 FFmpeg 写进视频，控件由运行中的播放器渲染。真值来自录制时的媒体状态及实际控件的字形边界。CSS 伪元素在原布局中通过开关字形绘制取差分得到边界。识别器只读取录屏像素。每段取末尾 0.8 秒中间隔 200 ms 的 5 帧，最后一帧评分，前四帧用于字幕复核。评分器校验源文件哈希、容器时间戳与每张解码帧的像素哈希。[录制工具](../packages/motionspec/bench/record_player_overlays.py) · [清单导出与评分](../packages/motionspec/bench/score_overlay_sequences.py)

xgplayer、Clappr 首先用于发现问题，后续也检查过 Media Chrome 和 OpenPlayerJS 的错例。Fluid Player 在图标规则确定后首次计分。最后用历史截图发现并修复了旧字体被误插空格的问题，随后按相同清单重跑字幕评分。每段仅一个评分时刻，35 段共享素材与控件，同款内多次出现的图标不能视作独立皮肤。本轮结论适用于这些样本和列出的版本。[xgplayer 3.0.21](https://cdn.jsdelivr.net/npm/xgplayer@3.0.21/dist/index.min.js) · [Clappr 0.3.13](https://cdn.jsdelivr.net/npm/clappr@0.3.13/dist/clappr.min.js) · [Media Chrome 4.11.1](https://registry.npmjs.org/media-chrome/-/media-chrome-4.11.1.tgz) · [OpenPlayerJS 2.14.3](https://cdn.jsdelivr.net/npm/openplayerjs@2.14.3/dist/openplayer.min.js) · [Fluid Player 3.45.0](https://cdn.fluidplayer.com/3.45.0/fluidplayer.min.js)

解压本轮证据包，将下列目录参数指向包中的 `new-recordings`、`title-repeat` 与 `preceding-player-refs`。在已安装项目及 OCR 依赖的仓库环境运行：

```shell
python packages/motionspec/bench/score_overlay_sequences.py new-recordings sampled
python packages/motionspec/bench/score_overlay_sequences.py new-recordings sampled --output default.jsonl
python packages/motionspec/bench/score_overlay_sequences.py new-recordings sampled --fallback-icon-dir preceding-player-refs --output supplemented.jsonl
python packages/motionspec/bench/score_overlay_sequences.py title-repeat title-sampled
python packages/motionspec/bench/score_overlay_sequences.py title-repeat title-sampled --all-text --output title-review.jsonl
```
