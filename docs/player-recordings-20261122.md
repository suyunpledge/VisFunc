# 字幕与播放器图标的录屏复测

在 Chrome 中实际运行 DPlayer 1.27.1、ArtPlayer 5.2.2、MediaElement.js 7.0.7，每款录制 6 段 WebM。每段固定取最后一张解码帧，基线 `af973f4` 与本轮代码读取相同像素。前两款用于发现问题和调参，MediaElement.js 的识别结果用于留出验证。下面的计数按画面或标注实例汇总。[逐帧转写、检测框、真值与哈希](../bench/results/20261122-player-recordings/player-recordings-20261122.jsonl)

## 字幕与片头

| 播放器 | 有字幕画面逐字正确，前 → 后 | 无字幕画面误收文字，前 → 后 |
|---|---:|---:|
| DPlayer | 0 → 3 / 4 | 2 → 0 / 2 |
| ArtPlayer | 1 → 4 / 4 | 1 → 0 / 2 |
| MediaElement.js（留出） | 1 → 3 / 4 | 1 → 0 / 2 |
| 合计 | 2 → 10 / 12 | 4 → 0 / 6 |

自动区域结合行的位置、相对字号、字宽和相邻字幕块，把疑似片头保存在 `detached_text_candidates`。使用 `--player-toolbar` 时，底部独立计时和成排的倍率读数也参与分流。显式 `--subtitle-region` 允许调用方读取指定位置的文字。编码空格复核在两种笔画阈值下测量空白列，再分别转写间隙两侧；确认原字符后才插入空格。源像素横坐标和原转写保存在 `spacing_review`。[识别实现](../svsg/recognition/subtitle_ocr.py)

新录屏中的 `RX 318`、`Line R 42`、`Deck Q 7` 都补回了空格，紧排对照 `RX318` 与 `N24` 保持原样。剩余两个逐字错误是 `EXIT NOW` 和 `TURN BACK` 的纯字母词间空格。有字幕的 12 帧共 96 个参考字符，字符编辑距离从 143 降到 2。旧版把播放器计时和背景读数拼入转写，插入错误可使编辑距离大于参考字数。

另对同一 `DEEP BLUE` 片头的两次录屏分别取末尾 0.8 秒，每隔 200 ms 读取一帧。旧版共 10/10 帧把片头列入字幕，新版为 **3/10**。主评分录屏中 5/5 帧都成功分流，另一次录屏中后 3 帧误收。相同分辨率下，末帧 OCR 框高度从 20 像素变到 17 像素，跨过了字号阈值。这个补测包含主表的一帧，单独报告，不与主表相加。[跨帧转写与位置](../bench/results/20261122-player-title-repeat/player-title-repeat-20261122.jsonl)

历史 54 张截图也按原像素哈希回放，其中含 3 张早期开发样例。整帧字幕完全一致从 45/54 升至 47/54，唯一改变的两张是单独的片头 `PRESENTS` 与字幕 `AB 204`。紧排的 `GateB12` 和 `AB12345` 都保留。该组通用底栏图标从 76/159 升至 111/159，两版误报均为 0。历史回放和新录屏分别计数。[历史画面索引](../bench/results/20261122-subtitle-icons-cross-player/subtitle-icons-cross-player-20261122.jsonl)

## 陌生图标样式

识别范围为画面底部 15%。每款有 5 段显示工具栏、1 段隐藏；每个可见工具栏标注播放或暂停、音量或静音、全屏共 3 个图标。未纳入标签集的设置键等仍出现在搜索区域，若被错报为已知标签则计误报。同标签的预测框与真值框按 IoU ≥ 0.5 一对一匹配。表中命中按实例计数，重复框计入误报。

| 播放器 | 通用模板命中，前 → 后 | 通用模板误报，前 → 后 | 上一轮双播放器模板，命中 / 误报 | 当前皮肤参考模板，命中 / 误报 |
|---|---:|---:|---:|---:|
| DPlayer | 5 → 7 / 15 | 0 → 0 | 5 / 0 | 14 / 1 |
| ArtPlayer | 8 → 13 / 15 | 0 → 0 | 14 / 0 | 15 / 0 |
| MediaElement.js（留出） | 2 → 7 / 15 | 0 → 0 | 12 / 0 | 15 / 0 |
| 合计 | 15 → 27 / 45 | 0 → 0 | 31 / 0 | 44 / 1 |

工具栏增加了对角箭头、三角角标、宽暂停条和扬声器变体，并在小尺寸区间逐像素搜索。静音叉号会单独复核，避免仅凭与音量图标共享的扬声器主体触发。通用条件只用程序生成的形状；旧库取自 Video.js 和 Plyr；当前皮肤参考来自目标播放器自身渲染的矢量字形。这最后一列使用了目标产品的先验信息。[匹配实现](../svsg/recognition/icon_matching.py)

通用模板的 18 个漏检中有 6 个静音、5 个全屏、4 个音量和 3 个暂停。DPlayer 的 5 个全屏角标全部漏掉；MediaElement.js 的 3 个暂停也未命中。三段隐藏工具栏的负例均为零图标误报。使用当前皮肤参考时，DPlayer 一帧把有声图标报成了静音，产生一次误报和一次漏检。该条件的命中总数由旧版 39/45 升至 44/45，误报都为 1；旧播放器模板两版均为 31/45，误报为 0。

## 素材、标注与复现

每款录屏覆盖播放、暂停、有声、静音与隐藏底栏。片头、字幕由 FFmpeg/libass 写入视频后进行 H.264 编码，播放器控件由浏览器实际绘制，Playwright 再录为 WebM。准备程序核对媒体状态和底栏范围，按实际 CSS 颜色、描边和亚像素位置单独光栅化矢量以取得标注框。曾发现 MediaElement.js 的容器未铺满视口，以及继承的描边在标注中丢失，均已修正并重新录制评分。评分程序核验 WebM 的 SHA-256、目标帧序号以及 PNG 与重新解码像素的一致性，识别器仅接收帧图。无字幕误收按画面计一次，与有字幕画面的逐字错误分开。

本轮有 18 个评分时刻，来自三个播放器、反复出现的五类控件与三种输入视频。前三分之二的录屏参与过调参。样本数不能用于估计任意播放器的总体准确率；逐帧评分也未衡量持续出现、闪烁或跨帧漏读。

在安装 Chrome、FFmpeg（含 libass）、Playwright、OpenCV 与项目 `overlays` 依赖的环境中，将固定资产放入 `assets/`。

- `dplayer.js` 和 `artplayer.js` 取自固定版本的发布包。[DPlayer 1.27.1](https://cdn.jsdelivr.net/npm/dplayer@1.27.1/dist/DPlayer.min.js) · [ArtPlayer 5.2.2](https://cdn.jsdelivr.net/npm/artplayer@5.2.2/dist/artplayer.js)
- `mejs.js`、`mediaelementplayer.css` 和 `mejs-controls.svg` 来自 MediaElement 7.0.7。[脚本](https://cdn.jsdelivr.net/npm/mediaelement@7.0.7/build/mediaelement-and-player.min.js) · [样式](https://cdn.jsdelivr.net/npm/mediaelement@7.0.7/build/mediaelementplayer.min.css) · [图标](https://cdn.jsdelivr.net/npm/mediaelement@7.0.7/build/mejs-controls.svg)
- `park.mp4` 是 W3C 的计时测试卡，`sintel.mp4` 是 Sintel 预告片，`oceans.mp4` 是海洋视频。测试卡用于检验数字、倍率与字幕共存的画面。[W3C 计时素材](https://media.w3.org/2010/05/video/movie_300.mp4) · [Sintel 预告片](https://media.w3.org/2010/05/sintel/trailer.mp4) · [海洋视频](https://vjs.zencdn.net/v/oceans.mp4)

```shell
python packages/motionspec/bench/record_player_overlays.py assets recordings
python packages/motionspec/bench/score_player_recordings.py recordings previous-player-icons scores.jsonl
```

`previous-player-icons` 使用上轮的 Video.js 与 Plyr 图标，目录结构为 `标签/播放器名.png`。本轮留存的录屏压缩包内已包含这一目录、每款播放器的专用模板、18 段评分 WebM、评分 PNG、`truth.json` 与 `ocr-repeat/` 中额外的片头重录。解压后可直接重跑评分。重新录屏时，播放中的背景可能随浏览器调度而变化。[录制与标注脚本](../packages/motionspec/bench/record_player_overlays.py) · [评分脚本](../packages/motionspec/bench/score_player_recordings.py)
