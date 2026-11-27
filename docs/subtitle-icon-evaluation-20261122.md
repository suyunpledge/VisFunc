# 字幕 OCR 与播放器图标识别

后续验证加入了 Plyr、H.264 烧录字幕、片头文字干扰及跨皮肤模板库。[跨播放器复测](subtitle-icons-cross-player-20261122.md)

在 17 张评分画面上，局部双尺度 OCR 将字幕逐字完全正确的画面从 **5/12 提高到 8/12**，两组在其余 5 张无字幕画面上均返回空结果。图标使用通用模板扫描整帧时命中 **30/57**、误报 7 个。把搜索限定到底部 15% 后仍命中 30 个，误报降至 0 个。同一范围内换成播放器自身的图标样本，命中 **54/57**。[逐图参考答案、输出框、错误数和输入哈希](../bench/results/20261122-subtitle-icons/subtitle-icons-20261122.jsonl)

## 在自己的画面上运行

```bash
python -m pip install -e ".[overlays]" -e "./packages/motionspec"
motionspec overlays screenshot.png --only subtitles
motionspec analyze recording.mp4 -o out --functions --evidence-every-ms 250
motionspec overlays out/functions.json --player-toolbar
motionspec overlays out/functions.json --frame 12 --icon-dir known-icons --player-toolbar
```

单幅图片直接读取。视频从 `functions.json` 读取已导出的帧，沿用实际帧号、采样时间和像素哈希。每个字幕或图标候选都有原画面半开像素框 `[左, 上, 右, 下)`。相邻采样帧像素相同时复用识别结果，各帧分别保留时间戳。导出间隔影响短暂字幕的可见性，仅能对已采样到的画面计数。

- **字幕**默认读取画面下方居中的文字区域，每帧以两种分辨率读取，按位置合并同一行，保留识别出的大小写、标点和分行。左对齐字幕、顶部字幕或多栏界面可用 `--subtitle-region L T R B` 指定源像素区域，解除居中筛选。字幕区域内的路牌、横幅也可能被读成字幕候选。
- **图标**内置播放、暂停、静音、音量、全屏、字幕、搜索和关闭八种几何模板。默认搜索整帧，`--player-toolbar` 适用于底部播放栏，其他布局可用 `--icon-region L T R B`。已知应用可用 `--icon-dir` 选用自己的 PNG 库，文件名就是输出类别，例如 `settings.png`。透明单色字形可在深浅底上匹配，含颜色的样本会额外核对颜色。数量统计只覆盖模板库中的类型。
- **证据**在 JSON 中保留字幕原文、OCR 引擎版本、图标样本哈希、搜索范围和分数。OCR 分数与图标匹配分数都适用于各自的候选排序。这里未做概率校准。[字幕实现](../svsg/recognition/subtitle_ocr.py) · [图标实现](../svsg/recognition/icon_matching.py) · [图像与视频入口](../svsg/recognition/overlays.py)

## 配对测试

背景取自 MOT20-01、02、03、05 四段实拍录像；字幕由测试脚本设定，在 Chrome 的 Video.js 8.23.4 播放器中实际渲染。输入是最终截屏。识别器只读像素，评分器独立取得字幕原文及控件类别。图标参考框来自同一控件在黑底重绘时的白色字形像素范围。三张开发画面用于检查裁剪与坐标，其余 17 张计分，包括后来补测的 4 张浅色工具栏画面。宽度为 480–960 像素，字幕字号为 10–19 像素。播放器模板单独从样式表对应的字体渲染，未从评分截屏裁取。

| 字幕指标 | 整帧单次 OCR | 字幕区域双尺度 OCR |
|---|---:|---:|
| 全句准确（12 张有字幕画面） | 5 | 8 |
| 字符编辑距离 / 239 个参考字符 | 23 / 239（9.6%） | 8 / 239（3.3%） |
| 无字幕画面出现文字 / 5 张 | 0 | 0 |

两组共用本地 RapidOCR 1.4.4 的中英文权重、居中筛选和置信门槛。参考文本与预测文本统一为 Unicode NFC，保留中英文间空格、换行及弯引号。8 个残余字符错误中包括中文数字两侧的空格、英文撇号，以及一张浅色主题双语画面中漏掉的“下一”二字。

| 图标检索条件 | 命中 / 57 | 误报 | 漏检 |
|---|---:|---:|---:|
| 通用模板，固定 16 像素，整帧 | 7 | 0 | 50 |
| 同一套模板，多尺寸，整帧 | 30 | 7 | 27 |
| 同一套模板，多尺寸，底部 15% | 30 | 0 | 27 |
| 播放器模板，多尺寸，底部 15% | 54 | 0 | 3 |

图标要求同类别、IoU ≥0.5、一对一匹配。限定底栏单独消除了本组 7 个误报；替换样本库又补回 24 个命中。余下 3 个漏检为两个全屏图标和一个字幕图标。最后补测的浅色工具栏中，播放器模板命中 15/15。上述样本共用一种播放器和四段背景录像，字幕为受控叠加。其他应用、影视作品原生烧录字幕和压缩质量的表现仍待验证。

本机 CPU 上，以一张开发画面预热后依次测量 17 张评分画面，单帧字幕 OCR 中位耗时为 0.41 秒。通用图标整帧与底栏搜索分别为 1.06 秒、0.08 秒，播放器样本底栏搜索为 0.07 秒。此处只计已加载模型和模板后的识别时间；实际视频导出、解码与模型加载需另计。计时期间本机也在执行回归测试。

## 复现

环境需要 Chrome、FFmpeg、Python Playwright、本项目的 `overlays` 和 MotionSpec。将固定版本 `video.js@8.23.4` 的 `dist` 目录放在本地。素材准备、截图渲染与离线评分可分别运行。播放状态下的截屏时刻可能随调度略有变化，逐图记录中的 PNG 哈希用于核对是否使用同一份输入。

```bash
python packages/motionspec/bench/pedestrian_clips.py clips/t3 --dataset MOT20 --start-seconds 1 --sequences 01 02 03 05
python packages/motionspec/bench/prepare_overlay_cases.py clips/t3 video.js/dist overlay-cases
python packages/motionspec/bench/overlay_eval.py overlay-cases overlay-scores.jsonl
```

生成的 `overlay-cases/icons` 可直接传给 `motionspec overlays --icon-dir`。逐图结果中的 `case=0..2` 属于开发组，`case=3..19` 计入上表。两张没有字幕也没有工具栏的负例同样进入评分。[素材与真值生成](../packages/motionspec/bench/prepare_overlay_cases.py) · [评分程序](../packages/motionspec/bench/overlay_eval.py)
