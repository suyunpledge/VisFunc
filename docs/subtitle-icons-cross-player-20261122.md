# 跨播放器复测：烧录字幕与多样式图标

在代码规则冻结后另外生成的 8 张 Plyr 播放画面上，6 张有字幕画面中逐字正确从 **2 张提高到 5 张**，字符编辑距离从 **20 降至 1**。两张无字幕画面中，两版都把单独出现的片头字样 `PRESENTS` 报成了字幕一次。图标方面，同一模板库同时覆盖 Video.js 和 Plyr 的字形；在这 8 张画面上自动匹配到 **21/21 个**底栏图标，误报为 0。[逐图结果](../bench/results/20261122-subtitle-icons-cross-player/subtitle-icons-cross-player-20261122.jsonl)

## 字幕

字幕框可能从第二或第三个字才开始，即使框内转写置信分很高。新流程用框内高局部对比度的笔画估计字色，检查框外同一行是否有连续笔画。只在原色扩框和单独提取的笔画图都读出相同扩展文字、原文被保留、扩展方向与出现笔画的一侧相符时，才扩大输出框。显式指定的裁剪区域仍是硬边界。Video.js 旧例中，“站：中央广场。”经复读补回了“下一”，框的左缘从 272 移到 246 像素。

对整行再做两种尺度的识别，只有原字符与已有空格都保留、两次高置信结果一致时，才接受补入的空格或句末标点。自动字幕区域里，若低处已有字幕，上方相隔明显的文字另列在 `detached_text_candidates`。垂直间距接近的多行字幕作为连续块保留。指定 `--subtitle-region` 时保留该区域内读出的所有行。[复核实现](../svsg/recognition/subtitle_ocr.py) · [输出协议](../svsg/recognition/overlays.py)

| 画面与使用阶段 | 有字幕画面完全正确，前 → 后 | 字符编辑距离，前 → 后 / 参考字符数 | 无字幕画面出现误报，前 → 后 |
|---|---:|---:|---:|
| Video.js 原评分集，17 张历史回放 | 8 → 10 / 12 | 8 → 4 / 239 | 0 → 0 / 5 |
| Plyr 首批，14 张，检查跨皮肤及字符错误 | 7 → 9 / 10 | 5 → 1 / 187 | 0 → 0 / 4 |
| Plyr 第二批，12 张，复查片头误收 | 5 → 6 / 10 | 35 → 8 / 139 | 0 → 0 / 2 |
| **Plyr 最终留出，8 张** | **2 → 5 / 6** | **20 → 1 / 87** | **1 → 1 / 2** |

编辑距离按有字幕画面汇总，大小写、空格、标点和分行都计分。无字幕误报单列。最终组有两张画面的上方片头文字与下方字幕正确分开；唯一未逐字读对的有字幕画面把 `AB 204` 读成了 `AB204`。单独出现的片头文字缺少与另一行字幕比较的依据，这种情况仍会误报。若应用有固定字幕区域，可按实际位置指定源像素范围。

## 图标

同一语义可在 `--icon-dir` 下提供多个画法，例如 `play/videojs.png` 与 `play/plyr.png`。两种画法输出同一 `label`，获选模板写入 `template_id`，可在 `icon_templates` 中按名称查到像素哈希。原有 `play.png` 布局继续可用。

小尺寸单色图形增加了笔画与背景的对比度检查。启用 `--player-toolbar` 时，对原始形状接近门槛的候选，还会比较平滑后的形状，并要求同一行有两个位置分离、大小相近的强匹配控件支持。复核结果写入 `row_support_px` 与 `smoothed_shape_score`。[图标实现](../svsg/recognition/icon_matching.py)

| 底栏图标配置 | Video.js 历史 17 张 / 57 个 | Plyr 全部 34 张 / 90 个 | 其中 Plyr 最终 8 张 / 21 个 |
|---|---:|---:|---:|
| 通用几何模板，前 → 后 | 30 → 31 | 36 → 36 | 9 → 9 |
| 使用另一款播放器的参考模板 | 16 | 30 | 7 |
| 手动选对该播放器的模板，前 → 后 | 54 → 54 | 90 → 90 | 21 → 21 |
| **固定混合库，所有输入共用** | **54** | **90** | **21** |

以上底栏条件的误报都是 0。取消范围限制后，历史组的通用模板整帧误报从 7 降至 2；Plyr 的 34 张整帧误报仍为 8。混合库使用两个产品的公开字形单独渲染参考图，视频识别时无需指定皮肤。未收入模板库的陌生控件仍受形状差异影响。

## 素材与复现

Plyr 3.7.8 在 Chrome 内实际播放输入视频。字幕由 FFmpeg/libass 先烧入视频并经 H.264 YUV420 编码，CRF 范围为 25–35。编码后的视频和真实播放器控件一起截屏；字幕真值来自编码前的文本，图标真值来自同一布局下渲染的字形掩膜。测试覆盖三个素材文件与两种播放器皮肤，中文和英文字幕为受控添加。每次版本对照均使用相同的 PNG 字节；播放中的重新截屏可能因调度产生背景差异，逐图记录保留了哈希。标注数量包含同一控件在多个画面上出现的次数。

在装有 Chrome、FFmpeg（含 libass）、Python Playwright 与 `svsg[overlays]` 的环境中，将以下源码资产按括号内名称放入 `assets`。播放器来自 [Plyr JS](https://cdn.jsdelivr.net/npm/plyr@3.7.8/dist/plyr.min.js)（plyr.js）、[样式表](https://cdn.jsdelivr.net/npm/plyr@3.7.8/dist/plyr.css)（plyr.css）、[图标集](https://cdn.jsdelivr.net/npm/plyr@3.7.8/dist/plyr.svg)（plyr.svg）。视频来自 [MDN Flower](https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4)（flower.mp4）、[Sintel 预告片](https://media.w3.org/2010/05/sintel/trailer.mp4)（trailer.mp4）、[Big Buck Bunny 预告片](https://media.w3.org/2010/05/bunny/trailer.mp4)（bunny.mp4）。

```bash
python packages/motionspec/bench/prepare_plyr_cases.py assets plyr-cases
python packages/motionspec/bench/overlay_eval.py plyr-cases scores.jsonl --mixed-icons mixed-icons
motionspec overlays plyr-cases/case26.png --player-toolbar --icon-dir mixed-icons
```

生成器会创建 `plyr-cases/icons`。要复现混合库，将该目录与[上一轮生成](subtitle-icon-evaluation-20261122.md)的 Video.js 图标分别存入 `mixed-icons/类别/plyr.png` 和 `mixed-icons/类别/videojs.png`。首批样本编号为 0–13，第二批 14–25，最终留出 26–33。新旧版本的原始输出、逐图真值和底栏一对一 IoU ≥0.5 评分保存在配对记录中。[生成器](../packages/motionspec/bench/prepare_plyr_cases.py) · [评分程序](../packages/motionspec/bench/overlay_eval.py)
