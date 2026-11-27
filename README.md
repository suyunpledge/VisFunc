# VisFunc

English · [中文](README.zh.md)

**Give a multimodal model the UI recording and its measurement summary.** VisFunc measures changes in source pixels, separates visible motion from estimated design timing, and writes the evidence to `summary.md`. A model can use the recording to identify the control and the summary to quantify what it did.

| Input or measurement method | Qt animations (12 held out) | Chrome recordings (12 held out) | Valid answers / 72 |
| --- | ---: | ---: | ---: |
| Video or timestamped sparse frames | 48.1% | 52.3% | 60 |
| Dense timestamped frames | 37.7% | 42.1% | 49 |
| Model measures the clip with OpenCV in a sandbox | 39.0% | 51.2% | 39 |
| **Video or sparse frames + VisFunc summary** | **82.7%** | **78.2%** | **69** |

The three models answered once per condition on the same clips. MiMo and Seed received video, GPT received timestamped frames. Each scene score counts the applicable motion fields and the number of animations. Scenes and models have equal weight. Missing fields earn no credit, and failed answers score zero. Actual input token counts varied between the dense-frame and summary conditions. The Qt clips were rendered and the Chrome clips recorded from local test pages. [Benchmark, confidence intervals and raw answers](bench/results/20261127-strong-baselines/report.md)

**MiMo and Seed gained the most from the summary.** GPT already scored 91.7% on Chrome with sparse frames and 95.8% with dense frames. Its OpenCV scripts also produced fully correct answers on all nine Chrome clips for which it returned a valid answer. MiMo hit its output limit on 21 dense-frame tasks. These delivery differences matter when a program consumes the result.

| Model | Qt, visual input alone → with summary | Chrome, visual input alone → with summary |
| --- | ---: | ---: |
| MiMo V2.6 Flash | 32.2% → 80.9% | 27.8% → 63.9% |
| Doubao Seed 2.1 Pro | 39.0% → 71.2% | 37.5% → 72.2% |
| GPT-6.1 Sol | 73.1% → 95.8% | 91.7% → 98.6% |

## Quick start

From the repository root with Python 3.11+, install the motion analyzer. Once installed, measurement runs locally without a model key.

```bash
python -m pip install -e "./packages/motionspec"
```

Generate the sample recording and measure it. For your own recording, replace `slide.mp4` in the last command.

```bash
python -c "from motionspec.synth import slide, write_video; frames, _ = slide(); write_video('slide.mp4', frames)"
python -m motionspec analyze slide.mp4 -o slide-out
```

Open `slide-out/summary.md`. The card's tracked anchor moves **299.4 px right**. The change is visible from **167 to 517 ms (350 ms)**. Fitting an `out-cubic` curve puts the design interval at **167 to 567 ms (400 ms)**. The sample was authored to move 300 px in 400 ms. Near the end it travels less than a pixel per frame.

Give the model the clip (or timestamped frames) and the contents of `summary.md`. A useful question for this sample is “Which element moved, how far, and what supports each duration?” The adjacent `spec.json` contains the underlying measurements, and the evidence images let you inspect the detected region. [Motion guide](packages/motionspec/README.md)

### How reliable are the measurements?

On the original analyzer-only holdout, mean scene scores were **69.8% on 30 Qt scenes** and **72.2% on 12 Chrome recordings**. All scored fields and the animation count were correct in 13/30 and 6/12 scenes. These scores compare the analyzer directly with the recorded ground truth. [Analyzer evaluation](docs/analyzer-evaluation-20261127.md)

For a panel that resizes and pushes a neighbour, add `--layout-edges` to track the four visible edges and the neighbour's displacement separately. On a fresh holdout using the same scene families, switching it on raised the analyzer score from 70.6% to 91.1% on Qt and from 70.8% to 100% on Chrome. All 16 reflow scenes passed. An earlier Chrome recording still produced an extra segment, and some Qt fades lacked opacity endpoints. [Layout measurements and remaining errors](bench/results/20261127-layout-edges/report.md)

### What happens when the summary marks a field as uncertain?

`--mark-uncertain` is an opt-in diagnostic. For example, a pixel blend can come from a colour change or a fade. The marked summary explains why the opacity endpoints are uncertain and keeps the observed change visible. On 27 development clips, the option flagged **5 of 30** wrong or missing object fields. It also flagged 11 fields that were already within tolerance.

| Model | Mean scene score, ordinary → marked summary | True gaps answered correctly, out of the 5 flagged |
| --- | ---: | ---: |
| GPT-6.1 Sol | 95.7% → 96.0% | 5/5 → 5/5 |
| MiMo V2.6 Flash | 85.0% → 72.0% | 4/5 → 3/5 |
| Doubao Seed 2.1 Pro | 89.0% → 87.2% after transport recovery | 4/5 → 3/5 |

Both groups received identical visual input in independent sessions. GPT already recovered all five gaps with the ordinary summary. On one fade MiMo originally got both opacity endpoints right, then left both blank with the marker. Its format failures rose from 3 to 6. Seed's ordinary-summary score was 81.5% when two network failures counted as zero, and 89.0% after independent retries. The ordinary summary remains the default. [Paired results, both Seed scores and field-level errors](bench/results/20261127-uncertainty/report.md)

## Advanced use

### Query pixels and motion by time

Use a function when you need a reproducible answer at a particular frame, coordinate or timestamp. In the same 24-clip holdout, visual input plus the function catalogue and optional queries scored **46.9% on Qt and 66.0% on Chrome**, compared with **82.7% and 78.2%** for the summary. The query workflow delivered 50/72 valid answers and the summary delivered 69/72. Extra turns and delivery failures contributed to the gap. [Function-query comparison](bench/results/20261127-strong-baselines/report.md)

Install the image runtime from the repository root, then export the measurements and evidence frames from the quick start.

```bash
python -m pip install -e ".[functions]"
python -m motionspec functions slide-out/spec.json -o slide-functions
python -m motionspec query slide-functions/functions.json --list
```

Save this as `query.json` to read the top-left pixel in frame 0 and the tracked anchor's x coordinate at 250 ms.

```json
[
  {"id": "color", "op": "pixel_rgba", "frame_index": 0, "x": 0, "y": 0},
  {"id": "position", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 250}
]
```

```bash
python -m motionspec query slide-functions/functions.json query.json
```

The pixel result is `[41, 45, 49, 255]` with a timestamp and pixel hash. The motion result is `261.0 source_px` with provenance `motion_estimate`. The same runtime represents still images as pixels, regions, contours and spatial relations. [Video function protocol](docs/video-functions.md) · [Still-image guide](svsg/README.md)

## Desktop and image tools

On Windows x64, extract the portable bundle and double-click `Start.cmd` to start the image API and desktop MCP server. [Deployment guide (中文)](deployment/windows/README.zh.md) · [Production hardening (中文)](docs/agent-production-20261123.md)

### Read subtitles and UI icons

Install local OCR to read Chinese and English captions and locate icons in screenshots or exported video frames. Results include source-pixel boxes and frame provenance. A verified word gap in a caption or code is recorded in `spacing_review`. Automatic subtitle search keeps likely titles in `detached_text_candidates`. For video manifests it can also resolve a fluctuating title box using nearby frames, with the supporting indices in `temporal_review`. A requested `--frame` receives the same review as that frame in the full sequence.

Use `--player-toolbar` for bottom-bar icons and player readouts. `--icon-dir` selects your own PNG template bank. `--fallback-icon-dir` first runs the built-in toolbar detector, then checks unused positions against your templates with support from nearby controls. [Evaluation and options](docs/subtitle-icon-evaluation-20261122.md) · [Earlier recording results](docs/player-recordings-20261122.md) · [Temporal OCR and unfamiliar icon styles](docs/temporal-player-overlays-20261122.md)

The latest recording benchmark measures opening-title false positives, raised captions, and volume and fullscreen controls across five players, including a held-out Shaka Player skin. [Results and reproduction steps](docs/volume-title-player-20261122.md)

```bash
python -m pip install -e ".[overlays]" -e "./packages/motionspec"
motionspec overlays screenshot.png
motionspec overlays slide-functions/functions.json --player-toolbar
```

These rules were tuned on video captions and player toolbars. On ordinary application screens they report a forward arrow as a play button and can give several different readings for the same text; use `screen-text` below instead.

### For agents: wait for the screen, find where to click, read the text

`motionspec agent-mcp` is a resident MCP server. Once an agent client registers it, the agent can call these tools:

- `wait_for_settle` observes longer operations or input from another tool. Built-in actions already wait and return their effect. The default quiet interval is 200 ms.
- `find_on_screen`: finds where to click by name or visible text. It checks Windows UI Automation first, which names icon-only buttons and knows which controls are disabled or checked, then OCR.
- `click_element`, `set_field_text`: press buttons and fill fields through UI Automation, only in windows allowed with `--allow-window` at start. Each action returns its effect directly.
- `read_screen_text`, `list_ui_elements`, `screenshot`.

`click_point` returns the dispatched physical screen coordinates in `operation_marker.screen_px`. A named UI Automation action labels the control centre as `uia_target_center`. Image-capable clients can request a marked pre-action crop with `marker_image=true`.

For a custom-drawn button with a known label, call `click_text(window_title=…, text=…)`. If the layout changes during OCR, the tool reads the new layout once and uses the updated coordinates. It verifies the entire recognition area before dispatch so a newly duplicated label is caught even when the original stays put. Missing, ambiguous or low-confidence labels and continuously changing layouts return `execution_state=not_started`. The receipt includes the recognized text, its box and the dispatched screen point. [Keystone desktop comparison](docs/agent-screen-targets-20261125.md) · [Repaint-during-OCR comparison](docs/agent-click-recheck-20261126.md)

Actions return a `screen_history.history_id`. Use `read_screen_history` with `compact=true` to recover a transient notification, changed field or dialog in that window. Continue from `after=next_after`. Request `read_text=true` for OCR of pixel evidence. The default recording lasts 20 seconds, sampling pixels at up to 8 Hz and controls at up to 4 Hz. `watch_screen_changes` starts a recording independently and `--history-seconds 0` disables it. [Event cache and token measurements](docs/agent-ui-events-20261123.md)

For HTTP retries, generate a UUID per intended action and reuse it in the `Idempotency-Key` header across reconnects. MCP clients can also send it in `params._meta["visfunc/operationId"]`. Within a negotiated session the same JSON-RPC request ID also identifies a retry. Receipts are retained for 24 hours. A replay reporting `unknown` calls for a read-only check of the window. [Retry and recovery details](docs/agent-ui-events-20261123.md)

Coordinates are physical screen pixels, the ones a click uses. The server learns areas that never stop changing, such as video or an animated background, once, and they no longer slow down the wait. Without MCP, use `motionspec settle --screen --read-text`, `motionspec screen-text` or the HTTP endpoint `POST /v1/screen-text`.

```bash
python -m pip install -e ".[overlays,uia]" -e "./packages/motionspec[capture]"
motionspec agent-mcp
```

On a real Qt window on the desktop, all 10 controls (6 of them icon-only) were found by name, most lookups in under 60 ms. After a status text changed, the screen was judged settled about 210 ms later and the whole line was read. On four kinds of web page rendered by Chrome, 239 of 240 pieces of text could be clicked by their text, including every cell of a 25-row table. Icon-only buttons in a browser are not found by OCR, and UI Automation on a Chrome window has not been checked yet. [Usage and evaluation](docs/agent-screen-20261122.md) (Chinese)

Connected to Pledge Keystone (`--http`), GPT-6.1 sol and terra completed four kinds of desktop task: a Qt settings window with a confirmation dialog, retrying a failed sync after fixing the port named in an error dialog, a sign-up form in Chrome, and a custom-drawn window that can only be clicked by coordinates. The latest 16 runs, with no turn limit, all succeeded in 20 to 65 seconds each. The retry task needs 13 to 15 tool calls; Keystone used to stop at 12 by default and now has no step limit unless one is set in its advanced settings. Keystone's own window driver sees no controls in Qt windows. On this machine, ESET's browser screen protection blocks screenshots of browser windows; the server then judges effects through UI Automation instead. [Keystone test](docs/agent-keystone-20261122.md) (Chinese)

## Historical results

Earlier development tests used six model configurations and a visual-input-only baseline. The six basic Qt animations scored 55% with visual input, 92% with added measurements and 94% with live Qt properties. The nine harder Qt scenes scored 58%, 82% and 100%, respectively. These were repeatedly used development scenes with field-by-field scoring. The analyzer alone scored 85% on those nine scenes. An earlier measurement-version comparison on those same scenes scored 68% and 86%. The holdout table at the top uses scene averages and includes dense frames and OpenCV. [Historical runs and scoring](bench/results/README.md) · [Sampling-rate ablation](docs/evidence-ablation-20261116.md)

In a separate two-animation, two-model test without images, across nine matched runs with valid answers, models got 145/147 items right from parameters and 147/147 from the text summary. In the basic Qt scenes, evaluating exported functions against the recorded properties gave position and entering/exiting size errors of at most 1 px, and zero opacity error. [Parameter-only retest](docs/timing-protocol-retest-20261117.md) · [Function evidence](docs/video-functions.md)

On COCO8 (8 images, 30 objects), multi-view review reduced false positives from 9 to 2 and increased misses from 10 to 16. At eight new sample times from four MOT20 videos, the local pedestrian detector recovered three more heavily occluded people (514 → 517 of 740), with 36 false positives in both versions. [Image evaluation](docs/recognition-evaluation-20261116.md) · [Pedestrian evaluation](docs/pedestrian-focused-review-20261122.md)

## Current limits

- 3D and game footage only gets summary statistics, with no usable motion curves.
- Overlapping elements can still merge into one segment. Four-edge tracking works best when panel boundaries remain visible.
- Colour transitions in compressed video are smeared by H.264 chroma subsampling.
- Objects are not tracked across frames. A segment ID refers to a change, not to an object.
- Object recognition does not guarantee that every object is found.

## Install and test

```bash
python -m pip install -e ".[api,llm]"                                  # HTTP service and model calls
python -m pip install -r requirements-dev.txt                          # development and tests
python -m pytest
```

The Qt scene tools and the probe demos also need PySide6. Monorepo installation, testing and code boundaries are described in
[docs/monorepo.md](docs/monorepo.md) (Chinese).

## Layout

```
svsg/                  VisFunc-Image
packages/motionspec/   VisFunc-Motion (its own package and command)
docs/                  function protocol, recognition, monorepo notes, evaluation records
tests/                 VisFunc-Image tests; VisFunc-Motion tests are in packages/motionspec/tests
scripts/               service launch and evaluation scripts
```

## License

[MIT](LICENSE)
