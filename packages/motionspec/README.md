# VisFunc-Motion

English · [中文](README.zh.md) · [Back to VisFunc](../../README.md)

VisFunc-Motion measures animation in a screen recording of a user interface: which element moves, when it starts, how far it
goes, how its size and opacity change, and which easing curve it follows. If the target is your own Qt app, it can skip the
recording and read widget properties every frame instead. The result is a numeric spec, a one-page summary and a few annotated
images, which can go to a model together with the original footage, or be exported as functions that can be queried by time.

The code is in `packages/motionspec/`; the Python package and command are `motionspec`. It depends only on numpy and
opencv-python; exporting functions also needs [VisFunc-Image](../../svsg/README.md) from the same repository.

```
recording ─▶ where and when things change ─▶ how (move, scale, rotate, size, opacity) ─▶ easing curves ─▶ spec.json, summary.md, sheets, functions
```

## Install and use

In this directory:

```bash
pip install -e .
```

```bash
motionspec analyze recording.mp4 -o out/
```

```bash
motionspec demo -o demo/
```

Install VisFunc-Image as well, from the repository root, to export and query functions:

```bash
python -m pip install -e ".[functions]" -e "./packages/motionspec"
motionspec analyze clip.mp4 -o out --functions --evidence-every-ms 500
motionspec query out/functions.json --list
```

From Python:

```python
from motionspec import analyze, check

spec = analyze("clip.mp4", "out/")   # also accepts a directory of images or a list of frames in memory
verdict = check(spec, [{"segment": "s1", "field": "easing", "value": "ease-out"}])
```

Common options:

| Option | Effect |
|---|---|
| `--start`, `--end` | analyse only this time range (seconds) |
| `--fps` | required for a directory of images; files are sorted naturally, `1.png`, `2.png`, `10.png` |
| `--max-dim` | longest side used for analysis, default 960 |
| `--min-area`, `--max-segments` | ignore smaller changes; keep at most this many segments |
| `--ambient-ms` | how long a region must keep changing to count as ambient, default 3000; 0 turns it off |
| `--max-frames`, `--max-crop-mb` | frame limit for the selected range; crop cache limit, default 150 MB |
| `--no-sheets`, `--no-raw-frames` | skip the annotated images; skip the source-resolution evidence frames |
| `--functions`, `--evidence-every-ms`, `--function-max-frames` | export functions; add evidence frames at a fixed interval; frame limit for the function bundle |

### Waiting for the screen to settle (for agents)

`analyze` fits curves and takes 1 to 3 seconds for a 0.6 second clip, too slow for an agent waiting after each click. `settle` only answers whether the screen has stopped and what changed. It judges frames as they arrive, returns `settled` after `--quiet-ms` (default 200) of stillness or `unchanged` if nothing changed within `--expect-ms`, and lists the regions that differ from the first frame. Each frame takes about 6 ms.

```bash
motionspec settle --screen --region 0 0 1920 1080 --read-text   # run after clicking; reads text in changed regions
motionspec settle recording.mp4                                  # also takes a recorded video or frame directory
```

Areas that keep changing (video, an animated background) stop counting after `--ambient-ms`. A loading spinner looks the same, so set `--ambient-ms 0` when you must wait for loading to finish; exclude a clock or similar with `--ignore`.

For agents, the resident MCP server `motionspec agent-mcp` is easier. It keeps the capture stream, the OCR model and the learned background areas, and offers `wait_for_settle`, `find_on_screen` (Windows UI Automation first, then OCR), `read_screen_text`, `list_ui_elements` and `screenshot`, plus `click_element` and `set_field_text` with `--allow-window TITLE`. It uses stdio by default; `--http HOST:PORT` serves HTTP instead, and Pledge Keystone also needs `--no-images`. Capture is faster with mss installed (`pip install -e ".[capture]"`). [Usage and evaluation](../../docs/agent-screen-20261122.md) (Chinese)

## Output

| File | Contents |
|---|---|
| `summary.md` | one row per segment: visible change, estimated design timing, kind, change, easing, confidence. Read this first |
| `spec.json` | the full spec, `motionspec/0.1`, with up to 40 samples per segment and a record of anything filtered or truncated |
| `s1.png` … | one per segment: key frames side by side, the motion path, measured samples against the fitted curve |
| `overview.png` | the first frame with each segment's region marked |
| `frame-<index>-<SHA-256>.png` | decoded source-resolution frames: the first and last of the range and each segment's key frames. `evidence_frames` records index, time, role and hash |
| `functions.json`, `raster-*.json` | the function bundle and pixel data, with `--functions` |

When nothing changes, the first and last frames are still kept.

### Segment kinds

| Kind | Meaning | Values reported |
|---|---|---|
| `transform` | one element moves, scales, rotates or changes width or height, possibly while changing opacity | timing and fit for each channel |
| `fade` | frames are close to a mix of the first and last frame: fades, colour transitions, a dimming overlay | duration and blend progress curve |
| `instant` | a change completed within one frame | when it happened |
| `nonrigid` | none of the above explains it: glow, blur, particles, 3D footage | statistics only, such as luminance range and period; confidence is always low |

### How it is measured

The first pass finds where and when things change. Each frame is compared in colour with a reference frame and the result is
pooled onto a coarse grid; a cell's reference is refreshed only after several unchanged frames, so even slow fades are caught.
Cells that keep changing for more than 3 seconds (animated backgrounds, videos, spinners, a blinking caret) are listed under
`ambient` and left out of segment detection; otherwise they would join everything around them into one segment spanning the
whole clip. The remaining changes are grouped into windows by connectivity in space and time.

The second pass explains each window, trying the paths below in order. The `model` field says which one was used:

| Path | `model` | When it applies | How |
|---|---|---|---|
| Backdrop split | none (two segments) | the whole window dims or changes colour along one curve while an element moves on top (a modal) | report the overlay as a `fade`, remove its change from every pixel, then measure the element on top |
| Solid box | `solid-box` | a block of colour that changes size along one axis: progress bars, tab indicators | find that colour's bounding box in each frame; `size` gives width and height from and to, `anchor` the edge it grows from, and `translate` is added when both edges move |
| Layer | `layer` | enter or exit: an element moves or scales while fading | use the end frame without the element as background and solve each frame as "background + opacity × transformed element", giving movement, scale, opacity and sub-pixel size |
| Tracking | none | an element present at both ends changes position or size | feature optical flow and a similarity transform |
| Multi-body | `multibody` | several elements in one area move in different directions: reordering, swapping | group feature points by their whole trajectory and follow each group with a template, forwards and backwards, through occlusion |
| Blend, instant, non-rigid | none | everything else | see the segment kinds above |

Elements that were already present and merely moved do not go through the layer path, including plain cards with little texture,
because those can also be fitted numerically as a fade-in.

The layer path has a problem that optical flow does not: at the end where opacity is 0, the element's position cannot be seen.
Fields on the channel say how that end was filled in:

| Field | Meaning |
|---|---|
| `follows_opacity: true` | movement is proportional to (1 − opacity), so both follow one curve and the hidden position follows from the ratio |
| `extrapolated: true` | the two curves differ; the hidden part is at most 30% of the motion and is extrapolated from the fit |
| `lower_bound: true` | too much is hidden; only the visible distance is reported and the true value is at least that |
| `visible_distance_px` | the distance that was actually measured |

Layer opacity is relative to the fully shown frame. An element that goes from 0 to 0.8 opacity reads as 0 to 1.

### Easing and timing

Each channel is fitted three ways: a free `cubic-bezier(...)`, an under-damped spring (when there is overshoot), and the nearest
named curve (common CSS, Material and easings.net curves). Named curves may start earlier and end later than the visible change,
because the slow ends of an eased motion move less than a pixel per frame and cannot be measured.

So every segment has two timings, in the format `svsg.motion-timing/1`:

- `visible_change`: the interval in which change can be seen. The segment's top-level `t_start_ms`, `t_end_ms` and
  `duration_ms` mean the same thing.
- `estimated_design`: the designed start and duration from the best-aligned named curve, with its RMSE and alignment basis.
  It is `null` when the fit is unreliable (RMSE above 0.05), the segment was truncated, or the estimated start would fall
  before 0; `design_unavailable_reason` says which.

When reading the numbers:

- Times are uncertain by about one frame: 17 ms at 60 fps, 33 ms at 30 fps.
- `duration_ms` is usually shorter than the designed duration. Use `estimated_design` to recover the designed values.
- With few samples the curve is not unique. A 150 ms animation at 60 fps has 9 samples and several curves fit; such segments
  carry a "few samples" note.
- Variable-frame-rate video uses the decoder's timestamps. If they repeat, go backwards or are invalid, the range is rescanned
  using frame indices and the nominal frame rate.
- On the tracking path, movement is that of a feature anchor, not necessarily the element's centre. `transform_reference`
  says what the reference point is.

## Functions and queries

`--functions` (or `motionspec functions out/spec.json -o functions` for existing results) writes `functions.json` in the
format `svsg.video-functions/1`, compiled by VisFunc-Image. Each kept frame gets a pixel function that restores it exactly,
plus region and contour descriptions; neighbouring kept frames get difference statistics; each motion segment gets sampled
functions (linear between samples) and fitted functions (Bézier or spring).

Read the catalogue first. It lists only what can be queried and is much smaller than the full manifest (about 4 KB for a
132 KB manifest):

```bash
motionspec query out/functions.json --list
```

Then query. A parameters file holds one call or a list of up to 1000; a failing call reports an error in its own result only:

```json
[
  {"id": "p", "op": "pixel_rgba", "frame_index": 0, "x": 100, "y": 80},
  {"id": "x", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 300},
  {"id": "e", "op": "motion_value", "segment_id": "s1", "name": "translate.progress", "t_ms": 300, "mode": "fitted"}
]
```

```bash
motionspec query out/functions.json query.json
```

`mean_rgba` and `delta_rgba` are also available. Frame indices, times and function names must come from this catalogue; a time
outside a function's support is an error, not an extrapolation.

Sampled functions exported by each path:

| Path | `x`, `y` | `width`, `height` | `opacity` | `*.progress` |
|---|---|---|---|---|
| Tracking | yes (feature anchor) | no | no | yes |
| Layer | yes (element centre) | yes (sub-pixel; omitted when the box touches the analysis window's edge) | yes | yes |
| Multi-body | yes (point-group centre) | no | no | yes |
| Solid box | yes (box centre) | yes | no | yes |
| Qt probe | yes | yes | yes | yes, plus `color.r/g/b/a` and `property.value` |

Exports default to at most 64 frames; change that with `--function-max-frames`. Missing frames, hash or size mismatches,
out-of-order times and exceeded budgets are errors. An offline check of encoding and recomputation:
`python tools/validate_functions.py --out checks`, with `--photo` to build a controlled panning video from a real photo.
It checks encoding and recomputation only and says nothing about recognition accuracy on natural video. Protocol details are in
the [video function notes](../../docs/video-functions.md) (Chinese).

## Qt probe

When the target is your own Qt app, widget properties can be read directly instead of inferred from pixels. Both QWidget and
QML (QQuickItem) are supported. There are two parts:

| | File | Depends on | What it does |
|---|---|---|---|
| In the app | `motionspec/qt_probe.py` | PySide6 or PyQt6 only | records each element's position, size, opacity, scale, rotation, visibility, colour and any named numeric properties, every frame |
| Offline | `motionspec/trace.py` | numpy, OpenCV | splits the trace into segments, fits easing, and writes the same `spec.json` and `summary.md` |

The in-app file imports nothing else from the package, so it can be copied into the target app or loaded by path (the package's
`__init__` imports numpy, which the app's environment may not have):

```python
from qt_probe import attach_from_env   # or load motionspec/qt_probe.py by path
attach_from_env()                      # active only when MOTIONSPEC_TRACE=<trace path> is set; writes the trace on exit
```

```bash
motionspec trace trace.json -o probe
motionspec functions probe/spec.json -o probe/functions
```

For explicit control, use `Recorder`: `start()` and `stop()` follow the app's frames (render frames for QML windows, a timer
otherwise), or call `sample(t_ms)` once per frame.

Colour comes from a QML `color` property, or from `background` / `background-color` in a widget's style sheet. Custom properties
must be named: `Recorder(properties=("hoverAmount", "emphasis"))`, or the environment variable
`MOTIONSPEC_PROPS=hoverAmount,emphasis`. Each becomes a `fade` segment with a `color` channel (RGBA from and to, 0 to 1) or a
`property` channel (the name and its from and to values).

In the spec, `model` is `qt-probe`, and each segment has `element` (class, object name, text) and an `element_id` within the
recording. Elements that move together at the same time keep separate segments, linked by `motion_groups[].segment_ids`.
Elements that grow from zero size (an expanding panel, a progress bar) are reported as size changes; being hidden at zero size
is not reported as an opacity change. Coordinates are device pixels (logical pixels times the screen scale), so they match
screenshots and recordings of the same window. The exported function format is `svsg.probe-functions/1`, with no pixel frames.

| | Visual measurement | Qt probe |
|---|---|---|
| Needs | a recording | access to the app's start-up code |
| Movement, opacity | measured, with sub-pixel to roughly 1 px error | the property values themselves |
| Which widget | region coordinates only | class, object name, text |
| Position at opacity 0 | inferred | read as usual |
| What it sees | everything on screen, including shaders and other people's apps | only the recorded properties |
| Easing curve | fitted | also fitted (the animation object itself is not visible), but from exact progress values |

The probe walks the whole widget tree every frame, which is noticeable with many elements, so it is meant for debugging sessions.
It has not been run for long periods inside a large application.

## Checking claims

Conclusions a model gives after reading a spec can be checked against it:

```bash
motionspec check out/spec.json claims.json
```

```json
[
  {"field": "count", "value": 2},
  {"segment": "s1", "field": "direction", "value": "right"},
  {"segment": "s1", "field": "duration_ms", "value": 400, "easing": "out-cubic"},
  {"segment": "s1", "field": "easing", "value": "out-cubic"}
]
```

Supported fields: `count`, `kind`, `duration_ms`, `easing`, `overshoot`, `distance_px`, `direction`, `scale_to`.
`duration_ms`, `easing` and `overshoot` accept a `channel` (`translate`, `scale`, `rotate`, `blend`, `opacity`, `size`).
Tolerances are set by the checker; a `tol` in the request is rejected. `count` only refers to segments found under the current
detection settings. Exit codes: 0 passed, 1 failed, 2 invalid input, 3 insufficient evidence.

## Accuracy

Synthetic animations with known values, at 60 fps:

| Sample | Truth | Measured |
|---|---|---|
| slide | 300 px, out-cubic, 400 ms | 299.8 px, out-cubic, 400 ms |
| fade-in | linear, 250 ms | linear, 250 ms |
| spring scale | damping ratio 0.35, angular frequency 28, ×1.667 | 0.355, 28.04, ×1.68 |
| move up and fade in | 24 px, decelerate, 300 ms | 23.9 px, material-decelerate, 300 ms |
| three identical list items entering in turn | starts at 167 / 233 / 300 ms, 20 px each | 3 segments, same starts, 19.9 to 20.0 px each |

Scenes rendered frame by frame with real Qt widgets (Chinese text, rounded corners, `QGraphicsOpacityEffect`, screen scale 1.5,
values in device pixels):

| Scene | Truth | Measured |
|---|---|---|
| card slide | 300 px, OutCubic, 300 ms | 300.0 px, out-cubic, 300 ms |
| notification enters | up 36 px and fades in, OutCubic, 250 ms | 36.1 px, out-cubic, 250 ms |
| notification exits | up 24 px and fades out, InCubic, 200 ms | 23.1 px, in-cubic, 200 ms |
| plain fade-in | InOutQuad, 300 ms | in-out-quad, 300 ms |
| card slide while something else fades in | 90 px InOutCubic 400 ms; Linear 200 ms | 89.9 px in-out-cubic 400 ms; linear 200 ms |

Nine harder Qt scenes (a modal, an expanding panel, a sidebar, scrolling, a tab indicator, a swap, an overshoot, a colour change,
a progress bar) are scored automatically by `tools/qt_hard_compare.py`: 85% field by field for visual measurement and 100% for
the probe. Evaluating the exported functions frame by frame against the properties the probe recorded
(`tools/function_accuracy.py`): on the basic scenes, position error is at most 1.0 px, the size of entering and exiting
elements is within 1.0 px, and opacity error is 0.

Speed: analysing and exporting functions for a sub-second 1350 × 840 clip takes about 2 to 4 seconds; the nine scenes above
take about 24 seconds in total on one development machine.

All of this footage is rendered frame by frame with perfectly regular timing; dropped frames and capture delay in real screen
recordings have had little testing. The nine harder scenes were used repeatedly during development and are not an independent
validation.

## Model benchmarks

`bench/` holds comparison scripts: the same animation is given to a model as video only, or with the measurements attached, and
both are scored. Keys are read from the local Forge configuration and used only inside the process. `bench/canary.py` first
confirms that a model actually received the video.

| Scenes | Video only | With visual measurements | With Qt probe |
|---|---|---|---|
| 6 basic scenes, 6 models | 55% | 92% | 94% |
| 9 harder scenes, 6 models | 58% | 82% | 100% |

The two GPT models received timestamped still frames, because the organisation behind the key does not have video input enabled.
Full results, scoring rules and raw answers are in the [benchmark notes](../../bench/results/README.md) (Chinese). `bench/pedestrian_*.py`
is an experiment on pedestrian detection in MOT17 street videos and is still in progress.

## Known limits

- Built for 2D interface animation. Complex footage may be marked `nonrigid`, or may happen to fit a 2D model.
  Audio, subtitles and object semantics are not analysed.
- A segment describes one overall motion. Several adjacent elements moving the same way at the same time become one segment.
  When they move in different directions the multi-body path handles translation only; frames where an element is fully
  covered are interpolated, so its duration and curve are unreliable. Two animations at the same place less than about 8 frames
  apart are merged.
- Layout reflow is measured wrongly. When an expanding panel pushes rows down or a collapsing sidebar pulls content along, the
  resizing block and the pushed block fall into one window; the former is reported as moving, or both become one segment.
  The solid-box path only applies when that colour forms its own region and is not cut by the window edge.
- The layer path solves translation and uniform scale only, not rotation; the element must be completely absent at one end;
  it is only tried on windows of at most 3 seconds and 180 frames.
- Animation drawn over a continuously changing area (a translucent panel over an animated background) cannot be separated and
  is marked low confidence. Slow animations lasting longer than `--ambient-ms` (a very long progress bar) end up in `ambient`.
- Back-and-forth motion (shakes, pulses) reports only the peak and its time, with no curve fit.
- H.264 chroma subsampling smears the curve of a colour transition.
- Mouse pointer movement is reported as a small element moving.
- A window holds at most 300 frames by default. Longer segments are measured only over what was kept, marked low confidence and
  flagged `truncated`.

## Code layout

```
motionspec/
  frames.py       frame sources: video, image directory, frames in memory; unified timestamps
  segment.py      pass one: colour differences → coarse grid → ambient removal → space-time components
  motion.py       pass two: backdrop split, solid box, layer, tracking, multi-body, classification
  layer.py        the layer model: joint solve of movement, scale and opacity for entering and exiting elements
  fit.py          easing fits: Bézier, named curves, spring (numpy only)
  timing.py       one calculation for both timings
  annotate.py     annotated images
  spec.py         spec format, validation, text summary
  claims.py       claim checking
  functions.py    function export, query catalogue, batch queries (via VisFunc-Image)
  settle.py       waiting for the screen to settle: judges frames as they arrive, lists changed regions; screen capture
  agent.py        screen tools for agents and the MCP server
  qt_probe.py     the in-app part of the Qt probe
  trace.py        the offline part of the Qt probe; trace_contract.py validates the trace format
  synth.py        synthetic animations for tests and demos
  cli.py          command line
tools/
  qt_scenes.py, qt_hard_scenes.py     render scenes with known values using real Qt widgets while the probe records (needs PySide6)
  qt_compare.py, qt_hard_compare.py   compare visual measurement, probe and truth; the latter scores automatically
  function_accuracy.py                evaluate exported functions per frame against the probe's values, with timings
  qml_probe_demo.py                   try the probe on a live QML window
  validate_functions.py               offline check of function encoding and recomputation
bench/
  run.py, run_hard.py, report*.py      model benchmarks and summaries
  providers.py, client.py, canary.py   local model configuration, API calls, video receipt check
  pedestrian_*.py                     MOT17 pedestrian experiment (in progress)
  agent_*.py                          agent scenarios: desktop screenshots and text, Chrome pages, MCP calls on a real window
```

Tests are in `tests/`; run `python -m pytest` in this directory.

## Not done yet

- Publishing results while analysis is still running; the spec is written only after the whole range has been analysed
- Visual measurement of layout reflow
- Comparing two specs, for example a reference animation against your implementation, or visual measurement against the probe

## License

[MIT](../../LICENSE)
