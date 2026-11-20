# VisFunc

English · [中文](README.zh.md)

VisFunc measures what can be measured in images and videos with ordinary code, then gives the measurements to an AI model.

Multimodal models can say what is in a picture, but they often get questions like "how many pixels did this card move",
"how long did the animation take" or "do these two regions touch" wrong. VisFunc measures these things locally and stores
them as functions that can be queried by frame, coordinate and time. The model answers from the numbers, and its answers
can be checked against the same data, one claim at a time.

The repository has two modules:

| Module | Code | What it does |
|---|---|---|
| [VisFunc-Image](svsg/README.md) | `svsg/`, package `svsg` | Compiles images into pixels, regions, contours and spatial relations; answers questions from those parameters and checks the model's claims; multi-view object recognition; the shared function format and query runtime for video |
| [VisFunc-Motion](packages/motionspec/README.md) | `packages/motionspec/`, package `motionspec` | Measures UI animation from a screen recording or from live properties of a Qt app: which element moved, when, how far, how its size and opacity changed, and which easing curve it followed |

The two modules can be used separately. Functions exported by VisFunc-Motion use VisFunc-Image's format and query runtime.

## An example

A 0.7-second screen recording: a notification in the bottom-right corner moves up 36 pixels and fades in over 250 ms.
We gave it to six multimodal models and asked how it moves.

| Model | Video only | Video plus VisFunc-Motion's measurements |
|---|---|---|
| MiMo v2.6 Flash | does not move | up 36.5 px, starting at 167 ms |
| MiMo v2.6 Pro | does not move, 400 ms | up 36.5 px, starting at 167 ms |
| Doubao Seed 2.1 Turbo | does not move, 100 ms | up 36.5 px, starting at 167 ms |
| Doubao Seed 2.1 Pro | left 30 px; in another run, left 300 px | up 36.5 px, starting at 167 ms |
| GPT 6.1 Sol / Terra | up 36 / 40 px | up 36.5 px, starting at 167 ms |

The two GPT models received timestamped still frames, because the organisation that owns the key does not have video input enabled.

Across nine scenes that are closer to real interfaces (a modal dialog, an expanding panel, a collapsing sidebar, scrolling,
a tab indicator, two rows swapping places, an overshoot, a button changing colour, a progress bar), the six models scored
58% field by field with the video alone, 82% with the visual measurements added, and 100% with widget properties read from
the Qt app. That is 216 calls; a call that returned no answer scores zero.

## Quick start

Python 3.11 or later. Everything below runs offline and needs no model keys.

```bash
python -m pip install -e ".[functions]" -e "./packages/motionspec"
```

Generate a sample video of a card sliding to the right, measure it, and export queryable functions:

```bash
python -c "from motionspec.synth import slide, write_video; frames, _ = slide(); write_video('slide.mp4', frames)"
```

```bash
python -m motionspec analyze slide.mp4 -o slide-out --functions --evidence-every-ms 250
```

`slide-out/summary.md` reports that the card moved about 299 px to the right, that the change is visible for about 350 ms,
and that the estimated design, from aligning a named curve, is out-cubic over 400 ms. The sample was generated with 300 px,
out-cubic, 400 ms. Near the end of an ease-out the card moves less than a pixel per frame, so the visible change is shorter
than the designed duration.

List the frames and functions that can be queried:

```bash
python -m motionspec query slide-out/functions.json --list
```

Save the following as `query.json` to ask for the colour of the top-left pixel in frame 0 and the card's x position at 250 ms:

```json
[
  {"id": "color", "op": "pixel_rgba", "frame_index": 0, "x": 0, "y": 0},
  {"id": "position", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 250}
]
```

```bash
python -m motionspec query slide-out/functions.json query.json
```

The result is `[41, 45, 49, 255]` with the frame index, timestamp and pixel hash, and `261.0` with unit `source_px` and
provenance `motion_estimate`.

## Results

| Test | Result |
|---|---|
| 6 models, 6 basic UI animations (216 calls) | video only 55%, with visual measurements 92%, with Qt properties 94% |
| 6 models, 9 harder UI animations (216 calls) | video only 58%, with visual measurements 82%, with Qt properties 100% |
| Same scenes, only the version of the visual measurements changed, two runs each | before the latest changes 68%, current 86% |
| Parameters only, no images (2 animations, 2 models) | 145/147 items from timing parameters, 147/147 from the text summary |
| The visual measurements themselves, 9 harder scenes | 85% field by field; Qt properties 100% |
| Exported functions evaluated per frame against properties recorded by Qt | basic scenes: position error ≤ 1.0 px, size of entering/exiting elements ≤ 1.0 px, opacity error 0 |
| Object recognition on COCO8 (8 images, 30 objects) | after multi-view review, false positives 9 → 2, misses 10 → 16 |

The UI scenes are rendered frame by frame with Qt, so frame timing is perfectly regular; dropped frames and capture delay in
real screen recordings have had little testing. The nine harder scenes were used repeatedly during development and are not an
independent validation. Data, scoring rules and raw answers are in the [model benchmark notes](packages/motionspec/bench/results/README.md),
the [parameter-only retest](docs/timing-protocol-retest-20261117.md), the [sampling-rate ablation](docs/evidence-ablation-20261116.md)
and the [recognition evaluation](docs/recognition-evaluation-20261116.md) (all in Chinese).
An experiment on pedestrian detection in MOT17 street videos is in progress.

## Current limits

- 3D and game footage only gets summary statistics, with no usable motion curves.
- Layout reflow is measured wrongly. When an expanding panel pushes the rows below it, or a collapsing sidebar pulls the content
  next to it, the resizing block is reported as moving and positions can be off by more than a hundred pixels.
- Colour transitions in compressed video are smeared by H.264 chroma subsampling.
- Objects are not tracked across frames. A segment ID refers to a change, not to an object.
- Object recognition does not guarantee that every object is found.

## Install and test

```bash
python -m pip install -e ".[functions]" -e "./packages/motionspec"   # measurement and function export, offline
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

MIT
