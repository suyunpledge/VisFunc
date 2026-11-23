# VisFunc-Image

English · [中文](README.zh.md) · [Back to VisFunc](../README.md)

VisFunc-Image compiles an image into parameters that can be computed with: the colour of every pixel, the extent and
outline of each same-colour region, and the spatial relations between regions. A model can answer questions from these
parameters alone, and the service then checks each of the model's claims against them. The module also handles multi-view
object recognition, and the shared function format and query runtime for video in VisFunc.

The code lives in `svsg/`; the Python package and command are `svsg` (the project's original name, Structured Visual
Semantic Gateway).

## An example

An 8 × 4 image, red on the left half and blue on the right. Compiling it gives two regions of 16 pixels each, with bounding
boxes `[0, 0, 4, 4]` and `[4, 0, 8, 4]`, a bounding-box IoU of 0 and a shared edge of 4 pixels. From these parameters alone
you can say that the red and blue regions touch along one edge and do not overlap, and anyone with the same image gets the
same numbers.

People and objects in a photo cannot be computed this way. That needs recognition, described under "Object recognition".

## What it is for

| Task | Entry point |
|---|---|
| Store an image as a pixel function that can be restored exactly | `python -m svsg.image_functions encode input.png raster.json`; restore with `decode` |
| Compile an image's regions, contours and spatial relations | `POST /v1/compile-functions`, or `svsg.image_functions` in Python |
| Have a model answer from those parameters and check its claims | `POST /v1/analyze-image`, `POST /v1/analyze` |
| Recognise objects, confirmed across several views | `POST /v1/recognize-image` |
| Read the text on an application screenshot, with boxes and click points (for agents; needs `.[overlays]`) | `POST /v1/screen-text`; on the command line `motionspec screen-text` |
| List a window's controls through Windows UI Automation and find click points by name (needs `.[uia]`) | `svsg.recognition.ui_elements`, or `list_ui_elements` and `find_on_screen` in `motionspec agent-mcp` |
| Query kept video frames and motion functions | `svsg.video_functions.VideoRuntime`; on the command line through `motionspec query` |

## Image functions

Images are rotated according to EXIF and decoded to RGBA8, then encoded as an `svsg.raster/1` raster function that restores
the exact bytes. On top of that the module computes:

- same-colour connected regions: area, bounding box, centroid, perimeter;
- closed contours, with outer rings and holes;
- relations between regions: `left_of`, `right_of`, `above`, `below`, `bbox_contains`, `bbox_iou`, `center_distance_px`,
  `shared_edge_px`, `touching`.

Images with too many colours (photos, gradients) are also analysed on a quantised grid. The grid size and the quantisation
method (`box_median_cut_16` or `box_octree_rgba_16`) are recorded, and lengths on the grid are in cells, not pixels.
The whole description uses the format `svsg.functions/1`.

There are two image queries: `pixel_rgba` (the colour of one pixel) and `mean_rgba` (the mean colour of a rectangle).
Both are evaluated locally; no code strings are executed.

## Question answering and claim checking

`/v1/analyze-image` handles a question in this order:

1. L1 compiles the image into an intermediate representation (IR). With `SVSG_DETECTOR=functions` the IR is the image
   functions above; a YOLO detector can be used instead (needs `.[ml]`).
2. L1.5 (optional) asks a vision model to re-check instances marked for verification. It has its own timeout; instances
   whose check times out are treated as lowest confidence and recorded, and the rest continue.
3. L3 has a text model answer from the IR only, stating its conclusions as structured claims such as "the class is X",
   "there are N", "A is left of B".
4. L2 checks each claim against the IR. Classes must match the detections, a count that is off by one fails, and spatial
   relations are recomputed with the same geometry code.
5. If a claim fails, the model gets one retry with the specific mismatch; if it fails again, the case goes to human review.

States and error codes:

| State | Codes | Handling |
|---|---|---|
| S2 invalid input | E1001–E1003 | returned directly, no model call |
| S3 missing dependency | E2001–E2003 | one soft retry |
| S4 conflict or failed check | E3001, E3002 | retry with corrective feedback |
| S5 timeout | E4001, E4003 | L1 or L3 timed out; the request stops |
| L1.5 timeout | E4002 | affected instances drop to lowest confidence; the rest continue |
| S6 human review | E5001 | returned as is, never swallowed |

## Object recognition

`/v1/recognize-image` sends one image per model call, with at most 5 calls by default:

1. the primary model looks at the full image;
2. the review model looks at a horizontal mirror, and its boxes are mapped back;
3. each model looks at one of two overlapping halves; if a candidate smaller than 2% of the image appears, that view is
   replaced by a zoomed view around it;
4. with budget left, the smallest unconfirmed candidate gets one more zoomed check.

Boxes of the same class with IoU ≥ 0.5 count as the same object. Confirmation needs support from two different views; when two
models are configured, both must be among the supporters. The same model repeating itself on the same encoded image is not
independent evidence. Results are split into `corroborated` and `needs_review` (kept for later checking), always with
`coverage=not_exhaustive`, since objects may still be missed. Results can be attached to evidence frames exported by
VisFunc-Motion; the sequence, frame index, time, size and hash are checked when attaching.

On COCO8 (8 images, 30 objects), review cut false positives from 9 to 2 while misses rose from 10 to 16.
See the [recognition protocol](../docs/recognition.md) and the [evaluation notes](../docs/recognition-evaluation-20261116.md) (Chinese).

## Video functions

VisFunc-Motion's measurements are turned into a single function bundle, `functions.json`, here:

| Format | Contents |
|---|---|
| `svsg.video-functions/1` | image functions for each kept frame (reusing the computation above), RGBA difference statistics between neighbouring kept frames, sampled and fitted functions for motion segments |
| `svsg.probe-functions/1` | built from a Qt property trace; no pixel frames; coordinates are window device pixels |
| `svsg.motion-timing/1` | two timings per segment: the interval in which change is visible, and the designed interval estimated by aligning a named curve |

Motion functions come in three kinds: linear interpolation between samples, a cubic Bézier, and an under-damped spring.
Each has a unit and a time support; evaluating outside the support is an error rather than an extrapolation.
Query operations: `pixel_rgba`, `mean_rgba`, `delta_rgba` (the pixel difference between two kept frames) and `motion_value`
(a sampled or fitted value).

Exports default to at most 64 frames and 64 × 1024² pixels in total (at most 16 × 1024² per frame). Missing frames, hash or
size mismatches, out-of-order times and exceeded budgets are errors. Full pixel data is stored in content-addressed
`raster-*.json` files and never needs to be sent to a model. Protocol details and accuracy records are in the
[video function notes](../docs/video-functions.md) (Chinese).

```python
from svsg.video_functions import VideoRuntime

runtime = VideoRuntime.load("out/functions.json")
runtime.call({"id": "x", "op": "motion_value", "segment_id": "s1", "name": "x", "t_ms": 250})
```

## HTTP service

```bash
python -m pip install -e ".[api,llm]"
python -m svsg
```

It listens on `127.0.0.1:3002` by default.

| Endpoint | Purpose |
|---|---|
| `POST /v1/compile-functions` | upload `file`, optionally with `queries`; returns image functions without calling a model |
| `POST /v1/analyze-image` | upload `file` and a question `query`; answered as described above |
| `POST /v1/analyze` | submit an existing `ir` with a new question |
| `POST /v1/recognize-image` | upload `file`, optionally with a `labels` list and a video `frame` reference |
| `POST /v1/screen-text` | upload `file`, optionally with `regions` (`[[l, t, r, b], ...]`); the OCR model loads on the first request and stays loaded |
| `GET /healthz` | health check, no authentication |

Settings go in `.env`; see [.env.example](../.env.example). The common ones:

| Setting | Purpose |
|---|---|
| `SVSG_DETECTOR` | how L1 compiles: `disabled`, `functions`, `demo`, `yolo` |
| `SVSG_LLM_PROVIDER`, `SVSG_LLM_BASE_URL`, `SVSG_LLM_API_KEY`, `SVSG_LLM_MODEL` | the L3 text model; provider `openai` means an OpenAI-compatible API |
| `SVSG_L15_*` | L1.5 visual verification; its timeout must be under a third of the L3 timeout |
| `SVSG_VISION_*` | recognition model, review model, call budget and number of views |
| `SVSG_API_KEY` | when set, `/v1/*` requires an `X-API-Key` header |
| `SVSG_AUTH_ENABLED=1` | use Bearer tokens issued at login instead |

On Windows, the [launch script](../scripts/start-svsg.cmd) loads settings from a separate runtime directory.

## Code layout

```
svsg/
  image_functions/   raster encoding, regions, contours, spatial relations, function claims
  video_functions/   compiling, exporting and evaluating video functions
  recognition/       multi-view recognition, coordinate mapping, agreement rules, video frame binding, evaluation;
                     subtitle and screen text reading; UI Automation element lists
  contracts/         data formats for IR, image functions, video functions and recognition results
  l1_compiler/       detector adapters and IR compilation
  l15_verifier/      visual verification and fallback
  l2_runtime/        state machine, intent routing, conflict handling, claim checking
  l3_orchestrator/   model calls and the retry loop
  api.py             HTTP service
```

Tests are in `tests/` at the repository root: `python -m pytest tests`.

## License

[MIT](../LICENSE)
