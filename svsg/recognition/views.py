"""Single-image views with invertible geometry and auditable encoded evidence."""

from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from PIL import Image, ImageOps

from svsg.contracts.recognition import RecognitionView, ViewObject, ViewPurpose, VisualObject
from svsg.image_functions.raster import RasterFunction

from .geometry import Box, pixel_box
from .images import image_part


@dataclass(frozen=True)
class View:
    id: str
    purpose: ViewPurpose
    box: Box
    mirrored: bool = False

    def map_object(self, obj: VisualObject, model: str, width: int, height: int) -> ViewObject:
        box = obj.bbox
        if self.mirrored:
            box = (1000 - box[2], box[1], 1000 - box[0], box[3])
        mapped = pixel_box(box, self.box)
        clipped = (
            (self.box[0] > 0 and mapped[0] <= self.box[0] + 1)
            or (self.box[1] > 0 and mapped[1] <= self.box[1] + 1)
            or (self.box[2] < width and mapped[2] >= self.box[2] - 1)
            or (self.box[3] < height and mapped[3] >= self.box[3] - 1)
        )
        from .geometry import canonical_label

        return ViewObject(
            view_id=self.id,
            model=model,
            label=canonical_label(obj.label),
            bbox_px=mapped,
            clipped=clipped,
        )

    def encode(self, raster: RasterFunction) -> tuple[list[dict], RecognitionView]:
        with raster.to_image() as image, image.crop(self.box) as crop:
            with ImageOps.mirror(crop) if self.mirrored else crop.copy() as oriented:
                part = image_part(oriented, 1536, min_side=768)
        encoded = b64decode(part["image_url"]["url"].partition(",")[2])
        with Image.open(BytesIO(encoded)) as sent:
            size = sent.size
        # No first-pass labels, object IDs, filenames or previous boxes enter this prompt.
        content = [{"type": "text", "text": "Identify visible objects in this image."}, part]
        return content, RecognitionView(
            view_id=self.id,
            purpose=self.purpose,
            source_box_px=self.box,
            mirrored=self.mirrored,
            encoded_sha256=sha256(encoded).hexdigest(),
            input_size=size,
            objects=[],
        )


def discovery_views(width: int, height: int) -> list[View]:
    """Two overlapping halves cover the entire source, with enlarged local detail."""
    if width >= height:
        boxes = [
            (0, 0, max(1, round(width * 0.65)), height),
            (min(width - 1, round(width * 0.35)), 0, width, height),
        ]
    else:
        boxes = [
            (0, 0, width, max(1, round(height * 0.65))),
            (0, min(height - 1, round(height * 0.35)), width, height),
        ]
    return [View(f"detail-{i}", "discovery", box) for i, box in enumerate(boxes)]


def focus_small_object(width: int, height: int, boxes: list[Box]) -> View | None:
    """Zoom around tiny proposals to discover their neighbours, without using class truth."""
    small = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) < width * height * 0.02]
    if not small:
        return None
    box = min(small, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    side = min(min(width, height), max(128, 4 * max(box[2] - box[0], box[3] - box[1])))
    x0 = max(0, min(width - side, (box[0] + box[2] - side) // 2))
    y0 = max(0, min(height - side, (box[1] + box[3] - side) // 2))
    return View("detail-focus", "discovery", (x0, y0, x0 + side, y0 + side))


def confirmation_view(box: Box, width: int, height: int, index: int = 0) -> View:
    """Expand only unresolved objects; an edge fragment needs more context, not lower IoU."""
    x0, y0, x1, y1 = box
    dx, dy = max(16, (x1 - x0) // 2), max(16, (y1 - y0) // 2)
    return View(
        f"confirm-{index}",
        "confirmation",
        (max(0, x0 - dx), max(0, y0 - dy), min(width, x1 + dx), min(height, y1 + dy)),
    )
