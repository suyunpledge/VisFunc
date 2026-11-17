"""A single coordinate convention for functions, detection and crop review."""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO

from PIL import Image
from PIL.Image import Image as PILImage

from svsg.image_functions.raster import RasterFunction


def canonical_raster(data: bytes) -> RasterFunction:
    return RasterFunction.from_image(data)


def image_part(image: PILImage, max_side: int, *, min_side: int = 0) -> dict:
    # Explicit white compositing prevents hidden RGB in transparent pixels becoming evidence.
    with Image.new("RGBA", image.size, "white") as background:
        background.alpha_composite(image)
        with background.convert("RGB") as rgb, BytesIO() as output:
            if not 0 <= min_side <= max_side:
                raise ValueError("invalid model-view size range")
            longest = max(rgb.size)
            if longest < min_side:
                # Ordinary interpolation increases model input resolution, not source evidence.
                resized = rgb.resize(
                    (
                        max(1, round(rgb.width * min_side / longest)),
                        max(1, round(rgb.height * min_side / longest)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                rgb.close()
                rgb = resized
            rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            rgb.save(output, format="JPEG", quality=90)
            encoded = b64encode(output.getvalue()).decode("ascii")
            rgb.close()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
