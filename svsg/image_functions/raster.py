"""Exact evaluation and serialization of the decoded RGBA8 raster function."""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from itertools import groupby
from pathlib import Path
from sys import byteorder
from typing import TYPE_CHECKING
from zlib import compress, decompressobj
from zlib import error as ZlibError

from svsg.contracts.image_functions import MAX_RASTER_PIXELS, Box

if TYPE_CHECKING:
    from PIL.Image import Image

RGBA8 = tuple[int, int, int, int]
CODEC = "svsg.raster/1"
MAX_PROGRAM_SPANS = 100_000


def _buffer_size(width: int, height: int) -> int:
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("raster dimensions must be positive integers")
    if width * height > MAX_RASTER_PIXELS:
        raise ValueError("raster exceeds the pixel budget")
    return width * height * 4


def region_mean(image: Image, box: Box) -> tuple[float, float, float, float]:
    from PIL import ImageStat

    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError("mean_rgba box is outside the raster")
    with image.crop(box) as crop:
        r, g, b, a = ImageStat.Stat(crop).mean
    return r, g, b, a


def color_hex(packed: int) -> str:
    return "#" + packed.to_bytes(4, byteorder).hex().upper()


@dataclass(frozen=True)
class RasterFunction:
    """f(x, y) gives the exact normalized RGBA8 value at integer pixel (x, y)."""

    width: int
    height: int
    samples: bytes

    def __post_init__(self) -> None:
        if len(self.samples) != _buffer_size(self.width, self.height):
            raise ValueError("RGBA sample count does not match raster dimensions")

    @classmethod
    def from_image(cls, image: bytes | Path) -> RasterFunction:
        from PIL import Image, ImageOps

        with Image.open(BytesIO(image) if isinstance(image, bytes) else image) as source:
            _buffer_size(*source.size)
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("the raster compiler requires a single-frame image")
            with ImageOps.exif_transpose(source) as oriented, oriented.convert("RGBA") as rgba:
                return cls(*rgba.size, rgba.tobytes())

    @property
    def pixel_sha256(self) -> str:
        return sha256(self.samples).hexdigest()

    def pixel(self, x: int, y: int) -> RGBA8:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("pixel is outside the raster")
        i = (y * self.width + x) * 4
        r, g, b, a = self.samples[i : i + 4]
        return r, g, b, a

    def mean(self, box: Box) -> tuple[float, float, float, float]:
        with self.to_image() as image:
            return region_mean(image, box)

    def to_image(self) -> Image:
        from PIL import Image

        return Image.frombytes("RGBA", (self.width, self.height), self.samples)

    def iter_rows(self) -> Iterator[Iterator[tuple[int, int, int]]]:
        """Yield (start, end, color) intervals for the step function on each row."""
        values = memoryview(self.samples).cast("I")
        for offset in range(0, len(values), self.width):
            yield self._intervals(values[offset : offset + self.width])

    @staticmethod
    def _intervals(row: memoryview) -> Iterator[tuple[int, int, int]]:
        start = 0
        for packed, group in groupby(row):
            end = start + sum(1 for _ in group)
            yield start, end, packed
            start = end

    def _interval_program(self) -> list[list[tuple[int, str]]] | None:
        # Keep long interval tables out of JSON for textured images.
        remaining = min(MAX_PROGRAM_SPANS, max(64, self.width * self.height // 8))
        if self.height > remaining:
            return None
        result = []
        for row in self.iter_rows():
            intervals = []
            for _, end, packed in row:
                if remaining == 0:
                    return None
                intervals.append((end, color_hex(packed)))
                remaining -= 1
            result.append(intervals)
        return result

    def to_program(self) -> dict:
        result = {
            "version": CODEC,
            "width_px": self.width,
            "height_px": self.height,
            "pixel_sha256": self.pixel_sha256,
        }
        rows = self._interval_program()
        if rows is not None:
            return {**result, "row_intervals": rows}
        return {**result, "rgba8_zlib_base64": b64encode(compress(self.samples)).decode("ascii")}

    @classmethod
    def from_program(cls, data: dict) -> RasterFunction:
        if data.get("version") != CODEC:
            raise ValueError("unsupported raster program")
        width, height = data.get("width_px"), data.get("height_px")
        if type(width) is not int or type(height) is not int:
            raise ValueError("raster dimensions must be integers")
        size = _buffer_size(width, height)
        # The two representations have an identical f(x, y) contract.
        if ("row_intervals" in data) == ("rgba8_zlib_base64" in data):
            raise ValueError("raster payload requires exactly one representation")
        if "row_intervals" in data:
            samples = _read_intervals(data["row_intervals"], width, height, size)
        else:
            samples = _read_packed(data.get("rgba8_zlib_base64"), size)
        raster = cls(width, height, samples)
        if raster.pixel_sha256 != data.get("pixel_sha256"):
            raise ValueError("raster pixel digest mismatch")
        return raster


def _read_packed(encoded: object, size: int) -> bytes:
    if not isinstance(encoded, str) or len(encoded) > 2 * size + 128:
        raise ValueError("invalid raster payload length")
    try:
        decoder = decompressobj()
        samples = decoder.decompress(b64decode(encoded, validate=True), size + 1)
    except (Base64Error, ZlibError) as exc:
        raise ValueError("invalid encoded raster samples") from exc
    if len(samples) != size or not decoder.eof or decoder.unused_data:
        raise ValueError("raster payload does not match its dimensions")
    return samples


def _read_intervals(rows: object, width: int, height: int, size: int) -> bytes:
    if not isinstance(rows, list) or len(rows) != height:
        raise ValueError("raster payload row count does not match its dimensions")
    samples = bytearray(size)
    spans = offset = 0
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("invalid raster payload row")
        spans += len(row)
        if spans > MAX_PROGRAM_SPANS:
            raise ValueError("raster payload exceeds the interval limit")
        start = 0
        for interval in row:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError("invalid raster payload interval")
            end, encoded = interval
            if type(end) is not int or not start < end <= width:
                raise ValueError("invalid raster payload interval bounds")
            if not isinstance(encoded, str) or len(encoded) != 9 or not encoded.startswith("#"):
                raise ValueError("invalid raster payload interval color")
            rgba = bytes.fromhex(encoded[1:])
            if len(rgba) != 4:
                raise ValueError("invalid raster payload interval color")
            length = end - start
            samples[offset : offset + 4 * length] = rgba * length
            offset += 4 * length
            start = end
        if start != width:
            raise ValueError("raster payload interval row is incomplete")
    return bytes(samples)
