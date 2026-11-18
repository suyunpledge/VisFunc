"""Row intervals must match the straightforward run-length definition for any raster."""

from __future__ import annotations

import random
from itertools import groupby
from sys import byteorder

import pytest

from svsg.image_functions.raster import RasterFunction


def reference_rows(raster: RasterFunction) -> list[list[tuple[int, int, int]]]:
    values = memoryview(raster.samples).cast("I")
    rows = []
    for offset in range(0, len(values), raster.width):
        start, row = 0, []
        for packed, group in groupby(values[offset : offset + raster.width]):
            end = start + sum(1 for _ in group)
            row.append((start, end, packed))
            start = end
        rows.append(row)
    return rows


@pytest.mark.parametrize("width,height", [(1, 1), (5, 1), (7, 5), (64, 3), (33, 17)])
@pytest.mark.parametrize("palette", [[0, 255], [0, 0, 0, 7, 255], list(range(256))])
def test_intervals_match_reference(width, height, palette):
    rng = random.Random(width * 1000 + height + len(palette))
    raster = RasterFunction(width, height, bytes(rng.choice(palette) for _ in range(width * height * 4)))
    assert [list(row) for row in raster.iter_rows()] == reference_rows(raster)


def test_runs_stay_pixel_aligned():
    # Byte patterns that repeat with a period other than four must not shift run boundaries.
    pixel_a, pixel_b = bytes([1, 2, 1, 2]), bytes([2, 1, 2, 1])
    raster = RasterFunction(6, 1, pixel_a * 2 + pixel_b + pixel_a * 3)
    assert [list(row) for row in raster.iter_rows()] == [[
        (0, 2, int.from_bytes(pixel_a, byteorder)),
        (2, 3, int.from_bytes(pixel_b, byteorder)),
        (3, 6, int.from_bytes(pixel_a, byteorder)),
    ]]
