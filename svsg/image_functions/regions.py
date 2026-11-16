"""Solve exact four-connected regions using constant-color row intervals."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest

from svsg.contracts.contour import Contour
from svsg.contracts.image_functions import MAX_REGIONS, RegionParameters

from .outlines import compile_outline
from .raster import RasterFunction, color_hex

MAX_SPANS = 100_000
MAX_CONTOURS = 16


@dataclass
class _Component:
    parent: int
    color: int
    left: int
    top: int
    right: int
    bottom: int
    area: int
    perimeter: int
    x2_moment: int
    y2_moment: int

    def merge(self, other: _Component) -> None:
        self.left = min(self.left, other.left)
        self.top = min(self.top, other.top)
        self.right = max(self.right, other.right)
        self.bottom = max(self.bottom, other.bottom)
        self.area += other.area
        self.perimeter += other.perimeter
        self.x2_moment += other.x2_moment
        self.y2_moment += other.y2_moment

    def parameters(self, iid: int, contour: Contour | None) -> RegionParameters:
        return RegionParameters(
            id=iid,
            rgba=color_hex(self.color),
            box=(self.left, self.top, self.right, self.bottom),
            area_px=self.area,
            perimeter_px=self.perimeter,
            centroid_x=self.x2_moment / (2 * self.area),
            centroid_y=self.y2_moment / (2 * self.area),
            contour=contour,
        )


@dataclass(frozen=True)
class _Span:
    left: int
    right: int
    color: int
    component: int


def _root(components: list[_Component], index: int) -> int:
    while components[index].parent != index:
        components[index].parent = components[components[index].parent].parent
        index = components[index].parent
    return index


def _connect(components: list[_Component], a: int, b: int, overlap: int) -> None:
    a, b = _root(components, a), _root(components, b)
    if a != b:
        if components[a].area < components[b].area:
            a, b = b, a
        components[a].merge(components[b])
        components[b].parent = a
    components[a].perimeter -= 2 * overlap


def solve_regions(
    raster: RasterFunction, *, span_limit: int = MAX_SPANS
) -> tuple[int | None, list[RegionParameters]]:
    """Calculate region moments; return unknown if the interval budget is exceeded."""
    components: list[_Component] = []
    rows: list[list[_Span]] = []
    previous: list[_Span] = []
    for y, intervals in enumerate(raster.iter_rows()):
        row: list[_Span] = []
        prev_index = 0
        for left, right, color in intervals:
            if len(components) >= span_limit:
                return None, []
            length, index = right - left, len(components)
            components.append(
                _Component(
                    index,
                    color,
                    left,
                    y,
                    right,
                    y + 1,
                    length,
                    2 * length + 2,
                    length * (left + right),
                    length * (2 * y + 1),
                )
            )
            row.append(_Span(left, right, color, index))
            while prev_index < len(previous) and previous[prev_index].right <= left:
                prev_index += 1
            for j in range(prev_index, len(previous)):
                above = previous[j]
                if above.left >= right:
                    break
                if above.color == color:
                    overlap = min(right, above.right) - max(left, above.left)
                    _connect(components, index, above.component, overlap)
        previous = row
        rows.append(row)
    count = sum(i == c.parent for i, c in enumerate(components))
    largest = nsmallest(
        MAX_REGIONS,
        (c for i, c in enumerate(components) if i == c.parent),
        key=lambda c: (-c.area, c.top, c.left, c.parent),
    )
    outlines: dict[int, dict[int, list[tuple[int, int]]]] = {
        c.parent: defaultdict(list) for c in largest[:MAX_CONTOURS]
    }
    for y, row in enumerate(rows):
        for span in row:
            root = _root(components, span.component)
            if root in outlines:
                outlines[root][y].append((span.left, span.right))
    results = []
    for iid, comp in enumerate(largest, start=1):
        rings = compile_outline(outlines[comp.parent]) if comp.parent in outlines else None
        contour = Contour.model_validate({"rings": rings}) if rings else None
        results.append(comp.parameters(iid, contour))
    return count, results
