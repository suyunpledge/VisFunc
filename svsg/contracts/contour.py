"""Typed contour primitive and geometry reproducible from its vertices."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cached_property
from math import atan2, degrees, hypot, pi, sqrt
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

AxisCoordinate = Annotated[StrictInt, Field(ge=0, le=16 * 1024 * 1024)]
Point = tuple[AxisCoordinate, AxisCoordinate]


def ring_edges(points: Sequence[Point]) -> Iterator[tuple[Point, Point]]:
    for i, point in enumerate(points):
        yield point, points[(i + 1) % len(points)]


def twice_area(points: Sequence[Point]) -> int:
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in ring_edges(points))


def _hull(points: list[Point]) -> list[Point]:
    def half(ordered: list[Point]) -> list[Point]:
        result: list[Point] = []
        for c in ordered:
            while len(result) >= 2:
                a, b = result[-2:]
                if (b[0] - a[0]) * (c[1] - a[1]) > (b[1] - a[1]) * (c[0] - a[0]):
                    break
                result.pop()
            result.append(c)
        return result

    ordered = sorted(set(points))
    return half(ordered)[:-1] + half(ordered[::-1])[:-1]


class Contour(BaseModel):
    """Implicitly closed rings with clockwise exterior in top-left image coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    rings: tuple[Annotated[tuple[Point, ...], Field(min_length=4, max_length=384)], ...] = Field(
        min_length=1, max_length=96
    )

    @model_validator(mode="after")
    def _orthogonal_rings(self) -> Self:
        if sum(map(len, self.rings)) > 384:
            raise ValueError("contour exceeds vertex budget")
        exterior_count = 0
        for ring in self.rings:
            if any((a[0] == b[0]) == (a[1] == b[1]) for a, b in ring_edges(ring)):
                raise ValueError("contour edges must be nonzero and axis-aligned")
            a2 = twice_area(ring)
            if a2 == 0:
                raise ValueError("contour rings must enclose an area")
            exterior_count += a2 > 0
        if exterior_count != 1 or self.area2 <= 0:
            raise ValueError("contour requires one exterior with positive net area")
        return self

    @cached_property
    def area2(self) -> int:
        return sum(twice_area(ring) for ring in self.rings)

    @cached_property
    def perimeter(self) -> int:
        return sum(
            abs(x1 - x0) + abs(y1 - y0)
            for ring in self.rings
            for (x0, y0), (x1, y1) in ring_edges(ring)
        )

    @cached_property
    def bounds(self) -> tuple[int, int, int, int]:
        xs, ys = zip(*(v for ring in self.rings for v in ring), strict=True)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def holes(self) -> int:
        return len(self.rings) - 1

    @cached_property
    def hull_area2(self) -> int:
        return abs(twice_area(_hull([p for ring in self.rings for p in ring])))

    @cached_property
    def centroid(self) -> tuple[float, float]:
        mx = my = 0
        for ring in self.rings:
            for (x0, y0), (x1, y1) in ring_edges(ring):
                cross = x0 * y1 - x1 * y0
                mx += (x0 + x1) * cross
                my += (y0 + y1) * cross
        return mx / (3 * self.area2), my / (3 * self.area2)

    @cached_property
    def axes(self) -> tuple[float | None, float]:
        # Integrate central second moments over the filled pixel cells.
        cx, cy = self.centroid
        sx = sy = sxy = 0.0
        for ring in self.rings:
            for (u0, v0), (u1, v1) in ring_edges(ring):
                x0, x1, y0, y1 = u0 - cx, u1 - cx, v0 - cy, v1 - cy
                cross = x0 * y1 - x1 * y0
                sx += (x0 * x0 + x0 * x1 + x1 * x1) * cross
                sy += (y0 * y0 + y0 * y1 + y1 * y1) * cross
                sxy += (2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1) * cross
        vx, vy, cov = sx / (6 * self.area2), sy / (6 * self.area2), sxy / (12 * self.area2)
        gap = hypot(vx - vy, 2 * cov)
        hi = (vx + vy + gap) / 2
        # A valid cell region is at least one unit cell thick.
        lo = max((vx + vy - gap) / 2, 1 / 12)
        if gap <= hi * 1e-9:
            return None, 1.0
        return degrees(0.5 * atan2(2 * cov, vx - vy)) % 180, sqrt(hi / lo)

    @property
    def solidity(self) -> float:
        return self.area2 / self.hull_area2

    @property
    def circularity(self) -> float:
        return 2 * pi * self.area2 / self.perimeter**2
