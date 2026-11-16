"""Compile region row intervals into oriented orthogonal polygon rings."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

Point = tuple[int, int]
Interval = tuple[int, int]
Ring = list[Point]
MAX_OUTLINE_EDGES = 8192
MAX_OUTLINE_VERTICES = 384


def _subtract(row: Sequence[Interval], neighbor: Sequence[Interval]) -> Iterator[Interval]:
    j = 0
    for left, right in row:
        while j < len(neighbor) and neighbor[j][1] <= left:
            j += 1
        cursor = left
        for k in range(j, len(neighbor)):
            start, end = neighbor[k]
            if start >= right:
                break
            if start > cursor:
                yield cursor, start
            cursor = max(cursor, end)
            if cursor >= right:
                break
        if cursor < right:
            yield cursor, right


def _direction(a: Point, b: Point) -> int:
    if a[1] == b[1]:
        return 0 if b[0] > a[0] else 2
    return 1 if b[1] > a[1] else 3


def _corners(ring: Ring) -> Ring:
    return [
        b
        for i, b in enumerate(ring)
        if _direction(ring[i - 1], b) != _direction(b, ring[(i + 1) % len(ring)])
    ]


def compile_outline(rows: dict[int, list[Interval]]) -> list[Ring] | None:
    """Trace boundary with the region on the right; points are on pixel-cell edges."""
    edges: dict[Point, list[Point]] = {}
    edge_count = 0

    def add(start: Point, end: Point) -> None:
        edges.setdefault(start, []).append(end)

    for y, row in rows.items():
        for lo, hi in row:
            add((lo, y + 1), (lo, y))
            add((hi, y), (hi, y + 1))
            edge_count += 2
            if edge_count > MAX_OUTLINE_EDGES:
                return None
        for lo, hi in _subtract(row, rows.get(y - 1, [])):
            add((lo, y), (hi, y))
            edge_count += 1
            if edge_count > MAX_OUTLINE_EDGES:
                return None
        for lo, hi in _subtract(row, rows.get(y + 1, [])):
            add((hi, y + 1), (lo, y + 1))
            edge_count += 1
            if edge_count > MAX_OUTLINE_EDGES:
                return None

    rings = []
    vertices = 0
    turn_priority = {1: 0, 0: 1, 3: 2, 2: 3}
    while edges:
        start = next(iter(edges))
        cur, heading = start, None
        ring: Ring = []
        while True:
            ring.append(cur)
            targets = edges.get(cur)
            if not targets:
                raise ValueError("region boundary cannot be closed")
            if heading is None:
                end = targets.pop()
            else:
                arrival = heading
                index = min(
                    range(len(targets)),
                    key=lambda i: turn_priority[(_direction(cur, targets[i]) - arrival) % 4],
                )
                end = targets.pop(index)
            if not targets:
                del edges[cur]
            heading = _direction(cur, end)
            if end == start:
                break
            cur = end
        corners = _corners(ring)
        vertices += len(corners)
        if vertices > MAX_OUTLINE_VERTICES:
            return None
        rings.append(corners)
    return rings
