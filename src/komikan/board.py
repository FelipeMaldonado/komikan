"""Komikan board geometry (Matus 1920 layout, as drawn on p.3 of the rules PDF).

The board is a 5x5 alquerque square with an inverted triangle whose apex is the
middle point of the square's top edge. Coordinates are integers with y growing
downwards. Square points are 3 units apart and triangle rows 2 units apart, which
keeps the triangle's sides at exactly 45 degrees, as on the printed board.
Jumps are derived from these coordinates, so straight lines through the apex need
no special handling.
"""

from __future__ import annotations

# Triangle rows counted upwards from the apex, as x positions on that row.
_TRIANGLE_ROWS = {1: (-2, 0, 2), 2: (-4, -2, 0, 2, 4), 3: (-6, -2, 0, 2, 6)}


def _square_xy(col: int, row: int) -> tuple[int, int]:
    return 3 * col - 6, 3 * row


def _build() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coords = [_square_xy(c, r) for r in range(5) for c in range(5)]
    coords += [(x, -2 * k) for k, xs in _TRIANGLE_ROWS.items() for x in xs]
    index = {xy: i for i, xy in enumerate(coords)}
    edges: set[tuple[int, int]] = set()

    def link(p: tuple[int, int], q: tuple[int, int]) -> None:
        a, b = index[p], index[q]
        edges.add((min(a, b), max(a, b)))

    # Square: orthogonal lines everywhere, diagonals from points with even col + row.
    for r in range(5):
        for c in range(5):
            here = _square_xy(c, r)
            if c < 4:
                link(here, _square_xy(c + 1, r))
            if r < 4:
                link(here, _square_xy(c, r + 1))
            if (c + r) % 2 == 0 and r < 4:
                if c < 4:
                    link(here, _square_xy(c + 1, r + 1))
                if c > 0:
                    link(here, _square_xy(c - 1, r + 1))

    # Triangle: each row is a horizontal line.
    for k, xs in _TRIANGLE_ROWS.items():
        for x0, x1 in zip(xs, xs[1:]):
            link((x0, -2 * k), (x1, -2 * k))
    # Three vertical lines run from the lowest row to the top row.
    for x in (-2, 0, 2):
        for k in (1, 2):
            link((x, -2 * k), (x, -2 * (k + 1)))
    # The two sides, and the middle vertical, start at the apex.
    for x in (-2, 0, 2):
        link((0, 0), (x, -2))
    for sign in (-1, 1):
        for k in (1, 2):
            link((sign * 2 * k, -2 * k), (sign * 2 * (k + 1), -2 * (k + 1)))

    return coords, sorted(edges)


COORDS, EDGES = _build()
NUM_POINTS = len(COORDS)
_INDEX = {xy: i for i, xy in enumerate(COORDS)}

NEIGHBORS: tuple[tuple[int, ...], ...] = tuple(
    tuple(sorted({b for a, b in EDGES if a == p} | {a for a, b in EDGES if b == p}))
    for p in range(NUM_POINTS)
)


def sq(col: int, row: int) -> int:
    """Point index of a square point. Column 0 is left, row 0 is the top (apex) row."""
    return _INDEX[_square_xy(col, row)]


def tri(rows_above_apex: int, x: int) -> int:
    """Point index of a triangle point, by its row above the apex and its x coordinate."""
    return _INDEX[(x, -2 * rows_above_apex)]


def _is_straight(a: int, b: int, c: int) -> bool:
    (ax, ay), (bx, by), (cx, cy) = COORDS[a], COORDS[b], COORDS[c]
    ux, uy, vx, vy = bx - ax, by - ay, cx - bx, cy - by
    return ux * vy - uy * vx == 0 and ux * vx + uy * vy > 0


# (src, over, dst): straight-line jumps along the board's lines.
JUMPS: tuple[tuple[int, int, int], ...] = tuple(
    (a, b, c)
    for a in range(NUM_POINTS)
    for b in NEIGHBORS[a]
    for c in NEIGHBORS[b]
    if c != a and _is_straight(a, b, c)
)

APEX = sq(2, 0)
KOMIKELU_START = tri(2, 0)
TREWAS_START: tuple[int, ...] = (
    sq(0, 2),
    sq(4, 2),
    *(sq(c, 3) for c in range(5)),
    *(sq(c, 4) for c in range(5)),
)
