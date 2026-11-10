"""Komikan rules engine: pure functions over an immutable state. See PLAN.md for the rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .board import COORDS, EDGES, JUMPS, KOMIKELU_START, MIRROR_POINT, NEIGHBORS, NUM_POINTS, TREWAS_START

KOMIKELU, TREWAS, DRAW = 0, 1, 2
ROLE_NAMES = {KOMIKELU: "komikelu", TREWAS: "trewas"}
NUM_TREWAS = len(TREWAS_START)


@dataclass(frozen=True, slots=True)
class Rules:
    captures_to_win: int = 9
    max_plies: int = 200  # turns before a draw; not in the traditional rules, see PLAN.md
    trewas_forward_only: bool = False  # the Matus/Leoncito variation
    trewas_no_backward: bool = False  # a variation the user knows: forward or sideways, never back; forward-only takes precedence


DEFAULT_RULES = Rules()
RULE_VARIANTS = {
    "standard": DEFAULT_RULES,
    "no-backward": Rules(trewas_no_backward=True),
    "forward-only": Rules(trewas_forward_only=True),
}


class GameState(NamedTuple):
    komikelu: int  # point index
    trewas: int  # bitmask of point indices
    to_move: int
    in_chain: bool  # the komikelu has just jumped and may continue or stop
    captured: int
    plies: int  # completed turns; a multi-jump counts as one


class Action(NamedTuple):
    src: int | None
    dst: int | None
    over: int | None  # the jumped point, for jumps


# Shared action space: directed steps, then directed jumps, then "stop chain".
ACTIONS: tuple[Action, ...] = (
    *(Action(a, b, None) for x, y in EDGES for a, b in ((x, y), (y, x))),
    *(Action(a, c, b) for a, b, c in JUMPS),
    Action(None, None, None),
)
NUM_ACTIONS = len(ACTIONS)
STOP = NUM_ACTIONS - 1

_ACTION_ID = {action: i for i, action in enumerate(ACTIONS)}
_STEPS: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple((1 << n, _ACTION_ID[Action(p, n, None)]) for n in NEIGHBORS[p]) for p in range(NUM_POINTS)
)
_FORWARD_STEPS: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple((1 << n, _ACTION_ID[Action(p, n, None)]) for n in NEIGHBORS[p] if COORDS[n][1] < COORDS[p][1])
    for p in range(NUM_POINTS)
)
_NO_BACKWARD_STEPS: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple((1 << n, _ACTION_ID[Action(p, n, None)]) for n in NEIGHBORS[p] if COORDS[n][1] <= COORDS[p][1])
    for p in range(NUM_POINTS)
)
_JUMPS_FROM: tuple[tuple[tuple[int, int, int], ...], ...] = tuple(
    tuple((1 << b, 1 << c, _ACTION_ID[Action(a, c, b)]) for x, b, c in JUMPS if x == a)
    for a in range(NUM_POINTS)
)


def bitmask(points) -> int:
    mask = 0
    for p in points:
        mask |= 1 << p
    return mask


def points_of(mask: int) -> list[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def mirror_action(action: int) -> int:
    """The same move reflected left to right."""
    src, dst, over = ACTIONS[action]
    if src is None:
        return action
    return _ACTION_ID[Action(MIRROR_POINT[src], MIRROR_POINT[dst], None if over is None else MIRROR_POINT[over])]


def mirror_state(state: GameState) -> GameState:
    """The same position reflected left to right."""
    return state._replace(
        komikelu=MIRROR_POINT[state.komikelu],
        trewas=bitmask(MIRROR_POINT[p] for p in points_of(state.trewas)),
    )


def initial_state() -> GameState:
    return GameState(KOMIKELU_START, bitmask(TREWAS_START), KOMIKELU, False, 0, 0)


def legal_actions(state: GameState, rules: Rules = DEFAULT_RULES) -> list[int]:
    if state.to_move == KOMIKELU:
        return _komikelu_actions(state)
    occupied = state.trewas | (1 << state.komikelu)
    table = _FORWARD_STEPS if rules.trewas_forward_only else _NO_BACKWARD_STEPS if rules.trewas_no_backward else _STEPS
    return [a for p in points_of(state.trewas) for bit, a in table[p] if not occupied & bit]


def _komikelu_actions(state: GameState) -> list[int]:
    trewas = state.trewas
    jumps = [
        a
        for over, dst, a in _JUMPS_FROM[state.komikelu]
        if trewas & over and not trewas & dst
    ]
    if state.in_chain:
        return [*jumps, STOP]
    return [*jumps, *(a for bit, a in _STEPS[state.komikelu] if not trewas & bit)]


def apply_action(state: GameState, action: int, rules: Rules = DEFAULT_RULES) -> GameState:
    """Apply a legal action. Legality is not re-checked; see `legal_actions`."""
    move = ACTIONS[action]
    if state.to_move == TREWAS:
        trewas = state.trewas ^ (1 << move.src) ^ (1 << move.dst)
        return state._replace(trewas=trewas, to_move=KOMIKELU, plies=state.plies + 1)
    if action == STOP:
        return state._replace(to_move=TREWAS, in_chain=False, plies=state.plies + 1)
    if move.over is None:
        return state._replace(komikelu=move.dst, to_move=TREWAS, plies=state.plies + 1)

    next_state = state._replace(
        komikelu=move.dst,
        trewas=state.trewas ^ (1 << move.over),
        captured=state.captured + 1,
        in_chain=True,
    )
    # The turn passes automatically when no further jump exists.
    if _komikelu_actions(next_state) == [STOP]:
        return apply_action(next_state, STOP, rules)
    return next_state


def outcome(state: GameState, rules: Rules = DEFAULT_RULES) -> int | None:
    """KOMIKELU, TREWAS or DRAW when the game is over, else None."""
    if state.captured >= rules.captures_to_win:
        return KOMIKELU
    if not state.in_chain and not legal_actions(state, rules):
        return TREWAS if state.to_move == KOMIKELU else KOMIKELU
    if state.plies >= rules.max_plies:
        return DRAW
    return None


def komikelu_reach(state: GameState) -> int:
    """How many points the komikelu could reach, stepping or jumping, if the trewas stood still."""
    trewas = state.trewas
    seen = 1 << state.komikelu
    stack = [state.komikelu]
    while stack:
        p = stack.pop()
        for bit, a in _STEPS[p]:
            if not (trewas | seen) & bit:
                seen |= bit
                stack.append(ACTIONS[a].dst)
        for over, dst, a in _JUMPS_FROM[p]:
            if trewas & over and not (trewas | seen) & dst:
                seen |= dst
                stack.append(ACTIONS[a].dst)
    return seen.bit_count()


def render_text(state: GameState) -> str:
    """Plain-text board: K komikelu, t trewa, . empty point."""
    xs = sorted({x for x, _ in COORDS})
    ys = sorted({y for _, y in COORDS})
    grid = {xy: "." for xy in COORDS}
    for p in points_of(state.trewas):
        grid[COORDS[p]] = "t"
    grid[COORDS[state.komikelu]] = "K"
    rows = ["".join(grid.get((x, y), " ") for x in xs).rstrip() for y in ys]
    return "\n".join(rows)
