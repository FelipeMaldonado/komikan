"""Positions to start training episodes from, so the agent practises situations it rarely reaches.

`human_positions` takes them from a human's recorded wins. `agent_positions` is the placebo: ordinary
agent games at the same turn numbers, which tests whether the human's positions matter or just varied starts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .bots import Bot, make_bot
from .game import KOMIKELU, GameState, apply_action, initial_state, mirror_state, outcome
from .recordings import RecordedGame


def human_wins(games: Sequence[RecordedGame], role: int) -> list[RecordedGame]:
    """The games that a human won playing `role`."""
    return [game for game in games if game.outcome == role and role in game.human_roles]


def game_positions(game: RecordedGame, role: int, min_progress: float = 0.0, max_progress: float = 1.0) -> list[GameState]:
    """The positions where `role` moved during a part of one game, as a share of that game's moves."""
    states = [state for state, _ in game.replay() if state.to_move == role]
    return states[int(len(states) * min_progress) : int(len(states) * max_progress)]


def human_positions(
    games: Sequence[RecordedGame],
    role: int,
    min_progress: float = 0.25,
    exclude: Sequence[int] = (),
) -> list[GameState]:
    """Positions from the games that `role` won with a human playing it, also mirrored left to right."""
    positions: dict[GameState, None] = {}
    for i, game in enumerate(games):
        if i in exclude or game.outcome != role or role not in game.human_roles:
            continue
        for state in game_positions(game, role, min_progress):
            positions[state] = None
            positions[mirror_state(state)] = None
    return list(positions)


def agent_positions(
    like: Sequence[GameState],
    role: int,
    komikelu_specs: Sequence[str],
    trewas_spec: str,
    seed: int,
    games: int = 60,
) -> list[GameState]:
    """For each position in `like`, a position from an agent game with the same turn number, or the nearest."""
    rng = np.random.default_rng(seed)
    trewas = make_bot(trewas_spec, seed=seed)
    komikelu: list[Bot] = [make_bot(spec, seed=seed + 1 + i) for i, spec in enumerate(komikelu_specs)]

    by_plies: dict[int, list[GameState]] = {}
    for game in range(games):
        state = initial_state()
        white = komikelu[game % len(komikelu)]
        while outcome(state) is None:
            if state.to_move == role:
                by_plies.setdefault(state.plies, []).append(state)
            bot = white if state.to_move == KOMIKELU else trewas
            state = apply_action(state, bot(state))

    available = np.array(sorted(by_plies))
    chosen = []
    for target in like:
        plies = int(available[np.abs(available - target.plies).argmin()])
        state = by_plies[plies][rng.integers(len(by_plies[plies]))]
        chosen.append(mirror_state(state) if rng.random() < 0.5 else state)
    return chosen
