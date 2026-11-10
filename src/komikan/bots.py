"""Scripted opponents, used as training partners and as fixed benchmarks.

A bot is any callable `bot(state) -> action`. Multi-jumps take one call per jump.
"""

from __future__ import annotations

import random
from typing import Callable

from .game import (
    DEFAULT_RULES,
    DRAW,
    KOMIKELU,
    TREWAS,
    GameState,
    Rules,
    apply_action,
    komikelu_reach,
    legal_actions,
    outcome,
)

Bot = Callable[[GameState], int]
_WIN = 1_000_000.0


class RandomBot:
    def __init__(self, seed: int | None = None, rules: Rules = DEFAULT_RULES):
        self.rng = random.Random(seed)
        self.rules = rules

    def __call__(self, state: GameState) -> int:
        return self.rng.choice(legal_actions(state, self.rules))


def evaluate(state: GameState, rules: Rules = DEFAULT_RULES) -> float:
    """Heuristic score, from the komikelu's point of view."""
    result = outcome(state, rules)
    if result == KOMIKELU:
        return _WIN - state.plies  # sooner wins score higher
    if result == TREWAS:
        return -_WIN + state.plies
    if result == DRAW:
        return 0.0
    return 100.0 * state.captured + 2.0 * komikelu_reach(state)


def squeeze_evaluate(state: GameState, rules: Rules = DEFAULT_RULES) -> float:
    """Like `evaluate` but with `mcts.squeeze_heuristic` for unfinished games, and a draw counted as good for the komikelu
    (0.5 on the heuristic's scale), so that a trewas player tries to win rather than settle."""
    from .mcts import squeeze_heuristic

    result = outcome(state, rules)
    if result == KOMIKELU:
        return _WIN - state.plies
    if result == TREWAS:
        return -_WIN + state.plies
    if result == DRAW:
        return 500.0
    return 1000.0 * squeeze_heuristic(state, rules)


EVALUATIONS = {"basic": evaluate, "squeeze": squeeze_evaluate}


class MinimaxBot:
    """Alpha-beta search. Depth counts turns, and a multi-jump is searched to its end as one turn.

    Equal moves are shuffled so that repeated games differ. `evaluation` names the scoring of positions where it stops.
    """

    def __init__(self, depth: int = 2, seed: int | None = None, rules: Rules = DEFAULT_RULES, evaluation: str = "basic"):
        self.depth = depth
        self.rng = random.Random(seed)
        self.rules = rules
        self.evaluate = EVALUATIONS[evaluation]

    def __call__(self, state: GameState) -> int:
        maximizing = state.to_move == KOMIKELU
        actions = legal_actions(state, self.rules)
        self.rng.shuffle(actions)
        best_action, best = actions[0], float("-inf") if maximizing else float("inf")
        alpha, beta = float("-inf"), float("inf")
        for action in actions:
            value = self._search(apply_action(state, action, self.rules), state.to_move, self.depth, alpha, beta)
            if maximizing and value > best:
                best_action, best, alpha = action, value, max(alpha, value)
            elif not maximizing and value < best:
                best_action, best, beta = action, value, min(beta, value)
        return best_action

    def _search(self, state: GameState, mover: int, depth: int, alpha: float, beta: float) -> float:
        # Only a change of side uses up depth, so chains are always followed through.
        if state.to_move != mover:
            depth -= 1
        if depth <= 0 or outcome(state, self.rules) is not None:
            return self.evaluate(state, self.rules)
        if state.to_move == KOMIKELU:
            value = float("-inf")
            for action in legal_actions(state, self.rules):
                value = max(value, self._search(apply_action(state, action, self.rules), KOMIKELU, depth, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        value = float("inf")
        for action in legal_actions(state, self.rules):
            value = min(value, self._search(apply_action(state, action, self.rules), TREWAS, depth, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def make_bot(spec: str, seed: int | None = None, rules: Rules = DEFAULT_RULES) -> Bot:
    """Build a bot from a spec: "random", "minimax:<depth>[:<evaluation>]", "policy:<path.npz>" or
    "mcts:<simulations>[:<draw>[:<playout>[:<heuristic>]]]" or "puct:<simulations>:<mode>[:<exploration>]" or "az:<simulations>:<directory>"."""
    kind, _, arg = spec.partition(":")
    if kind == "az":
        from .puct import Networks
        from .selfplay import AlphaZeroBot

        simulations, directory = arg.split(":", 1)
        return AlphaZeroBot(Networks.load(directory, directory), int(simulations), seed=seed, rules=rules)
    if kind == "puct":
        from .puct import Networks, PUCTBot

        simulations, mode, exploration = (arg.split(":") + ["", "", ""])[:3]
        return PUCTBot(Networks.load(), int(simulations or 400), mode or "both", float(exploration or 1.5), seed=seed, rules=rules)
    if kind == "mcts":
        from .mcts import MCTSBot

        simulations, draw, playout, heuristic = (arg.split(":") + ["", "", ""])[:4]
        return MCTSBot(
            int(simulations or 400), int(playout or 8), draw=float(draw or 0.0), seed=seed, rules=rules, heuristic=heuristic or "basic"
        )
    if kind == "random":
        return RandomBot(seed, rules)
    if kind == "minimax":
        depth, evaluation = (arg.split(":") + [""])[:2]
        return MinimaxBot(int(depth or 2), seed, rules, evaluation or "basic")
    if kind == "policy":
        from .policy import NumpyPolicy  # local import keeps scripted bots free of policy loading

        return NumpyPolicy.load(arg).as_bot(rules=rules, seed=seed)
    raise ValueError(f"unknown bot spec: {spec!r}")
