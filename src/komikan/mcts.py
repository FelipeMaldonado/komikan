"""Monte Carlo tree search with no network: how much does looking ahead alone help in this game?

Each simulation walks down the tree by UCT, adds one node, then estimates that position with a short random
playout followed by a heuristic. Values are always stored from the komikelu's point of view, and each node
picks by its own `to_move`. Signs must not alternate with depth, because a multi-jump moves the same side twice.

`draw` is the value of a draw to the komikelu. Above 0, the search treats a draw as bad for the trewas, which
a trewas player needs so that it doesn't settle for safe shuffling.
"""

from __future__ import annotations

import math
import random

from .board import COORDS, NUM_POINTS
from .game import (
    DEFAULT_RULES,
    DRAW,
    KOMIKELU,
    NUM_TREWAS,
    GameState,
    Rules,
    apply_action,
    komikelu_reach,
    legal_actions,
    outcome,
    points_of,
)

_MAX_HEIGHT = max(y for _, y in COORDS)


def heuristic(state: GameState, rules: Rules = DEFAULT_RULES) -> float:
    """An estimate in [-1, 1] for a position that isn't over, from the komikelu's point of view."""
    captures = state.captured / rules.captures_to_win
    freedom = komikelu_reach(state) / NUM_POINTS  # falls as the trewas box the komikelu in
    trewas = points_of(state.trewas)
    remaining = len(trewas) / NUM_TREWAS
    advance = 1.0 - sum(COORDS[p][1] for p in trewas) / (len(trewas) * _MAX_HEIGHT) if trewas else 0.0
    return max(-1.0, min(1.0, 1.2 * captures + 0.8 * (freedom - 0.5) - 0.2 * remaining * advance))


def squeeze_heuristic(state: GameState, rules: Rules = DEFAULT_RULES) -> float:
    """Like `heuristic`, but it still sees progress once the komikelu is shut in a small area.

    Its room counts on a log scale, so taking one of the triangle's 13 points matters about as much as taking 2 of 26 elsewhere.
    Its immediate moves count too, and advancing counts twice as much. Written after a human game in which search shut the komikelu
    in the triangle but never entered it.
    """
    captures = state.captured / rules.captures_to_win
    freedom = math.log(komikelu_reach(state)) / math.log(NUM_POINTS)
    moves = min(len(legal_actions(state._replace(to_move=KOMIKELU, in_chain=False), rules)), 8) / 8
    trewas = points_of(state.trewas)
    remaining = len(trewas) / NUM_TREWAS
    advance = 1.0 - sum(COORDS[p][1] for p in trewas) / (len(trewas) * _MAX_HEIGHT) if trewas else 0.0
    value = 1.2 * captures + 0.8 * (freedom - 0.75) + 0.2 * (moves - 0.5) - 0.4 * remaining * advance
    return max(-1.0, min(1.0, value))


HEURISTICS = {"basic": heuristic, "squeeze": squeeze_heuristic}


class _Node:
    __slots__ = ("state", "untried", "children", "visits", "total", "terminal")

    def __init__(self, state: GameState, rules: Rules, rng: random.Random):
        self.state = state
        self.terminal = outcome(state, rules)
        self.untried = [] if self.terminal is not None else legal_actions(state, rules)
        rng.shuffle(self.untried)
        self.children: dict[int, _Node] = {}
        self.visits = 0
        self.total = 0.0  # sum of values, from the komikelu's point of view


class MCTSBot:
    """Args:
    simulations: tree-search simulations per move.
    playout: random turns played before the heuristic is applied.
    exploration: UCT constant.
    draw: value of a draw to the komikelu.
    """

    def __init__(
        self,
        simulations: int = 400,
        playout: int = 8,
        exploration: float = 0.7,
        draw: float = 0.0,
        seed: int | None = None,
        rules: Rules = DEFAULT_RULES,
        heuristic: str = "basic",
    ):
        self.heuristic = HEURISTICS[heuristic]
        self.simulations = simulations
        self.playout = playout
        self.exploration = exploration
        self.draw = draw
        self.rng = random.Random(seed)
        self.rules = rules

    def __call__(self, state: GameState) -> int:
        actions = legal_actions(state, self.rules)
        if len(actions) == 1:
            return actions[0]
        root = _Node(state, self.rules, self.rng)
        for _ in range(self.simulations):
            self._simulate(root)
        # The most visited move is the most reliable; a mate found late can still have few visits, so check those first.
        return max(root.children, key=lambda action: self._final_rank(root, action))

    def _final_rank(self, root: _Node, action: int) -> tuple[bool, int]:
        child = root.children[action]
        mover_wins = child.terminal == root.state.to_move
        return mover_wins, child.visits

    def _simulate(self, root: _Node) -> None:
        path = [root]
        node = root
        while node.terminal is None and not node.untried:
            node = self._select(node)
            path.append(node)
        if node.terminal is None:
            action = node.untried.pop()
            child = _Node(apply_action(node.state, action, self.rules), self.rules, self.rng)
            node.children[action] = child
            node = child
            path.append(node)
        value = self._estimate(node)
        for visited in path:
            visited.visits += 1
            visited.total += value

    def _select(self, node: _Node) -> _Node:
        sign = 1.0 if node.state.to_move == KOMIKELU else -1.0
        log_visits = math.log(node.visits)
        c = self.exploration
        best, best_score = None, -math.inf
        for child in node.children.values():
            score = sign * child.total / child.visits + c * math.sqrt(log_visits / child.visits)
            if score > best_score:
                best, best_score = child, score
        return best

    def _estimate(self, node: _Node) -> float:
        if node.terminal is not None:
            return self._result_value(node.terminal)
        return self.estimate(node.state)

    def estimate(self, state: GameState) -> float:
        """A short random playout, then the heuristic, from the komikelu's point of view."""
        plies_left = self.playout
        while plies_left > 0:
            result = outcome(state, self.rules)
            if result is not None:
                return self._result_value(result)
            before = state.to_move
            state = apply_action(state, self.rng.choice(legal_actions(state, self.rules)), self.rules)
            plies_left -= state.to_move != before
        result = outcome(state, self.rules)
        return self._result_value(result) if result is not None else self.heuristic(state, self.rules)

    def _result_value(self, result: int) -> float:
        if result == DRAW:
            return self.draw
        return 1.0 if result == KOMIKELU else -1.0
