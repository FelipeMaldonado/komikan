"""PUCT search guided by the existing PPO networks: stage 2 of the AlphaZero plan.

Modes say what the networks supply:
  none:     uniform priors and the heuristic leaf estimate, a check against plain UCT in `mcts.py`.
  prior:    move priors from the policy networks, and the heuristic leaf estimate.
  value:    uniform priors, and the searching role's value network at the leaves.
  rawvalue: as value, without the correction below, to test whether the correction matters.
  both:     priors and value network.

A PPO value network wasn't trained for search, so it needs two adjustments:
1. It only saw positions with its own role to move, and its input doesn't say whose turn it is. A leaf with the
   opponent to move is first advanced by the opponent's most likely replies.
2. It predicts the rewards still to come, including shaping. A badly placed position therefore looks better,
   because there is more shaping left to earn. Adding the shaping already earned gives a comparable total.

Values are stored from the komikelu's point of view, and each node chooses by its own `to_move`.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from .env import RewardConfig, encode_observation, trewas_potential
from .game import (
    DEFAULT_RULES,
    DRAW,
    KOMIKELU,
    TREWAS,
    GameState,
    Rules,
    apply_action,
    initial_state,
    legal_actions,
    outcome,
)
from .mcts import MCTSBot
from .policy import NumpyPolicy

MODES = ("none", "prior", "value", "rawvalue", "both")
VALUE_DIR = Path("runs/guided/nets")
UNIFORM_MIX = 0.1  # share of each prior kept uniform, so that no legal move is ruled out by a confident network


def export_value_net(model_path: str | Path, out_path: str | Path) -> None:
    """Save the critic of a MaskablePPO model in the `NumpyPolicy` format."""
    import torch
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path, device="cpu")
    layers = [m for m in model.policy.mlp_extractor.value_net if isinstance(m, torch.nn.Linear)]
    layers.append(model.policy.value_net)
    NumpyPolicy(
        [layer.weight.detach().numpy() for layer in layers],
        [layer.bias.detach().numpy() for layer in layers],
    ).save(out_path)


class Networks:
    """The policy and value networks of both roles."""

    def __init__(self, policies: dict[int, NumpyPolicy], values: dict[int, NumpyPolicy]):
        self.policies = policies
        self.values = values

    @classmethod
    def load(cls, policy_dir: str | Path = "models", value_dir: str | Path = VALUE_DIR) -> "Networks":
        names = {KOMIKELU: "komikelu", TREWAS: "trewas"}
        return cls(
            {r: NumpyPolicy.load(Path(policy_dir) / f"{n}.npz") for r, n in names.items()},
            {r: NumpyPolicy.load(Path(value_dir) / f"{n}_value.npz") for r, n in names.items()},
        )


class _Node:
    __slots__ = ("state", "terminal", "actions", "priors", "children", "visits", "totals", "count", "total")

    def __init__(self, state: GameState, terminal: int | None, actions: list[int], priors: list[float]):
        self.state = state
        self.terminal = terminal
        self.actions = actions
        self.priors = priors
        self.children: list[_Node | None] = [None] * len(actions)
        self.visits = [0] * len(actions)
        self.totals = [0.0] * len(actions)
        self.count = 0
        self.total = 0.0  # sum of values, from the komikelu's point of view


class PUCTBot:
    def __init__(
        self,
        networks: Networks,
        simulations: int = 400,
        mode: str = "both",
        exploration: float = 1.5,
        draw: float = 0.5,
        playout: int = 2,
        seed: int | None = None,
        rules: Rules = DEFAULT_RULES,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.networks = networks
        self.simulations = simulations
        self.mode = mode
        self.exploration = exploration
        self.rules = rules
        self.rng = random.Random(seed)
        self.leaf = MCTSBot(playout=playout, draw=draw, seed=seed, rules=rules)
        self.reward = RewardConfig()
        self.start_potential = trewas_potential(initial_state(), rules, self.reward)
        self.role = KOMIKELU  # set per move, from the root

    def __call__(self, state: GameState) -> int:
        actions = legal_actions(state, self.rules)
        if len(actions) == 1:
            return actions[0]
        for action in actions:
            if outcome(apply_action(state, action, self.rules), self.rules) == state.to_move:
                return action
        root = self.search(state)
        best = max(range(len(root.actions)), key=lambda i: (root.visits[i], root.priors[i]))
        return root.actions[best]

    def search(self, state: GameState) -> _Node:
        """Run the simulations from `state` and return the root, whose visits say how the search rated each move."""
        self.role = state.to_move
        root = self._make_node(state)
        self._prepare_root(root)
        for _ in range(self.simulations):
            self._simulate(root)
        return root

    def _prepare_root(self, root: _Node) -> None:
        """A hook for changing the root's priors before the search."""

    def _make_node(self, state: GameState) -> _Node:
        terminal = outcome(state, self.rules)
        if terminal is not None:
            return _Node(state, terminal, [], [])
        actions = legal_actions(state, self.rules)
        self.rng.shuffle(actions)
        return _Node(state, None, actions, self._priors(state, actions))

    def _priors(self, state: GameState, actions: list[int]) -> list[float]:
        uniform = 1.0 / len(actions)
        if self.mode not in ("prior", "both"):
            return [uniform] * len(actions)
        logits = self.networks.policies[state.to_move].logits(encode_observation(state, self.rules))[actions]
        odds = np.exp(logits - logits.max())
        return list((1.0 - UNIFORM_MIX) * odds / odds.sum() + UNIFORM_MIX * uniform)

    def _simulate(self, root: _Node) -> None:
        path: list[tuple[_Node, int]] = []
        node = root
        while True:
            if node.terminal is not None:
                value = self._leaf_value(node.state, node.terminal)
                break
            index = self._select(node)
            path.append((node, index))
            child = node.children[index]
            if child is None:
                child = self._make_node(apply_action(node.state, node.actions[index], self.rules))
                node.children[index] = child
                value = self._leaf_value(child.state, child.terminal)
                child.count += 1
                child.total += value
                break
            node = child
        for visited, index in path:
            visited.count += 1
            visited.total += value
            visited.visits[index] += 1
            visited.totals[index] += value

    def _select(self, node: _Node) -> int:
        sign = 1.0 if node.state.to_move == KOMIKELU else -1.0
        parent_mean = node.total / node.count if node.count else 0.0
        scale = self.exploration * math.sqrt(node.count + 1)
        best, best_score = 0, -math.inf
        for i, prior in enumerate(node.priors):
            n = node.visits[i]
            mean = node.totals[i] / n if n else parent_mean  # an untried move starts at the parent's value
            score = sign * mean + scale * prior / (1 + n)
            if score > best_score:
                best, best_score = i, score
        return best

    def _leaf_value(self, state: GameState, terminal: int | None) -> float:
        if self.mode in ("value", "rawvalue", "both"):
            return self._network_value(state, terminal)
        if terminal is not None:
            return self.leaf._result_value(terminal)
        return self.leaf.estimate(state)

    def _network_value(self, state: GameState, terminal: int | None) -> float:
        result = terminal
        while result is None and state.to_move != self.role:
            state = apply_action(state, self._likely_reply(state), self.rules)
            result = outcome(state, self.rules)
        predicted = 0.0
        if result is None:
            predicted = float(self.networks.values[self.role].logits(encode_observation(state, self.rules))[0])
        total = predicted + self._return_so_far(state, result)
        return total if self.role == KOMIKELU else -total

    def _likely_reply(self, state: GameState) -> int:
        actions = legal_actions(state, self.rules)
        logits = self.networks.policies[state.to_move].logits(encode_observation(state, self.rules))[actions]
        return actions[int(np.argmax(logits))]

    def _return_so_far(self, state: GameState, result: int | None) -> float:
        """The role's total reward up to this position: shaping earned, and the final reward if it's over."""
        cfg = self.reward
        if result is None:
            final = 0.0
        elif result == DRAW:
            final = cfg.draw_komikelu if self.role == KOMIKELU else cfg.draw_trewas
        else:
            final = cfg.win if result == self.role else cfg.loss
        if self.mode == "rawvalue":
            return final
        if self.role == KOMIKELU:
            return final + cfg.capture * state.captured
        return final + trewas_potential(state, self.rules, cfg) - self.start_potential - cfg.capture * state.captured
