"""Gymnasium environment: one agent plays one role, and the opponent is part of the env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np

from .board import COORDS, NUM_POINTS
from .bots import Bot, RandomBot, make_bot
from .game import (
    DEFAULT_RULES,
    DRAW,
    KOMIKELU,
    NUM_ACTIONS,
    TREWAS,
    GameState,
    Rules,
    apply_action,
    initial_state,
    komikelu_reach,
    legal_actions,
    outcome,
    points_of,
    render_text,
)

OBS_SIZE = 2 * NUM_POINTS + 3
_ROW_HEIGHT = 3  # coordinate units per row of the square


def _komikelu_options(state: GameState, rules: Rules) -> list[int]:
    return legal_actions(state._replace(to_move=KOMIKELU), rules)


def encode_observation(state: GameState, rules: Rules = DEFAULT_RULES) -> np.ndarray:
    """Komikelu one-hot, trewas bitmap, in-chain flag, capture progress, turn progress."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[state.komikelu] = 1.0
    for p in points_of(state.trewas):
        obs[NUM_POINTS + p] = 1.0
    obs[-3] = float(state.in_chain)
    obs[-2] = state.captured / rules.captures_to_win
    obs[-1] = state.plies / rules.max_plies
    return obs


def action_mask(state: GameState, rules: Rules = DEFAULT_RULES) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[legal_actions(state, rules)] = True
    return mask


@dataclass(frozen=True)
class RewardConfig:
    """Only `win`/`loss` reflect the game's result; the rest is shaping to help learning."""

    win: float = 1.0
    loss: float = -1.0
    draw_komikelu: float = -0.25
    draw_trewas: float = -0.75  # a draw is what safe shuffling earns, so it must cost the trewas nearly a loss
    capture: float = 0.1  # gained by the komikelu and lost by the trewas, per capture
    # Trewas potentials: each pays for its change, so shuffling back and forth earns nothing.
    confine: float = 0.5  # the komikelu's reachable share of the board
    advance: float = 0.02  # per board row advanced by a trewa
    corner: float = 0.02  # per komikelu option removed


class KomikanEnv(gym.Env):
    """Komikan from one role's point of view.

    Args:
        role: "komikelu" or "trewas".
        opponents: bot specs (see `make_bot`) or bots. One is drawn per episode.
        opponent_weights: optional sampling weights for `opponents`.

    Each multi-jump takes one env step per jump. `action_masks()` gives the legal actions.
    An illegal action ends the episode with `reward.loss` and `info["illegal"] = True`.
    `info["outcome"]` holds the result: KOMIKELU, TREWAS or DRAW.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        role: str = "komikelu",
        opponents: Sequence[str | Bot] | None = None,
        opponent_weights: Sequence[float] | None = None,
        rules: Rules = DEFAULT_RULES,
        reward: RewardConfig = RewardConfig(),
        render_mode: str | None = None,
    ):
        if role not in ("komikelu", "trewas"):
            raise ValueError(f"role must be 'komikelu' or 'trewas', got {role!r}")
        self.role = KOMIKELU if role == "komikelu" else TREWAS
        self.rules = rules
        self.reward_config = reward
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self._cache: dict[str, Bot] = {}
        self.set_opponents(opponents or ["random"], opponent_weights)
        self.state = initial_state()
        self.opponent: Bot = RandomBot(rules=rules)

    def set_opponents(self, opponents: Sequence[str | Bot], weights: Sequence[float] | None = None) -> None:
        """Replace the opponent pool. It takes effect from the next episode."""
        self._opponents = list(opponents)
        self._weights = np.asarray(weights if weights is not None else [1.0] * len(self._opponents), dtype=float)
        self._weights = self._weights / self._weights.sum()

    def _draw_opponent(self) -> Bot:
        choice = self._opponents[self.np_random.choice(len(self._opponents), p=self._weights)]
        if callable(choice):
            return choice
        if choice not in self._cache:
            self._cache[choice] = make_bot(choice, seed=int(self.np_random.integers(2**31)), rules=self.rules)
        return self._cache[choice]

    def action_masks(self) -> np.ndarray:
        return action_mask(self.state, self.rules)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._cache.clear()  # rebuild bots so that their randomness follows the seed too
        self.opponent = self._draw_opponent()
        self.state = initial_state()
        self._play_opponent()
        return encode_observation(self.state, self.rules), {}

    def _play_opponent(self) -> None:
        while self.state.to_move != self.role and outcome(self.state, self.rules) is None:
            self.state = apply_action(self.state, self.opponent(self.state), self.rules)

    def step(self, action):
        action = int(action)
        cfg = self.reward_config
        if action not in legal_actions(self.state, self.rules):
            obs = encode_observation(self.state, self.rules)
            return obs, cfg.loss, True, False, {"illegal": True}

        before = self.state
        self.state = apply_action(self.state, action, self.rules)
        self._play_opponent()

        reward = self._shaping(before, self.state)
        result = outcome(self.state, self.rules)
        info = {}
        if result is not None:
            info["outcome"] = result
            draw = cfg.draw_komikelu if self.role == KOMIKELU else cfg.draw_trewas
            reward += draw if result == DRAW else cfg.win if result == self.role else cfg.loss
        return encode_observation(self.state, self.rules), reward, result is not None, False, info

    def _shaping(self, before: GameState, after: GameState) -> float:
        cfg = self.reward_config
        captures = after.captured - before.captured
        if self.role == KOMIKELU:
            return cfg.capture * captures
        return self._trewas_potential(after) - self._trewas_potential(before) - cfg.capture * captures

    def _trewas_potential(self, state: GameState) -> float:
        cfg = self.reward_config
        height = sum(COORDS[p][1] for p in points_of(state.trewas)) / _ROW_HEIGHT
        options = len(_komikelu_options(state, self.rules))
        reach = komikelu_reach(state) / NUM_POINTS
        return -cfg.advance * height - cfg.corner * options - cfg.confine * reach

    def render(self):
        if self.render_mode == "ansi":
            return render_text(self.state)
        return None
