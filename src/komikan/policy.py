"""Numpy-only inference for trained agents, so playing needs no torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bots import Bot
from .env import action_mask, encode_observation
from .game import DEFAULT_RULES, GameState, Rules


class NumpyPolicy:
    """A ReLU MLP mapping observations to action logits."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]):
        self.weights = weights
        self.biases = biases

    @classmethod
    def from_sb3(cls, model) -> "NumpyPolicy":
        """Export the actor of a MaskablePPO model built with ReLU layers."""
        import torch

        layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
        layers.append(model.policy.action_net)
        return cls(
            [layer.weight.detach().cpu().numpy() for layer in layers],
            [layer.bias.detach().cpu().numpy() for layer in layers],
        )

    def save(self, path: str | Path) -> None:
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)} | {f"b{i}": b for i, b in enumerate(self.biases)}
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        with np.load(path) as data:
            count = len(data.files) // 2
            return cls([data[f"w{i}"] for i in range(count)], [data[f"b{i}"] for i in range(count)])

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = obs
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(w @ x + b, 0.0)
        return self.weights[-1] @ x + self.biases[-1]

    def as_bot(
        self,
        rules: Rules = DEFAULT_RULES,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> Bot:
        rng = np.random.default_rng(seed)

        def bot(state: GameState) -> int:
            logits = self.logits(encode_observation(state, rules))
            mask = action_mask(state, rules)
            if deterministic:
                return int(np.flatnonzero(mask)[np.argmax(logits[mask])])
            legal = np.flatnonzero(mask)
            scores = logits[legal] - logits[legal].max()
            probs = np.exp(scores)
            return int(rng.choice(legal, p=probs / probs.sum()))

        return bot
