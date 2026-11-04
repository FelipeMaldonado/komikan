"""Self-play training with MaskablePPO: one model per role, in alternating generations.

Each generation trains each role against a mix of benchmark bots and past snapshots of the
other role, then scores it against fixed benchmarks by wins minus losses. The generation with
the best average is kept per role, because a rising self-play win rate doesn't show real
improvement, and a safe draw shouldn't count as success.

Output in --out: komikelu.{zip,npz}, trewas.{zip,npz} (best), metrics.csv, and snapshots/.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from .bots import make_bot
from .env import KomikanEnv
from .evaluate import play_match
from .game import KOMIKELU, ROLE_NAMES, TREWAS
from .policy import NumpyPolicy

SCRIPTED = ["random", "minimax:1", "minimax:2", "minimax:3"]
SCRIPTED_WEIGHTS = [0.15, 0.15, 0.2, 0.1]
SNAPSHOT_SHARE = 0.4
RECENT_SNAPSHOTS = 5


def _make_env(role: str, seed: int):
    def init():
        torch.set_num_threads(1)
        env = KomikanEnv(role=role, opponents=SCRIPTED, opponent_weights=SCRIPTED_WEIGHTS)
        env.reset(seed=seed)
        return env

    return init


def _opponent_pool(snapshots: list[Path]) -> tuple[list[str], list[float]]:
    recent = snapshots[-RECENT_SNAPSHOTS:]
    if not recent:
        return SCRIPTED, SCRIPTED_WEIGHTS
    specs = [*SCRIPTED, *(f"policy:{p}" for p in recent)]
    weights = [*SCRIPTED_WEIGHTS, *([SNAPSHOT_SHARE / len(recent)] * len(recent))]
    return specs, weights


def _benchmark(role: int, policy: NumpyPolicy, opponents: list[str], games: int, seed: int) -> dict[str, float]:
    """Score per benchmark opponent, playing stochastically as the agent would in practice."""
    agent = policy.as_bot(seed=seed)
    scores = {}
    for spec in opponents:
        other = make_bot(spec, seed + 1)
        match = play_match(agent, other, games) if role == KOMIKELU else play_match(other, agent, games)
        scores[spec] = match.score(role)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("runs/default"))
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--steps-per-generation", type=int, default=200_000, help="env steps per role per generation")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--benchmarks", nargs="+", default=["random", "minimax:1", "minimax:2", "minimax:3"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    snapshot_dir = args.out / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    roles = (KOMIKELU, TREWAS)
    envs, models = {}, {}
    for role in roles:
        name = ROLE_NAMES[role]
        envs[role] = SubprocVecEnv([_make_env(name, args.seed + 100 * role + i) for i in range(args.envs)])
        models[role] = MaskablePPO(
            "MlpPolicy",
            envs[role],
            n_steps=1024,
            batch_size=1024,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
            seed=args.seed + role,
            device="cpu",
        )

    snapshots: dict[int, list[Path]] = {KOMIKELU: [], TREWAS: []}
    best = {KOMIKELU: float("-inf"), TREWAS: float("-inf")}
    metrics_path = args.out / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        csv.writer(f).writerow(["generation", "role", "opponent", "score", "total_steps", "seconds"])

    start = time.time()
    try:
        for gen in range(args.generations):
            for role in roles:
                name = ROLE_NAMES[role]
                other = TREWAS if role == KOMIKELU else KOMIKELU
                specs, weights = _opponent_pool(snapshots[other])
                envs[role].env_method("set_opponents", specs, weights)
                models[role].learn(args.steps_per_generation, reset_num_timesteps=False)

                policy = NumpyPolicy.from_sb3(models[role])
                snapshot = snapshot_dir / f"{name}_{gen:03d}.npz"
                policy.save(snapshot)
                snapshots[role].append(snapshot)
                models[role].save(args.out / f"{name}_last.zip")  # to resume, or if a later check prefers it

                scores = _benchmark(role, policy, args.benchmarks, args.eval_games, args.seed + gen)
                headline = sum(scores.values()) / len(scores)
                if headline > best[role]:
                    best[role] = headline
                    shutil.copy(snapshot, args.out / f"{name}.npz")
                    models[role].save(args.out / f"{name}.zip")

                elapsed = time.time() - start
                with metrics_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    for spec, score in scores.items():
                        writer.writerow([gen, name, spec, f"{score:.3f}", models[role].num_timesteps, f"{elapsed:.0f}"])
                summary = "  ".join(f"{spec}={score:.2f}" for spec, score in scores.items())
                print(f"gen {gen:3d} {name:8} {summary}  (best {best[role]:.2f}, {elapsed:.0f}s)", flush=True)
    finally:
        for env in envs.values():
            env.close()


if __name__ == "__main__":
    main()
