"""Train an agent from recorded human games, then continue with PPO.

1. Behavioural cloning: fit the policy to the human's moves, each also mirrored left to right.
2. Value warm-up: PPO plays with the policy frozen, so the untrained value network can't
   damage the cloned policy with its first updates.
3. PPO continues against a fixed pool of bots and agents, scored against benchmarks each generation.

--no-clone skips steps 1 and 2 and spends their steps on ordinary PPO, as a fair control.
--init starts from a saved model, so cloning fine-tunes an agent that already plays safely.

Output in --out: <role>.npz (best by benchmark), <role>_cloned.npz (after step 1, before PPO),
<role>_last.zip, metrics.csv and snapshots/.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from .env import KomikanEnv, RewardConfig, action_mask, encode_observation
from .game import KOMIKELU, ROLE_NAMES, TREWAS, mirror_action, mirror_state
from .midgame import agent_positions, human_positions
from .policy import NumpyPolicy
from .recordings import RecordedGame, load_all
from .train import SCRIPTED, SCRIPTED_WEIGHTS, SNAPSHOT_SHARE, _benchmark, _make_env, make_ppo


@dataclass
class Examples:
    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    games: np.ndarray  # which game each example came from

    def __len__(self) -> int:
        return len(self.actions)

    def subset(self, keep: np.ndarray) -> "Examples":
        return Examples(self.observations[keep], self.masks[keep], self.actions[keep], self.games[keep])


def build_examples(games: list[RecordedGame], role: int, mirror: bool = True) -> Examples:
    """The human's moves in `role`. A mirrored move stays with its game, so folds never share a game."""
    observations, masks, actions, ids = [], [], [], []
    for game_id, game in enumerate(games):
        for state, action in game.human_moves():
            if state.to_move != role:
                continue
            variants = [(state, action)]
            if mirror:
                variants.append((mirror_state(state), mirror_action(action)))
            for s, a in variants:
                observations.append(encode_observation(s, game.rules))
                masks.append(action_mask(s, game.rules))
                actions.append(a)
                ids.append(game_id)
    if not actions:
        raise ValueError(f"no recorded moves for {ROLE_NAMES[role]}")
    return Examples(np.array(observations), np.array(masks), np.array(actions), np.array(ids))


def randomize_actions(examples: Examples, seed: int) -> Examples:
    """A placebo: the same positions with a random legal move in place of each human one."""
    rng = np.random.default_rng(seed)
    actions = np.array([rng.choice(np.flatnonzero(mask)) for mask in examples.masks])
    return Examples(examples.observations, examples.masks, actions, examples.games)


def load_ppo(path: Path, env, seed: int, learning_rate: float, ent_coef: float) -> MaskablePPO:
    """Load a model to continue training. The seed must be replaced too: loading re-applies the saved
    one to torch and the envs, which would make every run from the same model identical."""
    return MaskablePPO.load(
        path,
        env=env,
        device="cpu",
        custom_objects={
            "learning_rate": learning_rate,
            "lr_schedule": lambda _: learning_rate,
            "ent_coef": ent_coef,
            "seed": seed,
        },
    )


def _masked_logits(policy, observations: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    logits = policy.action_net(policy.mlp_extractor.forward_actor(observations))
    return logits.masked_fill(~masks, float("-inf"))


def behaviour_clone(
    policy,
    examples: Examples,
    epochs: int = 150,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> float:
    """Fit the actor to the examples, leaving the value network alone. Returns the final loss."""
    observations = torch.as_tensor(examples.observations)
    masks = torch.as_tensor(examples.masks)
    actions = torch.as_tensor(examples.actions)
    parameters = list(_actor_parameters(policy))
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator)
        for start in range(0, len(examples), batch_size):
            batch = order[start : start + batch_size]
            loss = torch.nn.functional.cross_entropy(_masked_logits(policy, observations[batch], masks[batch]), actions[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.item()


def accuracy(policy, examples: Examples) -> float:
    """How often the policy's top legal choice is the human's move."""
    with torch.no_grad():
        logits = _masked_logits(policy, torch.as_tensor(examples.observations), torch.as_tensor(examples.masks))
    return float((logits.argmax(dim=1).numpy() == examples.actions).mean())


def random_guess_accuracy(examples: Examples) -> float:
    """What picking a legal move at random would score, for comparison."""
    return float((1.0 / examples.masks.sum(axis=1)).mean())


def cross_validate(examples: Examples, role: str, epochs: int, seed: int) -> list[float]:
    """Accuracy on each game when trained on the others: what the agent generalises, not memorises."""
    results = []
    for game in np.unique(examples.games):
        model = make_ppo(KomikanEnv(role=role), seed=seed)
        behaviour_clone(model.policy, examples.subset(examples.games != game), epochs=epochs, seed=seed)
        results.append(accuracy(model.policy, examples.subset(examples.games == game)))
    return results


def _actor_parameters(policy):
    return chain(policy.mlp_extractor.policy_net.parameters(), policy.action_net.parameters())


def set_actor_frozen(policy, frozen: bool) -> None:
    for parameter in _actor_parameters(policy):
        parameter.requires_grad_(not frozen)


def _pool(partners: list[str]) -> tuple[list[str], list[float]]:
    if not partners:
        return SCRIPTED, SCRIPTED_WEIGHTS
    specs = [*SCRIPTED, *(f"policy:{p}" for p in partners)]
    return specs, [*SCRIPTED_WEIGHTS, *([SNAPSHOT_SHARE / len(partners)] * len(partners))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", nargs="*", type=Path, default=[])
    parser.add_argument("--role", choices=list(ROLE_NAMES.values()), default="trewas")
    parser.add_argument("--no-clone", action="store_true", help="control: skip cloning and the warm-up")
    parser.add_argument("--placebo", action="store_true", help="control: clone random legal moves instead of the human's")
    parser.add_argument("--init", type=Path, help="start from this saved model (.zip) instead of a new one")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clone-epochs", type=int, default=20, help="unseen-game accuracy levels off by about 20")
    parser.add_argument("--clone-learning-rate", type=float, default=1e-3, help="use far less with --init, to keep what it knows")
    parser.add_argument("--warmup-steps", type=int, default=200_000)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--steps-per-generation", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--partners", nargs="*", default=[], help="saved agents (.npz) to train against")
    parser.add_argument("--benchmarks", nargs="+", default=["random", "minimax:1", "minimax:2"])
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--envs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-from", choices=["human", "agent"], help="start some episodes from mid-game positions")
    parser.add_argument("--start-share", type=float, default=0.5, help="share of episodes that start mid-game")
    parser.add_argument("--start-min-progress", type=float, default=0.25, help="skip this first share of each game")
    parser.add_argument("--start-holdout", type=int, nargs="*", default=[], help="recorded games not to take positions from")
    defaults = RewardConfig()
    parser.add_argument("--reward-advance", type=float, default=defaults.advance)
    parser.add_argument("--reward-confine", type=float, default=defaults.confine)
    parser.add_argument("--reward-draw-trewas", type=float, default=defaults.draw_trewas)
    parser.add_argument("--relative-advance", action="store_true", help="no advance bonus for moving past the komikelu")
    args = parser.parse_args()

    role = KOMIKELU if args.role == "komikelu" else TREWAS
    snapshot_dir = args.out / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    env = SubprocVecEnv([_make_env(args.role, args.seed + i) for i in range(args.envs)])
    env.env_method("set_opponents", *_pool(args.partners))
    env.set_attr(
        "reward_config",
        replace(
            defaults,
            advance=args.reward_advance,
            confine=args.reward_confine,
            draw_trewas=args.reward_draw_trewas,
            relative_advance=args.relative_advance,
        ),
    )
    if args.start_from:
        positions = human_positions(load_all(args.recordings), role, args.start_min_progress, args.start_holdout)
        if args.start_from == "agent":
            own = args.init.with_suffix(".npz") if args.init else None
            if own is None or not args.partners:
                raise SystemExit("--start-from agent needs --init and --partners to play the placebo games")
            positions = agent_positions(positions, role, [f"policy:{p}" for p in args.partners], f"policy:{own}", args.seed)
        print(f"{len(positions)} {args.start_from} start positions, used for {args.start_share:.0%} of episodes", flush=True)
        env.env_method("set_start_positions", positions, args.start_share)
    model: MaskablePPO
    if args.init:
        model = load_ppo(args.init, env, args.seed, args.learning_rate, args.ent_coef)
    else:
        model = make_ppo(env, seed=args.seed, learning_rate=args.learning_rate, ent_coef=args.ent_coef)

    metrics_path = args.out / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        csv.writer(f).writerow(["generation", "opponent", "score", "total_steps", "seconds"])
    start = time.time()

    def evaluate(label: str, seed: int) -> float:
        policy = NumpyPolicy.from_sb3(model)
        scores = _benchmark(role, policy, args.benchmarks, args.eval_games, seed)
        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            for spec, score in scores.items():
                writer.writerow([label, spec, f"{score:.3f}", model.num_timesteps, f"{time.time() - start:.0f}"])
        summary = "  ".join(f"{spec.split('/')[-1]}={score:.2f}" for spec, score in scores.items())
        print(f"{label:>8} {summary}  ({time.time() - start:.0f}s)", flush=True)
        return sum(scores.values()) / len(scores)

    try:
        if args.no_clone:
            model.learn(args.warmup_steps)
        else:
            examples = build_examples(load_all(args.recordings), role)
            if args.placebo:
                examples = randomize_actions(examples, args.seed)
            games = len(np.unique(examples.games))
            print(f"{len(examples)} examples from {games} games", flush=True)
            if games > 1:
                folds = cross_validate(examples, args.role, args.clone_epochs, args.seed)
                folds_text = " ".join(f"{a:.2f}" for a in folds)
                print(
                    f"unseen-game accuracy {np.mean(folds):.2f} ({folds_text}), "
                    f"random guess {random_guess_accuracy(examples):.2f}",
                    flush=True,
                )
            loss = behaviour_clone(
                model.policy, examples, epochs=args.clone_epochs, learning_rate=args.clone_learning_rate, seed=args.seed
            )
            print(f"cloned: final loss {loss:.3f}, training accuracy {accuracy(model.policy, examples):.2f}", flush=True)
            NumpyPolicy.from_sb3(model).save(args.out / f"{args.role}_cloned.npz")
            evaluate("cloned", args.seed)

            set_actor_frozen(model.policy, True)
            model.learn(args.warmup_steps)
            set_actor_frozen(model.policy, False)

        best = float("-inf")
        for generation in range(args.generations):
            model.learn(args.steps_per_generation, reset_num_timesteps=False)
            snapshot = snapshot_dir / f"{args.role}_{generation:03d}.npz"
            NumpyPolicy.from_sb3(model).save(snapshot)
            model.save(args.out / f"{args.role}_last.zip")
            score = evaluate(f"gen {generation}", args.seed + generation + 1)
            if score > best:
                best = score
                shutil.copy(snapshot, args.out / f"{args.role}.npz")
    finally:
        env.close()


if __name__ == "__main__":
    main()
