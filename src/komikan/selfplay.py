"""AlphaZero-style self-play: stage 3 of the plan.

Each role has a policy network and a value network, both trained from scratch on search visit counts and game results, with no shaped
rewards. A network only sees positions with its own role to move, so a leaf is judged by the network of the side to move there.

Values are always from the komikelu's point of view: +1 if it wins, -1 if the trewas win, and `draw` for a draw. A positive `draw`
makes a draw bad for the trewas, because otherwise they settle for safe shuffling, as in run 1.

python -m komikan.selfplay --out runs/selfplay/test --generations 2 --games 8 --simulations 25
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .env import OBS_SIZE, action_mask, encode_observation
from .game import (
    DEFAULT_RULES,
    DRAW,
    KOMIKELU,
    NUM_ACTIONS,
    RULE_VARIANTS,
    TREWAS,
    GameState,
    Rules,
    apply_action,
    initial_state,
    legal_actions,
    mirror_action,
    mirror_state,
    outcome,
)
from .policy import NumpyPolicy
from .puct import Networks, PUCTBot, _Node

ROLE_NAMES = {KOMIKELU: "komikelu", TREWAS: "trewas"}
DIRICHLET_ALPHA = 0.3
DIRICHLET_SHARE = 0.25


def result_value(result: int, draw: float) -> float:
    """A finished game's value, from the komikelu's point of view."""
    if result == DRAW:
        return draw
    return 1.0 if result == KOMIKELU else -1.0


class AlphaZeroBot(PUCTBot):
    """Search with self-play networks. With `root_noise`, it explores as in training."""

    def __init__(
        self,
        networks: Networks,
        simulations: int = 400,
        exploration: float = 1.5,
        draw: float = 0.5,
        root_noise: bool = False,
        seed: int | None = None,
        rules: Rules = DEFAULT_RULES,
    ):
        super().__init__(networks, simulations, "both", exploration, seed=seed, rules=rules)
        self.draw = draw
        self.root_noise = root_noise
        self.noise_rng = np.random.default_rng(seed)

    def _prepare_root(self, root: _Node) -> None:
        if self.root_noise and root.actions:
            noise = self.noise_rng.dirichlet([DIRICHLET_ALPHA] * len(root.actions))
            root.priors = list((1.0 - DIRICHLET_SHARE) * np.array(root.priors) + DIRICHLET_SHARE * noise)

    def _network_value(self, state: GameState, terminal: int | None) -> float:
        if terminal is not None:
            return result_value(terminal, self.draw)
        value = float(self.networks.values[state.to_move].logits(encode_observation(state, self.rules))[0])
        return max(-1.0, min(1.0, value))


@dataclass
class Sample:
    role: int
    obs: np.ndarray
    mask: np.ndarray
    target: np.ndarray  # visit shares over all actions
    value: float


def _visit_target(root: _Node) -> np.ndarray:
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target[root.actions] = root.visits
    return target / target.sum()


def _mirror_target(target: np.ndarray) -> np.ndarray:
    mirrored = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for action in np.flatnonzero(target):
        mirrored[mirror_action(int(action))] = target[action]
    return mirrored


def play_game(
    bot: AlphaZeroBot,
    start: GameState,
    seed: int,
    temperature_turns: int = 30,
    rules: Rules = DEFAULT_RULES,
) -> tuple[list[Sample], int, GameState]:
    """One self-play game. Moves are sampled from the visits for the first turns, then the most visited is played.

    Forced moves teach nothing and aren't recorded. Every position is recorded mirrored as well.
    """
    rng = random.Random(seed)
    state, seen = start, []
    while (result := outcome(state, rules)) is None:
        actions = legal_actions(state, rules)
        if len(actions) == 1:
            state = apply_action(state, actions[0], rules)
            continue
        root = bot.search(state)
        seen.append((state, _visit_target(root)))
        if state.plies < temperature_turns:
            action = rng.choices(root.actions, weights=root.visits)[0]
        else:
            action = root.actions[max(range(len(root.actions)), key=lambda i: root.visits[i])]
        state = apply_action(state, action, rules)
    value = result_value(result, bot.draw)
    samples = []
    for position, target in seen:
        for s, t in ((position, target), (mirror_state(position), _mirror_target(target))):
            samples.append(Sample(s.to_move, encode_observation(s, rules), action_mask(s, rules), t, value))
    return samples, result, state


class ReplayBuffer:
    """The most recent samples of one role."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.masks = np.zeros((capacity, NUM_ACTIONS), dtype=bool)
        self.targets = np.zeros((capacity, NUM_ACTIONS), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.next = 0

    def add(self, sample: Sample) -> None:
        i = self.next
        self.obs[i], self.masks[i], self.targets[i], self.values[i] = sample.obs, sample.mask, sample.target, sample.value
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class Learner:
    """The trainable networks of one role."""

    def __init__(self, hidden: tuple[int, ...], lr: float, seed: int):
        import torch

        torch.manual_seed(seed)

        def mlp(outputs: int):
            sizes = (OBS_SIZE, *hidden)
            layers: list[torch.nn.Module] = []
            for a, b in zip(sizes[:-1], sizes[1:]):
                layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
            layers.append(torch.nn.Linear(sizes[-1], outputs))
            return torch.nn.Sequential(*layers)

        self.policy, self.value = mlp(NUM_ACTIONS), mlp(1)
        params = [*self.policy.parameters(), *self.value.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        self.rng = np.random.default_rng(seed)

    def train(self, buffer: ReplayBuffer, steps: int, batch: int) -> tuple[float, float]:
        """Returns the mean policy and value losses."""
        import torch

        policy_losses, value_losses = [], []
        for _ in range(steps):
            picks = self.rng.integers(0, buffer.size, batch)
            obs = torch.from_numpy(buffer.obs[picks])
            logits = self.policy(obs).masked_fill(~torch.from_numpy(buffer.masks[picks]), -1e9)
            policy_loss = -(torch.from_numpy(buffer.targets[picks]) * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(self.value(obs).squeeze(1), torch.from_numpy(buffer.values[picks]))
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        return float(np.mean(policy_losses)), float(np.mean(value_losses))

    def save(self, directory: Path, name: str) -> None:
        import torch

        for net, path in ((self.policy, directory / f"{name}.npz"), (self.value, directory / f"{name}_value.npz")):
            layers = [m for m in net if isinstance(m, torch.nn.Linear)]
            NumpyPolicy([m.weight.detach().numpy() for m in layers], [m.bias.detach().numpy() for m in layers]).save(path)


_worker_nets: tuple[str, Networks] | None = None


def _selfplay_job(job: tuple) -> tuple:
    """Plays one game in a worker. Each worker loads a generation's networks once."""
    global _worker_nets
    directory, simulations, exploration, draw, seed, start, source, variant = job
    if _worker_nets is None or _worker_nets[0] != directory:
        _worker_nets = (directory, Networks.load(directory, directory))
    rules = RULE_VARIANTS[variant]
    bot = AlphaZeroBot(_worker_nets[1], simulations, exploration, draw, root_noise=True, seed=seed, rules=rules)
    samples, result, final = play_game(bot, start, seed, rules=rules)
    return samples, result, final.plies - start.plies, source


def late_games(path: str | Path, rules: Rules = DEFAULT_RULES) -> list[list[GameState]]:
    """The positions of each trewas win in a JSON list of {"actions": [...]}, without the final position."""
    import json

    games = []
    for game in json.loads(Path(path).read_text()):
        state, states = initial_state(), []
        for action in game["actions"]:
            states.append(state)
            state = apply_action(state, action, rules)
        if outcome(state, rules) != TREWAS:
            raise ValueError(f"{path}: a game doesn't end in a trewas win")
        games.append(states)
    return games


def late_reach(generation: int, generations: int, first: int, last: int) -> int:
    """How far back from the end late starts may go, growing evenly from `first` to `last` over training."""
    if generations <= 1:
        return last
    return round(first + (last - first) * generation / (generations - 1))


def late_start(games: list[list[GameState]], reach: int, rng: random.Random) -> GameState:
    """A position up to `reach` moves before the end of a random win."""
    states = rng.choice(games)
    return states[max(0, len(states) - rng.randint(1, reach))]


def starting_positions(hold_out: int | None) -> list[GameState]:
    """Positions from the human's wins as trewas, without the held-out game."""
    from .midgame import human_positions, human_wins
    from .recordings import load_all

    wins = human_wins(load_all(glob("recordings/*.json")), TREWAS)
    return human_positions(wins, TREWAS, exclude=() if hold_out is None else (hold_out,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--games", type=int, default=100, help="self-play games per generation")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--exploration", type=float, default=1.5)
    parser.add_argument("--draw", type=float, default=0.5, help="value of a draw to the komikelu")
    parser.add_argument("--start-share", type=float, default=0.25, help="share of games starting from positions in the human's wins")
    parser.add_argument("--hold-out", type=int, default=None, help="index of a human win to keep out of the starting positions")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--reuse", type=float, default=2.0, help="how many times each new sample is trained on, on average")
    parser.add_argument("--buffer", type=int, default=150_000, help="samples kept per role")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rules", choices=RULE_VARIANTS, default="standard")
    parser.add_argument("--late", help="JSON file of trewas wins: some games start late in one of them")
    parser.add_argument("--late-share", type=float, default=0.5, help="share of games with a late start")
    parser.add_argument("--late-reach", type=int, nargs=2, default=[10, 200], metavar=("FIRST", "LAST"),
                        help="how many moves back from the end a start may be, in the first and the last generation")
    args = parser.parse_args()
    if (args.start_share > 0 or args.late) and args.rules != "standard":
        parser.error("the human's positions and late starts come from standard games, so use neither with a variant")
    if args.start_share > 0 and args.late:
        parser.error("use either --start-share or --late, so each run tests one kind of start")

    import torch

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    learners = {role: Learner(tuple(args.hidden), args.lr, args.seed + role) for role in ROLE_NAMES}
    buffers = {role: ReplayBuffer(args.buffer) for role in ROLE_NAMES}
    starts = starting_positions(args.hold_out) if args.start_share > 0 else []
    wins = late_games(args.late) if args.late else []
    began = time.time()

    with Pool(args.workers) as pool:
        for generation in range(args.generations + 1):  # generation 0 is the untrained networks, and the last isn't played
            directory = out / f"gen_{generation:03d}"
            directory.mkdir(exist_ok=True)
            for role, name in ROLE_NAMES.items():
                learners[role].save(directory, name)
            if generation == args.generations:
                break
            jobs = []
            reach = late_reach(generation, args.generations, *args.late_reach)
            for i in range(args.games):
                if starts and rng.random() < args.start_share:
                    source, start = "human", rng.choice(starts)
                elif wins and rng.random() < args.late_share:
                    source, start = "late", late_start(wins, reach, rng)
                else:
                    source, start = "start", initial_state()
                game_seed = args.seed * 1_000_003 + generation * 10_000 + i
                jobs.append((str(directory), args.simulations, args.exploration, args.draw, game_seed, start, source, args.rules))
            results = pool.map(_selfplay_job, jobs, chunksize=1)

            new = {role: 0 for role in ROLE_NAMES}
            for samples, *_ in results:
                for sample in samples:
                    buffers[sample.role].add(sample)
                    new[sample.role] += 1
            losses = {}
            for role, name in ROLE_NAMES.items():
                steps = math.ceil(new[role] * args.reuse / args.batch)
                if buffers[role].size:
                    losses[name] = learners[role].train(buffers[role], steps, args.batch)

            def summary(source: str) -> str:
                rows = [r for r in results if r[3] == source]
                if not rows:
                    return "-"
                count = lambda who: sum(r[1] == who for r in rows)
                return f"K{count(KOMIKELU)} T{count(TREWAS)} D{count(DRAW)} turns {np.mean([r[2] for r in rows]):.0f}"

            other = "late" if wins else "human"
            extra = f" reach {reach}" if wins else ""
            print(
                f"gen {generation} start [{summary('start')}] {other} [{summary(other)}]{extra} samples K{new[KOMIKELU]} T{new[TREWAS]} "
                + " ".join(f"{name} p={p:.2f} v={v:.3f}" for name, (p, v) in losses.items())
                + f" ({time.time() - began:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
