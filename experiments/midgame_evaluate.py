"""Score the mid-game experiment. Run from the project folder. `--baseline-only` skips the trained groups.

Two questions:
1. Full games from the start, against training partners and against agents never used in training.
2. Finishing: from positions in the recorded game each seed held out, how often do the trewas go on to win?
"""

import sys
from glob import glob
from multiprocessing import Pool

import numpy as np

from komikan.bots import make_bot
from komikan.evaluate import play_game, play_match
from komikan.game import TREWAS
from komikan.midgame import game_positions
from komikan.recordings import load_all

S, F = "runs/second/snapshots", "runs/first"
OPPONENTS = {
    "random": ("random", 300),
    "mm1": ("minimax:1", 300),
    "mm2": ("minimax:2", 200),
    "K2_005": (f"policy:{S}/komikelu_005.npz", 200),
    "K2_011*": (f"policy:{S}/komikelu_011.npz", 200),
    "K2_014": (f"policy:{S}/komikelu_014.npz", 200),
    "K1_best": (f"policy:{F}/komikelu.npz", 200),
    "K1_011": (f"policy:{F}/snapshots/komikelu_011.npz", 200),
}
FINISH_OPPONENTS = {"K2_011*": OPPONENTS["K2_011*"][0], "K1_best": OPPONENTS["K1_best"][0]}
PHASES = {"25-50%": (0.25, 0.5), "50-75%": (0.5, 0.75), "75-100%": (0.75, 1.0)}
PLAYS_PER_POSITION = 8
ARMS = ["baseline", "control", "agent", "human"]
SEEDS = range(20, 25)
BASELINE = "models/trewas.npz"
GAMES = load_all(glob("recordings/*.json"))  # each seed held out one of the first 5 games, all wins as trewas


def agent_path(arm: str, seed: int) -> str:
    return BASELINE if arm == "baseline" else f"runs/midgame/{arm}_{seed}/snapshots/trewas_005.npz"


def full_games(job):
    arm, seed, label = job
    spec, games = OPPONENTS[label]
    result = play_match(make_bot(spec, 4321 + seed), make_bot(f"policy:{agent_path(arm, seed)}", 8765 + seed), games)
    return arm, seed, label, result.trewas_wins / games, result.komikelu_wins / games


def finishing(job):
    arm, seed, label, phase = job
    states = game_positions(GAMES[seed - SEEDS[0]], TREWAS, *PHASES[phase])
    komikelu = make_bot(FINISH_OPPONENTS[label], 111 + seed)
    trewas = make_bot(f"policy:{agent_path(arm, seed)}", 222 + seed)
    results = [play_game(komikelu, trewas, start=state)[0] for state in states for _ in range(PLAYS_PER_POSITION)]
    return arm, seed, label, phase, float(np.mean([r == TREWAS for r in results]))


def summary(values: list[float]) -> str:
    return f"{np.mean(values):5.0%} [{min(values):3.0%}-{max(values):3.0%}]"


def main():
    arms = ARMS[:1] if "--baseline-only" in sys.argv else ARMS
    with Pool(8) as pool:
        finish = pool.map(finishing, [(a, s, l, p) for a in arms for s in SEEDS for l in FINISH_OPPONENTS for p in PHASES], chunksize=1)
        games = [] if "--baseline-only" in sys.argv else pool.map(full_games, [(a, s, l) for a in arms for s in SEEDS for l in OPPONENTS], chunksize=1)

    if games:
        print("Full games: trewas win rate, mean over 5 seeds [range]. * = training partner")
        print(f"{'':10}" + "".join(f"{label:>17}" for label in OPPONENTS))
        for arm in arms:
            cells = [summary([g[3] for g in games if g[0] == arm and g[2] == label]) for label in OPPONENTS]
            print(f"{arm:10}" + "".join(f"{c:>17}" for c in cells))
        print("\nFull games: trewas loss rate")
        for arm in arms:
            cells = [summary([g[4] for g in games if g[0] == arm and g[2] == label]) for label in OPPONENTS]
            print(f"{arm:10}" + "".join(f"{c:>17}" for c in cells))

    print("\nFinishing from held-out human positions: trewas win rate, by how far through the game")
    for label in FINISH_OPPONENTS:
        print(f"against {label}")
        for arm in arms:
            cells = [summary([f[4] for f in finish if f[0] == arm and f[2] == label and f[3] == phase]) for phase in PHASES]
            print(f"  {arm:10}" + "".join(f"{phase} {c:>17}   " for phase, c in zip(PHASES, cells)))


if __name__ == "__main__":
    main()
