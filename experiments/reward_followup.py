"""Are the reward groups' standout seeds strong in general, or exploiting single opponents? Run from the project folder.

Plays every seed against all saved komikelu generations of runs 1 and 2, with new game seeds, and describes how the wins end.
"""

from collections import Counter
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from komikan.board import COORDS
from komikan.bots import make_bot
from komikan.game import KOMIKELU, TREWAS, apply_action, initial_state, outcome

GROUPS = ["current", "harsh_draw", "confine", "relative"]
SEEDS = (30, 31, 32)
GAMES = 40
OPPONENTS = {
    **{"K1 " + Path(p).stem[-3:]: p for p in sorted(glob("runs/first/snapshots/komikelu_*.npz"))},
    "K1 best": "runs/first/komikelu.npz",
    **{"K2 " + Path(p).stem[-3:]: p for p in sorted(glob("runs/second/snapshots/komikelu_*.npz"))},
}


def play(job):
    group, seed, label = job
    komikelu = make_bot(f"policy:{OPPONENTS[label]}", 300 + seed)
    trewas = make_bot(f"policy:runs/reward/{group}_{seed}/snapshots/trewas_005.npz", 400 + seed)
    won, turns, rows = 0, [], []
    for _ in range(GAMES):
        state = initial_state()
        while outcome(state) is None:
            state = apply_action(state, (komikelu if state.to_move == KOMIKELU else trewas)(state))
        if outcome(state) == TREWAS:
            won += 1
            turns.append(state.plies)
            rows.append(COORDS[state.komikelu][1])
    return group, seed, label, won / GAMES, turns, rows


def main():
    with Pool(8) as pool:
        rows = pool.map(play, [(g, s, l) for g in GROUPS for s in SEEDS for l in OPPONENTS], chunksize=1)
    print(f"Trewas win rate per seed, {GAMES} games per cell")
    print(f"{'':16}" + "".join(f"{l:>8}" for l in OPPONENTS))
    for g in GROUPS:
        for s in SEEDS:
            print(f"{f'{g}_{s}':16}" + "".join(f"{r[3]:>8.0%}" for r in rows if r[:2] == (g, s) ))
    print("\nOpponents beaten at least half the time, how long the wins took, and the komikelu's row when trapped (12 bottom, -6 tip)")
    for g in GROUPS:
        for s in SEEDS:
            mine = [r for r in rows if r[:2] == (g, s)]
            beaten = [r[2] for r in mine if r[3] >= 0.5]
            turns = [t for r in mine for t in r[4]]
            where = Counter(y for r in mine for y in r[5])
            length = f"turns {np.mean(turns):.0f} [{min(turns)}-{max(turns)}]" if turns else "no wins"
            print(f"{g}_{s}: {len(beaten)} of {len(mine)} ({', '.join(beaten)}), {length}, rows {sorted(where.items())}")


if __name__ == "__main__":
    main()
