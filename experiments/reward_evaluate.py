"""Score the reward experiment: results against opponents, and whether the hiding habit has gone. Run from the project folder."""

from multiprocessing import Pool

import numpy as np

from komikan.board import COORDS
from komikan.bots import make_bot
from komikan.evaluate import play_match
from komikan.game import KOMIKELU, TREWAS, apply_action, initial_state, komikelu_reach, outcome, points_of

S, F = "runs/second/snapshots", "runs/first"
OPPONENTS = {
    "random": ("random", 200),
    "mm1": ("minimax:1", 200),
    "mm2": ("minimax:2", 100),
    "K2_011*": ("policy:models/komikelu.npz", 200),
    "K2_014": (f"policy:{S}/komikelu_014.npz", 200),
    "K1_best": (f"policy:{F}/komikelu.npz", 200),
    "K1_011": (f"policy:{F}/snapshots/komikelu_011.npz", 200),
}
HABIT_OPPONENTS = ("K1_best", "K2_011*")
GROUPS = ["models", "current", "harsh_draw", "confine", "relative"]
SEEDS = (30, 31, 32)
TRIANGLE = {p for p, (_, y) in enumerate(COORDS) if y < 0}


def path(group, seed):
    return "models/trewas.npz" if group == "models" else f"runs/reward/{group}_{seed}/snapshots/trewas_005.npz"


def games(job):
    group, seed, label = job
    spec, n = OPPONENTS[label]
    r = play_match(make_bot(spec, 40 + seed), make_bot(f"policy:{path(group, seed)}", 50 + seed), n)
    return group, seed, label, r.trewas_wins / n, r.komikelu_wins / n


def habit(job):
    """Turns 100-150: share of trewas in the triangle, and the points the komikelu can reach."""
    group, seed, label = job
    komikelu = make_bot(OPPONENTS[label][0], 60 + seed)
    trewas = make_bot(f"policy:{path(group, seed)}", 70 + seed)
    in_triangle, reach = [], []
    for _ in range(40):
        state = initial_state()
        while outcome(state) is None:
            if state.to_move == TREWAS and 100 <= state.plies < 150:
                pieces = points_of(state.trewas)
                in_triangle.append(np.mean([p in TRIANGLE for p in pieces]))
                reach.append(komikelu_reach(state))
            state = apply_action(state, (komikelu if state.to_move == KOMIKELU else trewas)(state))
    return group, seed, label, float(np.mean(in_triangle)) if in_triangle else None, float(np.mean(reach)) if reach else None


def cell(values, percent=True):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return f"{np.mean(values):.0%} [{min(values):.0%}-{max(values):.0%}]" if percent else f"{np.mean(values):.1f} [{min(values):.1f}-{max(values):.1f}]"


def main():
    with Pool(8) as pool:
        played = pool.map(games, [(g, s, l) for g in GROUPS for s in SEEDS for l in OPPONENTS], chunksize=1)
        habits = pool.map(habit, [(g, s, l) for g in GROUPS for s in SEEDS for l in HABIT_OPPONENTS], chunksize=1)
    for title, index in (("Trewas win rate, mean over seeds [range]. * = training partner", 3), ("Trewas loss rate", 4)):
        print(title)
        print(f"{'':11}" + "".join(f"{l:>17}" for l in OPPONENTS))
        for g in GROUPS:
            print(f"{g:11}" + "".join(f"{cell([p[index] for p in played if p[0] == g and p[2] == l]):>17}" for l in OPPONENTS))
        print()
    print("Turns 100-150: trewas in the triangle / points the komikelu can reach (your wins: 51% / 12.3)")
    for l in HABIT_OPPONENTS:
        print(f"against {l}")
        for g in GROUPS:
            rows = [h for h in habits if h[0] == g and h[2] == l]
            print(f"  {g:11}{cell([h[3] for h in rows]):>17}  {cell([h[4] for h in rows], False):>20}")


if __name__ == "__main__":
    main()
