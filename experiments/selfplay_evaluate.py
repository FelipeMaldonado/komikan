"""Score stage 3: the last generation of every seed, at 400 simulations. Run from the project folder.

None of the opponents was used in training. Fixed before seeing any result, along with the check: to go on, it must clearly
beat plain search at 1600 simulations as trewas (23 wins and 2 losses in 120 games against K2 gen 14, K1 best and K1 gen 11), and win
against minimax 2.
"""

from glob import glob
from multiprocessing import Pool

import numpy as np

from komikan.bots import make_bot
from komikan.evaluate import play_game, play_match
from komikan.game import TREWAS
from komikan.midgame import game_positions, human_wins
from komikan.recordings import load_all

S, F = "runs/second/snapshots", "runs/first"
GENERATION = 30
SIMULATIONS = 400
ARMS = ("human", "control")
SEEDS = (40, 41, 42)
KOMIKELU_OPPONENTS = {
    "random": ("random", 40),
    "minimax 1": ("minimax:1", 40),
    "minimax 2": ("minimax:2", 40),
    "K2 gen 11": ("policy:models/komikelu.npz", 40),
    "K2 gen 14": (f"policy:{S}/komikelu_014.npz", 40),
    "K1 best": (f"policy:{F}/komikelu.npz", 40),
    "K1 gen 11": (f"policy:{F}/snapshots/komikelu_011.npz", 40),
    "search 1600": ("mcts:1600:0:2", 20),
}
TREWAS_OPPONENTS = {
    "random": ("random", 40),
    "minimax 1": ("minimax:1", 40),
    "minimax 2": ("minimax:2", 40),
    "minimax 3": ("minimax:3", 40),
    "T2 gen 13": ("policy:models/trewas.npz", 40),
    "T1 best": (f"policy:{F}/trewas.npz", 40),
    "search 1600": ("mcts:1600:0.5:2", 20),
}
PHASES = {"25-50%": (0.25, 0.5), "50-75%": (0.5, 0.75), "75-100%": (0.75, 1.0)}
FINISH_OPPONENTS = ("K2 gen 11", "K1 best")
PLAYS = 2
WINS = human_wins(load_all(glob("recordings/*.json")), TREWAS)


def spec(arm, seed, root="runs/selfplay", simulations=SIMULATIONS):
    return f"az:{simulations}:{root}/{arm}_{seed}/gen_{GENERATION:03d}"


def full_games(job):
    arm, seed, role, label, root = job
    opponent, games = (KOMIKELU_OPPONENTS if role == "trewas" else TREWAS_OPPONENTS)[label]
    agent, other = make_bot(spec(arm, seed, root), 500 + seed), make_bot(opponent, 600 + seed)
    r = play_match(other, agent, games) if role == "trewas" else play_match(agent, other, games)
    won, lost = (r.trewas_wins, r.komikelu_wins) if role == "trewas" else (r.komikelu_wins, r.trewas_wins)
    return arm, seed, role, label, won, lost, games


def finishing(job):
    """From the win this seed held out, whichever arm: the control never trained on any human position."""
    arm, seed, label, phase = job
    agent, other = make_bot(spec(arm, seed), 700 + seed), make_bot(KOMIKELU_OPPONENTS[label][0], 800 + seed)
    states = game_positions(WINS[seed - SEEDS[0]], TREWAS, *PHASES[phase])
    results = [play_game(other, agent, start=state)[0] for state in states for _ in range(PLAYS)]
    return arm, seed, label, phase, float(np.mean([r == TREWAS for r in results]))


def table(rows, role, labels, arms, seeds):
    print(f"\nAs {role}: games won / lost per seed, of 40 (20 against search 1600)")
    for arm in arms:
        for seed in seeds:
            cells = []
            for label in labels:
                row = next(r for r in rows if r[:4] == (arm, seed, role, label))
                cells.append(f"{label} {row[4]}/{row[5]}")
            print(f"{arm}_{seed}: " + "  ".join(cells), flush=True)
        totals = [f"{label} {sum(r[4] for r in rows if r[0] == arm and r[2] == role and r[3] == label)}/{sum(r[5] for r in rows if r[0] == arm and r[2] == role and r[3] == label)}" for label in labels]
        print(f"{arm} total: " + "  ".join(totals), flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/selfplay")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--no-finish", action="store_true", help="skip finishing, for runs that trained on all the human's wins")
    args = parser.parse_args()
    with Pool(8) as pool:
        jobs = [(a, s, "trewas", l, args.root) for a in args.arms for s in args.seeds for l in KOMIKELU_OPPONENTS]
        jobs += [(a, s, "komikelu", l, args.root) for a in args.arms for s in args.seeds for l in TREWAS_OPPONENTS]
        rows = pool.map(full_games, jobs, chunksize=1)
        table(rows, "trewas", KOMIKELU_OPPONENTS, args.arms, args.seeds)
        table(rows, "komikelu", TREWAS_OPPONENTS, args.arms, args.seeds)
        if args.no_finish:
            return
        finish = pool.map(finishing, [(a, s, l, p) for a in ARMS for s in SEEDS for l in FINISH_OPPONENTS for p in PHASES], chunksize=1)
    print("\nFinishing from the held-out human win: trewas win rate, mean over seeds [range]")
    for label in FINISH_OPPONENTS:
        for arm in ARMS:
            cells = []
            for phase in PHASES:
                values = [f[4] for f in finish if f[0] == arm and f[2] == label and f[3] == phase]
                cells.append(f"{phase} {np.mean(values):.0%} [{min(values):.0%}-{max(values):.0%}]")
            print(f"vs {label:10} {arm:8} " + "  ".join(cells))


if __name__ == "__main__":
    main()
