"""Compare search bots by spec. Run from the project folder.

Settings are chosen with --set pilot, whose opponents and seeds are kept apart from --set test, the ones that get reported.

python experiments/guided_evaluate.py --set pilot --games 20 --role trewas --specs puct:400:both puct:400:none mcts:400:0.5:2
python experiments/guided_evaluate.py --set test --games 40 --role komikelu --specs puct:400:both mcts:400:0:2
python experiments/guided_evaluate.py --finish --specs puct:400:both mcts:400:0.5:2
"""

import argparse
import time
from glob import glob
from multiprocessing import Pool

import numpy as np

from komikan.bots import make_bot
from komikan.evaluate import play_game, play_match
from komikan.game import TREWAS
from komikan.midgame import game_positions, human_wins
from komikan.recordings import load_all

S, F = "runs/second/snapshots", "runs/first"
OPPONENTS = {
    ("pilot", "trewas"): {
        "random": "random",
        "minimax 1": "minimax:1",
        "K1 gen 5": f"policy:{F}/snapshots/komikelu_005.npz",
        "K1 gen 9": f"policy:{F}/snapshots/komikelu_009.npz",
    },
    ("test", "trewas"): {
        "random": "random",
        "minimax 1": "minimax:1",
        "minimax 2": "minimax:2",
        "K2 gen 11 (partner)": "policy:models/komikelu.npz",
        "K2 gen 14": f"policy:{S}/komikelu_014.npz",
        "K1 best": f"policy:{F}/komikelu.npz",
        "K1 gen 11": f"policy:{F}/snapshots/komikelu_011.npz",
    },
    ("pilot", "komikelu"): {
        "random": "random",
        "minimax 1": "minimax:1",
        "T1 gen 3": f"policy:{F}/snapshots/trewas_003.npz",
        "T1 gen 6": f"policy:{F}/snapshots/trewas_006.npz",
    },
    ("test", "komikelu"): {
        "random": "random",
        "minimax 1": "minimax:1",
        "minimax 2": "minimax:2",
        "minimax 3": "minimax:3",
        "T2 gen 13 (partner)": "policy:models/trewas.npz",
        "T1 best": f"policy:{F}/trewas.npz",
    },
}
SEED_BASE = {"pilot": 1000, "test": 5000}
PHASES = {"25-50%": (0.25, 0.5), "50-75%": (0.5, 0.75), "75-100%": (0.75, 1.0)}
FINISH_OPPONENTS = {"K2 gen 11": "policy:models/komikelu.npz", "K1 best": f"policy:{F}/komikelu.npz"}
GAMES = human_wins(load_all(glob("recordings/*.json")), TREWAS)


def full_games(job):
    spec, role, label, opponent, games, seed = job
    start = time.time()
    search, other = make_bot(spec, seed=seed), make_bot(opponent, seed=seed + 1)
    r = play_match(other, search, games) if role == "trewas" else play_match(search, other, games)
    return spec, label, r, time.time() - start


def finishing(job):
    spec, label, opponent, game, phase, plays = job
    search, other = make_bot(spec, seed=game + 1), make_bot(opponent, seed=game + 2)
    states = game_positions(GAMES[game], TREWAS, *PHASES[phase])
    results = [play_game(other, search, start=state)[0] for state in states for _ in range(plays)]
    return spec, label, phase, float(np.mean([r == TREWAS for r in results]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--set", dest="games_set", choices=["pilot", "test"], default="pilot")
    parser.add_argument("--role", choices=["trewas", "komikelu"], default="trewas")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--finish", action="store_true", help="play on from positions in the human's wins, as trewas")
    parser.add_argument("--plays", type=int, default=2)
    args = parser.parse_args()

    with Pool(args.workers) as pool:
        if args.finish:
            jobs = [
                (spec, label, opponent, game, phase, args.plays)
                for spec in args.specs for label, opponent in FINISH_OPPONENTS.items()
                for game in range(len(GAMES)) for phase in PHASES
            ]
            rows = pool.map(finishing, jobs, chunksize=1)
            for spec in args.specs:
                for label in FINISH_OPPONENTS:
                    cells = [f"{p} {np.mean([r[3] for r in rows if r[:3] == (spec, label, p)]):4.0%}" for p in PHASES]
                    print(f"finish {spec:20} vs {label:10} " + "  ".join(cells), flush=True)
            return

        opponents = OPPONENTS[args.games_set, args.role]
        base = SEED_BASE[args.games_set]
        jobs = [(spec, args.role, label, opp, args.games, base + 10 * i) for spec in args.specs for i, (label, opp) in enumerate(opponents.items())]
        for spec, label, r, seconds in pool.imap(full_games, jobs, chunksize=1):
            g = r.games
            won, lost = (r.trewas_wins, r.komikelu_wins) if args.role == "trewas" else (r.komikelu_wins, r.trewas_wins)
            print(
                f"{args.role:8} {spec:20} vs {label:20} won {won / g:4.0%} lost {lost / g:4.0%} drawn {r.draws / g:4.0%}"
                f" | captures {r.captures / g:4.1f} turns {r.plies / g:5.1f} ({seconds:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
