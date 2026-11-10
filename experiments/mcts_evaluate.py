"""How strong is search with no network? Run from the project folder.

Full games:  python experiments/mcts_evaluate.py --games 20 --simulations 100 400 --draws 0 0.5 --playouts 0 8
Finishing:   python experiments/mcts_evaluate.py --finish --simulations 400 --draws 0.5 --playouts 0
Agents marked K1 and T1 were never training partners of the current models.
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
KOMIKELU_OPPONENTS = {
    "random": "random",
    "minimax 1": "minimax:1",
    "minimax 2": "minimax:2",
    "minimax 3": "minimax:3",
    "K2 models": "policy:models/komikelu.npz",
    "K1 best": f"policy:{F}/komikelu.npz",
    "K1 gen 11": f"policy:{F}/snapshots/komikelu_011.npz",
}
TREWAS_OPPONENTS = {
    "random": "random",
    "minimax 1": "minimax:1",
    "minimax 2": "minimax:2",
    "minimax 3": "minimax:3",
    "T2 models": "policy:models/trewas.npz",
    "T1 best": f"policy:{F}/trewas.npz",
}
PHASES = {"25-50%": (0.25, 0.5), "50-75%": (0.5, 0.75), "75-100%": (0.75, 1.0)}
FINISH_OPPONENTS = {"K2 models": KOMIKELU_OPPONENTS["K2 models"], "K1 best": KOMIKELU_OPPONENTS["K1 best"]}


def run(job):
    search_role, simulations, draw, playout, label, spec, games = job
    start = time.time()
    search = make_bot(f"mcts:{simulations}:{draw}:{playout}", seed=simulations + 1)
    other = make_bot(spec, seed=simulations + 2)
    result = play_match(other, search, games) if search_role == TREWAS else play_match(search, other, games)
    return search_role, simulations, draw, playout, label, result, time.time() - start


def finish(job):
    simulations, draw, playout, label, spec, game, phase, plays = job
    search = make_bot(f"mcts:{simulations}:{draw}:{playout}", seed=game + 1)
    other = make_bot(spec, seed=game + 2)
    results = [play_game(other, search, start=state)[0] for state in game_positions(GAMES[game], TREWAS, *PHASES[phase]) for _ in range(plays)]
    return simulations, draw, playout, label, phase, float(np.mean([r == TREWAS for r in results]))


GAMES = human_wins(load_all(glob("recordings/*.json")), TREWAS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, nargs="+", default=[100, 400])
    parser.add_argument("--draws", type=float, nargs="+", default=[0.0, 0.5], help="draw values for the trewas search")
    parser.add_argument("--playouts", type=int, nargs="+", default=[8])
    parser.add_argument("--roles", nargs="+", default=["trewas", "komikelu"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="*", help="only opponents with these labels")
    parser.add_argument("--finish", action="store_true", help="play on from positions in the human's wins instead")
    parser.add_argument("--plays", type=int, default=2, help="with --finish: games per position")
    args = parser.parse_args()
    keep = lambda table: {k: v for k, v in table.items() if not args.only or k in args.only}

    with Pool(args.workers) as pool:
        if args.finish:
            jobs = [
                (n, d, p, label, spec, game, phase, args.plays)
                for n in args.simulations for d in args.draws for p in args.playouts
                for label, spec in FINISH_OPPONENTS.items() for game in range(len(GAMES)) for phase in PHASES
            ]
            rows = pool.map(finish, jobs, chunksize=1)
            for n in args.simulations:
                for d in args.draws:
                    for p in args.playouts:
                        for label in FINISH_OPPONENTS:
                            cells = []
                            for phase in PHASES:
                                values = [r[5] for r in rows if r[:5] == (n, d, p, label, phase)]
                                cells.append(f"{phase} {np.mean(values):4.0%} [{min(values):.0%}-{max(values):.0%}]")
                            print(f"finish sims={n:<5} draw={d:<4} playout={p:<2} vs {label:10} " + "  ".join(cells), flush=True)
            return

        jobs = []
        if "trewas" in args.roles:
            jobs += [(TREWAS, n, d, p, l, s, args.games) for n in args.simulations for d in args.draws for p in args.playouts for l, s in keep(KOMIKELU_OPPONENTS).items()]
        if "komikelu" in args.roles:
            jobs += [(0, n, 0.0, p, l, s, args.games) for n in args.simulations for p in args.playouts for l, s in keep(TREWAS_OPPONENTS).items()]
        for role, n, draw, playout, label, r, seconds in pool.imap(run, jobs, chunksize=1):
            name = "trewas  " if role == TREWAS else "komikelu"
            g = r.games
            won, lost = (r.trewas_wins, r.komikelu_wins) if role == TREWAS else (r.komikelu_wins, r.trewas_wins)
            print(
                f"{name} sims={n:<5} draw={draw:<4} playout={playout:<2} vs {label:10} won {won / g:4.0%} lost {lost / g:4.0%}"
                f" drawn {r.draws / g:4.0%} | captures {r.captures / g:4.1f} turns {r.plies / g:5.1f} ({seconds:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
