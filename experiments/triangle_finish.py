"""Can search finish the trap the human's search game left on turn 137? Run from the project folder.

In recordings/komikan-games-search.json, plain search shut the komikelu in the triangle by turn 135 but never entered it, and the
game was drawn. This plays on from turn 137 with several search settings as trewas, against komikelu opponents that look ahead.
"""

import json
import sys
from multiprocessing import Pool

from komikan.bots import make_bot
from komikan.evaluate import play_game
from komikan.game import TREWAS, apply_action, initial_state

TURN = 137
TREWAS_SPECS = sys.argv[1:] or ["mcts:1600:0.5:2", "mcts:1600:0.5:0", "mcts:6400:0.5:2", "mcts:6400:0.5:0"]
KOMIKELU_SPECS = {"search 1600": "mcts:1600:0:2", "K1 best": "policy:runs/first/komikelu.npz", "minimax 2": "minimax:2"}
GAMES = 8


def position():
    raw = json.loads(open("recordings/komikan-games-search.json").read())
    game = (raw["games"] if isinstance(raw, dict) else raw)[-1]
    state = initial_state()
    for action in game["actions"]:
        if state.plies >= TURN:
            break
        state = apply_action(state, action)
    assert state.to_move == TREWAS and state.plies == TURN
    return state


def run(job):
    trewas, label, seed = job
    result, captures, plies = play_game(make_bot(KOMIKELU_SPECS[label], seed), make_bot(trewas, seed + 1), start=position())
    return trewas, label, result, captures, plies


def main():
    jobs = [(t, l, 100 * i + g) for i, t in enumerate(TREWAS_SPECS) for l in KOMIKELU_SPECS for g in range(GAMES)]
    with Pool(8) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    print(f"From turn {TURN} of the human's game against search: trewas wins / komikelu wins / draws, of {GAMES}")
    for t in TREWAS_SPECS:
        cells = []
        for l in KOMIKELU_SPECS:
            mine = [r for r in rows if r[0] == t and r[1] == l]
            won = [r[4] for r in mine if r[2] == TREWAS]
            cells.append(f"{l}: {len(won)}/{sum(r[2] == 0 for r in mine)}/{sum(r[2] == 2 for r in mine)}" + (f" (by turn {max(won)})" if won else ""))
        print(f"{t:18} " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    sys.exit(main())
