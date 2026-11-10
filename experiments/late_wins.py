"""Collect trewas wins to take late training positions from. Run from the project folder.

Plain search plays trewas against komikelu agents that the stage 3 test never uses, so the test opponents stay unseen. Winning games are
saved as action lists, together with the human's wins as trewas.
"""

import json
from glob import glob
from multiprocessing import Pool
from pathlib import Path

from komikan.bots import make_bot
from komikan.game import TREWAS, apply_action, initial_state, outcome
from komikan.midgame import human_wins
from komikan.recordings import load_all

OPPONENTS = [f"runs/first/snapshots/komikelu_{g:03d}.npz" for g in (3, 5, 7, 9)] + [
    f"runs/second/snapshots/komikelu_{g:03d}.npz" for g in (3, 6, 8)
]
GAMES = 60
OUT = Path("runs/late/wins.json")


def collect(job):
    index, path = job
    komikelu, trewas = make_bot(f"policy:{path}", 1000 + index), make_bot("mcts:1600:0.5:2", 2000 + index)
    wins = []
    for _ in range(GAMES):
        state, actions = initial_state(), []
        while outcome(state) is None:
            action = (komikelu if state.to_move != TREWAS else trewas)(state)
            actions.append(int(action))
            state = apply_action(state, action)
        if outcome(state) == TREWAS:
            wins.append(actions)
    return path, wins


def main():
    with Pool(8) as pool:
        results = pool.map(collect, list(enumerate(OPPONENTS)), chunksize=1)
    games = []
    for path, wins in results:
        print(f"{path}: {len(wins)} of {GAMES} won")
        games += [{"source": path, "actions": w} for w in wins]
    human = [{"source": "human", "actions": list(g.actions)} for g in human_wins(load_all(glob("recordings/*.json")), TREWAS)]
    print(f"human wins: {len(human)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(games + human))


if __name__ == "__main__":
    main()
