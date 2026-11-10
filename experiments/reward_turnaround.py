"""Do trewas turn round when the komikelu gets behind them? Run from the project folder.

Starts just after the komikelu's jump to the bottom row on turn 32 of the human's 56-turn win. The human then made 10 backward moves
in 12 and trapped it on turn 55, 23 turns later.
"""

from glob import glob
from multiprocessing import Pool

import numpy as np

from komikan.board import COORDS
from komikan.bots import make_bot
from komikan.game import KOMIKELU, TREWAS, apply_action, initial_state, outcome, points_of
from komikan.midgame import human_wins
from komikan.recordings import load_all
from reward_evaluate import GROUPS, OPPONENTS, SEEDS, path

CAPTURE_GAME, CAPTURE_TURN = 3, 32  # the game's index among the human's wins, and the turn of the jump
FIRST_MOVES = 12
QUICK_TURNS = 40
PLAYS = 40
LABELS = ("K2_011*", "K1_best")


def start_position():
    state = initial_state()
    for action in human_wins(load_all(glob("recordings/*.json")), TREWAS)[CAPTURE_GAME].actions[: CAPTURE_TURN + 1]:
        state = apply_action(state, action)
    assert state.to_move == TREWAS and state.captured == 1 and COORDS[state.komikelu] == (0, 12)
    return state


def is_backward(before, after) -> bool:
    gone, new = set(points_of(before.trewas)) - set(points_of(after.trewas)), set(points_of(after.trewas)) - set(points_of(before.trewas))
    return COORDS[new.pop()][1] > COORDS[gone.pop()][1]


def play(job):
    group, seed, label = job
    komikelu = make_bot(OPPONENTS[label][0], 80 + seed)
    trewas = make_bot(f"policy:{path(group, seed)}", 90 + seed)
    start = start_position()
    backward, won, quick, lost = [], 0, 0, 0
    for _ in range(PLAYS):
        state, moves = start, []
        while outcome(state) is None:
            mover = state.to_move
            after = apply_action(state, (komikelu if mover == KOMIKELU else trewas)(state))
            if mover == TREWAS and len(moves) < FIRST_MOVES:
                moves.append(is_backward(state, after))
            state = after
        result = outcome(state)
        backward.append(np.mean(moves))
        won += result == TREWAS
        quick += result == TREWAS and state.plies - start.plies <= QUICK_TURNS
        lost += result == KOMIKELU
    return group, seed, label, float(np.mean(backward)), quick / PLAYS, won / PLAYS, lost / PLAYS


def main():
    with Pool(8) as pool:
        rows = pool.map(play, [(g, s, l) for g in GROUPS for s in SEEDS for l in LABELS], chunksize=1)
    print(f"From turn 32 of the 56-turn win. Mean over seeds [range]. Human: 83% backward, won in 23 turns. {PLAYS} games per seed.")
    for label in LABELS:
        print(f"against {label}")
        print(f"  {'':11}{'backward':>17}{f'won in {QUICK_TURNS} turns':>18}{'won':>17}{'lost':>17}")
        for group in GROUPS:
            cells = [[r[i] for r in rows if r[0] == group and r[2] == label] for i in range(3, 7)]
            print(f"  {group:11}" + "".join(f"{f'{np.mean(c):.0%} [{min(c):.0%}-{max(c):.0%}]':>17}" for c in cells))


if __name__ == "__main__":
    main()
