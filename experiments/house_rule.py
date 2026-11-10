"""The squeeze search's difficulty levels with the house rule (komikelu wins at 8 captures) against the original 9.

Run from the project folder. Same three unseen komikelu agents as the stage 2 test, 20 games each.
"""

from multiprocessing import Pool

from komikan.bots import make_bot
from komikan.evaluate import play_match
from komikan.game import Rules

S, F = "runs/second/snapshots", "runs/first"
KOMIKELU = {"K2 gen 14": f"policy:{S}/komikelu_014.npz", "K1 best": f"policy:{F}/komikelu.npz", "K1 gen 11": f"policy:{F}/snapshots/komikelu_011.npz"}
GAMES = 20


def run(job):
    captures, depth, label = job
    rules = Rules(captures_to_win=captures)
    r = play_match(make_bot(KOMIKELU[label], 31, rules), make_bot(f"minimax:{depth}:squeeze", 32, rules), GAMES, rules)
    return captures, depth, label, r.trewas_wins, r.komikelu_wins


if __name__ == "__main__":
    jobs = [(c, d, l) for c in (8, 9) for d in (3, 4, 5) for l in KOMIKELU]
    with Pool(8) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    for c in (8, 9):
        for d in (3, 4, 5):
            won = sum(r[3] for r in rows if r[:2] == (c, d))
            lost = sum(r[4] for r in rows if r[:2] == (c, d))
            print(f"komikelu wins at {c}, depth {d}: trewas won {won}, lost {lost}, of {3 * GAMES}", flush=True)
