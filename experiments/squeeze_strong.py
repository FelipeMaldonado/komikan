"""Best-play search with the squeeze score as trewas, against komikelu opponents that look ahead. Run from the project folder."""

from multiprocessing import Pool

from komikan.bots import make_bot
from komikan.evaluate import play_match

TREWAS = ["minimax:4:squeeze", "minimax:5:squeeze"]
KOMIKELU = {"minimax 3": "minimax:3", "search 1600": "mcts:1600:0:2", "search 6400": "mcts:6400:0:2"}
GAMES = 20


def run(job):
    trewas, label = job
    r = play_match(make_bot(KOMIKELU[label], 77), make_bot(trewas, 78), GAMES)
    return trewas, label, r


if __name__ == "__main__":
    with Pool(8) as pool:
        for trewas, label, r in pool.imap(run, [(t, l) for t in TREWAS for l in KOMIKELU]):
            print(f"{trewas:18} vs {label:12} won {r.trewas_wins}/{GAMES} lost {r.komikelu_wins} drawn {r.draws} | turns {r.plies / GAMES:.0f}", flush=True)
