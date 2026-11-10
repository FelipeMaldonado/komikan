"""How does confine_30 win? Shows one game, the end statistics, and the trick against opponents that look ahead."""
from collections import Counter
from multiprocessing import Pool

from komikan.board import COORDS
from komikan.bots import make_bot
from komikan.game import ACTIONS, KOMIKELU, TREWAS, apply_action, initial_state, legal_actions, outcome, render_text

CONFINE = "policy:runs/reward/confine_30/snapshots/trewas_005.npz"
HARSH = "policy:runs/reward/harsh_draw_31/snapshots/trewas_005.npz"
K1 = "policy:runs/first/komikelu.npz"


def game(komikelu, trewas, verbose=False):
    state, jumps = initial_state(), []
    while outcome(state) is None:
        mover = state.to_move
        action = (komikelu if mover == KOMIKELU else trewas)(state)
        if verbose and mover == KOMIKELU:
            a = ACTIONS[action]
            print(state.plies, "K", a.src is not None and COORDS[a.src], "->", a.dst is not None and COORDS[a.dst], "JUMP" if a.over is not None else "", "options", len(legal_actions(state)))
        state = apply_action(state, action)
    return state


def versus(job):
    label, spec, trewas_spec, n = job
    k, t = make_bot(spec, 7), make_bot(trewas_spec, 8)
    ends = [game(k, t) for _ in range(n)]
    wins = [s for s in ends if outcome(s) == TREWAS]
    return label, trewas_spec[-30:-19], len(wins), sum(outcome(s) == KOMIKELU for s in ends), n, Counter(COORDS[s.komikelu] for s in wins).most_common(3), Counter(s.captured for s in wins).most_common(4), sorted(s.plies for s in wins)[:: max(1, len(wins) // 6)]


if __name__ == "__main__":
    s = game(make_bot(K1, 1), make_bot(CONFINE, 2), verbose=True)
    print(render_text(s), "result", outcome(s), "captured", s.captured, "turns", s.plies)
    opponents = [("K1 best", K1, 200), ("K2 gen 5", "policy:runs/second/snapshots/komikelu_005.npz", 200), ("K2 gen 11", "policy:models/komikelu.npz", 200),
                 ("random", "random", 200), ("minimax 1", "minimax:1", 100), ("minimax 2", "minimax:2", 40), ("search 400", "mcts:400:0:2", 40), ("search 1600", "mcts:1600:0:2", 16)]
    jobs = [(l, sp, t, n) for l, sp, n in opponents for t in (CONFINE, HARSH)]
    with Pool(8) as pool:
        for row in pool.imap(versus, jobs, chunksize=1):
            print(*row)
