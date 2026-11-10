"""Score self-play under the forward-or-sideways variant. Run from the project folder.

All games use the variant, including the references: plain search, and the standard-rules PPO agents restricted to legal moves. The PPO
komikelu agents are fair opponents, since the variant doesn't change the komikelu's moves. Fixed before seeing any result: to go on, the
self-play agent at 400 simulations must clearly beat plain search at 1600 as trewas against K2 gen 14, K1 best and K1 gen 11, and win
against minimax 2. The self-play outcomes in the logs answer the other question: does the variant make games more decisive?
"""

from multiprocessing import Pool

from komikan.bots import make_bot
from komikan.evaluate import play_match
from komikan.game import RULE_VARIANTS

RULES = RULE_VARIANTS["no-backward"]
S, F = "runs/second/snapshots", "runs/first"
SEEDS = (50, 51, 52)
AGENTS = {
    **{f"selfplay_{s}": f"az:400:runs/variant/selfplay_{s}/gen_030" for s in SEEDS},
    "plain 400": None,  # the role's own search settings, below
    "plain 1600": None,
    "PPO models": None,
}
PLAIN = {"trewas": {"plain 400": "mcts:400:0.5:2", "plain 1600": "mcts:1600:0.5:2", "PPO models": "policy:models/trewas.npz"},
         "komikelu": {"plain 400": "mcts:400:0:2", "plain 1600": "mcts:1600:0:2", "PPO models": "policy:models/komikelu.npz"}}
KOMIKELU_OPPONENTS = {
    "random": "random",
    "minimax 1": "minimax:1",
    "minimax 2": "minimax:2",
    "K2 gen 11": "policy:models/komikelu.npz",
    "K2 gen 14": f"policy:{S}/komikelu_014.npz",
    "K1 best": f"policy:{F}/komikelu.npz",
    "K1 gen 11": f"policy:{F}/snapshots/komikelu_011.npz",
    "search 1600": "mcts:1600:0:2",
}
TREWAS_OPPONENTS = {
    "random": "random",
    "minimax 1": "minimax:1",
    "minimax 2": "minimax:2",
    "minimax 3": "minimax:3",
    "T2 gen 13": "policy:models/trewas.npz",
    "T1 best": f"policy:{F}/trewas.npz",
    "search 1600": "mcts:1600:0.5:2",
}
UNSEEN = ("K2 gen 14", "K1 best", "K1 gen 11")


def spec(agent, role):
    return AGENTS[agent] or PLAIN[role][agent]


def games_for(label):
    return 20 if label == "search 1600" else 40


def run(job):
    agent, role, label = job
    opponents = KOMIKELU_OPPONENTS if role == "trewas" else TREWAS_OPPONENTS
    n = games_for(label)
    mine, other = make_bot(spec(agent, role), 900, RULES), make_bot(opponents[label], 901, RULES)
    r = play_match(other, mine, n, RULES) if role == "trewas" else play_match(mine, other, n, RULES)
    won, lost = (r.trewas_wins, r.komikelu_wins) if role == "trewas" else (r.komikelu_wins, r.trewas_wins)
    return agent, role, label, won, lost, n


def main():
    jobs = []
    for agent in AGENTS:
        for role, opponents in (("trewas", KOMIKELU_OPPONENTS), ("komikelu", TREWAS_OPPONENTS)):
            for label in opponents:
                if label == "search 1600" and not agent.startswith("selfplay"):
                    continue  # the references aren't played against search
                jobs.append((agent, role, label))
    with Pool(8) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    for role, opponents in (("trewas", KOMIKELU_OPPONENTS), ("komikelu", TREWAS_OPPONENTS)):
        print(f"\nAs {role}, forward-or-sideways rules: games won/lost, of 40 (20 against search 1600)")
        for agent in AGENTS:
            cells = [f"{l} {r[3]}/{r[4]}" for l in opponents for r in rows if r[:3] == (agent, role, l)]
            print(f"{agent:12} " + "  ".join(cells))
        if role == "trewas":
            for agent in AGENTS:
                won = sum(r[3] for r in rows if r[0] == agent and r[1] == role and r[2] in UNSEEN)
                lost = sum(r[4] for r in rows if r[0] == agent and r[1] == role and r[2] in UNSEEN)
                print(f"{agent:12} against {', '.join(UNSEEN)}: won {won}, lost {lost}, of 120")


if __name__ == "__main__":
    main()
