# Komikan

Komikan is a traditional hunting board game of the Mapuche people. One komikelu ("the one who eats everything") tries to eat
trewas (dogs), and twelve trewas try to corner it. This project has:

- a rules engine and a Gymnasium environment,
- reinforcement-learning agents for both roles,
- a browser version, for two people or against an agent, that runs on GitHub Pages with no server.

**Play it: https://felipemaldonado.github.io/komikan/**

See [PLAN.md](PLAN.md) for the rules as implemented, the decisions where the sources are silent, and the design.

## Setup

```bash
uv sync --all-extras
```

```bash
uv run pytest
```

## The environment

```python
import numpy as np
import komikan  # registers Komikan-v0
import gymnasium as gym

env = gym.make("Komikan-v0", role="trewas", opponents=["minimax:2"]).unwrapped
obs, info = env.reset(seed=0)
done = False
while not done:
    action = np.random.choice(np.flatnonzero(env.action_masks()))
    obs, reward, done, truncated, info = env.step(action)
print(info["outcome"])  # 0 komikelu, 1 trewas, 2 draw
```

Opponents are given as specs: `random`, `minimax:<depth>` or `policy:<path.npz>`.

## Training and evaluating

```bash
uv run komikan-train --out runs/my-run --generations 12 --steps-per-generation 150000
```

```bash
uv run komikan-eval --komikelu policy:models/komikelu.npz minimax:2 --trewas policy:models/trewas.npz minimax:2 --games 100
```

A run writes the best agent per role (`.zip` to resume with SB3, `.npz` for play), `metrics.csv` and `snapshots/`.
Copy the agents you want to keep into `models/`.

## Current agents

`models/` holds komikelu generation 11 and trewas generation 13 from a run of 15 generations, 200k steps per role per generation.
Stochastic play, 60 games against minimax 3 and 200 or more against the others:

| Trained komikelu vs trewas played by | Komikelu wins | Trewas win | Draw |
| --- | ---: | ---: | ---: |
| Random | 100% | 0% | 0% |
| Minimax 1 | 100% | 0% | 0% |
| Minimax 2 | 5% | 1.5% | 93.5% |
| Minimax 3 | 3.3% | 3.3% | 93.3% |

| Trained trewas vs komikelu played by | Komikelu wins | Trewas win | Draw |
| --- | ---: | ---: | ---: |
| Random | 0% | 6.5-13% | 87-93.5% |
| Minimax 1 | 0-0.2% | 9.5-12% | 88-90% |
| Minimax 2 | 0% | 0% | 100% |
| Minimax 3 | 0% | 0% | 100% |

**These agents are a first baseline, and the trewas are the weak side.** They rarely lose a piece but rarely finish a game: they
draw most games, even against a random komikelu. They do trap their own komikelu partner about 95% of the time, which is
overfitting to that partner (against an earlier snapshot of the komikelu, generation 5, it's 50%). None of this has been
measured against human players. Minimax is only a benchmark, and its own scoring is passive.

## The web version

Export the board and agents, then serve the static site:

```bash
uv run komikan-export-web
```

```bash
python3 -m http.server 8321 --directory web
```

Then open http://localhost:8321. When you play the komikelu, the computer's trewas are best-play search with the squeeze score at
depth 5 (`minimax:5:squeeze`, see "Search without a network"), not the trained agent, which is much weaker. When you play the trewas,
the computer's komikelu is the trained agent. Options in the address, for testing: `?squeeze=4` for another depth, `?search=1600`
for Monte Carlo search, `?trewas=agent` for the trained trewas, and `?record` to record games outside `localhost`.

The live site is served from the `gh-pages` branch, which holds only `web/`. Browsers keep its files for up to 10 minutes, so
when a change to `web/` goes live, raise the `?v=` number on the file links in `web/index.html` too. To update it after committing
to `main`:

```bash
git push origin "$(git subtree split --prefix web main):refs/heads/gh-pages"
```

## Recording games for training

The site keeps every finished game in the browser that played it, and nothing is sent anywhere. Abandoned games aren't kept.
Storage is per site, so games played at `localhost` and on the live site are separate.

1. Choose "Trained agent", play as Trewas, and finish the game.
2. When you have some games, choose Download. The file holds all recorded games, so overlapping downloads are fine.
3. Save it in `recordings/`, which git ignores, then check it:

```bash
uv run komikan-recordings recordings/*.json
```

This replays every game through the engine and rejects any that are illegal or don't match the current rules.
`load_all` and `RecordedGame.human_moves()` in `recordings.py` give the (state, action) examples for training.

## Learning from recorded games

`komikan-pretrain` clones recorded moves (each also mirrored left to right), trains the value network with the policy frozen,
then continues with PPO. `--no-clone` and `--placebo` (random legal moves on the same positions) are controls.

```bash
uv run komikan-pretrain --recordings recordings/*.json --init models/trewas.zip --clone-epochs 10 --clone-learning-rate 1e-4 --out runs/my-run
```

**So far this hasn't produced a stronger agent.** With 5 recorded games, all wins by the human against the komikelu agent:

- A clone predicts the human's move in unseen games about 35% of the time (random guessing: 5%), but on its own it loses most
  games, because the human almost never hung a piece and so the games barely show recovery from bad positions.
- Cloned from scratch and continued with PPO, it won no games against any benchmark in the first experiment and lost more than
  its control. Fine-tuning the existing agent gently enough to keep its safe play matched the human only 6% of the time.
- Fine-tuned with the human's games, with random moves (placebo) or not at all, the agents were level: about 15% wins against a
  random komikelu, 0% against minimax 2 and 3, and 0% against the first run's best komikelu, with large gaps between seeds
  (3 seeds per group). The agents in `models/` are unchanged.

`--no-clone --start-from human` skips cloning and instead starts half of the episodes from positions in the human's wins.
This improved finishing against known training partners, but not against unseen agents or in full games.

## Search without a network

`mcts:<simulations>[:<draw>[:<playout>]]` is a bot spec that plays by tree search alone. `<draw>` is the value of a draw to the
komikelu, so use 0.5 when it plays the trewas. As trewas, it beats agents that the trained agent can't, though it mostly draws.

```bash
uv run komikan-eval --komikelu policy:models/komikelu.npz --trewas mcts:1600:0.5:2 --games 10
```

The strongest trewas so far is `minimax:<depth>:squeeze`: alpha-beta search with a score that rewards shrinking the komikelu's room. At
depth 5 it wins about 96% of games against the trained komikelu agents, and loses none. `mcts:<...>:squeeze` uses the same score in tree
search, which helps much less.

```bash
uv run komikan-eval --komikelu policy:runs/first/komikelu.npz --trewas minimax:5:squeeze --games 10
```

The site has both searches in `web/search.js`, not linked from the page: open it with `?squeeze=5`, or `?search=1600` for tree search.

`puct:<simulations>:<mode>[:<exploration>]` is search guided by the trained networks, with mode `none`, `prior`, `value`, `rawvalue` or
`both`. Modes that use a value network need the export described in `experiments/guided_experiment.sh`. As trewas it does worse than
plain search against agents the networks never trained with. As komikelu it does better, though it mostly draws.

## Self-play training

`python -m komikan.selfplay` trains a policy and a value network per role from scratch, AlphaZero-style, using search and game results
only. `az:<simulations>:<directory>` is the bot spec for playing with the result. In a small test the trewas learned not to lose, but
almost never won, so plain search is still stronger against unseen agents.

```bash
uv run python -m komikan.selfplay --out runs/selfplay/try --generations 2 --games 8 --simulations 25
```

## Sources

Rules, setup and board: ["El juego del Komikan"](https://www.chileparaninos.gob.cl/639/w3-article-639005.html) by Chile para Niños (CC BY-SA), based on Matus (1920), Molina and Manquilef.
The PDF's illustrations aren't used here. See also [Komikan on Wikipedia](https://es.wikipedia.org/wiki/Komikan).
