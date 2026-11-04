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

Then open http://localhost:8321. The live site is served from the `gh-pages` branch, which holds only `web/`. To update it after
committing to `main`:

```bash
git push origin "$(git subtree split --prefix web main):refs/heads/gh-pages"
```

## Sources

Rules, setup and board: "El juego del Komikan" by Chile para Niños (CC BY-SA), based on Matus (1920), Molina and Manquilef.
The PDF's illustrations aren't used here. See also [Komikan on Wikipedia](https://es.wikipedia.org/wiki/Komikan).
