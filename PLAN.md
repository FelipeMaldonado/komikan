# Komikan: Gymnasium environment, RL agents and web game

## Goal

1. A Gymnasium environment for Komikan.
2. Trained agents for both roles (komikelu and trewas), saved so they can be played against.
3. A web version with human vs human and human vs agent (in either role).

## Rules as implemented

Source: the ["El juego del Komikan"](https://www.chileparaninos.gob.cl/639/w3-article-639005.html) PDF by Chile para Niños (not included in this repository), following Matus (1920) for the board.

| Topic | Rule | Source |
| --- | --- | --- |
| Board | 5x5 alquerque plus an inverted triangle sharing its apex: 38 points, 79 lines | PDF p.3 |
| Pieces | 1 komikelu, 12 trewas | PDF p.2 |
| Setup | Komikelu at the triangle's centre; trewas fill the bottom two rows plus both ends of the row above | PDF p.4 |
| First move | Komikelu | PDF p.2 |
| Movement | One point along a line, forwards or backwards | PDF p.2, p.5 |
| Capture | Komikelu only: jump in a straight line over an adjacent trewa to an empty point | PDF p.2 |
| Multi-capture | Can continue jumping in the same turn | PDF p.2, p.6 |
| Komikelu wins | At least 9 trewas captured | PDF p.2 |
| Trewas win | Komikelu has no step or jump available | PDF p.2, p.7 |

### Decisions where the sources are silent

These are assumptions, and each is a single setting or a small change in `game.py`.

- **Capturing is optional, and so is continuing a chain.** The PDF says the komikelu "puede" (may) do both.
- **Captured trewas leave the board immediately.** The p.6 example needs a closed loop, and this is consistent with it.
- **Straight lines run through the apex.** A jump such as square (2,1) over the apex into the triangle is legal, because the
  points are collinear on the drawn board.
- **The site's default: the komikelu wins at 8 captures (house rule).** With 4 trewas left they can trap it only on the two
  outer corners of the triangle, so with the 200-turn draw they could only run out the clock. The site offers 9, the PDF's rule,
  too. Python's `Rules.captures_to_win` is still 9 by default, so all training and experiments used 9.
- **Draw after 200 turns (added rule).** The rules have no draw, but both sides can shuffle forever, so RL episodes and web
  games need a limit. A multi-jump counts as one turn. See `Rules.max_plies`.
- **Trewas with no legal move lose.** This is practically unreachable with the default rules.
- **Variant:** `Rules.trewas_forward_only` is the Matus/Leoncito variation (PDF p.8), read as "strictly towards the triangle".
  It is off by default.

## Architecture

```
src/komikan/
  board.py      geometry, generated from coordinates; jumps derived from collinearity
  game.py       pure rules engine, no dependencies: the single source of truth
  bots.py       random and alpha-beta minimax opponents (benchmarks)
  env.py        Gymnasium env with action masks and a pluggable opponent
  policy.py     numpy-only inference for trained agents (no torch needed to play)
  train.py      self-play training loop
  evaluate.py   matches and tournament tables
  export_web.py writes board data and agent weights for the web app
  recordings.py loads and validates games recorded on the site
  pretrain.py   learning from recorded games: behavioural cloning, then PPO
tests/          rules (including the PDF examples), env, policy export, JS engine agreement
models/         final saved agents
web/            static site for GitHub Pages
```

### Environment

- **Single-agent view:** `KomikanEnv(role=..., opponent=...)`. The opponent is part of the env dynamics.
- **Actions:** one `Discrete` space shared by both roles: every directed step, every directed jump, and "stop chain".
  `action_masks()` gives the legal ones. A multi-jump is several env steps.
- **Observation:** komikelu one-hot, trewas bitmap, in-chain flag, captures so far, turn progress.
- **Reward:** ±1 for win/loss and ±0.1 per capture. A draw costs the trewas -0.75 and the komikelu -0.25: the first trained
  trewas shuffled safely for a draw every game, even against a random komikelu, so a draw must cost them nearly a loss.
  Trewas also get potential-based shaping for advancing, reducing the komikelu's options and confining it. It pays only for
  changes, so shuffling back and forth earns nothing.

### Training

MaskablePPO (`sb3-contrib`) with one model per role, trained in alternating generations against a mix of the benchmark bots and
past snapshots of the other role.

**Self-play win rates alone can be misleading**, so each generation is scored against fixed benchmarks (random, minimax depth
1 to 3) by wins minus losses, and the best generation per role is kept. Each agent is saved twice: an SB3 `.zip` to resume training and a `.npz` for play.

If PPO levels off below minimax strength, the natural next step is AlphaZero-style MCTS, using this same engine.

### Web app

Target: GitHub Pages (https://felipemaldonado.github.io), so everything runs in the browser with no server.

- `komikan-export-web` writes the board, action table and agent weights from Python into `web/data/`.
- `web/engine.js` is a small JS port of the rules. The board and action ids come from the exported data, and
  `tests/test_web_engine.py` replays Python game traces through it (using the embedded `quickjs` interpreter) to check that both engines agree.
- Agents run as a few lines of plain JS matrix maths, so there is no torch, ONNX or build step.
- All paths are relative, so the site works under a subpath such as `/komikan/`.

## Milestones

1. [x] Rules engine and tests from the PDF examples
2. [x] Gymnasium environment
3. [x] Benchmark bots
4. [x] Self-play training and evaluation
5. [x] First trained agents saved
6. [x] Web app: human vs human and human vs agent
7. [ ] Stronger agents, the trewas above all. They draw most games (see README), though a human won 5 of 5 as the trewas against
   the komikelu agent. Cloning recorded games and then running PPO didn't beat its controls. Next, in order: start training
   episodes from mid-game positions of recorded wins; keep an imitation loss during PPO; AlphaZero-style MCTS. The komikelu agent
   needs work too.
8. [x] Publish the site to GitHub Pages: https://felipemaldonado.github.io/komikan/ (the `gh-pages` branch holds `web/`)
9. [ ] Optional: Spanish UI, Leoncito variant agents
