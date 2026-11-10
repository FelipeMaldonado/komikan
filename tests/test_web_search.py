"""Check that the JS search in web/search.js chooses the same moves as the Python search in mcts.py.

Both are given the same list of random numbers, so every shuffle and playout matches and any difference is a real one.
"""

import json
import math
import random
from pathlib import Path

import pytest

from komikan.board import COORDS
from komikan.bots import RandomBot
from komikan.export_web import board_data
from komikan.game import KOMIKELU, TREWAS, GameState, apply_action, initial_state, komikelu_reach, outcome
from komikan.mcts import MCTSBot, heuristic

quickjs = pytest.importorskip("quickjs")
WEB = Path(__file__).parent.parent / "web"
NUMBERS = [random.Random(7).random() for _ in range(200_000)]


class SequenceRandom:
    """Python's side of the shared random numbers, with the same shuffle and choice as search.js."""

    def __init__(self, numbers):
        self.numbers, self.next = numbers, 0

    def random(self) -> float:
        value = self.numbers[self.next]
        self.next += 1
        return value

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = math.floor(self.random() * (i + 1))
            items[i], items[j] = items[j], items[i]

    def choice(self, items):
        return items[math.floor(self.random() * len(items))]


@pytest.fixture(scope="module")
def js():
    ctx = quickjs.Context()
    ctx.eval((WEB / "engine.js").read_text())
    ctx.eval((WEB / "search.js").read_text())
    ctx.eval(f"var engine = Komikan.createEngine({json.dumps(board_data())});")
    ctx.eval(
        """
        var used = 0;
        function toState(s) {
          const trewas = new Array(engine.numPoints).fill(false);
          s.trewas.forEach((p) => { trewas[p] = true; });
          return { komikelu: s.komikelu, trewas, toMove: s.toMove, inChain: s.inChain, captured: s.captured, plies: s.plies };
        }
        function measures(json) {
          const search = Komikan.createSearch(engine);
          return JSON.stringify(JSON.parse(json).map((s) => {
            const state = toState(s);
            return [search.komikeluReach(state), search.heuristic(state)];
          }));
        }
        function choose(json, numbers, settings) {
          let next = 0;
          used = 0;
          const random = () => numbers[next++];
          const search = Komikan.createSearch(engine, { ...settings, random });
          const action = search.selectAction(toState(JSON.parse(json)));
          used = next;
          return action;
        }
        """
    )
    ctx.eval(f"var numbers = {json.dumps(NUMBERS)};")
    return ctx


def view(state: GameState) -> dict:
    return {
        "komikelu": state.komikelu,
        "trewas": [p for p in range(len(COORDS)) if state.trewas >> p & 1],
        "toMove": state.to_move,
        "inChain": state.in_chain,
        "captured": state.captured,
        "plies": state.plies,
    }


def positions(games: int = 12, seed: int = 3) -> list[GameState]:
    """Positions from random games, at many stages and with both sides to move."""
    out = []
    for g in range(games):
        komikelu, trewas = RandomBot(seed + g), RandomBot(seed + 100 + g)
        state = initial_state()
        while outcome(state) is None:
            if state.plies % 9 == g % 9:
                out.append(state)
            state = apply_action(state, (komikelu if state.to_move == KOMIKELU else trewas)(state))
    return out


def test_heuristic_and_reach_match(js):
    states = positions()
    measured = json.loads(js.eval(f"measures({json.dumps(json.dumps([view(s) for s in states]))})"))
    assert len(states) > 100
    for state, (reach, value) in zip(states, measured):
        assert reach == komikelu_reach(state)
        assert value == pytest.approx(heuristic(state), abs=1e-12)


@pytest.mark.parametrize("draw", [0.0, 0.5])
def test_search_chooses_the_same_moves(js, draw):
    states = positions(games=6, seed=40)
    assert {s.to_move for s in states} == {KOMIKELU, TREWAS}
    for state in states:
        bot = MCTSBot(simulations=150, playout=2, draw=draw)
        bot.rng = SequenceRandom(NUMBERS)
        expected = bot(state)
        settings = json.dumps({"simulations": 150, "playout": 2, "draw": draw})
        chosen = js.eval(f"choose({json.dumps(json.dumps(view(state)))}, numbers, {settings})")
        assert chosen == expected, f"different move at {view(state)}"
        assert js.eval("used") == bot.rng.next  # the same random numbers were drawn


def test_squeeze_heuristic_matches(js):
    from komikan.mcts import squeeze_heuristic

    states = positions()
    js.eval(
        """
        function squeezes(json) {
          const search = Komikan.createSearch(engine);
          return JSON.stringify(JSON.parse(json).map((s) => search.squeezeHeuristic(toState(s))));
        }
        """
    )
    values = json.loads(js.eval(f"squeezes({json.dumps(json.dumps([view(s) for s in states]))})"))
    for state, value in zip(states, values):
        assert value == pytest.approx(squeeze_heuristic(state), abs=1e-12)


@pytest.mark.parametrize("depth", [2, 3])
def test_minimax_chooses_the_same_moves(js, depth):
    from komikan.bots import MinimaxBot

    js.eval(
        """
        function chooseMinimax(json, numbers, depth) {
          let next = 0;
          const search = Komikan.createMinimax(engine, { depth, random: () => numbers[next++] });
          return search.selectAction(toState(JSON.parse(json)));
        }
        """
    )
    states = positions(games=4, seed=70)[::3]
    assert {s.to_move for s in states} == {KOMIKELU, TREWAS}
    for state in states:
        bot = MinimaxBot(depth, evaluation="squeeze")
        bot.rng = SequenceRandom(NUMBERS)
        expected = bot(state)
        chosen = js.eval(f"chooseMinimax({json.dumps(json.dumps(view(state)))}, numbers, {depth})")
        assert chosen == expected, f"different move at {view(state)}"
