"""Check that the JS engine in web/engine.js agrees with the Python engine on real game traces."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from komikan.board import COORDS, sq
from komikan.bots import MinimaxBot, RandomBot
from komikan.env import OBS_SIZE, encode_observation
from komikan.export_web import board_data, policy_to_web
from komikan.game import NUM_ACTIONS, KOMIKELU, apply_action, bitmask, initial_state, legal_actions, outcome
from komikan.game import GameState
from komikan.policy import NumpyPolicy

quickjs = pytest.importorskip("quickjs")
ENGINE_JS = Path(__file__).parent.parent / "web" / "engine.js"


@pytest.fixture(scope="module")
def js():
    ctx = quickjs.Context()
    ctx.eval(ENGINE_JS.read_text())
    ctx.eval(f"var engine = Komikan.createEngine({json.dumps(board_data())});")
    ctx.eval(
        """
        function view(s) {
          return {
            komikelu: s.komikelu,
            trewas: s.trewas.flatMap((on, p) => on ? [p] : []),
            toMove: s.toMove, inChain: s.inChain, captured: s.captured, plies: s.plies,
            legal: engine.legalActions(s).sort((a, b) => a - b),
            outcome: engine.outcome(s),
            obs: Array.from(engine.encodeObservation(s)),
          };
        }
        function replay(actions) {
          let s = engine.initialState();
          const out = [view(s)];
          for (const a of actions) { s = engine.applyAction(s, a); out.push(view(s)); }
          return JSON.stringify(out);
        }
        """
    )
    return ctx


def python_view(state: GameState) -> dict:
    trewas = [p for p in range(len(COORDS)) if state.trewas >> p & 1]
    return {
        "komikelu": state.komikelu,
        "trewas": trewas,
        "toMove": state.to_move,
        "inChain": state.in_chain,
        "captured": state.captured,
        "plies": state.plies,
        "legal": sorted(legal_actions(state)),
        "outcome": outcome(state),
    }


def assert_same(state: GameState, seen: dict) -> None:
    obs = seen.pop("obs")
    assert seen == python_view(state)
    assert np.allclose(obs, encode_observation(state))


@pytest.mark.parametrize(
    "make_bots",
    [
        lambda seed: (RandomBot(seed), RandomBot(seed + 1)),
        lambda seed: (MinimaxBot(1, seed), RandomBot(seed + 1)),
        lambda seed: (RandomBot(seed), MinimaxBot(1, seed + 1)),
    ],
    ids=["random-random", "minimax-random", "random-minimax"],
)
def test_engines_agree_along_whole_games(js, make_bots):
    final_outcomes = set()
    for seed in range(6):
        komikelu, trewas = make_bots(seed)
        state, states, actions = initial_state(), [initial_state()], []
        while outcome(state) is None:
            action = (komikelu if state.to_move == KOMIKELU else trewas)(state)
            state = apply_action(state, action)
            actions.append(action)
            states.append(state)
        seen = json.loads(js.eval(f"replay({json.dumps(actions)})"))
        assert len(seen) == len(states)
        for python_state, js_state in zip(states, seen):
            assert_same(python_state, js_state)
        final_outcomes.add(outcome(state))
    assert final_outcomes  # games reached a result


def test_js_detects_a_trapped_komikelu(js):  # PDF p.7
    trewas = [sq(0, 1), sq(0, 2), sq(1, 3), sq(2, 3), sq(0, 4), sq(1, 4), sq(3, 4)]
    state = GameState(sq(0, 3), bitmask(trewas), KOMIKELU, False, 5, 0)
    flags = [p in trewas for p in range(len(COORDS))]
    seen = json.loads(
        js.eval(
            "JSON.stringify(view({komikelu: %d, trewas: %s, toMove: 0, inChain: false, captured: 5, plies: 0}))"
            % (sq(0, 3), json.dumps(flags))
        )
    )
    assert_same(state, seen)


def test_js_policy_matches_numpy_policy(js):
    rng = np.random.default_rng(0)
    policy = NumpyPolicy(
        [rng.normal(size=(16, OBS_SIZE)), rng.normal(size=(8, 16)), rng.normal(size=(NUM_ACTIONS, 8))],
        [rng.normal(size=16), rng.normal(size=8), rng.normal(size=NUM_ACTIONS)],
    )
    blob, shapes = policy_to_web(policy)
    values = np.frombuffer(blob, dtype="<f4").tolist()
    js.eval(f"var policy = Komikan.createPolicy(engine, new Float32Array({json.dumps(values)}), {json.dumps(shapes)});")
    logits = json.loads(js.eval("JSON.stringify(Array.from(policy.logits(engine.encodeObservation(engine.initialState()))))"))
    assert np.allclose(logits, policy.logits(encode_observation(initial_state())), atol=1e-3)

    # Sampling stays legal, and an almost-certain draw picks the policy's preferred legal move.
    legal = set(legal_actions(initial_state()))
    random.seed(0)
    for _ in range(20):
        chosen = js.eval(f"policy.selectAction(engine.initialState(), () => {random.random()})")
        assert chosen in legal
    top = max(legal, key=lambda a: policy.logits(encode_observation(initial_state()))[a])
    probe = "(function(){const s=engine.initialState();const l=engine.legalActions(s);const c={};for(let i=0;i<400;i++){const a=policy.selectAction(s,Math.random);c[a]=(c[a]||0)+1;}return JSON.stringify(c);})()"
    counts = json.loads(js.eval(probe))
    assert max(counts, key=counts.get) == str(top)


def test_page_text_exists_in_both_languages():
    """Every key the page uses has English and Spanish text, and both languages have the same keys."""
    import re

    ctx = quickjs.Context()
    ctx.eval((ENGINE_JS.parent / "i18n.js").read_text())
    keys = json.loads(ctx.eval("JSON.stringify(Object.fromEntries(Komikan.i18n.languages.map((l) => [l, Object.keys(Komikan.i18n.strings[l]).sort()])))"))
    assert set(keys) == {"en", "es"} and keys["en"] == keys["es"]
    html = (ENGINE_JS.parent / "index.html").read_text()
    app = (ENGINE_JS.parent / "app.js").read_text()
    used = set(re.findall(r'data-i18n(?:-html|-aria)?="(\w+)"', html)) | set(re.findall(r'\bt\("(\w+)"', app))
    used |= {"moveTo", "jumpTo", "languageButton", "languageButtonLabel", "description"}  # chosen in code or set by key
    assert used <= set(keys["en"]), used - set(keys["en"])
    order = [re.search(rf'src="{name}\.js(\?v=\d+)?"', html).start() for name in ("engine", "i18n", "app")]
    assert order == sorted(order)  # engine.js replaces Komikan, so i18n.js must come after it
