"""The search must follow each node's own side to move, take wins and avoid losses."""

from komikan.bots import make_bot
from komikan.game import ACTIONS, KOMIKELU, TREWAS, GameState, Rules, apply_action, bitmask, initial_state, legal_actions, outcome
from komikan.mcts import MCTSBot, heuristic


def _chain_position() -> GameState:
    """The komikelu to move, with a jump available, found from the rules rather than from a fixed layout."""
    for action in range(len(ACTIONS)):
        src, dst, over = ACTIONS[action]
        if over is not None:
            return GameState(src, bitmask([over]), KOMIKELU, False, 0, 0)
    raise AssertionError("no jumps")


def test_returns_legal_moves_for_both_roles():
    state = initial_state()
    bot = MCTSBot(simulations=30, seed=0)
    for _ in range(12):
        action = bot(state)
        assert action in legal_actions(state)
        state = apply_action(state, action)


def test_komikelu_takes_the_winning_capture():
    src, dst, over = next(a for a in ACTIONS if a.over is not None)
    state = GameState(src, bitmask([over]), KOMIKELU, False, 8, 50)  # one capture from winning
    action = MCTSBot(simulations=100, seed=1)(state)
    assert ACTIONS[action].over == over
    assert outcome(apply_action(state, action)) == KOMIKELU


def test_trewas_fill_the_last_escape():
    """From a real game state: search with enough budget plays a move that ends the game when one exists."""
    rules = Rules()
    state = _trapped_but_for_one_move()
    wins = [a for a in legal_actions(state, rules) if outcome(apply_action(state, a, rules), rules) == TREWAS]
    assert wins
    assert MCTSBot(simulations=300, seed=2)(state) in wins


def _trapped_but_for_one_move() -> GameState:
    """Search the board for a komikelu with one escape, which a single trewa move can cover."""
    from komikan.board import NUM_POINTS

    for komikelu in range(NUM_POINTS):
        others = [p for p in range(NUM_POINTS) if p != komikelu]
        base = GameState(komikelu, 0, KOMIKELU, False, 0, 10)
        neighbours = [ACTIONS[a].dst for a in legal_actions(base)]
        if len(neighbours) > 4:
            continue
        for escape in neighbours:
            blockers = [n for n in neighbours if n != escape]
            behind = [ACTIONS[a].dst for p in blockers for a in range(len(ACTIONS)) if ACTIONS[a].src == komikelu and ACTIONS[a].over == p]
            spare = [p for p in others if p not in neighbours and p not in behind]
            guard = [*blockers, *behind]
            for mover in spare:
                state = GameState(komikelu, bitmask([*guard, mover]), TREWAS, False, 0, 10)
                for action in legal_actions(state):
                    if ACTIONS[action].dst == escape and ACTIONS[action].src == mover:
                        after = apply_action(state, action)
                        if outcome(after) == TREWAS:
                            return state
    raise AssertionError("no such position")


def test_chain_positions_keep_the_komikelu_to_move():
    state = _chain_position()
    bot = MCTSBot(simulations=50, seed=3)
    action = bot(state)
    assert action in legal_actions(state)


def test_draw_value_changes_a_terminal_estimate():
    state = GameState(0, bitmask([20]), TREWAS, False, 0, 199)
    drawn = apply_action(state, legal_actions(state)[0])
    assert outcome(drawn) is not None
    assert MCTSBot(draw=0.5)._result_value(outcome(drawn)) == 0.5


def test_heuristic_bounds_and_direction():
    assert -1.0 <= heuristic(initial_state()) <= 1.0
    state = initial_state()
    assert heuristic(state._replace(captured=5)) > heuristic(state)


def test_make_bot_spec():
    bot = make_bot("mcts:50:0.5", seed=0)
    assert bot.simulations == 50 and bot.draw == 0.5
    assert bot(initial_state()) in legal_actions(initial_state())


def test_seed_makes_games_repeatable():
    state = initial_state()
    assert MCTSBot(simulations=60, seed=9)(state) == MCTSBot(simulations=60, seed=9)(state)


def test_squeeze_heuristic_sees_progress_inside_the_triangle():
    from komikan.board import COORDS
    from komikan.game import komikelu_reach
    from komikan.mcts import squeeze_heuristic

    at = {xy: p for p, xy in enumerate(COORDS)}
    gate = [at[xy] for xy in [(0, 0), (0, 3), (3, 3), (-3, 3)]]  # the gate, and the points beyond it that stop a jump out
    far = [at[xy] for xy in [(-6, 12), (-3, 12), (0, 12), (3, 12), (6, 12), (-6, 9), (6, 9)]]
    shut_in = initial_state()._replace(komikelu=at[(0, -6)], trewas=bitmask(gate + far + [at[(-3, 9)]]), to_move=TREWAS)
    assert komikelu_reach(shut_in) == 13  # the whole triangle and nothing else
    entered = shut_in._replace(trewas=bitmask(gate + far + [at[(-2, -2)]]))  # the same piece, now inside
    assert komikelu_reach(entered) == 12
    squeeze_gain = squeeze_heuristic(shut_in) - squeeze_heuristic(entered)
    basic_gain = heuristic(shut_in) - heuristic(entered)
    assert squeeze_gain > 0 and basic_gain > 0
    assert squeeze_gain > basic_gain  # entering counts for more


def test_bot_spec_takes_a_heuristic():
    import pytest

    bot = make_bot("mcts:50:0.5:2:squeeze", seed=0)
    assert bot.heuristic.__name__ == "squeeze_heuristic"
    assert make_bot("mcts:50:0.5:2", seed=0).heuristic.__name__ == "heuristic"
    with pytest.raises(KeyError):
        make_bot("mcts:50:0.5:2:nonsense")


def test_minimax_spec_takes_an_evaluation():
    from komikan.bots import evaluate, squeeze_evaluate

    assert make_bot("minimax:2").evaluate is evaluate
    bot = make_bot("minimax:3:squeeze", seed=0)
    assert bot.depth == 3 and bot.evaluate is squeeze_evaluate
    state = initial_state()
    assert bot(state) in legal_actions(state)
    assert squeeze_evaluate(state._replace(plies=200)) == 500.0  # a draw is good for the komikelu
