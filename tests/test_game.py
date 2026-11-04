"""Rules tests. The examples come from the rules PDF (p.4 setup, p.6 and p.7 endings)."""

from komikan.board import APEX, EDGES, JUMPS, NEIGHBORS, NUM_POINTS, sq, tri
from komikan.game import (
    ACTIONS,
    DEFAULT_RULES,
    DRAW,
    KOMIKELU,
    STOP,
    TREWAS,
    Action,
    GameState,
    Rules,
    apply_action,
    bitmask,
    initial_state,
    komikelu_reach,
    legal_actions,
    outcome,
    points_of,
)

ID = {a: i for i, a in enumerate(ACTIONS)}


def jump(src, over, dst):
    return ID[Action(src, dst, over)]


def step(src, dst):
    return ID[Action(src, dst, None)]


def k_state(komikelu, trewas, captured=0, **kw):
    return GameState(komikelu, bitmask(trewas), KOMIKELU, False, captured, 0)._replace(**kw)


def test_board_size():
    assert NUM_POINTS == 38
    assert len(EDGES) == 79


def test_point_with_diagonals_has_eight_neighbors():  # p.5
    assert len(NEIGHBORS[sq(1, 1)]) == 8


def test_triangle_centre_moves_orthogonally():  # p.5
    assert set(NEIGHBORS[tri(2, 0)]) == {tri(3, 0), tri(1, 0), tri(2, -2), tri(2, 2)}


def test_triangle_is_only_connected_through_apex():
    triangle = {tri(k, x) for k, xs in {1: (-2, 0, 2), 2: (-4, -2, 0, 2, 4), 3: (-6, -2, 0, 2, 6)}.items() for x in xs}
    links = {(a, b) for a, b in EDGES if (a in triangle) != (b in triangle)}
    assert all(APEX in link for link in links)
    assert len(links) == 3


def test_initial_setup():  # p.4
    s = initial_state()
    assert s.komikelu == tri(2, 0)
    assert set(points_of(s.trewas)) == {sq(0, 2), sq(4, 2), *(sq(c, r) for r in (3, 4) for c in range(5))}
    assert s.to_move == KOMIKELU  # p.2
    assert len(points_of(s.trewas)) == 12


def test_jumps_run_straight_through_apex():
    assert (sq(2, 1), APEX, tri(1, 0)) in JUMPS
    assert (sq(1, 1), APEX, tri(1, 2)) in JUMPS
    assert (sq(1, 0), APEX, sq(3, 0)) in JUMPS


def test_no_jump_turns_a_corner():
    # (0,1) has no diagonal, so it cannot jump (1,1) towards a diagonal landing.
    assert (sq(0, 1), sq(1, 1), sq(2, 2)) not in JUMPS
    # Along the triangle's side, jumping from the apex lands on the side, not the row.
    assert (APEX, tri(1, -2), tri(2, -2)) not in JUMPS
    assert (APEX, tri(1, -2), tri(2, -4)) in JUMPS


def test_cannot_jump_when_landing_is_occupied():  # p.6, the trewa on (3,1)
    s = k_state(sq(2, 1), [sq(3, 1), sq(4, 1)])
    assert jump(sq(2, 1), sq(3, 1), sq(4, 1)) not in legal_actions(s)


def test_four_capture_circuit_wins_the_game():  # p.6
    trewas = [sq(1, 1), sq(0, 2), sq(2, 2), sq(1, 3), sq(3, 1), sq(4, 1), sq(4, 3)]
    s = k_state(sq(2, 1), trewas, captured=5)
    path = [
        (sq(2, 1), sq(2, 2), sq(2, 3)),
        (sq(2, 3), sq(1, 3), sq(0, 3)),
        (sq(0, 3), sq(0, 2), sq(0, 1)),
        (sq(0, 1), sq(1, 1), sq(2, 1)),
    ]
    for i, move in enumerate(path):
        assert outcome(s) is None, f"game ended early at jump {i}"
        assert jump(*move) in legal_actions(s)
        s = apply_action(s, jump(*move))
    assert s.komikelu == sq(2, 1)  # back on its starting point
    assert s.captured == 9
    assert set(points_of(s.trewas)) == {sq(3, 1), sq(4, 1), sq(4, 3)}  # 3 left, as the PDF says
    assert outcome(s) == KOMIKELU


def test_trapped_komikelu_loses():  # p.7
    trewas = [sq(0, 1), sq(0, 2), sq(1, 3), sq(2, 3), sq(0, 4), sq(1, 4), sq(3, 4)]
    s = k_state(sq(0, 3), trewas, captured=5)
    assert legal_actions(s) == []
    assert outcome(s) == TREWAS


def test_chain_is_optional_and_turn_passes_on_stop():
    s = k_state(sq(2, 1), [sq(2, 2), sq(1, 3), sq(4, 4)])
    s = apply_action(s, jump(sq(2, 1), sq(2, 2), sq(2, 3)))
    assert s.in_chain and s.to_move == KOMIKELU
    assert set(legal_actions(s)) == {jump(sq(2, 3), sq(1, 3), sq(0, 3)), STOP}  # no plain steps mid-chain
    s = apply_action(s, STOP)
    assert s.to_move == TREWAS and not s.in_chain and s.plies == 1


def test_turn_passes_automatically_after_last_jump():
    s = k_state(sq(2, 1), [sq(2, 2), sq(4, 4)])
    s = apply_action(s, jump(sq(2, 1), sq(2, 2), sq(2, 3)))
    assert s.to_move == TREWAS and not s.in_chain and s.plies == 1


def test_capture_is_optional():
    s = k_state(sq(2, 1), [sq(2, 2), sq(4, 4)])
    assert step(sq(2, 1), sq(1, 1)) in legal_actions(s)


def test_trewas_step_to_empty_points_and_never_jump():
    s = initial_state()._replace(to_move=TREWAS)
    for a in legal_actions(s):
        assert ACTIONS[a].over is None and a != STOP
        assert ACTIONS[a].src in points_of(s.trewas)
    assert step(sq(0, 2), sq(1, 1)) in legal_actions(s)
    assert step(sq(0, 3), sq(0, 2)) not in legal_actions(s)


def test_trewa_move_updates_only_that_piece():
    s = initial_state()._replace(to_move=TREWAS)
    after = apply_action(s, step(sq(0, 2), sq(1, 1)))
    assert sq(1, 1) in points_of(after.trewas) and sq(0, 2) not in points_of(after.trewas)
    assert len(points_of(after.trewas)) == 12
    assert after.to_move == KOMIKELU


def test_turn_limit_is_a_draw_but_a_trap_on_the_last_turn_still_wins():
    limit = Rules(max_plies=10)
    s = initial_state()._replace(plies=10)
    assert outcome(s, limit) == DRAW
    trapped = k_state(sq(0, 3), [sq(0, 1), sq(0, 2), sq(1, 3), sq(2, 3), sq(0, 4)], plies=10)
    assert outcome(trapped, limit) == TREWAS


def test_forward_only_variant():
    rules = Rules(trewas_forward_only=True)
    s = GameState(tri(2, 0), bitmask([sq(1, 1), sq(4, 4)]), TREWAS, False, 0, 0)
    moves = {ACTIONS[a] for a in legal_actions(s, rules)}
    assert Action(sq(1, 1), sq(1, 0), None) in moves
    assert Action(sq(1, 1), sq(0, 1), None) not in moves  # sideways
    assert Action(sq(1, 1), sq(1, 2), None) not in moves  # backwards
    assert Action(sq(1, 1), sq(1, 2), None) in {ACTIONS[a] for a in legal_actions(s, DEFAULT_RULES)}


def test_reach_is_confined_by_a_wall():
    assert komikelu_reach(k_state(tri(2, 0), [])) == NUM_POINTS
    # Doubled apex lines: the komikelu cannot step past or jump out of the triangle.
    wall = [APEX, sq(2, 1), sq(1, 1), sq(3, 1)]
    assert komikelu_reach(k_state(tri(2, 0), wall)) == 13
