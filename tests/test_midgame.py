"""Mid-game training starts: position selection and the env's use of them."""

from pathlib import Path

import pytest

from komikan.env import KomikanEnv
from komikan.evaluate import play_game
from komikan.bots import make_bot
from komikan.game import KOMIKELU, TREWAS, Rules, apply_action, initial_state, legal_actions, mirror_state, outcome
from komikan.midgame import game_positions, human_positions
from komikan.recordings import RecordedGame


def _game(result: int, human: int = TREWAS, turns: int = 40) -> RecordedGame:
    """A legal game, with `result` as its recorded outcome: only selection logic reads that field."""
    state, actions = initial_state(), []
    for i in range(turns):
        moves = legal_actions(state)
        actions.append(moves[i % len(moves)])
        state = apply_action(state, actions[-1])
        if outcome(state) is not None:
            break
    return RecordedGame(tuple(actions), (human,), result, None, Rules())


def _position() -> object:
    return apply_action(initial_state(), legal_actions(initial_state())[0])


def test_positions_come_only_from_the_roles_wins():
    won, lost, other_role = _game(TREWAS), _game(KOMIKELU), _game(TREWAS, human=KOMIKELU)
    assert human_positions([lost, other_role], TREWAS) == []
    positions = human_positions([won, lost, other_role], TREWAS)
    assert positions and all(state.to_move == TREWAS for state in positions)


def test_min_progress_drops_the_opening_and_mirrors_are_included():
    game = _game(TREWAS)
    moved = game_positions(game, TREWAS)
    positions = human_positions([game], TREWAS, min_progress=0.5)
    late = moved[len(moved) // 2 :]
    assert set(late) <= set(positions)
    assert {mirror_state(state) for state in late} <= set(positions)
    assert moved[0] not in positions


def test_holdout_excludes_a_game():
    kept, held_out = _game(TREWAS, turns=40), _game(TREWAS, turns=20)
    all_positions = human_positions([kept, held_out], TREWAS, min_progress=0.0)
    assert human_positions([held_out, kept], TREWAS, min_progress=0.0, exclude=[1]) != all_positions
    assert human_positions([kept, held_out], TREWAS, min_progress=0.0, exclude=[0, 1]) == []


def test_env_starts_from_given_positions():
    position = _position()
    env = KomikanEnv(role="trewas", opponents=["random"])
    env.set_start_positions([position], 1.0)
    env.reset(seed=0)
    assert env.state == position
    assert env.action_masks().any()


def test_share_zero_keeps_normal_starts():
    env = KomikanEnv(role="trewas", opponents=["random"])
    env.set_start_positions([_position()], 0.0)
    env.reset(seed=0)
    assert env.state.plies == 1  # the komikelu's first move from the initial position


def test_share_mixes_both_starts_and_follows_the_seed():
    position = _position()

    def starts(seed: int) -> list[bool]:
        env = KomikanEnv(role="trewas", opponents=["random"])
        env.set_start_positions([position], 0.5)
        result = []
        for i in range(60):
            env.reset(seed=seed if i == 0 else None)
            result.append(env.state == position)
        return result

    first = starts(3)
    assert first == starts(3)
    assert 10 < sum(first) < 50


def test_env_rejects_positions_for_the_other_role():
    env = KomikanEnv(role="komikelu", opponents=["random"])
    with pytest.raises(ValueError):
        env.set_start_positions([_position()], 0.5)


def test_play_game_can_start_from_a_position():
    position = _position()._replace(plies=199)
    _, _, turns = play_game(make_bot("random", 0), make_bot("random", 1), start=position)
    assert turns == 200


@pytest.mark.skipif(not list(Path("recordings").glob("*.json")), reason="the recordings are local only")
def test_real_recordings_give_positions():
    from komikan.recordings import load_all

    games = load_all(sorted(Path("recordings").glob("*.json")))
    assert len(human_positions(games, TREWAS)) > 100


def test_human_wins_keeps_only_the_humans_wins_in_that_role():
    from komikan.midgame import human_wins

    won, lost, as_other_role = _game(TREWAS), _game(KOMIKELU), _game(TREWAS, human=KOMIKELU)
    assert human_wins([won, lost, as_other_role], TREWAS) == [won]
