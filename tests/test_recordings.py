import json

import pytest

from komikan.bots import RandomBot
from komikan.game import DRAW, KOMIKELU, NUM_ACTIONS, TREWAS, apply_action, initial_state, outcome
from komikan.recordings import RecordingError, load_all, load_games, summarize


def record_game(seed=0, human_roles=(TREWAS,)):
    komikelu, trewas = RandomBot(seed), RandomBot(seed + 1)
    state, actions = initial_state(), []
    while outcome(state) is None:
        action = (komikelu if state.to_move == KOMIKELU else trewas)(state)
        actions.append(action)
        state = apply_action(state, action)
    return {
        "finishedAt": f"2026-11-04T00:00:{seed:02d}Z",
        "humanRoles": list(human_roles),
        "opponentAgent": "abc123" if len(human_roles) == 1 else None,
        "capturesToWin": 9,
        "maxPlies": 200,
        "actionCount": NUM_ACTIONS,
        "actions": actions,
        "outcome": outcome(state),
        "captured": state.captured,
        "plies": state.plies,
    }


def write(tmp_path, games, name="games.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"format": "komikan-games", "version": 1, "games": games}))
    return path


def test_loads_and_replays_a_game(tmp_path):
    raw = record_game()
    [game] = load_games(write(tmp_path, [raw]))
    assert game.outcome == raw["outcome"]
    assert [a for _, a in game.replay()] == raw["actions"]


def test_human_moves_are_only_the_humans(tmp_path):
    [game] = load_games(write(tmp_path, [record_game(human_roles=(TREWAS,))]))
    moves = list(game.human_moves())
    assert moves and all(state.to_move == TREWAS for state, _ in moves)
    assert len(moves) < len(game.actions)


def test_two_player_game_counts_every_move(tmp_path):
    [game] = load_games(write(tmp_path, [record_game(human_roles=(KOMIKELU, TREWAS))]))
    assert len(list(game.human_moves())) == len(game.actions)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda g: g.update(actions=[g["actions"][1], *g["actions"][1:]]),  # a trewas move on the komikelu's turn
        lambda g: g.update(actions=g["actions"][:5]),  # doesn't reach an end
        lambda g: g.update(actions=[*g["actions"], g["actions"][0]]),  # a move after the end
        lambda g: g.update(outcome=TREWAS if g["outcome"] != TREWAS else DRAW),
        lambda g: g.update(captured=g["captured"] + 1),
        lambda g: g.update(actionCount=NUM_ACTIONS + 1),
        lambda g: g.update(maxPlies=100),
        lambda g: g.pop("actions"),
    ],
    ids=["illegal", "unfinished", "after-end", "outcome", "captures", "action-count", "rules", "missing-field"],
)
def test_bad_recordings_are_rejected(tmp_path, tamper):
    raw = record_game()
    tamper(raw)
    with pytest.raises(RecordingError):
        load_games(write(tmp_path, [raw]))


def test_rejects_other_files(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"format": "something-else", "version": 1, "games": []}))
    with pytest.raises(RecordingError):
        load_games(path)


def test_overlapping_downloads_count_each_game_once(tmp_path):
    a, b = record_game(0), record_game(1)
    first = write(tmp_path, [a], "first.json")
    second = write(tmp_path, [a, b], "second.json")
    assert len(load_all([first, second])) == 2


def test_summary_reports_results_by_role(tmp_path):
    games = load_games(write(tmp_path, [record_game(0), record_game(1, (KOMIKELU,)), record_game(2, (KOMIKELU, TREWAS))]))
    text = summarize(games)
    assert text.startswith("3 games")
    assert "you as trewas vs agent" in text and "you as komikelu vs agent" in text and "two-player: 1 games" in text


def test_numbered_files_load_in_numeric_order():
    from komikan.recordings import _natural_key

    names = ["r/games-10.json", "r/games-2.json", "r/games-1.json"]
    assert sorted(names, key=_natural_key) == ["r/games-1.json", "r/games-2.json", "r/games-10.json"]
