"""Load games recorded by the web app, checking each one by replaying it through the engine."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .game import (
    DEFAULT_RULES,
    DRAW,
    KOMIKELU,
    NUM_ACTIONS,
    TREWAS,
    GameState,
    Rules,
    apply_action,
    initial_state,
    legal_actions,
    outcome,
)


class RecordingError(ValueError):
    """A recording doesn't match the current rules, or isn't a legal game."""


@dataclass(frozen=True)
class RecordedGame:
    actions: tuple[int, ...]
    human_roles: tuple[int, ...]
    outcome: int
    opponent_agent: str | None
    rules: Rules

    def replay(self) -> Iterator[tuple[GameState, int]]:
        """Each (state, action) of the game, in order. A loaded game is known to be legal."""
        state = initial_state()
        for action in self.actions:
            yield state, action
            state = apply_action(state, action, self.rules)

    def human_moves(self) -> Iterator[tuple[GameState, int]]:
        """The (state, action) pairs chosen by a human: the training examples."""
        return ((s, a) for s, a in self.replay() if s.to_move in self.human_roles)


def _parse_game(raw: dict) -> RecordedGame:
    if raw["actionCount"] != NUM_ACTIONS:
        raise RecordingError(f"recorded with {raw['actionCount']} actions, but the engine has {NUM_ACTIONS}")
    rules = Rules(captures_to_win=raw["capturesToWin"], max_plies=raw["maxPlies"])
    if rules != DEFAULT_RULES:
        raise RecordingError(f"recorded with different rules: {rules}")

    state = initial_state()
    for i, action in enumerate(raw["actions"]):
        if outcome(state, rules) is not None:
            raise RecordingError(f"move {i} comes after the end of the game")
        if action not in legal_actions(state, rules):
            raise RecordingError(f"move {i} (action {action}) is illegal")
        state = apply_action(state, action, rules)

    result = outcome(state, rules)
    if result is None:
        raise RecordingError("the game doesn't reach an end")
    if result != raw["outcome"] or state.captured != raw["captured"] or state.plies != raw["plies"]:
        raise RecordingError("the recorded result doesn't match the replay")
    return RecordedGame(tuple(raw["actions"]), tuple(raw["humanRoles"]), result, raw["opponentAgent"], rules)


def load_games(path: str | Path) -> list[RecordedGame]:
    data = json.loads(Path(path).read_text())
    if data.get("format") != "komikan-games" or data.get("version") != 1:
        raise RecordingError(f"{path}: not a version 1 komikan-games file")
    games = []
    for i, raw in enumerate(data["games"]):
        try:
            games.append(_parse_game(raw))
        except (RecordingError, KeyError, TypeError) as error:
            raise RecordingError(f"{path}: game {i}: {error!r}") from error
    return games


def load_all(paths: list[Path]) -> list[RecordedGame]:
    """Load several files, numbered ones in numeric order so that games keep their positions as files are added.
    Re-downloads overlap, so games repeated across files count once."""
    ordered = sorted(paths, key=_natural_key)
    return list(dict.fromkeys(game for path in ordered for game in load_games(path)))


def _natural_key(path: str | Path) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def summarize(games: list[RecordedGame]) -> str:
    lines = [f"{len(games)} games"]
    for role, name in ((KOMIKELU, "komikelu"), (TREWAS, "trewas")):
        played = [g for g in games if len(g.human_roles) == 1 and g.human_roles[0] == role]
        if played:
            won = sum(g.outcome == role for g in played)
            drawn = sum(g.outcome == DRAW for g in played)
            lost = len(played) - won - drawn
            moves = sum(sum(1 for _ in g.human_moves()) for g in played)
            lines.append(f"you as {name} vs agent: {won} won, {drawn} drawn, {lost} lost, {moves} moves")
    two_player = [g for g in games if len(g.human_roles) == 2]
    if two_player:
        moves = sum(len(g.actions) for g in two_player)
        lines.append(f"two-player: {len(two_player)} games, {moves} moves")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check recorded games and summarize them.")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        print(summarize(load_all(args.files)))
    except RecordingError as error:
        raise SystemExit(f"Invalid recording: {error}") from error


if __name__ == "__main__":
    main()
