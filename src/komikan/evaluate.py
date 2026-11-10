"""Play matches between bots and print win rates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .bots import Bot, make_bot
from .game import DEFAULT_RULES, DRAW, KOMIKELU, TREWAS, GameState, Rules, apply_action, initial_state, outcome


@dataclass
class MatchResult:
    komikelu_wins: int = 0
    trewas_wins: int = 0
    draws: int = 0
    captures: int = 0
    plies: int = 0

    @property
    def games(self) -> int:
        return self.komikelu_wins + self.trewas_wins + self.draws

    def score(self, role: int) -> float:
        """Wins minus losses per game for `role`, so a safe draw earns nothing."""
        margin = self.komikelu_wins - self.trewas_wins
        return (margin if role == KOMIKELU else -margin) / self.games

    def __str__(self) -> str:
        n = self.games
        return (
            f"komikelu {self.komikelu_wins / n:6.1%}  trewas {self.trewas_wins / n:6.1%}  draw {self.draws / n:6.1%}"
            f"  | avg captures {self.captures / n:4.1f}  avg turns {self.plies / n:5.1f}"
        )


def play_game(komikelu: Bot, trewas: Bot, rules: Rules = DEFAULT_RULES, start: GameState | None = None) -> tuple[int, int, int]:
    """Returns (outcome, captures, turns). Plays from `start` if given, else from the initial position."""
    state = start or initial_state()
    while (result := outcome(state, rules)) is None:
        bot = komikelu if state.to_move == KOMIKELU else trewas
        state = apply_action(state, bot(state), rules)
    return result, state.captured, state.plies


def play_match(komikelu: Bot, trewas: Bot, games: int, rules: Rules = DEFAULT_RULES) -> MatchResult:
    match = MatchResult()
    for _ in range(games):
        result, captures, plies = play_game(komikelu, trewas, rules)
        match.komikelu_wins += result == KOMIKELU
        match.trewas_wins += result == TREWAS
        match.draws += result == DRAW
        match.captures += captures
        match.plies += plies
    return match


def main() -> None:
    parser = argparse.ArgumentParser(description="Play bots against each other, every komikelu against every trewas side.")
    parser.add_argument("--komikelu", nargs="+", required=True, help='bot specs: "random", "minimax:2", "policy:models/komikelu.npz"')
    parser.add_argument("--trewas", nargs="+", required=True, help="bot specs, as above")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for k_spec in args.komikelu:
        for t_spec in args.trewas:
            result = play_match(make_bot(k_spec, args.seed), make_bot(t_spec, args.seed + 1), args.games)
            print(f"{k_spec:32} vs {t_spec:32} {result}")


if __name__ == "__main__":
    main()
