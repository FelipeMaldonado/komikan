"""Self-play training: search values, the recorded samples, mirroring and a short training run."""

import numpy as np
import pytest

from komikan.bots import make_bot
from komikan.env import OBS_SIZE, action_mask, encode_observation
from komikan.game import (
    DRAW,
    KOMIKELU,
    NUM_ACTIONS,
    TREWAS,
    apply_action,
    initial_state,
    legal_actions,
    mirror_action,
    mirror_state,
    outcome,
)
from komikan.policy import NumpyPolicy
from komikan.puct import Networks
from komikan.selfplay import (
    AlphaZeroBot,
    Learner,
    ReplayBuffer,
    Sample,
    _mirror_target,
    play_game,
    result_value,
)


def _net(outputs: int, bias: float = 0.0) -> NumpyPolicy:
    return NumpyPolicy([np.zeros((4, OBS_SIZE)), np.zeros((outputs, 4))], [np.zeros(4), np.full(outputs, bias)])


def _networks(komikelu_value: float = 0.0, trewas_value: float = 0.0) -> Networks:
    return Networks(
        {KOMIKELU: _net(NUM_ACTIONS), TREWAS: _net(NUM_ACTIONS)},
        {KOMIKELU: _net(1, komikelu_value), TREWAS: _net(1, trewas_value)},
    )


def test_result_values_are_from_the_komikelus_point_of_view():
    assert result_value(KOMIKELU, 0.5) == 1.0
    assert result_value(TREWAS, 0.5) == -1.0
    assert result_value(DRAW, 0.5) == 0.5


def test_a_leaf_is_judged_by_the_network_of_the_side_to_move_and_clamped():
    bot = AlphaZeroBot(_networks(komikelu_value=0.25, trewas_value=-3.0), simulations=1)
    komikelu_to_move = initial_state()
    assert komikelu_to_move.to_move == KOMIKELU
    assert bot._network_value(komikelu_to_move, None) == 0.25
    trewas_to_move = apply_action(komikelu_to_move, legal_actions(komikelu_to_move)[0])
    assert trewas_to_move.to_move == TREWAS
    assert bot._network_value(trewas_to_move, None) == -1.0


def test_final_results_outrank_the_networks():
    bot = AlphaZeroBot(_networks(komikelu_value=0.25), simulations=1, draw=0.5)
    assert bot._network_value(initial_state(), TREWAS) == -1.0
    assert bot._network_value(initial_state(), DRAW) == 0.5


def test_root_noise_changes_priors_only_when_asked():
    state = initial_state()
    plain = AlphaZeroBot(_networks(), simulations=2, seed=1)
    noisy = AlphaZeroBot(_networks(), simulations=2, root_noise=True, seed=1)
    assert len(set(plain.search(state).priors)) == 1
    root = noisy.search(state)
    assert len(set(root.priors)) > 1
    assert sum(root.priors) == pytest.approx(1.0)


def test_mirroring_a_target_moves_each_share_to_the_mirrored_action():
    state = initial_state()
    actions = legal_actions(state)
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target[actions[0]], target[actions[1]] = 0.75, 0.25
    mirrored = _mirror_target(target)
    assert mirrored[mirror_action(actions[0])] == 0.75
    assert mirrored[mirror_action(actions[1])] == 0.25
    assert mirrored.sum() == 1.0
    assert set(np.flatnonzero(mirrored)) <= set(legal_actions(mirror_state(state)))


def test_game_samples_are_consistent():
    start = initial_state()._replace(plies=180)  # a short game: the turn limit ends it
    bot = AlphaZeroBot(_networks(), simulations=8, root_noise=True, seed=3)
    samples, result, final = play_game(bot, start, seed=3, temperature_turns=190)
    assert result == outcome(final)
    assert samples and len(samples) % 2 == 0  # each position and its mirror
    for sample in samples:
        assert sample.value == result_value(result, bot.draw)
        assert sample.target.sum() == pytest.approx(1.0)
        assert not sample.target[~sample.mask].any()  # visits only on legal moves
    assert {sample.role for sample in samples} == {KOMIKELU, TREWAS}


def test_recorded_mask_and_observation_belong_to_the_same_position():
    start = initial_state()._replace(plies=196)
    samples, _, _ = play_game(AlphaZeroBot(_networks(), simulations=4, seed=1), start, seed=1)
    first = samples[0]
    assert (first.obs == encode_observation(start)).all()
    assert (first.mask == action_mask(start)).all()


def test_games_follow_the_seed():
    def play():
        bot = AlphaZeroBot(_networks(), simulations=6, root_noise=True, seed=5)
        return play_game(bot, initial_state()._replace(plies=190), seed=5)[2]

    assert play() == play()


def test_search_uses_real_results_over_the_network():
    state = initial_state()._replace(plies=199)  # the turn limit gives a draw whatever the komikelu does
    root = AlphaZeroBot(_networks(komikelu_value=-1.0), simulations=30, seed=0).search(state)
    assert sum(root.visits) == 30
    assert root.total / root.count == pytest.approx(0.5)  # every line is the terminal draw, not the network's -1


def test_self_play_follows_the_variant():
    from komikan.board import COORDS
    from komikan.game import ACTIONS, RULE_VARIANTS

    rules = RULE_VARIANTS["no-backward"]
    bot = AlphaZeroBot(_networks(), simulations=6, root_noise=True, seed=2, rules=rules)
    samples, _, _ = play_game(bot, initial_state()._replace(plies=170), seed=2, rules=rules)
    for sample in samples:
        if sample.role == TREWAS:
            for action in np.flatnonzero(sample.mask):
                move = ACTIONS[action]
                assert COORDS[move.dst][1] <= COORDS[move.src][1]


def test_late_starts_come_from_the_end_of_trewas_wins(tmp_path):
    import json
    import random

    from komikan.selfplay import late_games, late_reach, late_start

    def random_game(seed):
        rng, state, actions = random.Random(seed), initial_state(), []
        while outcome(state) is None:
            actions.append(rng.choice(legal_actions(state)))
            state = apply_action(state, actions[-1])
        return actions, outcome(state)

    win, result = random_game(1193)  # a rare random game that the trewas win
    assert result == TREWAS
    path = tmp_path / "wins.json"
    path.write_text(json.dumps([{"actions": win}]))
    games = late_games(path)
    assert len(games) == 1 and len(games[0]) == len(win)

    rng = random.Random(0)
    for _ in range(50):
        start = late_start(games, 3, rng)
        assert start in games[0][-3:]
        assert outcome(start) is None

    assert late_reach(0, 30, 10, 200) == 10
    assert late_reach(29, 30, 10, 200) == 200
    assert 10 < late_reach(15, 30, 10, 200) < 200

    lost = tmp_path / "lost.json"
    other, result = random_game(0)
    assert result != TREWAS
    lost.write_text(json.dumps([{"actions": other}]))
    with pytest.raises(ValueError):
        late_games(lost)


def test_buffer_keeps_the_most_recent_samples():
    buffer = ReplayBuffer(2)
    blank = lambda value: Sample(KOMIKELU, np.zeros(OBS_SIZE), np.ones(NUM_ACTIONS, bool), np.ones(NUM_ACTIONS) / NUM_ACTIONS, value)
    for value in (1.0, 2.0, 3.0):
        buffer.add(blank(value))
    assert buffer.size == 2
    assert sorted(buffer.values) == [2.0, 3.0]


def test_learner_fits_targets_and_saves_networks_the_search_can_load(tmp_path):
    buffer = ReplayBuffer(4)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[3, 7]] = True
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target[7] = 1.0
    for _ in range(4):
        buffer.add(Sample(KOMIKELU, encode_observation(initial_state()), mask, target, 0.5))
    learner = Learner((16,), 1e-2, seed=0)
    learner.train(buffer, 200, 4)
    learner.save(tmp_path, "komikelu")
    learner.save(tmp_path, "trewas")

    obs = encode_observation(initial_state())
    logits = NumpyPolicy.load(tmp_path / "komikelu.npz").logits(obs)
    assert logits[7] > logits[3]
    assert NumpyPolicy.load(tmp_path / "komikelu_value.npz").logits(obs)[0] == pytest.approx(0.5, abs=0.05)

    bot = make_bot(f"az:8:{tmp_path}", seed=0)
    assert bot(initial_state()) in legal_actions(initial_state())
