"""Guided search: signs, turn handling and the shaping correction, with small fixed networks."""

import numpy as np
import pytest

from komikan.env import OBS_SIZE, KomikanEnv, RewardConfig, trewas_potential
from komikan.game import (
    ACTIONS,
    KOMIKELU,
    NUM_ACTIONS,
    TREWAS,
    GameState,
    apply_action,
    bitmask,
    initial_state,
    legal_actions,
    outcome,
)
from komikan.policy import NumpyPolicy
from komikan.puct import MODES, Networks, PUCTBot


def _net(outputs: int, bias: np.ndarray | float = 0.0) -> NumpyPolicy:
    """A network with constant output."""
    return NumpyPolicy([np.zeros((4, OBS_SIZE)), np.zeros((outputs, 4))], [np.zeros(4), np.full(outputs, 0.0) + bias])


def _networks(policy_bias=0.0, value=0.0) -> Networks:
    return Networks(
        {KOMIKELU: _net(NUM_ACTIONS, policy_bias), TREWAS: _net(NUM_ACTIONS, policy_bias)},
        {KOMIKELU: _net(1, value), TREWAS: _net(1, value)},
    )


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_plays_legal_moves_for_both_roles(mode):
    bot = PUCTBot(_networks(), simulations=30, mode=mode, seed=0)
    state = initial_state()
    for _ in range(10):
        action = bot(state)
        assert action in legal_actions(state)
        state = apply_action(state, action)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        PUCTBot(_networks(), mode="other")


def test_takes_an_immediate_win_whatever_the_prior_says():
    src, dst, over = next(a for a in ACTIONS if a.over is not None)
    state = GameState(src, bitmask([over]), KOMIKELU, False, 8, 50)
    winning = ACTIONS.index(ACTIONS[next(i for i, a in enumerate(ACTIONS) if a == (src, dst, over))])
    bias = np.zeros(NUM_ACTIONS)
    bias[winning] = -50.0  # the network hates the winning move
    action = PUCTBot(_networks(bias), simulations=20, mode="both", seed=0)(state)
    assert outcome(apply_action(state, action)) == KOMIKELU


def test_strong_prior_gets_the_visits_with_a_flat_value():
    state = initial_state()
    favourite = legal_actions(state)[-1]
    bias = np.zeros(NUM_ACTIONS)
    bias[favourite] = 10.0
    assert PUCTBot(_networks(bias), simulations=50, mode="both", seed=0)(state) == favourite


def test_values_are_stored_from_the_komikelus_point_of_view():
    state = apply_action(initial_state(), legal_actions(initial_state())[0])
    assert state.to_move == TREWAS
    good_for_komikelu = PUCTBot(_networks(value=0.5), mode="rawvalue", seed=0)
    good_for_komikelu.role = KOMIKELU
    assert good_for_komikelu._network_value(state, None) == pytest.approx(0.5)
    good_for_trewas = PUCTBot(_networks(value=0.5), mode="rawvalue", seed=0)
    good_for_trewas.role = TREWAS
    assert good_for_trewas._network_value(state, None) == pytest.approx(-0.5)


def test_leaf_with_the_opponent_to_move_is_advanced_to_the_roles_turn():
    seen = []

    class Recording(NumpyPolicy):
        def logits(self, obs):
            seen.append(obs[-1])
            return super().logits(obs)

    networks = _networks()
    value = _net(1)
    networks.values[TREWAS] = Recording(value.weights, value.biases)
    bot = PUCTBot(networks, mode="rawvalue", seed=0)
    bot.role = TREWAS
    start = initial_state()  # komikelu to move, 0 turns played
    assert start.to_move == KOMIKELU
    bot._network_value(start, None)
    assert seen == [pytest.approx(1 / 200)]  # asked only after the komikelu's reply


def test_terminal_values_follow_the_result():
    state = initial_state()._replace(captured=9)
    bot = PUCTBot(_networks(), mode="rawvalue", seed=0)
    bot.role = TREWAS
    assert bot._network_value(state, KOMIKELU) == pytest.approx(RewardConfig().win)
    bot.role = KOMIKELU
    assert bot._network_value(state, KOMIKELU) == pytest.approx(RewardConfig().win)


def test_shaping_correction_makes_values_add_up_along_an_episode():
    """Corrected value = predicted future + reward so far, so it matches the env's actual rewards."""
    env = KomikanEnv(role="trewas", opponents=["random"])
    env.reset(seed=1)
    start = env.state
    earned = 0.0
    for _ in range(12):
        _, reward, done, _, _ = env.step(legal_actions(env.state)[0])
        if done:
            pytest.skip("episode ended early")
        earned += reward
    bot = PUCTBot(_networks(), mode="value", seed=0)
    bot.role = TREWAS
    first = bot._return_so_far(start, None)
    assert bot._return_so_far(env.state, None) - first == pytest.approx(earned)


def test_potential_is_unchanged_by_moving_it_out_of_the_env():
    env = KomikanEnv(role="trewas")
    assert env._trewas_potential(initial_state()) == trewas_potential(initial_state())
