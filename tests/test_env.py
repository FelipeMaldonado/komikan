import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import komikan  # noqa: F401  (registers Komikan-v0)
from komikan.board import NUM_POINTS, sq
from komikan.bots import MinimaxBot, RandomBot
from komikan.env import OBS_SIZE, KomikanEnv, RewardConfig, encode_observation
from komikan.evaluate import MatchResult, play_match
from komikan.game import ACTIONS, DRAW, KOMIKELU, NUM_ACTIONS, TREWAS, Action, apply_action, initial_state, legal_actions
from komikan.policy import NumpyPolicy

ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}


@pytest.mark.parametrize("role", ["komikelu", "trewas"])
def test_passes_gymnasium_checker(role):
    check_env(KomikanEnv(role=role), skip_render_check=True)


def test_registered_id():
    env = gym.make("Komikan-v0", role="trewas")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)


def test_trewas_agent_starts_after_komikelu_has_moved():
    env = KomikanEnv(role="trewas")
    env.reset(seed=1)
    assert env.state.to_move == TREWAS
    assert env.state.plies == 1


@pytest.mark.parametrize("role", ["komikelu", "trewas"])
def test_random_masked_episodes_terminate_with_a_result(role):
    env = KomikanEnv(role=role)
    rng = np.random.default_rng(0)
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            mask = env.action_masks()
            assert mask.shape == (NUM_ACTIONS,) and mask.any()
            obs, reward, done, truncated, info = env.step(rng.choice(np.flatnonzero(mask)))
        assert info["outcome"] in (KOMIKELU, TREWAS, DRAW)
        assert not info.get("illegal")


def test_illegal_action_ends_episode_with_a_loss():
    env = KomikanEnv(role="komikelu")
    env.reset(seed=0)
    illegal = int(np.flatnonzero(~env.action_masks())[0])
    _, reward, done, _, info = env.step(illegal)
    assert done and reward == -1.0 and info["illegal"]


def test_observation_encodes_pieces():
    obs = encode_observation(initial_state())
    assert obs[:NUM_POINTS].sum() == 1
    assert obs[NUM_POINTS : 2 * NUM_POINTS].sum() == 12


def test_minimax_beats_random_as_komikelu():
    assert play_match(MinimaxBot(1, seed=0), RandomBot(1), 5).komikelu_wins == 5


def test_numpy_policy_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    policy = NumpyPolicy(
        [rng.normal(size=(8, OBS_SIZE)), rng.normal(size=(NUM_ACTIONS, 8))],
        [rng.normal(size=8), rng.normal(size=NUM_ACTIONS)],
    )
    path = tmp_path / "p.npz"
    policy.save(path)
    obs = encode_observation(initial_state())
    assert np.allclose(NumpyPolicy.load(path).logits(obs), policy.logits(obs))
    state = initial_state()
    bot = policy.as_bot(seed=0)
    assert bot(state) in legal_actions(state)
    assert policy.as_bot(deterministic=True)(state) in legal_actions(state)


def test_numpy_export_matches_sb3_logits(tmp_path):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("sb3_contrib")
    model = sb3.MaskablePPO(
        "MlpPolicy",
        KomikanEnv(),
        n_steps=32,
        policy_kwargs={"net_arch": [16, 16], "activation_fn": torch.nn.ReLU},
        device="cpu",
    )
    obs = encode_observation(initial_state())
    with torch.no_grad():
        latent = model.policy.mlp_extractor.forward_actor(torch.as_tensor(obs[None]))
        expected = model.policy.action_net(latent).numpy()[0]
    assert np.allclose(NumpyPolicy.from_sb3(model).logits(obs), expected, atol=1e-5)


def test_trewas_shaping_rewards_advancing_but_not_shuffling():
    env = KomikanEnv(role="trewas")
    env.reset(seed=0)
    start = initial_state()._replace(to_move=TREWAS)
    forward = apply_action(start, ACTION_ID[Action(sq(0, 2), sq(0, 1), None)])
    assert env._shaping(start, forward) > 0
    assert env._shaping(start, forward) == pytest.approx(-env._shaping(forward, start))


def test_a_draw_costs_trewas_more_than_komikelu():
    cfg = RewardConfig()
    assert cfg.loss < cfg.draw_trewas < cfg.draw_komikelu < 0


def test_score_gives_draws_no_credit():
    assert MatchResult(draws=10).score(TREWAS) == 0
    assert MatchResult(trewas_wins=3, komikelu_wins=1, draws=6).score(TREWAS) == pytest.approx(0.2)
    assert MatchResult(trewas_wins=3, komikelu_wins=1, draws=6).score(KOMIKELU) == pytest.approx(-0.2)


def test_relative_advance_pays_nothing_for_walking_past_the_komikelu():
    from komikan.board import COORDS, NUM_POINTS, NEIGHBORS
    from komikan.env import RewardConfig, trewas_potential
    from komikan.game import GameState, KOMIKELU, bitmask

    relative = RewardConfig(relative_advance=True, confine=0.0, corner=0.0)
    absolute = RewardConfig(confine=0.0, corner=0.0)
    komikelu = next(p for p in range(NUM_POINTS) if COORDS[p][1] == 6)
    above = next(p for p in range(NUM_POINTS) if COORDS[p][1] == 12)
    level = next(p for p in range(NUM_POINTS) if COORDS[p][1] == 6 and p != komikelu)
    below = next(p for p in range(NUM_POINTS) if COORDS[p][1] == -6)

    def state(trewa):
        return GameState(komikelu, bitmask([trewa]), KOMIKELU, False, 0, 10)

    # Moving down to the komikelu's level still pays, in both.
    assert trewas_potential(state(level), cfg=relative) > trewas_potential(state(above), cfg=relative)
    # Past it, only the absolute bonus keeps paying.
    assert trewas_potential(state(below), cfg=relative) == trewas_potential(state(level), cfg=relative)
    assert trewas_potential(state(below), cfg=absolute) > trewas_potential(state(level), cfg=absolute)


def test_reward_config_can_be_replaced_on_a_running_env():
    from komikan.env import KomikanEnv, RewardConfig
    from komikan.game import initial_state

    env = KomikanEnv(role="trewas")
    plain = env._trewas_potential(initial_state())
    env.reward_config = RewardConfig(advance=0.0, confine=2.0)
    assert env._trewas_potential(initial_state()) != plain
