import numpy as np
import pytest
import torch

from komikan.bots import RandomBot
from komikan.env import KomikanEnv, action_mask, encode_observation
from komikan.game import (
    DEFAULT_RULES,
    KOMIKELU,
    STOP,
    TREWAS,
    apply_action,
    initial_state,
    legal_actions,
    mirror_action,
    mirror_state,
    outcome,
)
from komikan.pretrain import (
    Examples,
    accuracy,
    behaviour_clone,
    build_examples,
    cross_validate,
    load_ppo,
    random_guess_accuracy,
    randomize_actions,
    set_actor_frozen,
)
from komikan.recordings import RecordedGame
from komikan.train import make_ppo


def play(seed: int, human_roles=(TREWAS,)) -> RecordedGame:
    komikelu, trewas = RandomBot(seed), RandomBot(seed + 1)
    state, actions = initial_state(), []
    while outcome(state) is None:
        action = (komikelu if state.to_move == KOMIKELU else trewas)(state)
        actions.append(action)
        state = apply_action(state, action)
    return RecordedGame(tuple(actions), tuple(human_roles), outcome(state), None, DEFAULT_RULES)


def test_initial_position_is_its_own_mirror():
    assert mirror_state(initial_state()) == initial_state()


def test_mirroring_preserves_the_rules_along_real_games():
    for state, action in play(0).replay():
        reflected = mirror_state(state)
        assert sorted(mirror_action(a) for a in legal_actions(state)) == sorted(legal_actions(reflected))
        assert mirror_state(reflected) == state
        assert mirror_action(mirror_action(action)) == action
        assert apply_action(reflected, mirror_action(action)) == mirror_state(apply_action(state, action))


def test_stop_mirrors_to_itself():
    assert mirror_action(STOP) == STOP


def test_examples_cover_the_humans_moves_and_their_mirrors():
    game = play(0)
    examples = build_examples([game], TREWAS)
    moves = list(game.human_moves())
    assert len(examples) == 2 * len(moves)
    assert all(examples.masks[i, examples.actions[i]] for i in range(len(examples)))
    assert np.array_equal(examples.observations[0], encode_observation(moves[0][0]))
    assert np.array_equal(examples.masks[1], action_mask(mirror_state(moves[0][0])))


def test_examples_are_only_from_the_requested_role():
    assert len(build_examples([play(0)], TREWAS, mirror=False)) > 0
    with pytest.raises(ValueError):
        build_examples([play(0)], KOMIKELU)


def test_two_player_games_give_examples_for_each_role():
    game = play(0, human_roles=(KOMIKELU, TREWAS))
    total = len(build_examples([game], KOMIKELU, mirror=False)) + len(build_examples([game], TREWAS, mirror=False))
    assert total == len(game.actions)


def test_cloning_learns_the_examples_and_leaves_the_value_network_alone():
    examples = build_examples([play(0)], TREWAS)
    model = make_ppo(KomikanEnv(role="trewas"), seed=0)
    value_before = [p.clone() for p in model.policy.mlp_extractor.value_net.parameters()]
    before = accuracy(model.policy, examples)
    behaviour_clone(model.policy, examples, epochs=60)
    assert accuracy(model.policy, examples) > max(before + 0.3, 0.6)
    for old, new in zip(value_before, model.policy.mlp_extractor.value_net.parameters()):
        assert torch.equal(old, new)


def test_random_guess_accuracy_is_the_mean_chance():
    masks = np.zeros((2, 4), dtype=bool)
    masks[0, :2] = True
    masks[1, :4] = True
    examples = Examples(np.zeros((2, 3)), masks, np.zeros(2, dtype=int), np.zeros(2, dtype=int))
    assert random_guess_accuracy(examples) == pytest.approx(0.375)


def test_cross_validation_holds_out_whole_games():
    examples = build_examples([play(0), play(2)], TREWAS)
    results = cross_validate(examples, "trewas", epochs=1, seed=0)
    assert len(results) == 2 and all(0 <= r <= 1 for r in results)


def test_placebo_keeps_positions_but_replaces_moves_with_legal_ones():
    examples = build_examples([play(0)], TREWAS)
    placebo = randomize_actions(examples, seed=0)
    assert np.array_equal(placebo.observations, examples.observations)
    assert all(placebo.masks[i, placebo.actions[i]] for i in range(len(placebo)))
    assert (placebo.actions != examples.actions).mean() > 0.5


def test_runs_continued_from_one_model_differ_by_seed(tmp_path):
    make_ppo(KomikanEnv(role="trewas"), seed=99).save(tmp_path / "start.zip")

    def continue_with(seed):
        model = load_ppo(tmp_path / "start.zip", KomikanEnv(role="trewas"), seed, 1e-3, 0.01)
        assert model.seed == seed
        model.learn(1024)
        return next(model.policy.action_net.parameters()).detach().clone()

    assert torch.equal(continue_with(1), continue_with(1))
    assert not torch.equal(continue_with(1), continue_with(2))


def test_frozen_actor_is_unchanged_by_ppo_but_the_value_network_learns():
    model = make_ppo(KomikanEnv(role="trewas"), seed=0)
    set_actor_frozen(model.policy, True)
    actor = [p.clone() for p in model.policy.action_net.parameters()]
    value = [p.clone() for p in model.policy.value_net.parameters()]
    model.learn(1024)
    assert all(torch.equal(a, b) for a, b in zip(actor, model.policy.action_net.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(value, model.policy.value_net.parameters()))
    set_actor_frozen(model.policy, False)
    assert all(p.requires_grad for p in model.policy.action_net.parameters())
