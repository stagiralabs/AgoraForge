import torch
import torch.nn as nn
import torch.nn.functional as F

from agoraforge.training.ppo import index_staged, ppo_minibatch_loss


class _IdentityValueModel(nn.Module):
    def normalize_value(self, value):
        return value


def test_centralized_ppo_indexing_groups_action_heads_by_env():
    staged = {
        "actor_obs": {"formula_mask": torch.tensor([[1.0], [2.0]]),
                      "agent_scalars": torch.tensor([[10.0], [20.0]])},
        "old_log_probs": torch.arange(6, dtype=torch.float32),
        "advantages": torch.tensor([0.5, 1.5]),
        "returns": torch.tensor([2.0, 3.0]),
    }

    mb = index_staged(
        staged,
        agent_idx=torch.tensor([3, 4, 5]),
        env_idx=torch.tensor([1]),
        n_env=2,
    )

    assert mb["actor_obs"]["formula_mask"].tolist() == [[2.0]]
    assert mb["actor_obs"]["agent_scalars"].tolist() == [[20.0]]
    assert mb["old_log_probs"].tolist() == [3.0, 4.0, 5.0]
    assert mb["advantages"].tolist() == [1.5]
    assert mb["returns"].tolist() == [3.0]


def test_shared_ppo_loss_matches_population_objective():
    model = _IdentityValueModel()
    new_log_probs = torch.tensor([0.1, -0.2])
    new_kls = torch.tensor([0.3, 0.4])
    values = torch.tensor([0.2, -0.1])
    minibatch = {
        "old_log_probs": torch.tensor([0.0, -0.1]),
        "advantages": torch.tensor([1.0, -0.5]),
        "returns": torch.tensor([0.4, 0.3]),
    }

    losses = ppo_minibatch_loss(
        model,
        {},
        values,
        minibatch,
        lambda _logits, _mb: (new_log_probs, new_kls),
        group=1,
        clip_eps=0.2,
        kl_coeff=0.05,
        critic_huber_beta=0.4,
    )

    ratios = torch.exp(new_log_probs - minibatch["old_log_probs"])
    expected_actor = -torch.minimum(
        ratios * minibatch["advantages"],
        ratios.clamp(0.8, 1.2) * minibatch["advantages"],
    ).mean() + 0.05 * new_kls.mean()
    expected_critic = F.smooth_l1_loss(
        values, minibatch["returns"], beta=0.4)
    assert torch.allclose(losses.actor_loss, expected_actor)
    assert torch.allclose(losses.critic_loss, expected_critic)
    assert torch.allclose(losses.loss, expected_actor + expected_critic)


def test_shared_ppo_loss_forms_one_centralized_joint_ratio():
    model = _IdentityValueModel()
    head_log_probs = torch.tensor([0.1, 0.2, -0.2, 0.1])
    minibatch = {
        "old_log_probs": torch.zeros(4),
        "advantages": torch.tensor([1.0, -1.0]),
        "returns": torch.zeros(2),
    }

    losses = ppo_minibatch_loss(
        model,
        {},
        torch.zeros(2),
        minibatch,
        lambda _logits, _mb: (head_log_probs, torch.zeros(4)),
        group=2,
        clip_eps=0.2,
        kl_coeff=0.0,
        critic_huber_beta=0.4,
    )

    assert torch.allclose(losses.log_ratios, torch.tensor([0.3, -0.1]))
