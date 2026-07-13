"""Centralized planner dynamics over shared resident env state."""

from __future__ import annotations

import torch


class CentralizedPlanner:
    def __init__(self, env):
        self.env = env

    def step(self, actions) -> torch.Tensor:
        """Planner step: math actions only, knowledge shared, team fitness reward."""
        env = self.env
        with env._prof_section("cstep_math_actions"):
            env._process_math_actions(actions)
        with env._prof_section("cstep_share_knowledge"):
            self.share_knowledge()
        with env._prof_section("cstep_share_query"):
            self.share_query_memory()
        with env._prof_section("cstep_record_resolutions"):
            env._record_target_resolutions()
        with env._prof_section("cstep_reward"):
            rewards = self.reward()
        return rewards.unsqueeze(-1)

    def share_knowledge(self) -> None:
        env = self.env
        env.public_resolved |= env.resolved.any(dim=1)
        env.public_concrete |= env.concrete.any(dim=1)
        env.public_concrete |= env._gather_neg(env.public_resolved)
        env.resolved |= env.public_resolved.unsqueeze(1)
        env.concrete |= env.public_concrete.unsqueeze(1)

    def share_query_memory(self) -> None:
        env = self.env
        known = env.query_prob_known.any(dim=1, keepdim=True)
        prob = torch.where(env.query_prob_known, env.query_prob, torch.full_like(env.query_prob, -1.0))
        prob = prob.max(dim=1, keepdim=True).values
        time = torch.where(env.query_prob_known, env.query_time, torch.zeros_like(env.query_time))
        time = time.max(dim=1, keepdim=True).values
        env.query_prob = torch.where(known, prob, env.query_prob).expand(env.B, env.A, env.F).clone()
        env.query_time = torch.where(known, time, env.query_time).expand(env.B, env.A, env.F).clone()
        env.query_prob_known = known.expand(env.B, env.A, env.F).clone()
        env.query_related_in_known = (
            env.query_related_in_known.any(dim=1, keepdim=True).expand(env.B, env.A, env.F).clone()
        )
        env.query_related_out_known = (
            env.query_related_out_known.any(dim=1, keepdim=True).expand(env.B, env.A, env.F).clone()
        )

    def reward(self) -> torch.Tensor:
        env = self.env
        tau = float(env.cfg.fitness_tau)
        newly = env.target_theorem & (env.target_resolution_step == env.timestep.unsqueeze(1))
        contrib = torch.exp(-env.timestep.to(torch.float32) / tau).unsqueeze(1) * newly.to(torch.float32)
        team = contrib.sum(dim=1) / env.n_targets
        return team.unsqueeze(1).expand(env.B, env.A)
