# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum learning utilities for screwdriver manipulation."""

import torch

from .rewards import screwdriver_stability_reward, screwdriver_upright_reward


def _get_curriculum_stage(env) -> torch.Tensor:
    """Return per-env curriculum stage tensor, creating it if missing."""
    if not hasattr(env, "curriculum_stage"):
        env.curriculum_stage = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    return env.curriculum_stage


def init_curriculum_stage(env, env_ids: torch.Tensor = None) -> None:
    """Initialize per-env curriculum stages to 0."""
    stage = _get_curriculum_stage(env)
    if env_ids is None:
        stage.zero_()
    else:
        stage[env_ids] = 0


CURRICULUM_CHECK_INTERVAL = 5


def advance_curriculum_stage(env, env_ids: torch.Tensor = None) -> None:
    """Advance per-env curriculum stages based on performance."""
    if not hasattr(advance_curriculum_stage, "_counter"):
        advance_curriculum_stage._counter = 0
    advance_curriculum_stage._counter += 1
    if advance_curriculum_stage._counter < CURRICULUM_CHECK_INTERVAL:
        return
    advance_curriculum_stage._counter = 0

    stage = _get_curriculum_stage(env)
    upright_rew = screwdriver_upright_reward(env)
    stability_rew = screwdriver_stability_reward(env)

    promote_0_to_1 = (stage == 0) & (upright_rew > 0.95) & (stability_rew > 0.9)
    if torch.any(promote_0_to_1):
        stage[promote_0_to_1] = 1
