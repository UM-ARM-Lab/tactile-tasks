# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for screwdriver manipulation."""

import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from .constants import ALLEGRO_FINGER_JOINT_NAMES
from .observations import screwdriver_angular_velocity_z


def screwdriver_upright_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")
) -> torch.Tensor:
    """Reward for keeping screwdriver upright (z-axis aligned with world z-axis).
    Zero reward if tilted >20°; squared scale from threshold to 1 for smooth gradient.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    z_axis = rot_matrix[:, :, 2]
    upright_alignment = z_axis[:, 2]  # Dot with world Z [0,0,1]
    threshold_cos_20_deg = math.cos(math.radians(20.0))
    upright_reward = torch.where(
        upright_alignment >= threshold_cos_20_deg,
        ((upright_alignment - threshold_cos_20_deg) / (1.0 - threshold_cos_20_deg)) ** 2,
        torch.zeros_like(upright_alignment),
    )
    return upright_reward


def screwdriver_tilt_exceeds(
    env,
    threshold_deg: float = 25.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"),
) -> torch.Tensor:
    """Return per-env boolean: True when tilt exceeds threshold (for termination).
    cos(theta) < cos(threshold) means angle > threshold.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    cos_theta = torch.clamp(rot_matrix[:, :, 2][:, 2], -1.0, 1.0)
    threshold_cos = math.cos(math.radians(threshold_deg))
    return cos_theta < threshold_cos


def screwdriver_signed_yaw_velocity_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"),
    *,
    degrees: bool = False,
    vmax: float | None = None,
) -> torch.Tensor:
    """Reward negative (clockwise) yaw velocity, penalize positive (counter-clockwise).
    Clockwise = screwing in; only rewards when upright (within ~20°) to avoid wobble.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_vel = screwdriver_angular_velocity_z(env, asset_cfg=asset_cfg).squeeze(-1)
    if degrees:
        yaw_vel = yaw_vel * (180.0 / math.pi)
    base_reward = -torch.clamp(yaw_vel, -5.0, 5.0)
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    z_axis = rot_matrix[:, :, 2]
    upright_alignment = z_axis[:, 2]
    threshold_cos_20_deg = math.cos(math.radians(20.0))
    upright_mask = (upright_alignment >= threshold_cos_20_deg).float()  # Zero reward if tilted
    return base_reward * upright_mask


def screwdriver_stability_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"), degrees: bool = False
) -> torch.Tensor:
    """Penalize local x/y angular velocity (wobbling); reward pure Z-axis spin.
    Transform ang_vel to screwdriver frame, take xy components, return -||w_xy||^2.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_w = asset.data.root_ang_vel_w
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    ang_vel_local = torch.bmm(
        rot_matrix.transpose(-2, -1), ang_vel_w.unsqueeze(-1)
    ).squeeze(-1)
    vel_xy = ang_vel_local[:, :2]  # Tilt/wobble; Z is desired spin
    if degrees:
        vel_xy = vel_xy * (180.0 / math.pi)
    return -torch.sum(vel_xy * vel_xy, dim=1)


def torque_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
    weight: float = 1e-2,
) -> torch.Tensor:
    """Penalize squared actuator torques per env."""
    asset: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        joint_ids = torch.arange(asset.num_joints, device=env.device)
    else:
        joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    tau = asset.data.applied_torque[:, joint_ids]
    return -weight * torch.sum(tau * tau, dim=1)


def energy_penalty_abs(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
    scale: float = 5e-2,
) -> torch.Tensor:
    """Penalize per-step mechanical energy: -scale * sum(|tau * qd|) * dt.
    Power = tau * qd; |power| avoids canceling positive/negative work.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        joint_ids = torch.arange(asset.num_joints, device=env.device)
    else:
        joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    tau = asset.data.applied_torque[:, joint_ids]
    qd = asset.data.joint_vel[:, joint_ids]
    power_abs = torch.sum(torch.abs(tau * qd), dim=1)
    dt = env.physics_dt
    return -scale * power_abs * dt


def finger_joint_deviation_penalty(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), degrees: bool = False
) -> torch.Tensor:
    """Penalty for finger joints deviating from initial positions.
    Encourages finger gaiting: small penalty nudges policy to move fingers for rotation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    current_joint_pos = asset.data.joint_pos[:, joint_ids]
    initial_joint_pos = asset.data.default_joint_pos[:, joint_ids]
    delta = current_joint_pos - initial_joint_pos
    if degrees:
        delta = delta * (180.0 / math.pi)
    joint_deviation = torch.sum(delta * delta, dim=1)
    return -joint_deviation


def add_action_noise(env, std: float = 0.02):
    """Add Gaussian noise to actions (pre_physics_step event) for exploration."""
    a = env.action_manager.action
    env.action_manager.action = torch.clamp(a + std * torch.randn_like(a), -1.0, 1.0)
