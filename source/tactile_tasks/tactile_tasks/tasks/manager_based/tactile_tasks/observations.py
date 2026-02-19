# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions and configurations for screwdriver manipulation."""

import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from .constants import ALLEGRO_FINGER_JOINT_NAMES
from .screwdriver_utils import get_camera_point_cloud


def joint_pos_in_order(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return joint positions of the asset in configured order."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    return asset.data.joint_pos[:, joint_ids]


def screwdriver_pose(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")
) -> torch.Tensor:
    """Screwdriver pose (position + quaternion) in the environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w - env.scene.env_origins
    quat = asset.data.root_quat_w
    return torch.cat([pos, quat], dim=-1)


def screwdriver_orientation_z_axis(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")
) -> torch.Tensor:
    """Z-axis direction of the screwdriver in world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    rot_matrix = matrix_from_quat(quat)
    return rot_matrix[:, :, 2]


def screwdriver_yaw_angle_from_quaternion(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")
) -> torch.Tensor:
    """Compute yaw angle of the screwdriver relative to its initial orientation."""
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    R = matrix_from_quat(quat)

    if not hasattr(env, "_screwdriver_init_quat"):
        env._screwdriver_init_quat = quat.clone()

    R_init = matrix_from_quat(env._screwdriver_init_quat)
    R_rel = torch.bmm(R, R_init.transpose(-2, -1))
    return torch.atan2(R_rel[:, 1, 0], R_rel[:, 0, 0])


def screwdriver_angular_velocity_z(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")
) -> torch.Tensor:
    """Angular velocity around screwdriver's z-axis computed from yaw angle differences."""
    if not hasattr(env, "_screwdriver_prev_yaw"):
        env._screwdriver_prev_yaw = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_screwdriver_last_yaw_seen"):
        env._screwdriver_last_yaw_seen = None

    current_yaw = screwdriver_yaw_angle_from_quaternion(env, asset_cfg=asset_cfg)
    is_new_step = False
    if env._screwdriver_last_yaw_seen is None:
        is_new_step = True
    else:
        yaw_diff = torch.abs(current_yaw - env._screwdriver_last_yaw_seen)
        if torch.any(yaw_diff > 1e-5):
            is_new_step = True
            env._screwdriver_prev_yaw = env._screwdriver_last_yaw_seen.clone()

    yaw_diff = current_yaw - env._screwdriver_prev_yaw
    yaw_diff = yaw_diff - 2 * math.pi * torch.round(yaw_diff / (2 * math.pi))
    dt = env.physics_dt
    yaw_vel = yaw_diff / dt
    env._screwdriver_last_yaw_seen = current_yaw.clone()
    return yaw_vel.unsqueeze(-1)


def reset_screwdriver_yaw_tracking(env, env_ids: torch.Tensor = None) -> None:
    """Reset yaw tracking state for specified environments."""
    asset: RigidObject = env.scene["screwdriver"]
    quat = asset.data.root_quat_w

    if not hasattr(env, "_screwdriver_init_quat"):
        env._screwdriver_init_quat = quat.clone()
        env._screwdriver_prev_yaw = torch.zeros(env.num_envs, device=env.device)
        env._screwdriver_last_yaw_seen = None

    if env_ids is None:
        env._screwdriver_init_quat = quat.clone()
        env._screwdriver_prev_yaw = torch.zeros(env.num_envs, device=env.device)
        env._screwdriver_last_yaw_seen = None
    else:
        env._screwdriver_init_quat[env_ids] = quat[env_ids].clone()
        env._screwdriver_prev_yaw[env_ids] = 0.0
        if env._screwdriver_last_yaw_seen is not None:
            env._screwdriver_last_yaw_seen[env_ids] = 0.0
        else:
            env._screwdriver_last_yaw_seen = torch.zeros(env.num_envs, device=env.device)
            env._screwdriver_last_yaw_seen[env_ids] = 0.0


def contact_forces_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Observation function for contact forces from contact sensors."""
    sensor = env.scene[sensor_cfg.name]
    data = sensor.data

    if hasattr(data, "net_forces_w"):
        forces = data.net_forces_w
    elif hasattr(data, "forces_w"):
        forces = data.forces_w
    else:
        num_envs = env.num_envs
        num_bodies = 4
        forces = torch.zeros((num_envs, num_bodies, 3), device=env.device)

    return torch.clamp(forces.flatten(start_dim=1), -5.0, 5.0)


def point_cloud_obs(
    env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera")
) -> torch.Tensor:
    """Observation function for point cloud from tiled camera."""
    point_cloud = get_camera_point_cloud(env, env_frame=True, asset_cfg=sensor_cfg)
    return point_cloud.reshape(point_cloud.shape[0], -1)


from isaaclab.utils import configclass

JOINT_POS_OBS_TERM = ObsTerm(
    func=joint_pos_in_order,
    noise=None,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALLEGRO_FINGER_JOINT_NAMES)},
)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = JOINT_POS_OBS_TERM
        screwdriver_pose = ObsTerm(
            func=screwdriver_pose,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )
        screwdriver_orientation_z = ObsTerm(
            func=screwdriver_orientation_z_axis,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )
        screwdriver_angular_velocity_z = ObsTerm(
            func=screwdriver_angular_velocity_z,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ContactObservationCfg:
    """Observation specifications with contact sensor data."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = JOINT_POS_OBS_TERM
        screwdriver_pose = ObsTerm(
            func=screwdriver_pose,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )
        screwdriver_orientation_z = ObsTerm(
            func=screwdriver_orientation_z_axis,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )
        screwdriver_angular_velocity_z = ObsTerm(
            func=screwdriver_angular_velocity_z,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )
        contact_forces = ObsTerm(
            func=contact_forces_obs,
            noise=None,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PointCloudObservationCfg:
    """Observation specifications with point cloud and contact data."""

    @configclass
    class PolicyCfg(ObsGroup):
        point_cloud = ObsTerm(
            func=point_cloud_obs,
            noise=None,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera")},
        )
        joint_pos = JOINT_POS_OBS_TERM
        screwdriver_pose = ObsTerm(
            func=screwdriver_pose,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )
        screwdriver_orientation_z = ObsTerm(
            func=screwdriver_orientation_z_axis,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )
        screwdriver_angular_velocity_z = ObsTerm(
            func=screwdriver_angular_velocity_z,
            noise=None,
            params={"asset_cfg": SceneEntityCfg("screwdriver")},
        )
        contact_forces = ObsTerm(
            func=contact_forces_obs,
            noise=None,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
