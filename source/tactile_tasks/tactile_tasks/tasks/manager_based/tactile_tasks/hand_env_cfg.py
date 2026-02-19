# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment configuration for screwdriver manipulation tasks.

This module composes scene configs, actions, observations, rewards, terminations,
and events into full environment configurations. Implementation details live in:
  - scene_configs, screwdriver_utils, observations, rewards, curriculum
"""

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .constants import ALLEGRO_FINGER_JOINT_NAMES
from .curriculum import advance_curriculum_stage
from .observations import (
    ContactObservationCfg,
    ObservationsCfg,
    PointCloudObservationCfg,
    reset_screwdriver_yaw_tracking,
)
from .rewards import (
    add_action_noise,
    energy_penalty_abs,
    finger_joint_deviation_penalty,
    screwdriver_signed_yaw_velocity_reward,
    screwdriver_stability_reward,
    screwdriver_tilt_exceeds,
    torque_penalty,
)
from .scene_configs import ScrewdriverSceneCfg, ScrewdriverSceneWithCameraCfg
from .screwdriver_utils import (
    add_screwdriver_rotation_markers,
    randomize_screwdriver_geometry_prestartup,
    setup_screwdriver_tip_pivots,
)


@configclass
class ActionsCfg:
    """Action specifications: relative joint position control for 16 Allegro fingers."""

    hand_joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ALLEGRO_FINGER_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
        clip={name: (-2.0, 2.0) for name in ALLEGRO_FINGER_JOINT_NAMES},
    )


@configclass
class CurriculumCfg:
    """Curriculum learning configuration."""

    pass


@configclass
class EventCfg:
    """Configuration for environment events."""

    reset_hand_root_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )

    reset_hand_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALLEGRO_FINGER_JOINT_NAMES),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    add_action_noise = EventTerm(func=add_action_noise, mode="pre_physics_step", params={"std": 0.02})

    randomize_screwdriver_usd = EventTerm(
        func=randomize_screwdriver_geometry_prestartup,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )

    reset_screwdriver_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("screwdriver"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )

    reset_yaw_tracking = EventTerm(func=reset_screwdriver_yaw_tracking, mode="reset")

    create_tip_pivots = EventTerm(
        func=setup_screwdriver_tip_pivots,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )

    create_rotation_markers = EventTerm(
        func=add_screwdriver_rotation_markers,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )

    create_rotation_markers_startup = EventTerm(
        func=add_screwdriver_rotation_markers,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )

    curriculum_advancement = EventTerm(
        func=advance_curriculum_stage,
        mode="interval",
        interval_range_s=(5.0, 5.0),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    screwdriver_rotation = RewTerm(
        func=screwdriver_signed_yaw_velocity_reward,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("screwdriver"), "vmax": 4.0},
    )

    screwdriver_stability = RewTerm(
        func=screwdriver_stability_reward,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("screwdriver")},
    )

    finger_deviation_penalty = RewTerm(
        func=finger_joint_deviation_penalty,
        weight=8.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALLEGRO_FINGER_JOINT_NAMES)},
    )

    torque_mag_penalty = RewTerm(
        func=torque_penalty,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": None, "weight": 1e-3},
    )

    energy_penalty = RewTerm(
        func=energy_penalty_abs,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": None, "scale": 5e-4},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    screwdriver_tilt_limit = DoneTerm(
        func=screwdriver_tilt_exceeds,
        params={"threshold_deg": 20.0, "asset_cfg": SceneEntityCfg("screwdriver")},
    )


##
# Environment configurations
##


@configclass
class TurnScrewdriverEnvCfg(ManagerBasedRLEnvCfg):
    """Basic screwdriver rotation task (proprioceptive observations)."""

    scene = ScrewdriverSceneCfg(num_envs=256, env_spacing=4.0, clone_in_fabric=False)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (2.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "world"
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_position_iteration_count = 32
        self.sim.physx.solver_velocity_iteration_count = 8


@configclass
class TurnScrewdriverContactEnvCfg(ManagerBasedRLEnvCfg):
    """Contact-aware screwdriver manipulation (includes tactile/contact observations)."""

    scene = ScrewdriverSceneCfg(num_envs=256, env_spacing=4.0, clone_in_fabric=False)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    observations: ContactObservationCfg = ContactObservationCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (2.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "world"
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@configclass
class TurnScrewdriverPointCloudEnvCfg(ManagerBasedRLEnvCfg):
    """Point cloud-based screwdriver manipulation (visual perception)."""

    scene = ScrewdriverSceneWithCameraCfg(num_envs=256, env_spacing=4.0, clone_in_fabric=False)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    observations: PointCloudObservationCfg = PointCloudObservationCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (2.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "world"
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_position_iteration_count = 16
        self.sim.physx.solver_velocity_iteration_count = 4


# Backward compatibility aliases (old names still work for scripts)
TestEnvCfg = TurnScrewdriverEnvCfg
TestContactEnvCfg = TurnScrewdriverContactEnvCfg
TestPointCloudEnvCfg = TurnScrewdriverPointCloudEnvCfg
ScrewdriverCurriculumEnvCfg = TurnScrewdriverEnvCfg