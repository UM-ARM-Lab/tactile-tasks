# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene configurations for screwdriver manipulation environments."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass

from .arm_allegro import AllegroCfg
from .screwdriver import ScrewdriverCfg


@configclass
class ScrewdriverSceneCfg(InteractiveSceneCfg):
    """Configuration for the screwdriver manipulation scene."""

    replicate_physics = False

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    screwdriver: AssetBaseCfg = ScrewdriverCfg(prim_path="{ENV_REGEX_NS}/Screwdriver")
    robot: ArticulationCfg = AllegroCfg(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_aftc_base_link$",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        force_threshold=0.01,
    )


@configclass
class ScrewdriverSceneWithCameraCfg(ScrewdriverSceneCfg):
    """Configuration for the screwdriver scene with tiled camera for point cloud extraction."""

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        data_types=["rgb", "distance_to_image_plane"],
        width=32,
        height=32,
        colorize_semantic_segmentation=False,
        colorize_instance_segmentation=False,
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.2, 0.3, 0.3),
            rot=(0.224144, -0.129410, 0.836516, -0.482963),
            convention="ros",
        ),
        debug_vis=False,
    )
