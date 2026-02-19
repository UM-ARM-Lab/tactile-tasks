# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for screwdriver setup and manipulation."""

import glob
import pathlib

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.utils import bind_physics_material
from isaacsim.core.utils.stage import get_current_stage
from omni.usd import get_context
from pxr import Sdf, UsdGeom, UsdPhysics, PhysxSchema

from .screwdriver import ScrewdriverCfg

# Package root and assets directory
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PACKAGE_ROOT.parent / "assets"


def recursively_uninstance_prim(prim):
    """Recursively uninstance a prim and its children."""
    if prim.IsInstance():
        prim.SetInstanceable(False)
    for child in prim.GetChildren():
        recursively_uninstance_prim(child)


def setup_screwdriver_tip_pivots(
    env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")
) -> None:
    """
    Connects screwdriver tip to a fixed point in world space using a D6 joint with damping.
    """
    from pxr import Gf

    stage = get_current_stage()
    screwdriver = env.scene[asset_cfg.name]
    screwdriver_cfg = ScrewdriverCfg()
    tip_offset_local = screwdriver_cfg.tip_offset_local
    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))

    for env_i in env_indices:
        screwdriver_prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        base = f"/World/envs/env_{env_i}"

        root_pose = screwdriver.data.root_state_w[env_i, :7]
        pos = root_pose[:3].cpu().numpy()
        qw, qx, qy, qz = root_pose[3:].cpu().numpy()
        q = Gf.Quatd(float(qw), float(qx), float(qy), float(qz))
        tip_off = Gf.Vec3d(*[float(v) for v in tip_offset_local])
        tip_world = Gf.Vec3d(*(pos.tolist())) + q.Transform(tip_off)

        joint_path = f"{base}/TipSphericalJoint"
        if not stage.GetPrimAtPath(joint_path).IsValid():
            joint = UsdPhysics.Joint.Define(stage, joint_path)
        else:
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(joint_path))

        jp = stage.GetPrimAtPath(joint_path)
        joint.CreateBody1Rel().SetTargets([Sdf.Path(screwdriver_prim_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in tip_world]))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in tip_offset_local]))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        PhysxSchema.PhysxJointAPI.Apply(jp)
        for axis in ["transX", "transY", "transZ"]:
            limit = UsdPhysics.LimitAPI.Apply(jp, axis)
            limit.CreateLowAttr().Set(0.0)
            limit.CreateHighAttr().Set(0.0)

        for axis in ["rotZ"]:
            drive = UsdPhysics.DriveAPI.Apply(jp, axis)
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.1)


def add_screwdriver_rotation_markers(
    env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver")
) -> None:
    """Add a small red visual marker to each screwdriver to make rotation visible."""
    from pxr import Gf

    stage = get_current_stage()
    screwdriver = env.scene[asset_cfg.name]
    marker_local_offset = Gf.Vec3f(0.04, 0.0, 0.03)
    marker_radius = 0.004

    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))
    for env_i in env_indices:
        screwdriver_prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        screw_prim = stage.GetPrimAtPath(screwdriver_prim_path)
        if screw_prim.IsInstanceable():
            screw_prim.SetInstanceable(False)
        marker_prim_path = f"{screwdriver_prim_path}/RotationMarker"

        marker_prim = stage.GetPrimAtPath(marker_prim_path)
        if not marker_prim.IsValid():
            marker = UsdGeom.Sphere.Define(stage, Sdf.Path(marker_prim_path))
            marker.CreateRadiusAttr(marker_radius)
            gprim = UsdGeom.Gprim(marker.GetPrim())
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            xformable = UsdGeom.Xformable(marker.GetPrim())
            xformable.AddTranslateOp().Set(marker_local_offset)
        else:
            marker = UsdGeom.Sphere(stage.GetPrimAtPath(marker_prim_path))
            if not marker.GetRadiusAttr():
                marker.CreateRadiusAttr(marker_radius)
            else:
                marker.GetRadiusAttr().Set(marker_radius)
            gprim = UsdGeom.Gprim(marker.GetPrim())
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            xformable = UsdGeom.Xformable(marker.GetPrim())
            ops = xformable.GetOrderedXformOps()
            translate_op = None
            for op in ops:
                if op.GetOpName().startswith("xformOp:translate"):
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            translate_op.Set(marker_local_offset)


def _discover_random_screwdriver_usds() -> list[str]:
    """Return all screwdriver USDs from the attached random set on disk."""
    base_dir = ASSETS_DIR / "usd" / "screwdriver" / "variants" / "train"
    pattern = str(base_dir / "*.usd")
    return sorted(glob.glob(pattern))


def randomize_screwdriver_geometry_prestartup(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("screwdriver"),
    usd_paths: list[str] | None = None,
) -> None:
    """Swap each env's screwdriver USD reference before play starts."""
    import isaaclab.sim as sim_utils

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    if not usd_paths:
        usd_paths = _discover_random_screwdriver_usds()
    if not usd_paths:
        return

    env_indices = list(range(len(prim_paths))) if env_ids is None else env_ids.tolist()
    with Sdf.ChangeBlock():
        for env_i in env_indices:
            prim_path = prim_paths[env_i]
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
            refs = prim.GetReferences()
            choice_idx = (env_i * 131 + (env.cfg.seed or 0)) % len(usd_paths)
            usd_path = usd_paths[choice_idx]
            refs.ClearReferences()
            refs.AddReference(usd_path)


def get_camera_point_cloud(
    env,
    env_ids: torch.Tensor = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    env_frame: bool = False,
) -> torch.Tensor:
    """Extract point cloud from tiled camera using distance_to_image_plane data."""
    from isaaclab.sensors.camera.utils import create_pointcloud_from_depth

    camera = env.scene[asset_cfg.name]
    camera_data = camera.data
    depth = camera_data.output["distance_to_image_plane"]
    num_envs = depth.shape[0]

    if env_ids is not None:
        env_indices = env_ids.cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    else:
        env_indices = list(range(num_envs))

    point_clouds = []
    for env_idx in env_indices:
        if env_idx >= num_envs:
            continue
        env_depth = depth[env_idx]
        point_cloud = create_pointcloud_from_depth(
            intrinsic_matrix=camera.data.intrinsic_matrices[env_idx],
            depth=env_depth,
            device=env.device,
        )
        num_samples = 1024
        if point_cloud.shape[0] >= num_samples:
            idx = torch.randperm(point_cloud.shape[0], device=point_cloud.device)[:num_samples]
            point_cloud = point_cloud[idx]
        else:
            repeat = num_samples - point_cloud.shape[0]
            extra = point_cloud[
                torch.randint(point_cloud.shape[0], (repeat,), device=point_cloud.device)
            ]
            point_cloud = torch.cat([point_cloud, extra], dim=0)
        point_clouds.append(point_cloud)

    if point_clouds:
        max_points = max(pc.shape[0] for pc in point_clouds)
        padded_point_clouds = []
        for pc in point_clouds:
            if pc.shape[0] < max_points:
                padding = torch.zeros(
                    (max_points - pc.shape[0], pc.shape[1]), device=pc.device, dtype=pc.dtype
                )
                pc = torch.cat([pc, padding], dim=0)
            padded_point_clouds.append(pc)
        point_cloud_batch = torch.stack(padded_point_clouds, dim=0)
        if env_frame:
            env_indices_tensor = torch.tensor(env_indices, device=env.device, dtype=torch.long)
            env_origins = env.scene.env_origins[env_indices_tensor]
            point_cloud_batch = point_cloud_batch - env_origins.unsqueeze(1)
        return point_cloud_batch
    else:
        num_selected = len(env_indices) if env_ids is not None else num_envs
        return torch.zeros((num_selected, 0, 3), device=env.device)


def randomize_screwdriver_mass(
    env,
    env_ids=None,
    asset_cfg=SceneEntityCfg("screwdriver"),
    mass_range=(0.05, 0.20),
):
    """Randomize screwdriver mass per environment."""
    from pxr import UsdPhysics

    stage = get_current_stage()
    screwdriver = env.scene[asset_cfg.name]
    env_indices = env_ids.tolist() if env_ids is not None else list(range(env.scene.num_envs))
    for env_i in env_indices:
        prim_path = screwdriver.root_physx_view.prim_paths[env_i]
        prim = stage.GetPrimAtPath(prim_path)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        m = float(torch.empty(1, device=env.device).uniform_(mass_range[0], mass_range[1]).cpu())
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr:
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(m)
