import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

# Package root: .../tactile_tasks/source/tactile_tasks/tactile_tasks
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# Assets directory: .../tactile_tasks/source/tactile_tasks/assets
ASSETS_DIR = PACKAGE_ROOT.parent / "assets"

@configclass
class ScrewdriverCfg(RigidObjectCfg):
    spawn = sim_utils.UsdFileCfg(
        usd_path=str(ASSETS_DIR / "usd" / "screwdriver" / "screwdriver_fric.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # Enable gravity so it rests on table
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
            enable_gyroscopic_forces=True,
        )
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.015),
        # pos=(0.0, 0.0, 0.155),  # Position above table
        rot=(1.0, 0.0, 0.0, 0.0),
    )
    
    # Tip offset in local frame (meters) - adjust based on your screwdriver USD
    tip_offset_local = (0.0, 0.0, -0.045) 
    # tip_offset_local = (0.0, 0.0, -0.13)  # use for random screwdrivers
