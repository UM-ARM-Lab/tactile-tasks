# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone script to run the screwdriver manipulation environment."""

import argparse

import torch
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv

from hand_env_cfg import TurnScrewdriverEnvCfg

parser = argparse.ArgumentParser(description="Run the screwdriver manipulation environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    """Main function."""
    env_cfg = TurnScrewdriverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # Ensure gravity is enabled for the simulation
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    
    # Ensure viewer is properly configured for testing
    if hasattr(env_cfg, 'viewer'):
        env_cfg.viewer.eye = (2.0, 2.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.origin_type = "world"
    
    # Print configuration being loaded
    print("\n" + "="*80)
    print("HAND ENVIRONMENT CONFIGURATION")
    print("="*80)
    print("Robot configuration joint positions:")
    for joint_name, pos in env_cfg.scene.robot.init_state.joint_pos.items():
        if joint_name != ".*":  # Skip the default pattern
            print(f"  {joint_name}: {pos:.4f} rad ({pos*180/3.14159:.2f} deg)")
    print("="*80 + "\n")
    
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Initial reset so PhysX views are valid and defaults are applied
    env.reset()
    
    # Apply configured hand joint positions immediately after reset
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # Print actual robot joint configuration after environment setup
    print("\n" + "="*80)
    print("LOADED ROBOT JOINT CONFIGURATION")
    print("="*80)
    print(f"Number of joints: {robot.num_joints}")
    print("Default joint positions loaded in environment:")
    for i, name in enumerate(robot.joint_names):
        if i < robot.data.default_joint_pos.shape[1]:
            pos = robot.data.default_joint_pos[0, i].item()
            print(f"  {name}: {pos:.4f} rad ({pos*180/3.14159:.2f} deg)")
    print("="*80 + "\n")

    # simulate physics
    
    count = 0
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                
                # Manually set joint positions after reset
                robot = env.scene["robot"]
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                
                print("-" * 80)
                print("[INFO]: Resetting environment and applying configured joint positions...")
                
                # Print joint positions after reset
                # robot = env.scene["robot"]
                print(f"[RESET] Joint positions after reset:")
                current_joint_pos = robot.data.joint_pos[0]  # First environment
                for i, name in enumerate(robot.joint_names):
                    if "allegro_hand" in name and i < current_joint_pos.shape[0]:
                        pos = current_joint_pos[i].item()
                        print(f"  {name}: {pos:.4f} rad ({pos*180/3.14159:.2f} deg)")
                        
            # sample random actions
            joint_efforts = torch.zeros_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            count += 1
            
            

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()