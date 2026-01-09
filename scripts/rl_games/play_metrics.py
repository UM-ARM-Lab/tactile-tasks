# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games, with cleaned up metrics for rotation and drops."""

# ==================================================================================================
#  FIX: GLIBCXX COMPATIBILITY SHIM
#  This block forces the script to use the system libstdc++ instead of the incompatible Conda one.
#  This must be at the VERY TOP before any other imports (even argparse or isaaclab).
# ==================================================================================================
import os
import sys

def _enforce_system_libstdcxx():
    """
    Detects if the incompatible Conda libstdc++ is active and forces a reload
    using the system library via LD_PRELOAD.
    """
    if sys.platform != "linux":
        return

    # Prevent infinite recursion if the fix is already applied
    if os.environ.get("ISAAC_GLIBCXX_FIXED") == "1":
        return

    # Common paths for the system libstdc++ on Ubuntu 20.04/22.04
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib64/libstdc++.so.6"
    ]
    
    system_lib = None
    for path in candidates:
        if os.path.exists(path):
            system_lib = path
            break

    if system_lib:
        # Check if we are currently running without the preload
        current_preload = os.environ.get("LD_PRELOAD", "")
        if system_lib not in current_preload:
            print(f"[GLIBC FIX] Restarting script with system libstdc++: {system_lib}")
            
            # Update environment variables
            new_env = os.environ.copy()
            new_env["LD_PRELOAD"] = f"{system_lib}:{current_preload}" if current_preload else system_lib
            new_env["ISAAC_GLIBCXX_FIXED"] = "1"
            
            # Restart the current script with the new environment
            try:
                os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)
            except OSError as e:
                print(f"[GLIBC FIX] Failed to restart: {e}")
                # If restart fails, we just continue and hope for the best
    else:
        print("[GLIBC FIX] Warning: Could not find system libstdc++.so.6. Continuing without fix.")

# Run the fix immediately
_enforce_system_libstdcxx()
# ==================================================================================================


import argparse
import math
import random
import time
import torch
import csv
import gymnasium as gym

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games (with metrics).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--metrics_out", type=str, default="metrics_summary.csv", help="CSV path for per-episode metrics.")
parser.add_argument("--runs", type=int, default=50, help="Number of completed episodes to collect before stopping.")
parser.add_argument("--fall_angle_deg", type=float, default=19.0, help="Angle threshold (deg) for drop detection.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import tactile_tasks.tasks  # noqa: F401
from isaaclab.utils.math import matrix_from_quat

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent with metrics."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    base_env = env.unwrapped
    dt = base_env.step_dt
    num_envs = base_env.num_envs
    device = base_env.device

    # =========================================================================
    # METRIC TRACKING BUFFERS
    # =========================================================================
    
    # Per-env running buffers
    current_env_yaw = torch.zeros(num_envs, device=device, dtype=torch.float32)
    prev_raw_yaw = torch.zeros(num_envs, device=device, dtype=torch.float32)
    
    # Drop Latch: If true at ANY point in the episode, it counts as a drop.
    # We rely on this because the final frame might be auto-reset.
    has_fallen_latch = torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    # Track which envs just reset to prevent calculating rotation jumps
    just_reset_mask = torch.ones(num_envs, device=device, dtype=torch.bool)

    # Storage for completed episodes
    # We store: [net_yaw_radians, is_dropped (0/1)]
    completed_episodes_data = []

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    
    # Initialize previous yaw with the starting state
    with torch.inference_mode():
        try:
            screwdriver = base_env.scene["screwdriver"]
            R = matrix_from_quat(screwdriver.data.root_quat_w)
            prev_raw_yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        except:
            pass

    timestep = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    print(f"[INFO] Starting evaluation. Target: {args_cli.runs} episodes.")

    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # 1. PRE-STEP: Measure State and Detect Falls
            # We do this BEFORE env.step() to capture the state before any potential auto-reset
            try:
                screwdriver = base_env.scene["screwdriver"]
                # Get orientation quaternion (w, x, y, z)
                quat_w = screwdriver.data.root_quat_w
                R = matrix_from_quat(quat_w)
                
                # --- Rotation Logic ---
                # Calculate Raw Yaw (Z-rotation) using atan2(R[1,0], R[0,0])
                # This gives the global yaw orientation in [-pi, pi]
                raw_yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
                
                # Calculate Delta Yaw (Unwrapped)
                yaw_diff = raw_yaw - prev_raw_yaw
                # Wrap diff to [-pi, pi] to handle the jump boundary
                # e.g., if jumping from 3.1 to -3.1, diff is -6.2 -> wrapped is +0.08
                yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
                
                # Accumulate, but ONLY for envs that didn't just reset in the last frame
                # (If they just reset, yaw_diff is the jump from End -> Start, which is garbage)
                valid_step_mask = ~just_reset_mask
                current_env_yaw[valid_step_mask] += yaw_diff[valid_step_mask]
                
                # Update previous buffer
                prev_raw_yaw = raw_yaw.clone()

                # --- Fall detection logic ---
                # Z-axis of screwdriver is the 3rd column of R
                z_axis = R[:, :, 2] # (N, 3)
                # Dot product with world Up (0,0,1) is just the z component
                cos_theta = z_axis[:, 2]
                # Check if below threshold
                fall_thresh_cos = math.cos(math.radians(args_cli.fall_angle_deg))
                current_frame_fallen = cos_theta < fall_thresh_cos # Boolean tensor

                # Latch it: If it was ever fallen in this episode, keep it True
                # (But ignore falls calculated on the very first frame of a reset)
                has_fallen_latch[valid_step_mask] |= current_frame_fallen[valid_step_mask]
                
                # Clear the reset mask now that we've processed the first frame
                just_reset_mask[:] = False

            except Exception as e:
                # Fail gracefully if object not found (e.g. unexpected env config)
                # print(f"[WARN] {e}")
                pass

            # 3. Agent Step
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

            # 5. Process Completed Episodes
            # 'dones' identifies envs that finished this step (timeout or dropped)
            done_indices = torch.nonzero(dones, as_tuple=False).flatten()
            
            if len(done_indices) > 0:
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, done_indices, :] = 0.0
                
                for env_idx in done_indices:
                    # Get final metrics for this specific env
                    final_yaw = float(current_env_yaw[env_idx].item())
                    dropped = bool(has_fallen_latch[env_idx].item())
                    
                    # Store data
                    completed_episodes_data.append({
                        "net_yaw": final_yaw,
                        "dropped": dropped
                    })
                    
                    # Reset metric buffer for this env
                    current_env_yaw[env_idx] = 0.0
                    has_fallen_latch[env_idx] = False
                    
                    # Mark this env as "Just Reset" so we skip the metrics calc 
                    # on the very next loop iteration (start of new episode)
                    just_reset_mask[env_idx] = True

                count = len(completed_episodes_data)
                if count % 10 == 0:
                    print(f"[PROGRESS] Collected {count}/{args_cli.runs} episodes...")

            # 6. Stopping Condition
            if len(completed_episodes_data) >= args_cli.runs:
                print(f"[INFO] Reached target of {args_cli.runs} episodes.")
                break

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # Real-time delay
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # =========================================================================
    # SUMMARY AND LOGGING
    # =========================================================================
    
    total_eps = len(completed_episodes_data)
    if total_eps > 0:
        # Filter only episodes that DID NOT drop for rotation calculation
        successful_episodes = [d for d in completed_episodes_data if not d["dropped"]]
        successful_yaws = [d["net_yaw"] for d in successful_episodes]
        drops = [d["dropped"] for d in completed_episodes_data]
        
        # Calculate Statistics
        num_success = len(successful_yaws)
        if num_success > 0:
            avg_yaw = sum(successful_yaws) / num_success
            avg_turns = avg_yaw / (2 * math.pi)
        else:
            avg_yaw = 0.0
            avg_turns = 0.0

        drop_count = sum(drops)
        drop_rate = (drop_count / total_eps) * 100.0
        
        print("\n" + "="*40)
        print(" FINAL METRICS SUMMARY ")
        print("="*40)
        print(f"Total Episodes    : {total_eps}")
        print(f"Successful Episodes: {num_success}")
        print(f"Avg Rotation (Success Only): {avg_yaw:.4f} rad ({avg_turns:.2f} turns)")
        print(f"Total Drops       : {drop_count}")
        print(f"Drop Rate         : {drop_rate:.2f}%")
        print("="*40 + "\n")

        # Write CSV
        if args_cli.metrics_out:
            with open(args_cli.metrics_out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode_id", "net_yaw_rad", "net_turns", "dropped"])
                for i, data in enumerate(completed_episodes_data):
                    writer.writerow([
                        i, 
                        f"{data['net_yaw']:.5f}", 
                        f"{data['net_yaw']/(2*math.pi):.5f}", 
                        1 if data['dropped'] else 0
                    ])
            print(f"[INFO] Metrics saved to {args_cli.metrics_out}")
    else:
        print("[WARN] No episodes completed.")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()