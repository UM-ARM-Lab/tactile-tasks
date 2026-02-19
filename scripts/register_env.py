import os

import gymnasium as gym

from tactile_tasks.tasks.manager_based.tactile_tasks.hand_env_cfg import TurnScrewdriverEnvCfg

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config_path = os.path.join(script_dir, "hand_env_ppo_config.yaml")

gym.register(
    id="HandManipulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tactile_tasks.tasks.manager_based.tactile_tasks.hand_env_cfg:TurnScrewdriverEnvCfg",
        "rl_games_cfg_entry_point": yaml_config_path,
    },
)