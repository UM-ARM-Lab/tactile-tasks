# Tactile Tasks - Isaac Lab Extension

## Overview

This repository contains a tactile manipulation project built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/), focusing on robotic manipulation tasks involving screwdriver manipulation with tactile sensing. The project provides reinforcement learning environments where an Allegro robotic hand learns to manipulate and rotate screwdrivers with and without tactile feedback.

**Key Features:**
- **Tactile Manipulation**: Environments for learning dexterous manipulation tasks with tactile sensing
- **Screwdriver Tasks**: Multiple task variants including pure proprioceptive rotation and contact-aware rotation
- **Pre-trained Models**: Includes trained checkpoints for demonstration purposes ()
- **Coming Soon: Point Cloud Integration**: Eventual support for point cloud extraction as policyt observation.

**Available Tasks:**
- `TurnScrewdriver-v0`: Basic screwdriver rotation task
- `TurnScrewdriverContact-v0`: Contact-aware screwdriver manipulation
- `TurnScrewdriverPointCloud-v0`: (Coming soon) Point cloud-based manipulation with visual perception

## Installation

### Prerequisites

1. **Install Isaac Lab**: Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
   - We recommend using the conda (use Python 3.11)installation as it simplifies calling Python scripts from the terminal.

2. **Clone this repository**: Clone or copy this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory).

### Setup

1. **Install the extension**: Using a Python interpreter that has Isaac Lab installed, install the library in editable mode:

   ```bash
   # Use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
   python -m pip install -e source/tactile_tasks
   ```

2. **Verify installation**: List available tasks to verify the extension is correctly installed:

   ```bash
   # Use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
   python scripts/list_envs.py
   ```

   Note: If the task names change, you may need to update the search pattern `"Template-"` in `scripts/list_envs.py` to list them correctly.

## Running the Demo

### Play Policy Demo

The repository includes a pre-trained checkpoint for the `TurnScrewdriver-v0`and `TurnScrewdriverContact-v0` task. Run the demo using the provided script:

```bash
# Run with default task (TurnScrewdriver-v0) and baseline checkpoint
./scripts/play_policy.sh

# Run with baseline/proprioceptive checkpoint:

./scripts/play_policy.sh --proprio

# Run with contact-aware checkpoint
./scripts/play_policy.sh --contact

# Specify a different task with checkpoint type
./scripts/play_policy.sh TurnScrewdriver-v0 --contact
./scripts/play_policy.sh --baseline TurnScrewdriver-v0
```

**What the demo does:**
- Loads a pre-trained reinforcement learning policy from the selected checkpoint:
  - **Baseline (Proprioceptive)**: `scripts/rl_games/successful_checkpoints/FinalProprio.pth` (default)
  - **Contact-aware**: `scripts/rl_games/successful_checkpoints/FinalContact.pth`
- Runs the policy in a simulated environment with 16 parallel environments
- Shows the Allegro hand manipulating and rotating a screwdriver using learned tactile manipulation skills, with a variety of different screwdriver geometries.
- Demonstrates the learned dexterous manipulation capabilities

**Demo Parameters:**
- **Checkpoint selection**: Use `--proprio` for proprioceptive-only policy (default) or `--contact` for contact-aware policy
- **Task**: `TurnScrewdriver-v0` (default) or `TurnScrewdriverContact-v0`
- **Number of environments**: 16
- **Rendering**: Interactive visualization in Isaac Sim

### Additional Demo Options

You can also run the play script directly with custom parameters:

```bash
# Run with baseline/proprioceptive checkpoint
python scripts/rl_games/play.py \
    --task TurnScrewdriver-v0 \
    --checkpoint scripts/rl_games/successful_checkpoints/FinalProprio.pth \
    --num_envs 16

# Run with contact-aware checkpoint
python scripts/rl_games/play.py \
    --task TurnScrewdriver-v0 \
    --checkpoint scripts/rl_games/successful_checkpoints/FinalContact.pth \
    --num_envs 16
```

## Training

To train your own policy, use the training scripts:

```bash
# Train with RL-Games
python scripts/rl_games/train.py --task=TurnScrewdriver-v0

# Train with contact sensing
python scripts/rl_games/train.py --task=TurnScrewdriverContact-v0

# Train with PointNet (for point cloud-based tasks)
python scripts/rl_games/train_with_pointnet.py --task=TurnScrewdriverPointCloud-v0
```

## Project Structure

```
tactile-tasks/
├── source/
│   └── tactile_tasks/          # Main extension package
│       ├── tasks/               # Task implementations
│       │   ├── direct/          # Direct RL environments
│       │   └── manager_based/   # Manager-based RL environments
│       └── assets/              # 3D assets (screwdrivers, hands, etc.)
├── scripts/
│   ├── play_policy.sh          # Demo script
│   ├── rl_games/               # RL-Games training/playback scripts
│   │   └── successful_checkpoints/  # Pre-trained models
│   └── list_envs.py            # List available environments
└── README.md                    # This file
```

## Troubleshooting

### Checkpoint Not Found

If you encounter an error about the checkpoint not being found:
- Ensure the checkpoint file exists:
  - `scripts/rl_games/successful_checkpoints/FinalProprio.pth` (for baseline/proprioceptive)
  - `scripts/rl_games/successful_checkpoints/FinalContact.pth` (for contact-aware)
- Check that you're running the script from the project root directory
- Verify you're using the correct flag (`--baseline` or `--contact`)

### Environment Not Found

If the task environment is not recognized:
- Verify the extension is installed: `python -m pip list | grep tactile-tasks`
- Reinstall if needed: `python -m pip install -e source/tactile_tasks`
- Check that Isaac Lab is properly installed and activated

## Additional Resources
- **Isaac Lab Documentation**: https://isaac-sim.github.io/IsaacLab/
- **Extension Setup**: See the original template README sections below for IDE setup and Omniverse extension configuration

## License

This project is licensed under the Apache-2.0 License.

## Acknowledgments

Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) by NVIDIA.
