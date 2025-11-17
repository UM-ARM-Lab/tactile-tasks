#!/bin/bash

# Script to playback a checkpoint from successful_checkpoints directory
# Usage: ./play_policy.sh [task_name]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set default task name if not provided
TASK_NAME="${1:-TurnScrewdriver-v0}"

# Path to checkpoint
CHECKPOINT_PATH="$SCRIPT_DIR/rl_games/successful_checkpoints/BaselineFull.pth"

# Number of environments
NUM_ENVS=16

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Change to project root directory
cd "$PROJECT_ROOT" || exit 1

# Run the play script
python scripts/rl_games/play.py \
    --task "$TASK_NAME" \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_envs "$NUM_ENVS"

