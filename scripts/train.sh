#!/bin/bash

# Run ./scripts/train.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ========================================
# Training Parameters - Edit these values
# ========================================

# Enable video recording during training (true/false)
VIDEO=false

# Number of parallel environments
NUM_ENVS=1

# Enable Weights & Biases tracking (true/false)
TRACK=false

# Root directory for logging experiments
LOG_ROOT_PATH="/home/yifan/workspace/ndf_robot/narstie/model_weights/tactile-tasks/TurnScrewdriver_debug"

# Task name
TASK="TurnScrewdriver-v0"

# Maximum training iterations
MAX_ITERATIONS=10000

# Random seed
SEED=42

# Optional: Path to checkpoint for resuming training (leave empty for new training)
CHECKPOINT=""

# ========================================

# Change to project root directory
if ! cd "$PROJECT_ROOT"; then
    echo "Error: Failed to change to project root directory: $PROJECT_ROOT"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Task:             $TASK"
echo "Num Envs:         $NUM_ENVS"
echo "Max Iterations:   $MAX_ITERATIONS"
echo "Seed:             $SEED"
echo "Video Recording:  $VIDEO"
echo "W&B Tracking:     $TRACK"
echo "Log Root Path:    $LOG_ROOT_PATH"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint:       $CHECKPOINT"
fi
echo "=========================================="
echo ""

# Build the command
CMD="python scripts/rl_games/train.py \
    --task $TASK \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --seed $SEED"

# Add video flag if enabled
if [ "$VIDEO" = true ]; then
    CMD="$CMD --video"
fi

# Add track flag if enabled
if [ "$TRACK" = true ]; then
    CMD="$CMD --track"
fi

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Run the training script
echo "Running command:"
echo "$CMD"
echo ""
eval "$CMD"
