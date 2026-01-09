#!/bin/bash

# Script to playback a checkpoint from successful_checkpoints directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to print usage information
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [TASK_NAME]

Playback a trained policy checkpoint for tactile manipulation tasks.

OPTIONS:
    --baseline, --proprio    Use baseline/proprioceptive checkpoint (default)
    --contact                 Use contact-aware checkpoint
    --help, -h               Show this help message

TASK_NAME:
    Name of the task to run (default: TurnScrewdriver-v0)

EXAMPLES:
    $0                                    # Run default task with baseline checkpoint
    $0 --contact                          # Run default task with contact checkpoint
    $0 TurnScrewdriver-v0 --baseline      # Run specific task with baseline checkpoint
    $0 --contact TurnScrewdriver-v0       # Run specific task with contact checkpoint

Available checkpoints:
    Baseline:  scripts/rl_games/successful_checkpoints/FinalProprio.pth
    Contact:   scripts/rl_games/successful_checkpoints/FinalContact.pth
EOF
}

# Parse arguments
TASK_NAME="TurnScrewdriver-v0"
CHECKPOINT_TYPE="baseline"  # default to baseline/proprioceptive
INVALID_ARG=""

# Parse positional and flag arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --contact)
            CHECKPOINT_TYPE="contact"
            shift
            ;;
        --baseline|--proprio)
            CHECKPOINT_TYPE="baseline"
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            # Check if it's an invalid flag (starts with --)
            if [[ "$1" =~ ^-- ]]; then
                INVALID_ARG="$1"
                break
            else
                # Assume it's the task name
                TASK_NAME="$1"
            fi
            shift
            ;;
    esac
done

# Check for invalid arguments
if [ -n "$INVALID_ARG" ]; then
    echo "Error: Unknown option: $INVALID_ARG"
    echo ""
    print_usage
    exit 1
fi

# Path to checkpoints
CHECKPOINT_PATH_PROPRIOCEPTIVE="$SCRIPT_DIR/rl_games/successful_checkpoints/FinalProprio.pth"
CHECKPOINT_PATH_CONTACT="$SCRIPT_DIR/rl_games/successful_checkpoints/FinalContact.pth"

# Select checkpoint based on flag
if [ "$CHECKPOINT_TYPE" = "contact" ]; then
    CHECKPOINT_PATH="$CHECKPOINT_PATH_CONTACT"
    CHECKPOINT_NAME="Contact"
else
    CHECKPOINT_PATH="$CHECKPOINT_PATH_PROPRIOCEPTIVE"
    CHECKPOINT_NAME="Baseline (Proprioceptive)"
fi

# Number of environments
NUM_ENVS=16

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo ""
    echo "Please ensure the checkpoint file exists. Available checkpoints:"
    echo "  Baseline:  $CHECKPOINT_PATH_PROPRIOCEPTIVE"
    echo "  Contact:   $CHECKPOINT_PATH_CONTACT"
    echo ""
    echo "Use --help for usage information."
    exit 1
fi

# Change to project root directory
if ! cd "$PROJECT_ROOT"; then
    echo "Error: Failed to change to project root directory: $PROJECT_ROOT"
    exit 1
fi

# Print which checkpoint is being used
echo "Using $CHECKPOINT_NAME checkpoint: $CHECKPOINT_PATH"
echo "Running task: $TASK_NAME"

# Run the play script
python scripts/rl_games/play.py \
    --task "$TASK_NAME" \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_envs "$NUM_ENVS"

