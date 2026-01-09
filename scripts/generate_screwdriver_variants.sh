#!/bin/bash

# Script to generate URDF variants by randomizing handle geometry (lines 48 and 57),
# mass (line 42), and friction (lines 28-29)
# Randomly chooses box or cylinder with proper handle dimensions:
#   - Cylinder: radius 0.01-0.03, length fixed at 0.1
#   - Box: cross-section 0.025-0.04 (x,y), length fixed at 0.1 (z along shaft)
#   - Mass: 0.05-0.2 kg (screwdriver_body link)
#   - Friction: mu1/mu2 0.3-1.5 (static/dynamic friction coefficients)
#   - Minimum inertia enforced at 1e-5 for physics stability
#   - Box minimum size ensures compatibility with convex hull collision conversion
# Uses IsaacLab's convert_urdf.py for USD conversion
# Usage: ./generate_screwdriver_variants.sh [num_variants] [base_urdf] [output_dir] [--convert-to-usd] [usd_output_dir] [--train-split=0.8]

set -e

# Parse arguments - separate flags from positional args
CONVERT_TO_USD=false
USD_OUTPUT_DIR=
TRAIN_SPLIT=0.8
POSITIONAL_ARGS=()

for arg in "$@"; do
    case $arg in
        --convert-to-usd)
            CONVERT_TO_USD=true
            ;;
        --usd-output=*)
            USD_OUTPUT_DIR="${arg#*=}"
            CONVERT_TO_USD=true
            ;;
        --train-split=*)
            TRAIN_SPLIT="${arg#*=}"
            ;;
        *)
            POSITIONAL_ARGS+=("$arg")
            ;;
    esac
done

# Set positional arguments
NUM_VARIANTS="${POSITIONAL_ARGS[0]:-10}"
BASE_URDF="${POSITIONAL_ARGS[1]:-source/tactile_tasks/assets/urdf/screwdriver/screwdriver_free.urdf}"
OUTPUT_DIR="${POSITIONAL_ARGS[2]:-source/tactile_tasks/assets/urdf/screwdriver/variants}"

# Set default USD output directory (can be overridden by --usd-output or 4th positional arg)
DEFAULT_USD_OUTPUT_DIR="source/tactile_tasks/assets/usd/screwdriver/variants"

# Handle USD output directory
if [ -z "$USD_OUTPUT_DIR" ]; then
    # Check if there's a 4th positional arg (could be USD output dir)
    if [ -n "${POSITIONAL_ARGS[3]}" ]; then
        USD_OUTPUT_DIR="${POSITIONAL_ARGS[3]}"
    else
        USD_OUTPUT_DIR="$DEFAULT_USD_OUTPUT_DIR"
    fi
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT is tactile_tasks directory (for relative paths)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# ISAACLAB_ROOT is the parent directory containing IsaacLab
ISAACLAB_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"

# Create output directories for train and test
TRAIN_DIR="${OUTPUT_DIR}/train"
TEST_DIR="${OUTPUT_DIR}/test"
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

# Check if base URDF exists
if [ ! -f "$BASE_URDF" ]; then
    echo "Error: Base URDF not found at $BASE_URDF"
    exit 1
fi

# Calculate train and test counts
TRAIN_COUNT=$(python3 -c "import math; print(int(math.floor($NUM_VARIANTS * $TRAIN_SPLIT)))")
TEST_COUNT=$((NUM_VARIANTS - TRAIN_COUNT))

echo "Generating $NUM_VARIANTS variants (train: $TRAIN_COUNT, test: $TEST_COUNT)..."
echo "Base URDF: $BASE_URDF"
echo "Train directory: $TRAIN_DIR"
echo "Test directory: $TEST_DIR"
echo "Train split: $TRAIN_SPLIT"
echo ""

# Generate variants using Python for reliable XML manipulation
python3 << EOF
import random
import os
import math

base_urdf = "$BASE_URDF"
train_dir = "$TRAIN_DIR"
test_dir = "$TEST_DIR"
num_variants = int("$NUM_VARIANTS")
train_split = float("$TRAIN_SPLIT")

# Calculate train and test counts
train_count = int(math.floor(num_variants * train_split))
test_count = num_variants - train_count

# Read base URDF
with open(base_urdf, 'r') as f:
    base_lines = f.readlines()

# Generate train set with seed 42
print("Generating train set ($train_count variants)...")
random.seed(42)
for i in range(train_count):
    # Copy base lines
    variant_lines = base_lines.copy()
    
    # Randomly choose geometry type (cylinder or box)
    use_cylinder = random.choice([True, False])
    
    # Fixed height (length along shaft direction) for all variants
    height = 0.1
    
    # Randomize mass: 0.05 to 0.2 kg (reasonable range for screwdriver handle)
    mass = random.uniform(0.05, 0.2)
    
    # Randomize friction: mu1 and mu2 (static/dynamic friction) 0.3 to 1.5
    # Typical range: rubber (0.6-1.0), plastic (0.3-0.5), metal (0.5-1.2)
    mu1 = random.uniform(0.3, 1.5)
    mu2 = random.uniform(0.3, 1.5)
    
    if use_cylinder:
        # Cylinder geometry for handle
        # Radius: 0.01 to 0.03 (cross-section, wider than shaft which is 0.005)
        # Length: fixed at 0.1 (along shaft direction)
        radius = random.uniform(0.01, 0.025)
        new_line = f'            <cylinder radius="{radius:.6f}" length="{height}"/>\n'
        geom_type_str = f"cylinder r={radius:.4f} l={height:.4f}"
        
        # Calculate inertia tensor for cylinder
        # For cylinder: Ixx = Iyy = m/12 * (3*r² + L²), Izz = m/2 * r²
        ixx_iyy = (mass / 12.0) * (3 * radius**2 + height**2)
        izz = (mass / 2.0) * radius**2
    else:
        # Box geometry for handle
        # Box size format in URDF: "x y z" where:
        #   x, y: cross-section width/height (perpendicular to shaft)
        #   z: length along shaft direction (fixed at 0.1)
        # Cross-section: 0.025 to 0.04 (wider than shaft, minimum 0.025 for physics stability)
        # Increased minimum to avoid thin box issues with convex hull conversion in PhysX
        cross_section = random.uniform(0.025, 0.04)
        # Allow slightly different x and y for rectangular cross-section (optional)
        width_x = cross_section
        width_y = random.uniform(cross_section * 0.9, cross_section * 1.1)  # ±10% variation
        # Ensure minimum dimension to avoid degenerate boxes (safety check)
        width_x = max(width_x, 0.025)
        width_y = max(width_y, 0.025)
        size = f"{width_x:.6f} {width_y:.6f} {height:.6f}"
        new_line = f'            <box size="{width_x:.6f} {width_y:.6f} {height:.6f}"/>\n'
        geom_type_str = f"box {size}"
        
        # Calculate inertia tensor for box
        # For box with dimensions (a, b, c): 
        # Ixx = m/12 * (b² + c²), Iyy = m/12 * (a² + c²), Izz = m/12 * (a² + b²)
        ixx = (mass / 12.0) * (width_y**2 + height**2)
        iyy = (mass / 12.0) * (width_x**2 + height**2)
        izz = (mass / 12.0) * (width_x**2 + width_y**2)
        ixx_iyy = ixx  # For boxes, Ixx and Iyy are different
    
    # Minimum inertia threshold to prevent physics instability (PhysX requirement)
    # Increased to 1e-5 to ensure boxes have sufficient rotational inertia
    min_inertia = 1e-5
    
    # Apply minimum inertia to all components
    if use_cylinder:
        ixx_iyy = max(ixx_iyy, min_inertia)
        izz = max(izz, min_inertia)
    else:
        ixx = max(ixx, min_inertia)
        iyy = max(iyy, min_inertia)
        izz = max(izz, min_inertia)
    
    # Replace line 48 (visual geometry) - index 47 (0-indexed)
    if len(variant_lines) > 47:
        variant_lines[47] = new_line
    
    # Replace line 57 (collision geometry) - index 56 (0-indexed)
    if len(variant_lines) > 56:
        variant_lines[56] = new_line
    
    # Replace line 42 (mass) - index 41 (0-indexed)
    if len(variant_lines) > 41:
        mass_line = f'      <mass value="{mass:.6f}"/>\n'
        variant_lines[41] = mass_line
    
    # Replace line 43 (inertia tensor) - index 42 (0-indexed)
    if len(variant_lines) > 42:
        if use_cylinder:
            # Cylinder: Ixx = Iyy
            inertia_line = f'      <inertia ixx="{ixx_iyy:.9f}" ixy="0.0" ixz="0.0" iyy="{ixx_iyy:.9f}" iyz="0.0" izz="{izz:.9f}"/>\n'
        else:
            # Box: Ixx and Iyy are different
            inertia_line = f'      <inertia ixx="{ixx:.9f}" ixy="0.0" ixz="0.0" iyy="{iyy:.9f}" iyz="0.0" izz="{izz:.9f}"/>\n'
        variant_lines[42] = inertia_line
    
    # Replace line 28 (mu1 friction) - index 27 (0-indexed)
    if len(variant_lines) > 27:
        mu1_line = f'    <mu1>{mu1:.6f}</mu1>\n'
        variant_lines[27] = mu1_line
    
    # Replace line 29 (mu2 friction) - index 28 (0-indexed)
    if len(variant_lines) > 28:
        mu2_line = f'    <mu2>{mu2:.6f}</mu2>\n'
        variant_lines[28] = mu2_line
    
    # Write variant to train directory
    variant_path = os.path.join(train_dir, f"screwdriver_variant_{i}.urdf")
    with open(variant_path, 'w') as f:
        f.writelines(variant_lines)
    
    print(f"Train {i:3d}: {geom_type_str}, mass={mass:.4f}kg, mu1={mu1:.3f}, mu2={mu2:.3f} -> {variant_path}")

# Generate test set with different seed (123) to ensure no overlap
print(f"\nGenerating test set ({test_count} variants)...")
random.seed(123)  # Different seed for test set
for i in range(test_count):
    # Copy base lines
    variant_lines = base_lines.copy()
    
    # Randomly choose geometry type (cylinder or box)
    use_cylinder = random.choice([True, False])
    
    # Fixed height (length along shaft direction) for all variants
    height = 0.1
    
    # Randomize mass: 0.05 to 0.2 kg (reasonable range for screwdriver handle)
    mass = random.uniform(0.05, 0.2)
    
    # Randomize friction: mu1 and mu2 (static/dynamic friction) 0.3 to 1.5
    # Typical range: rubber (0.6-1.0), plastic (0.3-0.5), metal (0.5-1.2)
    mu1 = random.uniform(0.3, 1.5)
    mu2 = random.uniform(0.3, 1.5)
    
    if use_cylinder:
        # Cylinder geometry for handle
        # Radius: 0.01 to 0.03 (cross-section, wider than shaft which is 0.005)
        # Length: fixed at 0.1 (along shaft direction)
        radius = random.uniform(0.01, 0.03)
        new_line = f'            <cylinder radius="{radius:.6f}" length="{height}"/>\n'
        geom_type_str = f"cylinder r={radius:.4f} l={height:.4f}"
        
        # Calculate inertia tensor for cylinder
        # For cylinder: Ixx = Iyy = m/12 * (3*r² + L²), Izz = m/2 * r²
        ixx_iyy = (mass / 12.0) * (3 * radius**2 + height**2)
        izz = (mass / 2.0) * radius**2
    else:
        # Box geometry for handle
        # Box size format in URDF: "x y z" where:
        #   x, y: cross-section width/height (perpendicular to shaft)
        #   z: length along shaft direction (fixed at 0.1)
        # Cross-section: 0.025 to 0.04 (wider than shaft, minimum 0.025 for physics stability)
        # Increased minimum to avoid thin box issues with convex hull conversion in PhysX
        cross_section = random.uniform(0.025, 0.04)
        # Allow slightly different x and y for rectangular cross-section (optional)
        width_x = cross_section
        width_y = random.uniform(cross_section * 0.9, cross_section * 1.1)  # ±10% variation
        # Ensure minimum dimension to avoid degenerate boxes (safety check)
        width_x = max(width_x, 0.025)
        width_y = max(width_y, 0.025)
        size = f"{width_x:.6f} {width_y:.6f} {height:.6f}"
        new_line = f'            <box size="{width_x:.6f} {width_y:.6f} {height:.6f}"/>\n'
        geom_type_str = f"box {size}"
        
        # Calculate inertia tensor for box
        # For box with dimensions (a, b, c): 
        # Ixx = m/12 * (b² + c²), Iyy = m/12 * (a² + c²), Izz = m/12 * (a² + b²)
        ixx = (mass / 12.0) * (width_y**2 + height**2)
        iyy = (mass / 12.0) * (width_x**2 + height**2)
        izz = (mass / 12.0) * (width_x**2 + width_y**2)
        ixx_iyy = ixx  # For boxes, Ixx and Iyy are different
    
    # Minimum inertia threshold to prevent physics instability (PhysX requirement)
    # Increased to 1e-5 to ensure boxes have sufficient rotational inertia
    min_inertia = 1e-5
    
    # Apply minimum inertia to all components
    if use_cylinder:
        ixx_iyy = max(ixx_iyy, min_inertia)
        izz = max(izz, min_inertia)
    else:
        ixx = max(ixx, min_inertia)
        iyy = max(iyy, min_inertia)
        izz = max(izz, min_inertia)
    
    # Replace line 48 (visual geometry) - index 47 (0-indexed)
    if len(variant_lines) > 47:
        variant_lines[47] = new_line
    
    # Replace line 57 (collision geometry) - index 56 (0-indexed)
    if len(variant_lines) > 56:
        variant_lines[56] = new_line
    
    # Replace line 42 (mass) - index 41 (0-indexed)
    if len(variant_lines) > 41:
        mass_line = f'      <mass value="{mass:.6f}"/>\n'
        variant_lines[41] = mass_line
    
    # Replace line 43 (inertia tensor) - index 42 (0-indexed)
    if len(variant_lines) > 42:
        if use_cylinder:
            # Cylinder: Ixx = Iyy
            inertia_line = f'      <inertia ixx="{ixx_iyy:.9f}" ixy="0.0" ixz="0.0" iyy="{ixx_iyy:.9f}" iyz="0.0" izz="{izz:.9f}"/>\n'
        else:
            # Box: Ixx and Iyy are different
            inertia_line = f'      <inertia ixx="{ixx:.9f}" ixy="0.0" ixz="0.0" iyy="{iyy:.9f}" iyz="0.0" izz="{izz:.9f}"/>\n'
        variant_lines[42] = inertia_line
    
    # Replace line 28 (mu1 friction) - index 27 (0-indexed)
    if len(variant_lines) > 27:
        mu1_line = f'    <mu1>{mu1:.6f}</mu1>\n'
        variant_lines[27] = mu1_line
    
    # Replace line 29 (mu2 friction) - index 28 (0-indexed)
    if len(variant_lines) > 28:
        mu2_line = f'    <mu2>{mu2:.6f}</mu2>\n'
        variant_lines[28] = mu2_line
    
    # Write variant to test directory
    variant_path = os.path.join(test_dir, f"screwdriver_variant_{i}.urdf")
    with open(variant_path, 'w') as f:
        f.writelines(variant_lines)
    
    print(f"Test  {i:3d}: {geom_type_str}, mass={mass:.4f}kg, mu1={mu1:.3f}, mu2={mu2:.3f} -> {variant_path}")

print(f"\nDone! Generated {train_count} train variants and {test_count} test variants")
print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")
EOF

# Convert to USD if requested
if [ "$CONVERT_TO_USD" = true ]; then
    echo ""
    echo "Converting URDF variants to USD..."
    
    # Create train and test subdirectories in USD output
    USD_TRAIN_DIR="${USD_OUTPUT_DIR}/train"
    USD_TEST_DIR="${USD_OUTPUT_DIR}/test"
    mkdir -p "$USD_TRAIN_DIR"
    mkdir -p "$USD_TEST_DIR"
    
    # Path to convert_urdf.py in IsaacLab
    CONVERT_SCRIPT="$ISAACLAB_ROOT/IsaacLab/scripts/tools/convert_urdf.py"
    
    # Check if convert_urdf.py exists
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        echo "Error: convert_urdf.py not found at $CONVERT_SCRIPT"
        exit 1
    fi
    
    # Function to convert URDF files in a directory
    convert_directory() {
        local urdf_dir=$1
        local usd_dir=$2
        local set_name=$3
        
        echo ""
        echo "=========================================="
        echo "Converting $set_name set..."
        echo "=========================================="
        
        # Find URDF files
        mapfile -t urdf_files < <(find "$urdf_dir" -maxdepth 1 -name "*.urdf" -type f | sort)
        urdf_count=${#urdf_files[@]}
        
        if [ $urdf_count -eq 0 ]; then
            echo "No URDF files found in $urdf_dir"
            return
        fi
        
        echo "Found $urdf_count URDF file(s) in $set_name set"
        
        local converted=0
        local failed=0
        
        # Process each file
        for urdf_file in "${urdf_files[@]}"; do
            if [ ! -f "$urdf_file" ]; then
                echo "Warning: Skipping invalid file: $urdf_file"
                continue
            fi
            
            # Get base filename without extension
            base_name=$(basename "$urdf_file" .urdf)
            usd_file="$usd_dir/${base_name}.usd"
            
            echo ""
            echo "Converting: $(basename "$urdf_file") -> $(basename "$usd_file")"
            
            # Convert using convert_urdf.py with headless mode and merge-joints
            set +e
            python "$CONVERT_SCRIPT" \
                "$urdf_file" \
                "$usd_file" \
                --headless \
                --merge-joints
            convert_status=$?
            set -e
            
            if [ $convert_status -eq 0 ]; then
                converted=$((converted + 1))
                echo "  ✓ Success ($converted/$urdf_count)"
            else
                failed=$((failed + 1))
                echo "  ✗ Failed (exit code: $convert_status)"
            fi
        done
        
        echo ""
        echo "$set_name set: Converted: $converted, Failed: $failed"
    }
    
    # Convert train and test sets
    convert_directory "$TRAIN_DIR" "$USD_TRAIN_DIR" "Train"
    convert_directory "$TEST_DIR" "$USD_TEST_DIR" "Test"
    
    echo ""
    echo "✓ USD conversion complete!"
fi

