#!/usr/bin/env python3
"""
Script to convert URDF files to USD format using Isaac Lab's UrdfConverter.
Can convert a single file or batch convert multiple files.
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Parse arguments before launching app
parser = argparse.ArgumentParser(description="Convert URDF files to USD format.")
parser.add_argument("--input", type=str, help="Input URDF file path")
parser.add_argument("--output", type=str, help="Output USD file path (directory + filename)")
parser.add_argument("--output-dir", type=str, help="Output directory for USD files")
parser.add_argument("--batch-dir", type=str, help="Directory containing URDF files to batch convert")
parser.add_argument("--force", action="store_true", default=True, help="Force USD conversion")
parser.add_argument("--make-instanceable", action="store_true", default=True, help="Make USD instanceable")
parser.add_argument("--collider-type", type=str, default="convex_hull", help="Collider type (convex_hull, mesh, etc.)")
parser.add_argument("--merge-fixed-joints", action="store_true", default=True, help="Merge fixed joints")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Set headless mode for batch conversion
args_cli.headless = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.dict import print_dict


def convert_urdf_to_usd(urdf_path, usd_output_path, force=True, make_instanceable=True, 
                        collider_type="convex_hull", merge_fixed_joints=True):
    """Convert a single URDF file to USD format."""
    urdf_path = os.path.abspath(urdf_path)
    
    # Determine output directory and filename
    if os.path.isdir(usd_output_path):
        # If output is a directory, use input filename
        usd_dir = usd_output_path
        usd_filename = Path(urdf_path).stem + ".usd"
    else:
        # If output is a file path
        usd_dir = os.path.dirname(usd_output_path)
        usd_filename = os.path.basename(usd_output_path)
        if not usd_filename.endswith('.usd'):
            usd_filename += ".usd"
    
    # Create output directory if it doesn't exist
    os.makedirs(usd_dir, exist_ok=True)
    
    # Create UrdfConverter config
    # Note: When merge_fixed_joints=True, joints are merged so joint_drive is not needed
    converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        usd_file_name=usd_filename,
        force_usd_conversion=force,
        make_instanceable=make_instanceable,
        fix_base=False,
        collider_type=collider_type,
        merge_fixed_joints=merge_fixed_joints,
        link_density=0.0,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
    )
    
    # Convert URDF to USD
    print(f"Converting: {urdf_path}")
    print(f"Output: {os.path.join(usd_dir, usd_filename)}")
    
    converter = UrdfConverter(converter_cfg)
    print(f"✓ Generated: {converter.usd_path}")
    
    return converter.usd_path


def batch_convert(urdf_dir, output_dir, **kwargs):
    """Batch convert all URDF files in a directory."""
    urdf_dir = os.path.abspath(urdf_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Find all URDF files
    urdf_files = list(Path(urdf_dir).glob("*.urdf"))
    
    if not urdf_files:
        print(f"No URDF files found in {urdf_dir}")
        return
    
    print(f"Found {len(urdf_files)} URDF files to convert")
    print(f"Output directory: {output_dir}\n")
    
    for i, urdf_file in enumerate(urdf_files, 1):
        print(f"[{i}/{len(urdf_files)}] ", end="")
        try:
            convert_urdf_to_usd(str(urdf_file), output_dir, **kwargs)
        except Exception as e:
            print(f"✗ Error converting {urdf_file}: {e}")
        print()


def main():
    """Main function."""
    if args_cli.batch_dir:
        # Batch conversion mode
        if not args_cli.output_dir:
            print("Error: --output-dir required when using --batch-dir")
            sys.exit(1)
        batch_convert(
            args_cli.batch_dir,
            args_cli.output_dir,
            force=args_cli.force,
            make_instanceable=args_cli.make_instanceable,
            collider_type=args_cli.collider_type,
            merge_fixed_joints=args_cli.merge_fixed_joints
        )
    elif args_cli.input:
        # Single file conversion mode
        if not args_cli.output:
            print("Error: --output required when using --input")
            sys.exit(1)
        convert_urdf_to_usd(
            args_cli.input,
            args_cli.output,
            force=args_cli.force,
            make_instanceable=args_cli.make_instanceable,
            collider_type=args_cli.collider_type,
            merge_fixed_joints=args_cli.merge_fixed_joints
        )
    else:
        print("Error: Must specify either --input or --batch-dir")
        parser.print_help()
        sys.exit(1)
    
    # Close simulation app
    simulation_app.close()


if __name__ == "__main__":
    main()

