#!/usr/bin/env python3
"""
Visualize 3D point cloud data from reconstructed NPZ files.

Uses Open3D for interactive 3D visualization of point clouds with
RGB colors or semantic label colors.

Usage:
    # Visualize a specific file
    python -m visualizations.visualize_reconstructed_3d_data \
        --config reconstruction/config/kitti360.yaml \
        --file /path/to/specific.npz

    # Visualize every nth file from output directory
    python -m visualizations.visualize_reconstructed_3d_data \
        --config reconstruction/config/kitti360.yaml \
        --every-nth 10

    # Visualize with semantic labels instead of RGB
    python -m visualizations.visualize_reconstructed_3d_data \
        --config reconstruction/config/kitti360.yaml \
        --color-mode labels
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import yaml


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_label_colormap(config: dict) -> Dict[int, Tuple[float, float, float]]:
    """
    Create a mapping from label IDs to RGB colors (normalized to [0, 1]).

    Args:
        config: Configuration dict with labels.class_definitions

    Returns:
        Dictionary mapping label ID to (R, G, B) tuple in [0, 1] range
    """
    colormap = {}
    ignore_index = config.get("labels", {}).get("ignore_index", 255)

    # Default color for ignore index (white)
    colormap[ignore_index] = (1.0, 1.0, 1.0)

    class_definitions = config.get("labels", {}).get("class_definitions", [])
    for class_def in class_definitions:
        # Format: [name, original_id, mapped_id, [R, G, B]]
        if len(class_def) >= 4:
            mapped_id = class_def[2]
            rgb = class_def[3]
            if mapped_id != ignore_index:
                colormap[mapped_id] = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

    return colormap


def load_npz_file(npz_path: str) -> dict:
    """
    Load NPZ file and return its contents.

    Handles nested data structures where 3D data may be stored
    inside a 'data_3d' dictionary.
    """
    data = np.load(npz_path, allow_pickle=True)
    result = {key: data[key] for key in data.files}

    # Check if data is nested inside 'data_3d'
    if "data_3d" in result:
        data_3d = result["data_3d"].item()  # .item() extracts dict from 0-d array
        # Merge 3D data into top-level dict
        result.update(data_3d)

    return result


def create_point_cloud(
    data: dict,
    color_mode: str = "rgb",
    colormap: Optional[Dict[int, Tuple[float, float, float]]] = None,
    label_key: str = "student_segment",
) -> o3d.geometry.PointCloud:
    """
    Create an Open3D point cloud from NPZ data.

    Args:
        data: Dictionary with NPZ data
        color_mode: "rgb" for original colors, "labels" for semantic colors
        colormap: Label ID to RGB mapping (required if color_mode="labels")
        label_key: Which label field to use ("student_segment" or "original_segment")

    Returns:
        Open3D PointCloud object
    """
    # Get coordinates
    if "student_coord" in data:
        coords = data["student_coord"]
    elif "coord" in data:
        coords = data["coord"]
    else:
        raise KeyError("No coordinate data found in NPZ file")

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))

    # Set colors based on mode
    if color_mode == "rgb":
        if "student_colors" in data:
            colors = data["student_colors"]
        elif "color" in data:
            colors = data["color"]
        else:
            # Default to white if no colors
            colors = np.ones((len(coords), 3), dtype=np.float32)

        # Ensure colors are in [0, 1] range
        if colors.max() > 1.0:
            colors = colors / 255.0

        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    elif color_mode == "labels":
        if colormap is None:
            raise ValueError("colormap required for label visualization")

        labels = data.get(label_key, data.get("segment", None))
        if labels is None:
            raise KeyError(f"No label data found (tried '{label_key}' and 'segment')")

        # Map labels to colors
        colors = np.zeros((len(labels), 3), dtype=np.float64)
        for label_id, rgb in colormap.items():
            mask = labels == label_id
            colors[mask] = rgb

        # Default color for unknown labels (gray)
        unknown_mask = ~np.isin(labels, list(colormap.keys()))
        colors[unknown_mask] = (0.5, 0.5, 0.5)

        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def get_npz_files(
    output_dir: str,
    split: str = "training",
    every_nth: int = 1,
    max_files: Optional[int] = None,
) -> List[str]:
    """
    Get list of NPZ files from output directory.

    Args:
        output_dir: Path to output directory
        split: Dataset split ("training" or "validation")
        every_nth: Only return every nth file
        max_files: Maximum number of files to return

    Returns:
        List of NPZ file paths
    """
    split_dir = os.path.join(output_dir, split)
    if not os.path.exists(split_dir):
        # Try without split subdirectory
        split_dir = output_dir

    pattern = os.path.join(split_dir, "*.npz")
    files = sorted(glob.glob(pattern))

    # Apply every_nth filter
    files = files[::every_nth]

    # Apply max_files limit
    if max_files is not None:
        files = files[:max_files]

    return files


def visualize_point_cloud(
    pcd: o3d.geometry.PointCloud,
    window_name: str = "Point Cloud Visualization",
    point_size: float = 1.0,
):
    """
    Visualize a point cloud using Open3D.

    Args:
        pcd: Open3D PointCloud object
        window_name: Window title
        point_size: Size of points in visualization
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1920, height=1080)

    # Add geometry
    vis.add_geometry(pcd)

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background

    # Set view control
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)

    # Run visualizer
    vis.run()
    vis.destroy_window()


def visualize_multiple_clouds(
    pcds: List[o3d.geometry.PointCloud], names: List[str], point_size: float = 1.0
):
    """
    Visualize multiple point clouds sequentially.

    Args:
        pcds: List of Open3D PointCloud objects
        names: List of names for each point cloud
        point_size: Size of points in visualization
    """
    print(f"\nVisualization controls:")
    print("  - Left mouse: Rotate")
    print("  - Middle mouse / Scroll: Zoom")
    print("  - Right mouse: Pan")
    print("  - Q / Esc: Close current and show next")
    print(f"\nShowing {len(pcds)} point clouds...")

    for i, (pcd, name) in enumerate(zip(pcds, names)):
        print(f"\n[{i + 1}/{len(pcds)}] {name}")
        print(f"  Points: {len(pcd.points)}")
        visualize_point_cloud(
            pcd, window_name=f"[{i + 1}/{len(pcds)}] {name}", point_size=point_size
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D point cloud data from reconstructed NPZ files"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (for label colors and output directory)",
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Path to specific NPZ file to visualize"
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="Visualize every nth file from output directory (default: 1)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of files to visualize (default: 10)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "validation"],
        help="Dataset split to visualize (default: training)",
    )
    parser.add_argument(
        "--color-mode",
        type=str,
        default="rgb",
        choices=["rgb", "labels", "student_labels", "original_labels"],
        help="Color mode: rgb (original), labels/student_labels (scribble), original_labels (full mask)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Point size for visualization (default: 2.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    output_dir = args.output_dir or config["dataset"]["output_dir"]

    # Create colormap for label visualization
    colormap = create_label_colormap(config)

    # Determine label key based on color mode
    if args.color_mode in ["labels", "student_labels"]:
        label_key = "student_segment"
        color_mode = "labels"
    elif args.color_mode == "original_labels":
        label_key = "original_segment"
        color_mode = "labels"
    else:
        label_key = "student_segment"
        color_mode = "rgb"

    # Get files to visualize
    if args.file:
        npz_files = [args.file]
    else:
        npz_files = get_npz_files(
            output_dir,
            split=args.split,
            every_nth=args.every_nth,
            max_files=args.max_files,
        )

    if not npz_files:
        print(f"No NPZ files found in {output_dir}/{args.split}/")
        return 1

    print(f"Found {len(npz_files)} files to visualize")
    print(f"Color mode: {args.color_mode}")

    # Load and create point clouds
    pcds = []
    names = []
    for npz_path in npz_files:
        try:
            data = load_npz_file(npz_path)
            pcd = create_point_cloud(
                data, color_mode=color_mode, colormap=colormap, label_key=label_key
            )
            pcds.append(pcd)
            names.append(os.path.basename(npz_path))
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue

    if not pcds:
        print("No point clouds could be loaded")
        return 1

    # Visualize
    visualize_multiple_clouds(pcds, names, point_size=args.point_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
