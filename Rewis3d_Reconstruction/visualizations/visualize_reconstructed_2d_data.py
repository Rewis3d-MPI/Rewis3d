#!/usr/bin/env python3
# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Visualize 2D data from reconstructed NPZ files.

Creates matplotlib visualizations showing:
- Original RGB image
- Depth map
- Semantic segmentation (student/scribble labels)
- Semantic segmentation (original/full labels)

Usage:
    # Visualize a specific file
    python -m visualizations.visualize_reconstructed_2d_data \
        --config reconstruction/config/kitti360.yaml \
        --file /path/to/specific.npz

    # Visualize every nth file from output directory
    python -m visualizations.visualize_reconstructed_2d_data \
        --config reconstruction/config/kitti360.yaml \
        --every-nth 10

    # Save visualizations to disk instead of displaying
    python -m visualizations.visualize_reconstructed_2d_data \
        --config reconstruction/config/kitti360.yaml \
        --save-dir ./visualizations_output
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_label_colormap(config: dict) -> Tuple[ListedColormap, Dict[int, str]]:
    """
    Create matplotlib colormap and label names from config.

    Args:
        config: Configuration dict with labels.class_definitions

    Returns:
        Tuple of (ListedColormap, dict mapping label ID to name)
    """
    ignore_index = config.get("labels", {}).get("ignore_index", 255)
    class_definitions = config.get("labels", {}).get("class_definitions", [])

    # Find max label ID for colormap size
    max_id = ignore_index
    for class_def in class_definitions:
        if len(class_def) >= 3:
            mapped_id = class_def[2]
            if mapped_id != ignore_index and mapped_id > max_id:
                max_id = mapped_id

    # Create color array (default gray for unmapped)
    colors = np.ones((max_id + 2, 4)) * 0.5  # RGBA, default gray
    colors[:, 3] = 1.0  # Full opacity

    # Map names
    label_names = {ignore_index: "ignore"}

    for class_def in class_definitions:
        if len(class_def) >= 4:
            name = class_def[0]
            mapped_id = class_def[2]
            rgb = class_def[3]

            if mapped_id != ignore_index and mapped_id <= max_id:
                colors[mapped_id, :3] = [c / 255.0 for c in rgb]
                label_names[mapped_id] = name

    # Set ignore index color (white)
    if ignore_index <= max_id:
        colors[ignore_index, :3] = [1.0, 1.0, 1.0]

    return ListedColormap(colors), label_names


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


def reconstruct_2d_image(
    data: dict, key: str, shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Reconstruct 2D image from NPZ data if available.

    Args:
        data: Dictionary with NPZ data
        key: Key to look for in data
        shape: Expected (H, W) shape

    Returns:
        2D numpy array or None if not available
    """
    if key in data:
        arr = data[key]
        if arr.ndim == 1:
            # Try to reshape to 2D
            try:
                return arr.reshape(shape)
            except ValueError:
                return None
        elif arr.ndim == 2:
            return arr
    return None


def get_npz_files(
    output_dir: str,
    split: str = "training",
    every_nth: int = 1,
    max_files: Optional[int] = None,
) -> List[str]:
    """
    Get list of NPZ files from output directory.
    """
    split_dir = os.path.join(output_dir, split)
    if not os.path.exists(split_dir):
        split_dir = output_dir

    pattern = os.path.join(split_dir, "*.npz")
    files = sorted(glob.glob(pattern))
    files = files[::every_nth]

    if max_files is not None:
        files = files[:max_files]

    return files


def visualize_npz_2d(
    data: dict,
    config: dict,
    title: str = "NPZ Visualization",
    save_path: Optional[str] = None,
):
    """
    Create 2D visualization of NPZ data.

    Args:
        data: Dictionary with NPZ data
        config: Configuration dict for label colors
        title: Figure title
        save_path: If provided, save figure to this path instead of displaying
    """
    colormap, label_names = create_label_colormap(config)

    # Determine what data is available
    has_image = "image" in data
    has_depth = "depth" in data
    has_student_seg = "student_segment_2d" in data or "student_segment" in data
    has_original_seg = "original_segment_2d" in data or "original_segment" in data
    has_coords = "student_coord" in data or "coord" in data

    # Count available panels
    panels = []
    if has_image:
        panels.append(("image", "RGB Image"))
    if has_depth:
        panels.append(("depth", "Depth Map"))
    if has_student_seg:
        panels.append(("student_seg", "Student Labels (Scribbles)"))
    if has_original_seg:
        panels.append(("original_seg", "Original Labels (Full)"))

    if not panels:
        # Fall back to showing 3D data statistics
        print(f"No 2D data found in {title}")
        print(f"  Available keys: {list(data.keys())}")
        if has_coords:
            coords = data.get("student_coord", data.get("coord"))
            print(f"  3D Points: {len(coords)}")
        return

    # Create figure
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=12)

    for ax, (panel_type, panel_title) in zip(axes, panels):
        if panel_type == "image":
            img = data["image"]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            ax.set_title(panel_title)

        elif panel_type == "depth":
            depth = data["depth"]
            # Handle various depth formats
            if depth.ndim == 3:
                depth = depth.squeeze()
            # Clip extreme values for better visualization
            vmin, vmax = (
                np.percentile(depth[depth > 0], [5, 95])
                if (depth > 0).any()
                else (0, 1)
            )
            im = ax.imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(panel_title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth")

        elif panel_type == "student_seg":
            seg = data.get("student_segment_2d", data.get("student_segment"))
            if seg.ndim == 1:
                # Can't visualize 1D segment data as image
                ax.text(
                    0.5,
                    0.5,
                    f"1D data\n{len(seg)} points",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(panel_title)
            else:
                ax.imshow(seg, cmap=colormap, interpolation="nearest")
                ax.set_title(panel_title)

        elif panel_type == "original_seg":
            seg = data.get("original_segment_2d", data.get("original_segment"))
            if seg.ndim == 1:
                ax.text(
                    0.5,
                    0.5,
                    f"1D data\n{len(seg)} points",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(panel_title)
            else:
                ax.imshow(seg, cmap=colormap, interpolation="nearest")
                ax.set_title(panel_title)

        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def print_npz_info(data: dict, name: str):
    """Print information about NPZ file contents."""
    print(f"\n{'=' * 60}")
    print(f"File: {name}")
    print(f"{'=' * 60}")

    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if value.size < 10:
                print(f"    values: {value}")
            elif value.ndim == 1:
                print(f"    range: [{value.min():.3f}, {value.max():.3f}]")
                unique = np.unique(value)
                if len(unique) <= 20:
                    print(f"    unique values: {unique}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2D data from reconstructed NPZ files"
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
        default=5,
        help="Maximum number of files to visualize (default: 5)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "validation"],
        help="Dataset split to visualize (default: training)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (if not set, displays interactively)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print NPZ file info, don't visualize",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    output_dir = args.output_dir or config["dataset"]["output_dir"]

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

    # Create save directory if needed
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Process each file
    for i, npz_path in enumerate(npz_files):
        try:
            data = load_npz_file(npz_path)
            name = os.path.basename(npz_path)

            # Print info
            print_npz_info(data, name)

            if not args.info_only:
                # Create visualization
                save_path = None
                if args.save_dir:
                    save_path = os.path.join(
                        args.save_dir, f"{name.replace('.npz', '')}_2d.png"
                    )

                visualize_npz_2d(data, config, title=name, save_path=save_path)

        except Exception as e:
            print(f"Error processing {npz_path}: {e}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
