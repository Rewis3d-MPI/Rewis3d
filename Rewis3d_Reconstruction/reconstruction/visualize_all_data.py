#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import yaml

# Visualization imports
import matplotlib.pyplot as plt
from PIL import Image

# Local utilities for label color mappings
from .label_utils import create_label_mappings

# Color to use for ignore_index (default: white)
IGNORE_COLOR_RGB = (255, 255, 255)


def print_stats(name, value):
    if isinstance(value, np.ndarray):
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            try:
                vmin = value.min()
                vmax = value.max()
            except Exception:
                vmin = "n/a"
                vmax = "n/a"
        else:
            vmin = "n/a"
            vmax = "n/a"
        print(
            f"{name:25s} shape={value.shape} dtype={value.dtype} min={vmin} max={vmax}"
        )
    else:
        print(f"{name:25s} type={type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="Print contents (keys, shapes, min/max) of a reconstruction NPZ file"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to reconstruction YAML config (not strictly needed, kept for parity)",
    )
    parser.add_argument(
        "--npz", required=True, help="Path to NPZ file produced by the pipeline"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"NPZ file not found: {args.npz}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(2)

    # Load (config kept just to mirror previous signature; we do not use it here)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    npz = np.load(args.npz, allow_pickle=True)

    print(f"File: {args.npz}")
    print("Top-level keys:", list(npz.keys()))

    # data_3d
    if "data_3d" in npz:
        try:
            data_3d = npz["data_3d"].item()
            print("\n=== data_3d ===")
            for k, v in data_3d.items():
                print_stats(k, v)
        except Exception as e:
            print(f"Failed to parse data_3d: {e}")
    else:
        print("\n(no data_3d key)")

    # data_2d
    if "data_2d" in npz:
        try:
            data_2d = npz["data_2d"].item()
            print("\n=== data_2d ===")
            for k, v in data_2d.items():
                print_stats(k, v)

            # Try to visualize image and student_mask_1
            try:
                _visualize_image_and_mask(data_2d, config)
            except Exception as e:
                print(f"Visualization failed (image/mask): {e}")
        except Exception as e:
            print(f"Failed to parse data_2d: {e}")
    else:
        print("\n(no data_2d key)")

    # Any other top-level arrays (excluding the dict-wrapped ones)
    print("\n=== other top-level arrays ===")
    for k in npz.keys():
        if k in ("data_2d", "data_3d"):
            continue
        arr = npz[k]
        print_stats(k, arr)


# ----------------------------- Helpers ---------------------------------


def _load_image_from_data2d(data_2d: dict):
    """Return an RGB uint8 image from data_2d if present.

    Prefers 'student_image_1' array. Falls back to *path keys if available.
    Returns None if no image found.
    """
    # Case 1: Image array is embedded
    if "student_image_1" in data_2d:
        img = data_2d["student_image_1"]
        if isinstance(img, np.ndarray):
            arr = img
            # Normalize/convert to uint8 and ensure 3 channels
            if arr.dtype != np.uint8:
                # If in [0,1], scale; if out of range, clip to [0,255]
                a_min, a_max = float(arr.min()), float(arr.max())
                if a_max <= 1.0 and a_min >= 0.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                # grayscale -> RGB
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                pass  # already RGB
            else:
                return None
            return arr
        # If it's not a numpy array, ignore and try paths

    # Case 2: Path-based keys (best-effort)
    path_key = None
    for k in ("image_path", "img_path", "rgb_path"):
        if k in data_2d:
            path_key = k
            break
    if path_key is not None:
        p = data_2d.get(path_key)
        if isinstance(p, np.ndarray):
            try:
                p = p.item()
            except Exception:
                p = None
        if isinstance(p, str) and os.path.isfile(p):
            try:
                return np.array(Image.open(p).convert("RGB"))
            except Exception:
                return None
    return None


def _colorize_mask_student(
    mask: np.ndarray, config: dict, ignore_color=IGNORE_COLOR_RGB
) -> np.ndarray:
    """Colorize a student mask using config-defined trainId colors.

    Steps:
    - Map raw IDs (from PNG) -> trainIDs using config mapping
    - Build a color LUT over trainIDs and apply
    - Ignore index (usually 255) becomes black
    """
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray):
        return None
    if mask.ndim != 2:
        # If a 3-channel mask sneaks in, reduce to first channel
        mask = mask[..., 0]

    mappings = create_label_mappings(config)
    id2train = mappings["id2trainId_2d"]
    ignore_index = int(mappings["ignore_index"]) if "ignore_index" in mappings else 255
    trainId2color = mappings["trainId2color_2d"]

    # Build ID -> trainID LUT (0..255). Unmapped -> ignore_index
    lut_size = 256
    id2train_lut = np.full(lut_size, ignore_index, dtype=np.uint8)
    for src_id, tid in id2train.items():
        if 0 <= int(src_id) < lut_size:
            id2train_lut[int(src_id)] = np.uint8(
                tid if 0 <= int(tid) < 256 else ignore_index
            )

    # Vectorized remap
    mask_u8 = mask.astype(np.uint8)
    train_ids = id2train_lut[mask_u8]

    # Build trainID -> color LUT
    color_lut = np.zeros((lut_size, 3), dtype=np.uint8)
    for tid, col in trainId2color.items():
        if 0 <= int(tid) < lut_size:
            color_lut[int(tid)] = np.array(col, dtype=np.uint8)
    # Force ignore_index to a predefined constant color
    if 0 <= ignore_index < lut_size:
        color_lut[int(ignore_index)] = np.array(ignore_color, dtype=np.uint8)

    colorized = color_lut[train_ids]
    return colorized


def _visualize_image_and_mask(data_2d: dict, config: dict):
    """Show the RGB image and the colorized student_mask_1 side-by-side."""
    img = _load_image_from_data2d(data_2d)
    student_mask = data_2d.get("student_mask_1", None)

    if img is None and student_mask is None:
        print("No image or student_mask_1 found to visualize.")
        return

    # Prepare figures
    if (
        img is not None
        and student_mask is not None
        and isinstance(student_mask, np.ndarray)
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = axes.ravel()

        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        colorized = _colorize_mask_student(student_mask, config)
        if colorized is None:
            axes[1].text(
                0.5, 0.5, "student_mask_1 not available", ha="center", va="center"
            )
            axes[1].axis("off")
        else:
            axes[1].imshow(colorized)
            axes[1].set_title("student_mask_1 (colored)")
            axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        return

    # Fallbacks: show whichever exists
    if img is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")
        plt.show()

    if isinstance(student_mask, np.ndarray):
        colorized = _colorize_mask_student(student_mask, config)
        plt.figure(figsize=(6, 5))
        if colorized is None:
            plt.text(0.5, 0.5, "student_mask_1 not available", ha="center", va="center")
            plt.axis("off")
        else:
            plt.imshow(colorized)
            plt.title("student_mask_1 (colored)")
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
