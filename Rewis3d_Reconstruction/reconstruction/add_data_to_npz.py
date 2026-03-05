#!/usr/bin/env python3
"""
Add additional 2D mask arrays into existing NPZ files without re-running reconstruction.

Usage examples:

  python reconstruction/add_data_to_npz.py \
    --config reconstruction/config/waymo.yaml \
    --key sam_mask_1 \
    --mask-pattern sam_mask_{cam_id}.png

This will:
  - Scan the output_dir from the given config for existing .npz files
  - For each NPZ, reconstruct the original image path using dataset_dir, split, drive, image folder and image file name
  - Build the mask file path using --mask-pattern (supports {cam_id}) in the image folder
  - Load the mask and resize with NEAREST to match the NPZ's student_image_1 shape
  - Insert it into data_2d under the provided --key, and re-save the NPZ

Notes:
  - If a mask file is missing, the NPZ will be updated with a value of None for the key (warning logged)
  - Overwrites NPZ in-place by writing to a temporary path then renaming (atomic-ish for local filesystems)

"""

import argparse
import os
import re
import sys
import tempfile
import shutil
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import yaml
except Exception:
    print("Error: PyYAML is required. Please install with `pip install pyyaml`.")
    raise


def parse_args():
    p = argparse.ArgumentParser(
        description="Add additional 2D mask to existing NPZ files."
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to dataset/reconstruction YAML config (to get dataset_dir, output_dir, file_extension, scaling).",
    )
    p.add_argument(
        "--key",
        required=True,
        help="Key name to store the new mask in data_2d (e.g., sam_mask_1).",
    )
    p.add_argument(
        "--mask-pattern",
        required=True,
        help="Mask filename pattern relative to the image folder (supports {cam_id}, e.g., sam_mask_{cam_id}.png).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir in config. If omitted, uses config['dataset']['output_dir'].",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print what would be done.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many NPZ files (0 = no limit).",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_npz_to_image_path(
    npz_path: str, output_dir: str, dataset_dir: str, file_extension: str
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Given an NPZ path like:
        <output_dir>/<split>/<drive>_<image_folder>_cam{id}_image_{id}.npz

    Reconstruct the original image path:
        <dataset_dir>/<split>/<drive>/<image_folder>/image_{id}.<file_extension>

    Important: The filename component after cam{id} is 'image_{id}' (contains an underscore),
    so we cannot naïvely split by a fixed number of underscores. We use a regex to capture
    the trailing pattern and then split the remaining prefix on the last underscore to
    separate <drive> and <image_folder>.

    Returns (image_path, image_folder_dir, cam_id)
      - image_folder_dir is the directory that should contain the mask files
    """
    # Find split as the immediate subdirectory under output_dir
    try:
        rel = os.path.relpath(npz_path, output_dir)
    except Exception:
        return None, None, None

    parts = rel.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None, None, None
    split = parts[0]
    fname = os.path.splitext(parts[-1])[0]

    # Expect suffix pattern: _cam{ID}_image_{N}
    m = re.match(
        r"^(?P<prefix>.+)_(?P<camTag>cam(?:\d+|X))_(?P<image>image_\d+)$", fname
    )
    if not m:
        return None, None, None

    prefix = m.group("prefix")
    cam_tag = m.group("camTag")
    image_name = m.group("image")

    # Extract drive and image_folder from prefix by splitting at last underscore
    if "_" not in prefix:
        return None, None, None
    drive_part, image_folder = prefix.rsplit("_", 1)

    # Extract camera id from cam_tag (expected cam\d+ or camX)
    mcam = re.match(r"cam(\d+|X)", cam_tag)
    cam_id = None
    if mcam:
        cam_token = mcam.group(1)
        cam_id = int(cam_token) if cam_token.isdigit() else None

    # Construct image path and folder dir
    image_filename = f"{image_name}.{file_extension}"
    image_folder_dir = os.path.join(dataset_dir, split, drive_part, image_folder)
    image_path = os.path.join(image_folder_dir, image_filename)
    return image_path, image_folder_dir, cam_id


def load_mask_resize(
    mask_path: str, target_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Load a mask and resize with NEAREST to target (H, W). Return None if missing or failed."""
    if not os.path.exists(mask_path):
        print(f"[WARN] Mask file not found: {mask_path}")
        return None
    try:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != target_shape:
            H, W = target_shape
            mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
        return mask
    except Exception as e:
        print(f"[WARN] Failed to load/resize mask {mask_path}: {e}")
        return None


def process_npz(
    npz_path: str, key: str, mask_pattern: str, config: dict, dry_run: bool = False
) -> bool:
    """
    Insert mask into data_2d[key] for the given NPZ. Returns True if updated, False otherwise.
    """
    dataset_dir = config["dataset"]["dataset_dir"]
    output_dir = config["dataset"]["output_dir"]
    file_extension = config["dataset"].get("file_extension", "png")

    # Derive original image path and cam id from npz filename
    image_path, image_folder_dir, cam_id = parse_npz_to_image_path(
        npz_path, output_dir, dataset_dir, file_extension
    )
    if image_path is None or image_folder_dir is None:
        print(f"[WARN] Could not parse image path from NPZ filename: {npz_path}")
        return False

    # Build mask filename from pattern
    cam_token = str(cam_id if cam_id is not None else "1")
    mask_name = mask_pattern.replace("{cam_id}", cam_token)
    mask_path = os.path.join(image_folder_dir, mask_name)

    # Load NPZ and extract data dicts
    try:
        with np.load(npz_path, allow_pickle=True) as npz:
            data_2d = npz.get("data_2d", None)
            data_3d = npz.get("data_3d", None)
            if data_2d is None or data_3d is None:
                print(f"[WARN] Missing data_2d or data_3d in {npz_path}")
                return False
            # Unwrap object arrays to dicts
            data_2d = data_2d.item() if isinstance(data_2d, np.ndarray) else data_2d
            data_3d = data_3d.item() if isinstance(data_3d, np.ndarray) else data_3d
    except Exception as e:
        print(f"[WARN] Failed to open NPZ {npz_path}: {e}")
        return False

    # Determine target shape from stored student_image_1
    if "student_image_1" not in data_2d:
        print(
            f"[WARN] data_2d.student_image_1 missing in {npz_path}; cannot determine target shape"
        )
        return False
    target_shape = data_2d["student_image_1"].shape[:2]

    # Load & resize mask
    mask_arr = load_mask_resize(mask_path, target_shape)

    # Update data_2d
    data_2d[key] = mask_arr  # can be None; mirrors behavior in save_dataset

    if dry_run:
        print(
            f"[DRY] Would update {npz_path}: set data_2d['{key}'] from {mask_path} -> shape {None if mask_arr is None else mask_arr.shape}"
        )
        return True

    # Write to temp in the same directory as the target to avoid cross-device link errors
    target_dir = os.path.dirname(npz_path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".npz", prefix="tmpaddmask_", dir=target_dir
    )
    os.close(tmp_fd)
    try:
        np.savez_compressed(tmp_path, data_3d=data_3d, data_2d=data_2d)
        try:
            os.replace(tmp_path, npz_path)
        except OSError:
            # Fallback in case of EXDEV or similar errors
            try:
                if os.path.exists(npz_path):
                    os.remove(npz_path)
            except Exception:
                pass
            shutil.move(tmp_path, npz_path)
        print(
            f"[OK] Updated {npz_path} with key '{key}' from {os.path.basename(mask_path)}"
        )
        return True
    except Exception as e:
        print(f"[ERR] Failed to write updated NPZ for {npz_path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or config["dataset"]["output_dir"]

    if not os.path.isdir(output_dir):
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    # Gather NPZ files (under split subfolders)
    npz_files = []
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".npz"):
                npz_files.append(os.path.join(root, f))

    if not npz_files:
        print(f"No NPZ files found in {output_dir}")
        return

    npz_files.sort()
    total = len(npz_files)
    limit = args.limit if args.limit and args.limit > 0 else total
    processed = 0
    updated = 0

    print(f"Found {total} NPZ files. Processing up to {limit}...")
    for npz_path in npz_files[:limit]:
        ok = process_npz(
            npz_path, args.key, args.mask_pattern, config, dry_run=args.dry_run
        )
        processed += 1
        updated += int(ok)

    print(f"Done. Processed: {processed}, Updated: {updated}, Dry-run: {args.dry_run}")


if __name__ == "__main__":
    main()
