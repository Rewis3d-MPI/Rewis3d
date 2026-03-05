# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

import os
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

from .point_sampling import (
    sample_indices_random_radius,
    sample_indices_random_uniform,
)


def unproject_points_non_vec(
    point_indices_array, pixel_coords_array, segmentation_mask, num_points
):
    """
    Assign labels from a 2D segmentation mask to 3D points based on their pixel projections.
    Replicating the exact logic from the working old reconstruction method.
    """
    # Initialize labels with default value 255
    point_labels = np.full((num_points,), 255, dtype=segmentation_mask.dtype)

    for idx in range(point_indices_array.shape[0]):
        point_idx = point_indices_array[idx]
        if point_idx == -1:
            continue  # Skip invalid or padded entries

        v = int(pixel_coords_array[idx, 0])  # y coordinate
        u = int(pixel_coords_array[idx, 1])  # x coordinate

        # Check bounds
        if 0 <= v < segmentation_mask.shape[0] and 0 <= u < segmentation_mask.shape[1]:
            # Retrieve the segmentation label from the mask
            label = segmentation_mask[v, u]
            # Assign the label to the corresponding 3D point
            point_labels[point_idx] = label

    return point_labels


def _extract_camera_id(image_path: str) -> int:
    m = re.search(r"image_(\d+)\.[^.]+$", image_path)
    return int(m.group(1)) if m else -1


def _extract_split_from_path(image_path: str) -> str:
    """Infer split name from image path components.

    Returns one of {"training", "validation", "train", "val"} if found, else "".
    """
    parts = image_path.replace("\\", "/").split("/")
    for p in parts:
        lp = p.lower()
        if lp in {"training", "validation", "train", "val"}:
            return lp
    return ""


def save_dataset_for_chunk(unified_data, predictions, config):
    """
    Apply view-aware sampling and save NPZ files for each image in the chunk.

    Args:
        unified_data: Dictionary containing unified pointcloud data
        predictions: MapAnything predictions for the chunk
        config: Configuration dictionary
    """
    output_dir = config["dataset"]["output_dir"]
    num_points = config["point_sampling"]["num_points"]
    sampling_method = config["point_sampling"]["method"]
    image_ratio = config["point_sampling"].get("image_ratio", 0.5)

    os.makedirs(output_dir, exist_ok=True)

    chunk_images = unified_data["chunk_images"]
    view_point_mapping = unified_data["view_point_mapping"]

    tqdm.write(f"Saving dataset for {len(chunk_images)} images…")

    for view_idx in tqdm(range(len(chunk_images)), desc="Saving NPZs", unit="img"):
        image_path = chunk_images[view_idx]
        pred = predictions[view_idx]
        has_label = unified_data.get("view_has_label", [True] * len(chunk_images))[
            view_idx
        ]
        if not has_label:
            tqdm.write(
                f"Skipping save for view {view_idx} (no labels found) -> {os.path.basename(image_path)}"
            )
            continue
        try:
            # Generate output filename from image path
            output_filename = generate_output_filename(image_path, config)
            output_path = os.path.join(output_dir, output_filename)

            # Get view-specific data
            view_data = view_point_mapping[view_idx]
            view_point_indices = view_data["point_indices"]

            # Apply view-aware sampling
            if sampling_method == "random_radius":
                sampled_indices = sample_indices_random_radius(
                    unified_data["student_coord"],
                    view_point_indices,
                    num_points,
                    image_ratio,
                )
            elif sampling_method == "random_uniform":
                sampled_indices = sample_indices_random_uniform(
                    unified_data["student_coord"], num_points, replace=False
                )
            else:
                raise ValueError(f"Unsupported sampling method: {sampling_method}")

            # Create 3D data dictionary
            data_dict_3d = {
                "conf": unified_data["conf"][sampled_indices],
                "student_coord": unified_data["student_coord"][sampled_indices],
                "student_segment": unified_data["student_segment"][sampled_indices],
                "original_segment": unified_data["original_segment"][sampled_indices],
                "student_colors": unified_data["student_colors"][sampled_indices],
            }

            # Create 2D data dictionary
            data_dict_2d = create_2d_data(
                image_path, pred, sampled_indices, view_data, unified_data, config
            )

            # Filter the desired keys (as specified in requirements)
            keep_3d_keys = [
                "conf",
                "student_coord",
                "student_segment",
                "original_segment",
                "student_colors",
            ]
            keep_2d_keys = [
                "student_image_1",
                "student_mask_1",
                "point_indices_array",
                "pixel_coords_array",
                "original_mask_1",
                "depth",
            ]

            data_dict_3d = {k: v for k, v in data_dict_3d.items() if k in keep_3d_keys}
            data_dict_2d = {k: v for k, v in data_dict_2d.items() if k in keep_2d_keys}

            # Ensure output subdirectory exists (handles nested split/drive/image_folder structure)
            output_subdir = os.path.dirname(output_path)
            os.makedirs(output_subdir, exist_ok=True)
            # Save NPZ file
            np.savez_compressed(output_path, data_3d=data_dict_3d, data_2d=data_dict_2d)
            tqdm.write(f"Saved: {output_path}")

        except Exception as e:
            tqdm.write(f"Error saving data for {image_path}: {e}")
            exit()


def create_2d_data(image_path, pred, sampled_indices, view_data, unified_data, config):
    """
    Create 2D data dictionary for the current view.

    Args:
        image_path: Path to the current image
        pred: MapAnything prediction for this view
        sampled_indices: Indices of sampled points in unified pointcloud
        view_data: View-specific mapping data
        unified_data: Unified pointcloud data
        config: Configuration dictionary

    Returns:
        dict: 2D data dictionary
    """
    # Get scale factor from config (default 1.0 = no resize)
    image_scale_factor = config.get("preprocessing", {}).get("image_scale_factor", 1.0)

    # Load image at original resolution
    image = np.array(Image.open(image_path))
    orig_h, orig_w = image.shape[:2]

    # Dynamically resolve camera id from filename (pattern: image_<camid>.<ext>)
    cam_match = re.search(r"image_(\d+)\.[^.]+$", os.path.basename(image_path))
    cam_id = cam_match.group(1) if cam_match else "1"  # fallback to 1 for legacy

    # Get mask naming patterns from config with defaults
    masks_config = config.get("masks", {})
    student_2d_pattern = masks_config.get("student_mask_2d", "scribble_{cam_id}.png")
    original_2d_pattern = masks_config.get("original_mask_2d", "mask_{cam_id}.png")

    # Replace {cam_id} placeholder with actual camera ID
    student_2d_name = student_2d_pattern.replace("{cam_id}", str(cam_id))
    original_2d_name = original_2d_pattern.replace("{cam_id}", str(cam_id))

    # Build expected scribble / mask filenames for this camera
    base_dir = os.path.dirname(image_path)
    scribble_mask_path = os.path.join(base_dir, student_2d_name)
    original_mask_path = os.path.join(base_dir, original_2d_name)

    scribble_mask = load_mask_safe(scribble_mask_path, target_shape=image.shape[:2])
    original_mask = load_mask_safe(original_mask_path, target_shape=image.shape[:2])

    # Optionally override student mask source per split (e.g., use original on validation)
    split_name = _extract_split_from_path(image_path)
    masks_cfg = config.get("masks", {})
    use_original_splits = set(
        s.lower() for s in masks_cfg.get("use_original_for_student_in_splits", [])
    )
    # Normalize split names to match config entries
    split_alias = {
        "train": "training",
        "val": "validation",
        "training": "training",
        "validation": "validation",
    }.get(split_name, split_name)

    if split_alias in use_original_splits:
        student_mask = original_mask
    else:
        student_mask = scribble_mask

    # Get depth from prediction and resize to match original image
    depth_data = pred["depth_z"]
    # Handle both torch tensors and numpy arrays
    if hasattr(depth_data, "cpu"):
        depth = depth_data.squeeze().cpu().numpy()  # PyTorch tensor
    else:
        depth = np.squeeze(depth_data)  # Already numpy array
    if depth.ndim == 3:
        depth = depth.squeeze(-1)

    # CRITICAL: Resize depth to match original image dimensions
    if depth.shape != image.shape[:2]:
        tqdm.write(
            f"Resizing depth from {depth.shape} to {image.shape[:2]} to match original image"
        )
        from PIL import Image as PILImage

        depth_pil = PILImage.fromarray(depth.astype(np.float32))
        depth = np.array(
            depth_pil.resize((image.shape[1], image.shape[0]), PILImage.BILINEAR)
        )

    # Create correspondence arrays at original resolution
    point_indices_array, pixel_coords_array = create_correspondence_arrays(
        sampled_indices, view_data, unified_data, (orig_h, orig_w)
    )

    # Apply resizing if scale factor != 1.0
    if image_scale_factor != 1.0:
        # Calculate target dimensions
        tgt_w = max(1, int(round(orig_w * image_scale_factor)))
        tgt_h = max(1, int(round(orig_h * image_scale_factor)))

        # Resize image with BILINEAR interpolation
        image_pil = Image.fromarray(image)
        image = np.array(image_pil.resize((tgt_w, tgt_h), Image.BILINEAR))

        # Resize masks with NEAREST interpolation (preserve label values)
        if scribble_mask is not None:
            scribble_pil = Image.fromarray(scribble_mask)
            scribble_mask = np.array(scribble_pil.resize((tgt_w, tgt_h), Image.NEAREST))

        if original_mask is not None:
            original_pil = Image.fromarray(original_mask)
            original_mask = np.array(original_pil.resize((tgt_w, tgt_h), Image.NEAREST))

        # Resize depth with BILINEAR interpolation
        depth_pil = Image.fromarray(depth.astype(np.float32))
        depth = np.array(depth_pil.resize((tgt_w, tgt_h), Image.BILINEAR))

        # Scale pixel coordinates
        # pixel_coords_array is (N, 2) in (v, u) format
        s_x = tgt_w / orig_w
        s_y = tgt_h / orig_h

        # Create a mask for valid correspondences (>= 0)
        valid_mask = pixel_coords_array[:, 0] >= 0

        # Scale valid coordinates
        pixel_coords_scaled = pixel_coords_array.astype(np.float32)
        pixel_coords_scaled[valid_mask, 1] *= s_x  # u (x) coordinates
        pixel_coords_scaled[valid_mask, 0] *= s_y  # v (y) coordinates

        # Round to nearest integer
        pixel_coords_scaled = np.rint(pixel_coords_scaled).astype(np.int32)

        # Clamp to new bounds and invalidate out-of-bounds
        for idx in range(len(pixel_coords_scaled)):
            if valid_mask[idx]:
                v, u = pixel_coords_scaled[idx]
                if u < 0 or u >= tgt_w or v < 0 or v >= tgt_h:
                    pixel_coords_scaled[idx] = -1
                else:
                    pixel_coords_scaled[idx, 0] = np.clip(v, 0, tgt_h - 1)
                    pixel_coords_scaled[idx, 1] = np.clip(u, 0, tgt_w - 1)

        pixel_coords_array = pixel_coords_scaled

    # NOTE: Keys retain the *_1 suffix for backward compatibility with downstream code.
    # If desired, we can later generalize naming (e.g., include cam id) once consumers are updated.
    data_dict_2d = {
        "student_image_1": image,
        "student_mask_1": student_mask,
        "point_indices_array": point_indices_array,
        "pixel_coords_array": pixel_coords_array,
        "original_mask_1": original_mask,
        "depth": depth,
    }

    return data_dict_2d


def create_correspondence_arrays(sampled_indices, view_data, unified_data, image_shape):
    """
    Create correspondence arrays between 3D points and 2D pixels.
    Uses the same approach as the working old reconstruction method.

    Args:
        sampled_indices: Indices of sampled points in unified pointcloud
        view_data: View-specific mapping data
        unified_data: Unified pointcloud data
        image_shape: (H, W) shape of the original image

    Returns:
        tuple: (point_indices_array, pixel_coords_array)
    """
    num_sampled = len(sampled_indices)

    # Initialize arrays with -1 (no correspondence)
    point_indices_array = -np.ones(num_sampled, dtype=int)
    pixel_coords_array = -np.ones((num_sampled, 2), dtype=int)

    # Get view-specific point indices in unified cloud and pixel coords in ORIGINAL image space
    view_point_indices = view_data[
        "point_indices"
    ]  # indices into unified pointcloud (sorted)
    view_pixel_coords = view_data[
        "pixel_coords"
    ]  # (N, 2) in (y, x) ORIGINAL image space

    # Find which sampled points correspond to this view
    view_mask = np.isin(sampled_indices, view_point_indices)

    if np.any(view_mask):
        # Subset of sampled_indices that belong to this view (unified indices)
        sampled_view_points = sampled_indices[view_mask]

        # view_point_indices is sorted; use searchsorted to locate pixel coords
        positions = np.searchsorted(view_point_indices, sampled_view_points)
        positions = np.clip(positions, 0, len(view_point_indices) - 1)
        pix_coords = view_pixel_coords[positions]

        # Clamp to image bounds defensively
        H, W = image_shape
        pix_coords = np.asarray(pix_coords, dtype=int)
        pix_coords[:, 0] = np.clip(pix_coords[:, 0], 0, H - 1)
        pix_coords[:, 1] = np.clip(pix_coords[:, 1], 0, W - 1)

        # Reindex: for legacy consumers, point_indices_array should be the sampled array indices (0..N-1)
        # Positions within sampled array are simply the indices where view_mask is True
        sampled_positions = np.nonzero(view_mask)[0]
        point_indices_array[sampled_positions] = sampled_positions
        pixel_coords_array[sampled_positions] = pix_coords

    return point_indices_array, pixel_coords_array


def load_mask_safe(mask_path, target_shape=None):
    """Safely load mask file, return zeros if not found. Optionally resize to target shape."""
    if not os.path.exists(mask_path):
        tqdm.write(f"Warning: Mask file {mask_path} not found, using zeros")
        return None

    try:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Resize to target shape if provided and shapes don't match
        if target_shape is not None and mask.shape != target_shape:
            print(f"Resizing mask from {mask.shape} to {target_shape}")
            from PIL import Image as PILImage

            mask_pil = PILImage.fromarray(mask)
            mask = np.array(
                mask_pil.resize((target_shape[1], target_shape[0]), PILImage.NEAREST)
            )

        return mask
    except Exception as e:
        tqdm.write(f"Error loading mask {mask_path}: {e}")
        return None


def generate_output_filename(image_path, config):
    """
    Generate output filename from image path.
    Format: split_drive_imagefolder_imageid.npz
    """
    # Extract components from path
    path_parts = image_path.split(os.sep)

    # Find split, drive, and image info
    # Expected structure: .../split/drive/image_folder/image_1.png
    try:
        # Go backwards to find the relevant parts
        filename = os.path.splitext(os.path.basename(image_path))[0]  # image_1
        cam_id = _extract_camera_id(image_path)
        image_folder = path_parts[-2]  # folder containing the image
        drive = path_parts[-3]  # drive folder
        split = path_parts[-4]  # split folder
        cam_tag = f"cam{cam_id}" if cam_id != -1 else "camX"
        output_filename = os.path.join(
            split, f"{drive}_{image_folder}_{cam_tag}_{filename}.npz"
        )
        return output_filename
    except IndexError:
        # Fallback to simple filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return f"{base_name}.npz"
