import os
import re
from typing import Any, Dict, Tuple, Union

import numpy as np
from PIL import Image
from tqdm import tqdm


def _to_numpy(tensor_or_array: Union[np.ndarray, Any]) -> np.ndarray:
    """
    Convert a tensor or array to numpy, handling both torch tensors and numpy arrays.

    Args:
        tensor_or_array: Input that may be a torch.Tensor or numpy.ndarray.

    Returns:
        Numpy array with batch dimensions squeezed.
    """
    if isinstance(tensor_or_array, np.ndarray):
        return np.squeeze(tensor_or_array)
    # Assume it's a torch tensor
    return tensor_or_array.squeeze().detach().cpu().numpy()


def _compute_fixed_mapping_affine(
    orig_h: int, orig_w: int, tgt_h: int, tgt_w: int
) -> Tuple[float, float, float, float]:
    """
    Compute scale (sx, sy) and crop offsets (off_x, off_y) that map target-space pixels
    (produced by load_images with resize_mode="fixed_mapping") back to original image pixels.

    Assumes center-crop to match target aspect, then resize to (tgt_w, tgt_h).

    Mapping from target (u_t, v_t) -> original (u0, v0):
      u0 = u_t / sx + off_x
      v0 = v_t / sy + off_y
    """
    Rt = tgt_w / float(tgt_h)
    R0 = orig_w / float(orig_h)

    if R0 >= Rt:
        # Wider than target: crop width, keep full height
        crop_w = orig_h * Rt
        off_x = (orig_w - crop_w) / 2.0
        off_y = 0.0
        sx = tgt_w / crop_w
        sy = tgt_h / float(orig_h)
    else:
        # Taller than target: crop height, keep full width
        crop_h = orig_w / Rt
        off_x = 0.0
        off_y = (orig_h - crop_h) / 2.0
        sx = tgt_w / float(orig_w)
        sy = tgt_h / crop_h

    return float(sx), float(sy), float(off_x), float(off_y)


def _map_target_to_original(
    x_t: np.ndarray, y_t: np.ndarray, sx: float, sy: float, off_x: float, off_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized mapping of target pixel coords (x_t, y_t) to original coords (x0, y0)."""
    x0 = x_t.astype(np.float32) / sx + off_x
    y0 = y_t.astype(np.float32) / sy + off_y
    return x0, y0


def _extract_camera_id(image_path: str) -> int:
    """Extract camera id from filename (pattern: image_<id>.<ext>). Returns -1 if not found."""
    m = re.search(r"image_(\d+)\.[^.]+$", image_path)
    return int(m.group(1)) if m else -1


def _label_paths(image_path: str, cam_id: int, config: dict) -> tuple:
    """
    Get paths for student and original segment masks for 3D unprojection.
    Uses configurable mask names from config, with fallback to defaults.
    """
    base_dir = os.path.dirname(image_path)

    # Get mask naming patterns from config with defaults
    masks_config = config.get("masks", {})
    student_3d_pattern = masks_config.get("student_segment_3d", "scribble_{cam_id}.png")
    original_3d_pattern = masks_config.get("original_segment_3d", "mask_{cam_id}.png")

    # Replace {cam_id} placeholder with actual camera ID
    student_3d_name = student_3d_pattern.replace("{cam_id}", str(cam_id))
    original_3d_name = original_3d_pattern.replace("{cam_id}", str(cam_id))

    return (
        os.path.join(base_dir, student_3d_name),
        os.path.join(base_dir, original_3d_name),
    )


def create_unified_pointcloud(predictions, chunk_images, config):
    """
    Create a unified pointcloud from all views in the chunk and unproject 2D labels.

    Args:
        predictions: MapAnything predictions for the chunk
        chunk_images: List of image paths in the chunk
        config: Configuration dictionary

    Returns:
        dict: Contains unified pointcloud data with labels
    """
    confidence_percentile = config["reconstruction"].get("confidence_percentile", 30)

    all_points = []
    all_colors = []
    all_confidence = []
    all_student_segments = []
    all_original_segments = []
    view_point_mapping: Dict[int, Dict[str, np.ndarray]] = {}
    view_has_label = []  # Track if any label (scribble or full) exists for each view

    tqdm.write(f"Creating unified pointcloud from {len(predictions)} views…")

    for view_idx, (pred, image_path) in enumerate(zip(predictions, chunk_images)):
        # Extract data from prediction - handle both torch tensors and numpy arrays
        pts3d = _to_numpy(pred["pts3d"])  # (H_t, W_t, 3)
        confidence = _to_numpy(pred["conf"])  # (H_t, W_t)
        mask = _to_numpy(pred["mask"])  # (H_t, W_t)
        img_no_norm = _to_numpy(pred["img_no_norm"])  # (H_t, W_t, 3) in [0,1]

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[..., 0]

        # Load original image and masks at original resolution (no resizing)
        original_image = np.array(Image.open(image_path))
        cam_id = _extract_camera_id(image_path)
        scribble_mask_path, original_mask_path = _label_paths(
            image_path, cam_id, config
        )
        scribble_mask = load_mask(scribble_mask_path)
        original_mask = load_mask(original_mask_path)

        recon_h, recon_w = pts3d.shape[:2]
        img_h, img_w = original_image.shape[:2]

        tqdm.write(f"View {view_idx + 1}: Image shape: {original_image.shape}")
        tqdm.write(f"View {view_idx + 1}: pts3d shape: {pts3d.shape}")
        tqdm.write(f"View {view_idx + 1}: Mask shape: {mask.shape}")
        tqdm.write(f"View {view_idx + 1}: Confidence shape: {confidence.shape}")

        # Compute mapping parameters from target grid to original image
        sx, sy, off_x, off_y = _compute_fixed_mapping_affine(
            img_h, img_w, recon_h, recon_w
        )

        # 1) Get ALL valid reconstruction pixels
        valid_rc = mask > 0
        y_t, x_t = np.nonzero(valid_rc)
        num_valid = y_t.shape[0]
        tqdm.write(f"View {view_idx + 1}: Total reconstruction points: {num_valid}")

        if num_valid == 0:
            view_point_mapping[view_idx] = {
                "point_indices": np.zeros((0,), dtype=int),
                "pixel_coords": np.zeros((0, 2), dtype=int),
                "recon_shape": (recon_h, recon_w),
                "image_shape": (img_h, img_w),
            }
            continue

        # 2) Map to original pixel coordinates
        x0, y0 = _map_target_to_original(x_t, y_t, sx, sy, off_x, off_y)
        x0_i = np.clip(np.rint(x0).astype(int), 0, img_w - 1)
        y0_i = np.clip(np.rint(y0).astype(int), 0, img_h - 1)

        # 3) Collect 3D points, colors, confidence at valid_rc
        pts_valid = pts3d[y_t, x_t, :]  # (N,3)
        cols_valid = img_no_norm[y_t, x_t, :]  # (N,3) already [0,1]
        conf_valid = confidence[y_t, x_t]

        # 4) Sample labels from original-resolution masks
        view_labeled = False
        if scribble_mask is not None:
            scribble_labels = scribble_mask[y0_i, x0_i]
            view_labeled = True
        else:
            scribble_labels = np.full((num_valid,), 255, dtype=np.uint8)

        if original_mask is not None:
            original_labels = original_mask[y0_i, x0_i]
            view_labeled = True
        else:
            original_labels = np.full((num_valid,), 255, dtype=np.uint8)
        view_has_label.append(view_labeled)

        # 5) Apply confidence filtering AFTER unprojection
        conf_valid_only = conf_valid[np.isfinite(conf_valid)]
        if conf_valid_only.size == 0:
            thr = -np.inf
        else:
            thr = np.percentile(conf_valid_only, confidence_percentile)
        keep = conf_valid >= thr
        tqdm.write(
            f"View {view_idx + 1}: Confidence threshold: {thr:.3f}; kept {keep.sum()} / {num_valid}"
        )

        pts_kept = pts_valid[keep]
        cols_kept = cols_valid[keep]
        conf_kept = conf_valid[keep]
        scribble_kept = scribble_labels[keep]
        original_kept = original_labels[keep]
        y0_kept = y0_i[keep]
        x0_kept = x0_i[keep]

        # 6) Append to unified arrays and build mapping
        base_idx = len(all_points)
        all_points.extend(pts_kept.tolist())
        all_colors.extend(cols_kept.tolist())
        all_confidence.extend(conf_kept.tolist())
        all_student_segments.extend(scribble_kept.tolist())
        all_original_segments.extend(original_kept.tolist())

        unified_indices = base_idx + np.arange(pts_kept.shape[0], dtype=int)
        pix_coords = np.stack([y0_kept, x0_kept], axis=1).astype(int)
        view_point_mapping[view_idx] = {
            "point_indices": unified_indices,
            "pixel_coords": pix_coords,
            "recon_shape": (recon_h, recon_w),
            "image_shape": (img_h, img_w),
        }

    # Convert to numpy arrays
    unified_data = {
        "student_coord": np.array(all_points, dtype=np.float32),
        "student_colors": np.array(all_colors, dtype=np.float32),
        "conf": np.array(all_confidence, dtype=np.float32),
        "student_segment": np.array(all_student_segments, dtype=np.uint8),
        "original_segment": np.array(all_original_segments, dtype=np.uint8),
        "view_point_mapping": view_point_mapping,
        "chunk_images": chunk_images,
        "view_has_label": view_has_label,
    }

    tqdm.write(f"Unified pointcloud created with {len(all_points)} total points")
    return unified_data


def load_mask(mask_path):
    """Load mask file and return as numpy array (grayscale)."""
    if not os.path.exists(mask_path):
        return None

    try:
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr
    except Exception:
        return None
