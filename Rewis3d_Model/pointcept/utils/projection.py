# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

import numpy as np
import torch


def unproject_points(
    point_indices,
    pixel_coords,
    segmentation_mask,
    batch_indices,
    starting_indices,
    num_points,
    ignore_index=255,
    output_dtype=None,
):
    if output_dtype is None:
        output_dtype = segmentation_mask.dtype

    device = segmentation_mask.device
    is_4d_mask = segmentation_mask.dim() == 4

    if is_4d_mask:
        # segmentation_mask is [B, C, H, W] (e.g., logits/softmax)
        B, C, H_img, W_img = segmentation_mask.shape
        fill_value = 0.0  # For float/logits
        point_labels = torch.full(
            (num_points, C), fill_value, dtype=output_dtype, device=device
        )
    else:
        # segmentation_mask is [B, H, W] (e.g., labels/confidence)
        B, H_img, W_img = segmentation_mask.shape
        fill_value = ignore_index if output_dtype == torch.long else 0.0
        point_labels = torch.full(
            (num_points,), fill_value, dtype=output_dtype, device=device
        )

    v = pixel_coords[:, 0].long()  # Image H coordinates
    u = pixel_coords[:, 1].long()  # Image W coordinates

    # Valid coordinates mask
    valid_coords = (v >= 0) & (v < H_img) & (u >= 0) & (u < W_img)

    valid_point_indices_in_batch = point_indices[
        valid_coords
    ]  # Local point indices within their batch
    valid_v = v[valid_coords]
    valid_u = u[valid_coords]
    valid_batch_for_coords = batch_indices[valid_coords]

    # Global point indices across all batches
    global_point_indices = (
        valid_point_indices_in_batch + starting_indices[valid_batch_for_coords]
    )

    if is_4d_mask:
        # Gather [C] channels for each valid correspondence
        # segmentation_mask[batch_idx, :, H_coord, W_coord]
        labels_to_assign = segmentation_mask[
            valid_batch_for_coords, :, valid_v, valid_u
        ]  # Shape: [num_valid_correspondences, C]
    else:
        # Gather scalar value for each valid correspondence
        # segmentation_mask[batch_idx, H_coord, W_coord]
        labels_to_assign = segmentation_mask[
            valid_batch_for_coords, valid_v, valid_u
        ]  # Shape: [num_valid_correspondences]

    labels_to_assign = labels_to_assign.to(output_dtype)

    # Assign gathered values to the corresponding global point indices
    if labels_to_assign.numel() > 0:  # Ensure there are valid labels to assign
        point_labels[global_point_indices] = labels_to_assign

    return point_labels


def project_points(
    point_indices,
    pixel_coords,
    segment_3d,
    batch_indices,
    B,
    H,
    W,
    ignore_index=255,
    output_dtype=None,
):
    if output_dtype is None:
        output_dtype = segment_3d.dtype

    device = segment_3d.device
    is_2d_segment = segment_3d.dim() == 2  # segment_3d is [N, C]

    if is_2d_segment:
        # segment_3d is [N, C] (e.g., logits/softmax)
        N_points, C = segment_3d.shape
        fill_value = 0.0  # For float/logits
        projected_image = torch.full(
            (B, C, H, W), fill_value, dtype=output_dtype, device=device
        )
    else:
        # segment_3d is [N] (e.g., labels/confidence)
        fill_value = ignore_index if output_dtype == torch.long else 0.0
        projected_image = torch.full(
            (B, H, W), fill_value, dtype=output_dtype, device=device
        )

    v = pixel_coords[:, 0].long()  # Image H coordinates
    u = pixel_coords[:, 1].long()  # Image W coordinates

    # Valid coordinates mask
    valid_coords = (v >= 0) & (v < H) & (u >= 0) & (u < W)

    # Filter based on valid image coordinates
    valid_point_indices_for_projection = point_indices[
        valid_coords
    ]  # These are global point indices
    valid_v = v[valid_coords]
    valid_u = u[valid_coords]
    valid_batch_for_projection = batch_indices[valid_coords]

    # Ensure that the point indices themselves are valid (not -1, if that's a convention)
    # This step might be redundant if point_indices are always valid global indices from correspondences
    # but good for robustness if point_indices can contain sentinels.
    # For this example, assuming point_indices are already filtered global indices from valid correspondences.

    if valid_point_indices_for_projection.numel() == 0:
        return projected_image

    # Gather values from segment_3d for valid points
    values_3d_to_project = segment_3d[valid_point_indices_for_projection].to(
        output_dtype
    )  # Shape: [num_valid_proj, C] or [num_valid_proj]

    if is_2d_segment:
        # Assign [C] channels for each valid correspondence
        # projected_image[batch_idx, :, H_coord, W_coord] = values_3d_to_project[i] (effectively)
        # Need to handle potential multiple points projecting to the same pixel.
        # A simple assignment overwrites; averaging or max-pooling might be alternatives.
        # For now, direct assignment (last one wins for duplicates).
        # A loop is clearer for this type of assignment if scatter is not used.
        for i in range(values_3d_to_project.shape[0]):
            b_idx = valid_batch_for_projection[i]
            h_idx = valid_v[i]
            w_idx = valid_u[i]
            projected_image[b_idx, :, h_idx, w_idx] = values_3d_to_project[i]
    else:
        # Assign scalar value for each valid correspondence
        projected_image[valid_batch_for_projection, valid_v, valid_u] = (
            values_3d_to_project
        )

    return projected_image
