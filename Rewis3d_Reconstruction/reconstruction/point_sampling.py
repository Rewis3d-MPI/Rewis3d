# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

import numpy as np


def sample_indices_random_uniform(coords, num_samples=120000, replace=False):
    """
    Uniformly sample point indices from coords.

    Args:
        coords (np.ndarray): [N, 3] or [N, D] array of point coordinates.
        num_samples (int): Number of points to sample (default 120000).
        replace (bool): Whether to sample with replacement (default False).

    Returns:
        np.ndarray: Sampled indices (<= num_samples if not enough points).
    """
    N = coords.shape[0]
    if N <= num_samples:
        return np.arange(N)
    return np.random.choice(np.arange(N), size=num_samples, replace=replace)


def sample_indices_random_radius(
    coords,
    point_indices_array,
    num_samples,
    image_ratio=0.4,
    radius_expansion_factor=1.5,
):
    """
    Samples a subset of 3D point indices, prioritizing points near the image
    correspondences within an expanded radius.

    Args:
        coords (np.ndarray): [N, 3] array of 3D coordinates.
        point_indices_array (np.ndarray): Indices of points relevant to the current image.
        num_samples (int): Total number of points to sample.
        image_ratio (float): Target fraction of samples to keep from image-specific points.
        radius_expansion_factor (float): Factor to expand the radius calculated
                                         from image points for sampling.

    Returns:
        np.ndarray: Final sampled indices.
    """
    N = coords.shape[0]
    all_indices = np.arange(N)

    # Ensure image point indices are unique and valid
    point_indices_array = np.unique(point_indices_array)
    point_indices_array = point_indices_array[
        point_indices_array < N
    ]  # Filter out invalid indices if any

    if len(point_indices_array) == 0:
        # Handle case with no image points: fall back to random sampling from all points
        print(
            "Warning: No valid image points found. Performing random sampling on all points."
        )
        if N <= num_samples:
            return all_indices
        else:
            return np.random.choice(all_indices, size=num_samples, replace=False)

    # Step 1: Calculate the centroid and radius based on image points
    image_points_coords = coords[point_indices_array]
    if image_points_coords.shape[0] == 0:
        # Handle case where indices exist but corresponding coords don't (shouldn't happen with filtering)
        print(
            "Warning: Image point indices found, but no corresponding coordinates. Performing random sampling on all points."
        )
        if N <= num_samples:
            return all_indices
        else:
            return np.random.choice(all_indices, size=num_samples, replace=False)

    centroid = np.mean(image_points_coords, axis=0)
    distances_from_centroid = np.linalg.norm(image_points_coords - centroid, axis=1)
    image_radius = (
        np.max(distances_from_centroid) if len(distances_from_centroid) > 0 else 0
    )

    # Step 2: Define the sampling radius
    sampling_radius = image_radius * radius_expansion_factor

    # Step 3: Identify all points within the sampling radius
    all_distances = np.linalg.norm(coords - centroid, axis=1)
    within_radius_mask = all_distances <= sampling_radius
    indices_within_radius = all_indices[within_radius_mask]

    if len(indices_within_radius) == 0:
        # Handle case where no points are within the radius (unlikely but possible)
        print(
            "Warning: No points found within the calculated sampling radius. Performing random sampling on all points."
        )
        if N <= num_samples:
            return all_indices
        else:
            return np.random.choice(all_indices, size=num_samples, replace=False)

    # Step 4: Separate image points and other points *within the radius*
    image_points_in_radius = np.intersect1d(
        point_indices_array, indices_within_radius, assume_unique=True
    )
    other_points_in_radius = np.setdiff1d(
        indices_within_radius, image_points_in_radius, assume_unique=True
    )

    # Step 5: Sample image points within the radius
    max_image_points = int(num_samples * image_ratio)
    if len(image_points_in_radius) > max_image_points:
        keep_image_points = np.random.choice(
            image_points_in_radius, size=max_image_points, replace=False
        )
    else:
        keep_image_points = (
            image_points_in_radius  # Keep all available image points in radius
        )

    # Step 6: Sample remaining points from *other points within the radius*
    remaining_budget = num_samples - len(keep_image_points)
    if remaining_budget <= 0:
        # Budget filled by image points alone
        combined_indices = keep_image_points
    else:
        if len(other_points_in_radius) > remaining_budget:
            sampled_others_in_radius = np.random.choice(
                other_points_in_radius, size=remaining_budget, replace=False
            )
        else:
            # Take all other points within radius if budget allows
            sampled_others_in_radius = other_points_in_radius

        combined_indices = np.concatenate([keep_image_points, sampled_others_in_radius])

    # Step 7: Fix possible undersampling if not enough points *within the radius*
    if len(combined_indices) < num_samples:
        extra_needed = num_samples - len(combined_indices)
        # Pool for extra points: all points *outside* the currently sampled set
        extra_pool = np.setdiff1d(all_indices, combined_indices, assume_unique=True)

        if len(extra_pool) >= extra_needed:
            extra_sampled = np.random.choice(
                extra_pool, size=extra_needed, replace=False
            )
        else:
            # Take all remaining points if still not enough
            extra_sampled = extra_pool

        combined_indices = np.concatenate([combined_indices, extra_sampled])

    # Ensure exact number of samples if possible, shuffle final result
    final_indices = np.random.choice(
        combined_indices, size=min(num_samples, len(combined_indices)), replace=False
    )
    np.random.shuffle(final_indices)  # Shuffle to mix image and non-image points

    return final_indices
