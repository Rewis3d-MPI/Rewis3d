# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Reconstruction module for generating 3D reconstructions from images.

This module provides a unified interface for different reconstruction methods.
The reconstruction method is specified in the config and loaded dynamically.
"""

# Optional config for better memory efficiency
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from typing import Any, Dict, List, Optional

from .methods import get_reconstruction_method, list_available_methods


def generate_reconstruction(
    config: Dict[str, Any], chunk_images: List[str], device_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Process a single chunk of images through the configured reconstruction method.

    Args:
        config: Configuration dictionary containing reconstruction parameters.
                Must have config["reconstruction"]["method"] specifying the method name.
        chunk_images: List of image paths for a single chunk.
        device_id: GPU device ID for multi-GPU support. If None, uses default device.

    Returns:
        predictions: List of per-view prediction dicts from the reconstruction method.
                    Each dict contains at minimum:
                    - pts3d: 3D points array of shape (H, W, 3)
                    - conf: Confidence scores of shape (H, W)
                    - mask: Valid point mask of shape (H, W)
                    - img_no_norm: Original image in [0, 1] range of shape (H, W, 3)

    Raises:
        ValueError: If the specified reconstruction method is not available.
    """
    # Get reconstruction method name from config
    method_name = config.get("reconstruction", {}).get("method", "map_anything")

    # Get the reconstruction method instance
    method = get_reconstruction_method(method_name, device_id=device_id)

    # Run reconstruction
    output = method.reconstruct(chunk_images, config)

    return output.predictions


def get_available_methods() -> List[str]:
    """
    Get list of available reconstruction methods.

    Returns:
        List of method names that can be used in config["reconstruction"]["method"].
    """
    return list_available_methods()
