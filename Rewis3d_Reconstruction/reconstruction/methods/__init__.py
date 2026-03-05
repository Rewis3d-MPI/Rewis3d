# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Reconstruction methods package.

This package provides a pluggable architecture for different 3D reconstruction methods.
Each method is implemented as a subclass of BaseReconstructionMethod and registered
via the method registry.

Usage:
    from reconstruction.methods import get_reconstruction_method

    method = get_reconstruction_method("map_anything", device_id=0)
    predictions = method.reconstruct(image_paths)
"""

from .base import BaseReconstructionMethod
from .registry import get_reconstruction_method, list_available_methods, register_method

__all__ = [
    "BaseReconstructionMethod",
    "get_reconstruction_method",
    "list_available_methods",
    "register_method",
]
