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
