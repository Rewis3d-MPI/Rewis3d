# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Reconstruction package for 3D point cloud generation from images.

This package provides a modular architecture for different reconstruction methods.
The main entry point is `generate_reconstruction()` which uses the method specified
in the config file.

Available reconstruction methods:
    - map_anything: Meta's MapAnything model for dense multi-view reconstruction

Example usage:
    from reconstruction import generate_reconstruction, get_available_methods

    # List available methods
    print(get_available_methods())  # ['map_anything', ...]

    # Generate reconstruction using config
    config = {
        "reconstruction": {
            "method": "map_anything",
            "confidence_percentile": 30
        }
    }
    predictions = generate_reconstruction(config, image_paths, device_id=0)

To add a new reconstruction method:
    1. Create a new folder under reconstruction/methods/your_method/
    2. Implement a class inheriting from BaseReconstructionMethod
    3. Register it in reconstruction/methods/registry.py
"""

from .create_reconstructions import generate_reconstruction, get_available_methods
from .methods import (
    BaseReconstructionMethod,
    get_reconstruction_method,
    list_available_methods,
    register_method,
)

__all__ = [
    "generate_reconstruction",
    "get_available_methods",
    "BaseReconstructionMethod",
    "get_reconstruction_method",
    "list_available_methods",
    "register_method",
]
