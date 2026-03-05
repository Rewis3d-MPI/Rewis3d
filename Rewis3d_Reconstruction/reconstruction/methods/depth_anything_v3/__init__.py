# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Depth Anything V3 reconstruction method.

This module provides the Depth Anything V3-based reconstruction implementation
for metric depth estimation with pose estimation and sky segmentation.
"""

from .method import DepthAnythingV3Method

__all__ = ["DepthAnythingV3Method"]
