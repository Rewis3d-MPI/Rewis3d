# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
MapAnything reconstruction method.

This module provides the MapAnything-based reconstruction implementation,
wrapping the original MapAnything model from Meta.
"""

from .method import MapAnythingMethod

__all__ = ["MapAnythingMethod"]
