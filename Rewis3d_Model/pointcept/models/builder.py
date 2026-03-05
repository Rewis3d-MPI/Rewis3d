# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
from pointcept.utils.registry import Registry

MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    """Build models."""
    return MODELS.build(copy.deepcopy(cfg))
