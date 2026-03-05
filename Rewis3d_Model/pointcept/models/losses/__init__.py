# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

from .builder import build_criteria

from .misc import CrossEntropyLoss, SmoothCELoss, DiceLoss, FocalLoss, BinaryFocalLoss

# from .lovasz import LovaszLoss
from .student_teacher_losses import *
from .cmc_loss import *
