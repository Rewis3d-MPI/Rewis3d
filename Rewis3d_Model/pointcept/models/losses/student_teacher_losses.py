from typing import Optional
from itertools import filterfalse
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss

from .builder import LOSSES
from .lovasz import LovaszLoss

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"


@LOSSES.register_module()
class PartialConsistencyLoss(nn.Module):
    def __init__(
        self,
        h_name,
        ignore_index=255,
        beta=0.5,
        loss_type="cross_entropy",
        sky_id=None,
        sky_boost: float = 1.0,
    ):
        super().__init__()

        # -------------------------------------------------------------
        # keep your original init …
        # -------------------------------------------------------------
        if h_name == "cross_entropy":
            H = nn.CrossEntropyLoss
        else:
            raise ValueError(f"Wrong h_name {h_name}.")  # small typo fix

        self.ignore_index = ignore_index
        self.supervised_loss = H(ignore_index=ignore_index, reduction="mean")
        self.hard_label_loss = H(ignore_index=ignore_index, reduction="mean")
        self.consistency_loss = nn.KLDivLoss(reduction="mean")
        self.beta = beta
        self.loss_type = loss_type

        # ⇩⇩⇩   store the two NEW parameters  ⇩⇩⇩
        self.sky_id = sky_id
        self.sky_boost = sky_boost

    def forward(
        self,
        student_output,
        teacher_output,
        student_label,
        teacher_weight,
        teacher_hard_label=None,
    ):
        loss_s = self.compute_supervised_loss(student_output, student_label)
        mask = student_label == self.ignore_index

        # soft_consistency_mask = mask & ~teacher_hard_label
        loss_u = self.compute_consistency_loss(
            student_output, teacher_output, mask=mask
        )

        if self.loss_type == "cross_entropy":
            return loss_s
        elif self.loss_type == "fixed_consistency":
            return (1 - self.beta) * loss_s + self.beta * loss_u
        elif self.loss_type == "weighted_consistency":
            return (1 - self.beta) * loss_s + self.beta * teacher_weight * loss_u
        elif self.loss_type == "hard_label":
            if torch.all(teacher_hard_label == False):
                return (1 - self.beta) * loss_s + self.beta * teacher_weight * loss_u
            student_output_reshaped = student_output.permute(0, 2, 3, 1)
            teacher_output_reshaped = (
                teacher_output.permute(0, 2, 3, 1).softmax(-1).max(dim=-1)[1]
            )
            loss_t_hard = self.hard_label_loss(
                student_output_reshaped[teacher_hard_label],
                teacher_output_reshaped[teacher_hard_label],
            )
            return (
                (1 - self.beta) * loss_s + self.beta * loss_t_hard + self.beta * loss_u
            )
        else:
            raise ValueError("Invalid loss type")

    def compute_supervised_loss(self, student_output, student_label):
        """
        CE per pixel, with an optional multiplicative boost on the 'sky' class.
        """
        # apply boost only on valid (non-ignore) sky pixels
        if self.sky_id is not None and self.sky_boost != 1.0:
            per_pixel = F.cross_entropy(
                student_output,
                student_label,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            sky_mask = (student_label == self.sky_id) & (
                student_label != self.ignore_index
            )
            per_pixel = per_pixel + (self.sky_boost - 1) * per_pixel * sky_mask
            loss = per_pixel.mean()
        else:
            # standard CE loss
            loss = self.supervised_loss(student_output, student_label)

        return loss

    def compute_consistency_loss(self, student_output, teacher_output, mask=None):
        student_output_reshaped = student_output.permute(0, 2, 3, 1).log_softmax(-1)
        teacher_output_reshaped = teacher_output.permute(0, 2, 3, 1).softmax(-1)

        return self.consistency_loss(
            student_output_reshaped[mask], teacher_output_reshaped[mask]
        )


@LOSSES.register_module()
class PartialConsistencyLoss3D(nn.Module):
    def __init__(self, h_name, ignore_index=255, beta=0.5, loss_type="cross_entropy"):
        super().__init__()

        if h_name == "cross_entropy":
            H = nn.CrossEntropyLoss
        else:
            raise ValueError(f"Wrong h_name {h_name}.")

        self.ignore_index = ignore_index
        self.supervised_loss = H(ignore_index=ignore_index, reduction="mean")
        self.hard_label_loss = H(ignore_index=ignore_index, reduction="mean")
        self.consistency_loss = nn.KLDivLoss(
            reduction="batchmean"
        )  # Changed to 'batchmean' for flattened tensors
        self.beta = beta
        self.loss_type = loss_type

    def forward(
        self,
        student_output,
        teacher_output,
        student_label,
        teacher_weight,
        teacher_hard_label=None,
    ):
        """
        Args:
            student_output: Tensor of shape [N_voxels, C]
            teacher_output: Tensor of shape [N_voxels, C]
            student_label: Tensor of shape [N_voxels]
            teacher_weight: Scalar tensor
            teacher_hard_label: Tensor of shape [N_voxels], bool or int
        """
        loss_s = self.compute_supervised_loss(student_output, student_label)
        mask = student_label == self.ignore_index  # Shape: [N_voxels]

        loss_u = self.compute_consistency_loss(
            student_output, teacher_output, mask=mask
        )

        if self.loss_type == "cross_entropy":
            return loss_s
        elif self.loss_type == "fixed_consistency":
            return (1 - self.beta) * loss_s + self.beta * loss_u
        elif self.loss_type == "weighted_consistency":
            return (1 - self.beta) * loss_s + self.beta * teacher_weight * loss_u
        elif self.loss_type == "hard_label":
            if not teacher_hard_label.any():
                return (1 - self.beta) * loss_s + self.beta * teacher_weight * loss_u
            loss_t_hard = self.hard_label_loss(
                student_output[teacher_hard_label],
                teacher_output.argmax(dim=1)[teacher_hard_label],
            )
            return (
                (1 - self.beta) * loss_s + self.beta * loss_t_hard + self.beta * loss_u
            )
        else:
            raise ValueError("Invalid loss type")

    def compute_supervised_loss(self, student_output, student_label):
        """
        Compute supervised cross-entropy loss on labeled voxels.
        """
        return self.supervised_loss(student_output, student_label)

    def compute_consistency_loss(self, student_output, teacher_output, mask=None):
        """
        Compute consistency loss on unlabeled voxels.
        """
        # Apply mask to select unlabeled voxels
        student_output_masked = student_output[mask].log_softmax(dim=1)
        teacher_output_masked = teacher_output[mask].softmax(dim=1)

        return self.consistency_loss(student_output_masked, teacher_output_masked)
