import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss
from typing import Dict, Tuple, Any, Optional, List

from pointcept.utils.projection import project_points, unproject_points
from pointcept.utils.logger import get_root_logger
from .builder import LOSSES


def cross_entropy_with_confidence(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidence_weights: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Computes cross-entropy loss weighted by confidence scores.

    Args:
        logits (torch.Tensor): Predictions [N, C] or [B, C, H, W].
        labels (torch.Tensor): Ground truth labels [N] or [B, H, W].
        confidence_weights (torch.Tensor): Confidence weights [N] or [B, H, W].
        ignore_index (int): Label index to ignore.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    loss = loss * confidence_weights

    # Normalize by sum of weights for valid elements
    valid_mask = labels != ignore_index
    valid_weights_sum = (confidence_weights * valid_mask).sum()

    if valid_weights_sum > 0:
        return loss.sum() / valid_weights_sum
    return loss.sum() * 0.0


# Helper function renamed
def get_cmc_weight(cfg, epoch, which="2d"):
    ramp_start = cfg.cmc_epoch  # e.g., 15
    ramp_length = cfg.cmc_ramp_epochs  # e.g., 5
    max_weight = cfg.cmc_max_weight_2d if which == "2d" else cfg.cmc_max_weight_3d
    if epoch < ramp_start:
        return 0.0
    elif epoch >= ramp_start + ramp_length:
        return max_weight
    else:
        progress = (epoch + 1 - ramp_start) / ramp_length
        return max_weight * progress


@LOSSES.register_module()
class CMCLoss(_Loss):
    """
    Cross-Modal Consistency (CMC) Loss.

    Enforces consistency between 2D and 3D predictions using teacher models
    and confidence-weighted cross-entropy.
    """

    def __init__(self):
        super().__init__()
        self.logger = get_root_logger()  # Cache logger

    # --------------------------------------------------------------------------
    # Helper Methods for Forward Pass Logic
    # --------------------------------------------------------------------------

    def _unpack_inputs(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        data = {}
        data["device"] = input_dict["student_segment"].device
        data["labels_3d"] = input_dict["student_segment"]  # [total_points]
        data["labels_2d"] = input_dict["student_labels_1"]  # [B, H, W]
        data["student_offsets"] = input_dict["student_offset"]  # [B], cumulative count
        data["conf"] = input_dict["conf"].view(-1)
        data["H"], data["W"] = data["labels_2d"].shape[-2:]
        data["B"] = data["labels_2d"].shape[0]
        data["total_points"] = (
            data["student_offsets"][-1].item()
            if isinstance(data["student_offsets"][-1], torch.Tensor)
            else data["student_offsets"][-1]
        )
        data["starting_indices"] = torch.cat(
            [torch.tensor([0], device=data["device"]), data["student_offsets"][:-1]]
        )  # shape [B]

        # Optional correspondence data for CMC
        data["pixel_coords_array"] = input_dict.get("pixel_coords_array", None)
        data["point_indices_array"] = input_dict.get("point_indices_array", None)

        return data

    def _run_student_models(
        self,
        model_3d: nn.Module,
        model_2d: nn.Module,
        input_dict: Dict[str, Any],
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """Runs forward pass for both student models."""
        output_dict_3d_student = model_3d(input_dict)
        student_logits_3d = output_dict_3d_student["seg_logits"]  # [total_points, C]

        output_dict_2d_student = model_2d(input_dict)
        student_logits_2d = output_dict_2d_student["seg_logits"]  # [B, C, H2, W2]

        upsampled_logits_student_2d = F.interpolate(
            student_logits_2d, size=(H, W), mode="bilinear", align_corners=False
        )
        return (
            student_logits_3d,
            upsampled_logits_student_2d,
            output_dict_3d_student,
            output_dict_2d_student,
        )

    def _run_teacher_models_and_process(
        self,
        model_3d: nn.Module,
        model_2d: nn.Module,
        input_dict: Dict[str, Any],
        cfg: Any,
        H: int,
        W: int,
    ) -> Dict[str, Any]:
        """
        Runs teacher models to generate pseudo-labels and confidence scores.

        Returns a dictionary containing logits, hard labels based on thresholds,
        weights, and raw probabilities/predictions for CMC loss calculation.
        """
        with torch.no_grad():
            # ---------------------------
            # Teacher 3D
            # ---------------------------
            output_dict_3d_teacher = model_3d(input_dict)
            teacher_logits_3d = output_dict_3d_teacher["seg_logits"]
            softmax_teacher_3d = F.softmax(teacher_logits_3d, dim=1)
            teacher_prob_3d, teacher_pred_3d = softmax_teacher_3d.max(dim=1)

            # threshold-based for supervised portion
            teacher_hard_label_3d = teacher_prob_3d.ge(
                cfg.model_3d.teacher_conf_threshold
            )
            teacher_weight_3d = teacher_hard_label_3d.float().mean()

            # ---------------------------
            # Teacher 2D
            # ---------------------------
            output_dict_2d_teacher = model_2d(input_dict)
            teacher_logits_2d = output_dict_2d_teacher["seg_logits"]
            upsampled_logits_teacher_2d = F.interpolate(
                teacher_logits_2d, size=(H, W), mode="bilinear", align_corners=False
            )
            softmax_teacher_2d = F.softmax(upsampled_logits_teacher_2d, dim=1)
            teacher_prob_2d, teacher_pred_2d = softmax_teacher_2d.max(dim=1)

            # threshold-based for supervised portion
            teacher_hard_label_2d = teacher_prob_2d.ge(
                cfg.model_2d.teacher_conf_threshold
            )
            teacher_weight_2d = teacher_hard_label_2d.float().mean()

        return {
            # Full logits
            "logits_3d": teacher_logits_3d,
            "upsampled_logits_2d": upsampled_logits_teacher_2d,
            # Hard labels & scalar weights for supervised portion
            "hard_label_3d": teacher_hard_label_3d,
            "hard_label_2d": teacher_hard_label_2d,
            "weight_3d": teacher_weight_3d,
            "weight_2d": teacher_weight_2d,
            # Also store prob & pred so we can do confidence-based CMC
            "prob_3d": teacher_prob_3d,
            "pred_3d": teacher_pred_3d,
            "prob_2d": teacher_prob_2d,
            "pred_2d": teacher_pred_2d,
        }

    def _calculate_supervised_loss(
        self,
        cfg: Any,
        loss_fn_3d: callable,
        loss_fn_2d: callable,
        student_logits_3d: torch.Tensor,
        upsampled_logits_student_2d: torch.Tensor,
        labels_3d: torch.Tensor,
        labels_2d: torch.Tensor,
        teacher_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates supervised loss using ground truth or teacher pseudo-labels."""
        loss_3d_total = torch.tensor(0.0, device=student_logits_3d.device)
        loss_2d_total = torch.tensor(0.0, device=upsampled_logits_student_2d.device)

        # 3D Loss
        if "Fake" in cfg.model_3d.model_class:
            loss_3d_total = (student_logits_3d**2).mean() * 0.0
        elif teacher_info:
            # Teacher-based 3D loss
            loss_3d = loss_fn_3d(
                student_logits_3d,
                teacher_info["logits_3d"],
                labels_3d,
                teacher_info["weight_3d"],
                teacher_info["hard_label_3d"],
            )
            loss_3d_total += loss_3d
        else:
            loss_3d_total = loss_fn_3d(student_logits_3d, labels_3d)

        # 2D Loss
        if "Fake" in cfg.model_2d.model_class:
            loss_2d_total = (upsampled_logits_student_2d**2).mean() * 0.0
        elif teacher_info:
            # Teacher-based 2D loss
            loss_2d = loss_fn_2d(
                upsampled_logits_student_2d,
                teacher_info["upsampled_logits_2d"],
                labels_2d,
                teacher_info["weight_2d"],
                teacher_info["hard_label_2d"],
            )
            loss_2d_total += loss_2d
        else:
            loss_2d_total = loss_fn_2d(upsampled_logits_student_2d, labels_2d)

        return loss_3d_total, loss_2d_total

    # --------------------------------------------------------------------------
    # CMC portion using teacher's confidence instead of threshold-based masks
    # --------------------------------------------------------------------------
    def _calculate_cmc_loss(
        self,
        cfg: Any,
        epoch: int,
        data: Dict[str, Any],
        student_logits_3d: torch.Tensor,
        upsampled_logits_student_2d: torch.Tensor,
        teacher_info: Dict[str, Any],
        output_dict_3d_student: Dict[str, Any],
        output_dict_2d_student: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates Cross-Modal Consistency (CMC) loss.

        Computes confidence-weighted cross-entropy:
        - 2D->3D using teacher 2D predictions and confidence combined with 3D reconstruction confidence.
        - 3D->2D using teacher 3D predictions and confidence combined with 3D reconstruction confidence.
        """
        device = data["device"]
        cmc_loss_3d_val = torch.tensor(0.0, device=device)
        cmc_loss_2d_val = torch.tensor(0.0, device=device)

        # Check if CMC is active
        cmc_active = cfg.use_cmc_loss and epoch >= cfg.cmc_epoch
        correspondences_available = (
            data["pixel_coords_array"] is not None
            and data["point_indices_array"] is not None
        )
        if not (cmc_active and correspondences_available):
            return cmc_loss_3d_val, cmc_loss_2d_val

        # Weights for the overall CMC objective
        cmc_weight_2d = get_cmc_weight(cfg, epoch, which="2d")
        cmc_weight_3d = get_cmc_weight(cfg, epoch, which="3d")

        pixel_coords_array = data["pixel_coords_array"]  # [B, M, 2]
        point_indices_array = data["point_indices_array"]  # [B, M]
        B, M, _ = pixel_coords_array.shape

        batch_indices = (
            torch.arange(B, device=device).unsqueeze(1).expand(-1, M).reshape(-1)
        )
        pixel_coords_flat = pixel_coords_array.view(-1, 2)
        point_indices_flat = point_indices_array.view(-1)

        valid_corr_mask = (
            (pixel_coords_flat[:, 0] != -1)
            & (pixel_coords_flat[:, 1] != -1)
            & (point_indices_flat != -1)
        )
        valid_pixel_coords = pixel_coords_flat[valid_corr_mask]
        valid_batch_indices = batch_indices[valid_corr_mask]
        valid_point_indices = point_indices_flat[valid_corr_mask]
        # Convert to "global" point indices:
        global_point_indices = (
            valid_point_indices + data["starting_indices"][valid_batch_indices]
        )

        # -------------------------------------------------------------------------
        # 2D -> 3D
        # -------------------------------------------------------------------------
        teacher_pred_2d = teacher_info["pred_2d"]  # [B, H, W]
        teacher_prob_2d = teacher_info["prob_2d"]  # [B, H, W]

        # Unproject teacher's labels
        point_labels_from_2d = unproject_points(
            valid_point_indices,
            valid_pixel_coords,
            teacher_pred_2d.long(),
            valid_batch_indices,
            data["starting_indices"],
            data["total_points"],
            output_dtype=torch.long,
        )
        # Unproject teacher's per-pixel confidence
        point_conf_from_2d = unproject_points(
            valid_point_indices,
            valid_pixel_coords,
            teacher_prob_2d.float(),
            valid_batch_indices,
            data["starting_indices"],
            data["total_points"],
            output_dtype=torch.float,
        )

        # Also retrieve the 3D reconstruction confidence: shape [total_points]
        recon_conf_3d = data["conf"]

        # Combine teacher’s confidence (unprojected) with reconstruction confidence
        combined_conf_3d = point_conf_from_2d * recon_conf_3d

        # (Optional) ignore certain classes
        point_labels_from_2d[point_labels_from_2d == cfg.sky_label] = 255

        # Weighted CE in 3D
        cmc_loss_3d_raw = cross_entropy_with_confidence(
            logits=student_logits_3d,
            labels=point_labels_from_2d,
            confidence_weights=combined_conf_3d,
            ignore_index=255,
        )
        cmc_loss_3d_val = cmc_loss_3d_raw * cmc_weight_3d
        output_dict_3d_student["unprojected_labels"] = point_labels_from_2d

        # -------------------------------------------------------------------------
        # 3D -> 2D
        # -------------------------------------------------------------------------
        teacher_pred_3d = teacher_info["pred_3d"]  # [total_points]
        teacher_prob_3d = teacher_info["prob_3d"]  # [total_points]

        # Project teacher's labels into 2D
        projected_labels_from_3d = project_points(
            global_point_indices,
            valid_pixel_coords,
            teacher_pred_3d.long(),
            valid_batch_indices,
            B,
            data["H"],
            data["W"],
            output_dtype=torch.long,
        )
        # Project teacher's 3D confidence into 2D
        projected_conf_from_3d = project_points(
            global_point_indices,
            valid_pixel_coords,
            teacher_prob_3d.float(),
            valid_batch_indices,
            B,
            data["H"],
            data["W"],
            output_dtype=torch.float,
        )

        # Also project the 3D reconstruction confidence into 2D
        projected_recon_conf_3d = project_points(
            global_point_indices,
            valid_pixel_coords,
            recon_conf_3d,  # [total_points]
            valid_batch_indices,
            B,
            data["H"],
            data["W"],
            output_dtype=torch.float,
        )

        # Combine teacher’s projected confidence with reconstruction confidence
        combined_conf_2d_from_3d = projected_conf_from_3d * projected_recon_conf_3d

        # -------------------------------------------------------------------------
        # Sky Teacher Integration (for 3D -> 2D direction)
        # -------------------------------------------------------------------------
        sky_teacher_info = teacher_info.get("sky_teacher", None)

        if sky_teacher_info is not None:
            sky_teacher_pred_2d = sky_teacher_info["pred_2d"]  # [B, H, W]
            sky_teacher_prob_2d = sky_teacher_info["prob_2d"]  # [B, H, W]

            # Calculate density ratio for weight balancing
            total_pixels = B * data["H"] * data["W"]
            projected_pixels = (
                (projected_labels_from_3d != 255).sum().item()
            )  # Non-ignore pixels
            density_ratio = projected_pixels / total_pixels if total_pixels > 0 else 0.0

            self.logger.info(
                f"3D->2D projection density: {density_ratio:.4f} ({projected_pixels}/{total_pixels})"
            )

            # Create combined labels and confidence maps
            combined_labels_2d = projected_labels_from_3d.clone()
            combined_conf_2d = combined_conf_2d_from_3d.clone()

            # Create sky mask where we want to add sky teacher predictions
            sky_mask = sky_teacher_pred_2d == cfg.sky_label

            # For sky pixels, use sky teacher predictions with density-weighted confidence
            sky_confidence_weighted = sky_teacher_prob_2d * density_ratio

            # Only apply sky teacher where 3D projections don't exist (are 255/ignore)
            sky_apply_mask = sky_mask & (projected_labels_from_3d == 255)

            combined_labels_2d[sky_apply_mask] = cfg.sky_label
            combined_conf_2d[sky_apply_mask] = sky_confidence_weighted[sky_apply_mask]

            # Log statistics
            sky_pixels_added = sky_apply_mask.sum().item()
            self.logger.info(
                f"Sky teacher pixels added: {sky_pixels_added}, weighted confidence factor: {density_ratio:.4f}"
            )
        else:
            # Fallback to original method if no sky teacher
            combined_labels_2d = projected_labels_from_3d
            combined_conf_2d = combined_conf_2d_from_3d

        # Weighted CE in 2D with combined predictions
        cmc_loss_2d_raw = cross_entropy_with_confidence(
            logits=upsampled_logits_student_2d,
            labels=combined_labels_2d,
            confidence_weights=combined_conf_2d,
            ignore_index=255,
        )
        cmc_loss_2d_val = cmc_loss_2d_raw * cmc_weight_2d
        output_dict_2d_student["projected_image"] = combined_labels_2d

        return cmc_loss_3d_val, cmc_loss_2d_val

    # --------------------------------------------------------------------------
    # Main Forward Method
    # --------------------------------------------------------------------------

    def forward(
        self,
        cfg: Any,
        model_3d: List[nn.Module],
        model_2d: List[nn.Module],
        input_dict: Dict[str, Any],
        epoch: int,
        loss_fn_3d: callable,
        loss_fn_2d: callable,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """
        Forward pass computing supervised and CMC losses.

        1. Supervised losses:
           - Using teacher info (weight/hard_label) if available.
           - Otherwise using ground-truth labels.
        2. CMC loss:
           - Active if teacher is present and epoch >= cmc_epoch.
           - Uses confidence-based cross-entropy.
        """
        # 1. Unpack inputs and basic info
        data = self._unpack_inputs(input_dict)
        H, W = data["H"], data["W"]

        # 2. Run student models
        (
            student_logits_3d,
            upsampled_logits_student_2d,
            output_dict_3d_student,
            output_dict_2d_student,
        ) = self._run_student_models(model_3d[0], model_2d[0], input_dict, H, W)

        # 3. Initialize total losses
        loss_3d_total = torch.tensor(0.0, device=data["device"])
        loss_2d_total = torch.tensor(0.0, device=data["device"])

        # 4. Teacher logic
        has_3d_teacher = len(model_3d) > 1
        has_2d_teacher = len(model_2d) > 1
        teacher_info = None

        sup_weight_2d = 1.0
        sup_weight_3d = 1.0
        if "sup_weight_2d" in cfg.model_2d.model_class:
            sup_weight_2d = cfg.model_2d.model_class["sup_weight_2d"]
        if "sup_weight_3d" in cfg.model_3d.model_class:
            sup_weight_3d = cfg.model_3d.model_class["sup_weight_3d"]

        if has_3d_teacher and has_2d_teacher:
            # Run teachers and store teacher info (including threshold-based & confidence)
            teacher_info = self._run_teacher_models_and_process(
                model_3d[1], model_2d[1], input_dict, cfg, H, W
            )

            # Add sky teacher info if available (model_2d[2] is the frozen sky teacher)
            has_sky_teacher = len(model_2d) > 2
            if has_sky_teacher:
                with torch.no_grad():
                    sky_output_dict = model_2d[2](input_dict)
                    sky_logits_2d = sky_output_dict["seg_logits"]
                    sky_upsampled_logits = F.interpolate(
                        sky_logits_2d, size=(H, W), mode="bilinear", align_corners=False
                    )
                    sky_softmax = F.softmax(sky_upsampled_logits, dim=1)
                    sky_prob, sky_pred = sky_softmax.max(dim=1)

                    teacher_info["sky_teacher"] = {
                        "pred_2d": sky_pred,
                        "prob_2d": sky_prob,
                        "logits_2d": sky_upsampled_logits,
                    }

            # (Optional) store teacher info in student dict if needed
            output_dict_2d_student["teacher_upsampled_logits_2d"] = teacher_info[
                "upsampled_logits_2d"
            ]

            # 4a. Supervised loss using teacher weighting
            sup_loss_3d, sup_loss_2d = self._calculate_supervised_loss(
                cfg,
                loss_fn_3d,
                loss_fn_2d,
                student_logits_3d,
                upsampled_logits_student_2d,
                data["labels_3d"],
                data["labels_2d"],
                teacher_info=teacher_info,
            )
            loss_3d_total += sup_loss_3d * sup_weight_3d
            loss_2d_total += sup_loss_2d * sup_weight_2d

        else:
            # 4b. Supervised loss using ground-truth labels only
            sup_loss_3d, sup_loss_2d = self._calculate_supervised_loss(
                cfg,
                loss_fn_3d,
                loss_fn_2d,
                student_logits_3d,
                upsampled_logits_student_2d,
                data["labels_3d"],
                data["labels_2d"],
                teacher_info=None,
            )
            loss_3d_total += sup_loss_3d * sup_weight_3d
            loss_2d_total += sup_loss_2d * sup_weight_2d

        # 5. CMC (only if teacher_info is present and epoch >= cfg.cmc_epoch)
        if teacher_info and cfg.use_cmc_loss and epoch >= cfg.cmc_epoch:
            cmc_loss_3d, cmc_loss_2d = self._calculate_cmc_loss(
                cfg,
                epoch,
                data,
                student_logits_3d,
                upsampled_logits_student_2d,
                teacher_info,
                output_dict_3d_student,
                output_dict_2d_student,
            )
            self.logger.info(
                f"CMC Loss 2D: {cmc_loss_2d.item()}, 3D: {cmc_loss_3d.item()}, "
                f"sup_loss_2d: {sup_loss_2d.item()}, sup_loss_3d: {sup_loss_3d.item()}"
            )
            loss_3d_total += cmc_loss_3d
            loss_2d_total += cmc_loss_2d

        # 6. Return final losses and output dicts
        return (
            loss_2d_total,
            loss_3d_total,
            output_dict_2d_student,
            output_dict_3d_student,
        )


@LOSSES.register_module()
class CMCSkyTeacherLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.logger = get_root_logger()  # Cache logger

    # --------------------------------------------------------------------------
    # Helper Methods for Forward Pass Logic
    # --------------------------------------------------------------------------

    def _unpack_inputs(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        data = {}
        data["device"] = input_dict["student_segment"].device
        data["labels_3d"] = input_dict["student_segment"]  # [total_points]
        data["labels_2d"] = input_dict["student_labels_1"]  # [B, H, W]
        data["student_offsets"] = input_dict["student_offset"]  # [B], cumulative count
        data["conf"] = input_dict["conf"].view(-1)
        data["H"], data["W"] = data["labels_2d"].shape[-2:]
        data["B"] = data["labels_2d"].shape[0]
        data["total_points"] = (
            data["student_offsets"][-1].item()
            if isinstance(data["student_offsets"][-1], torch.Tensor)
            else data["student_offsets"][-1]
        )
        data["starting_indices"] = torch.cat(
            [torch.tensor([0], device=data["device"]), data["student_offsets"][:-1]]
        )  # shape [B]

        # Optional correspondence data for UCDC
        data["pixel_coords_array"] = input_dict.get("pixel_coords_array", None)
        data["point_indices_array"] = input_dict.get("point_indices_array", None)

        return data

    def _run_student_models(
        self,
        model_3d: nn.Module,
        model_2d: nn.Module,
        input_dict: Dict[str, Any],
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """Runs forward pass for both student models."""
        output_dict_3d_student = model_3d(input_dict)
        student_logits_3d = output_dict_3d_student["seg_logits"]  # [total_points, C]

        output_dict_2d_student = model_2d(input_dict)
        student_logits_2d = output_dict_2d_student["seg_logits"]  # [B, C, H2, W2]

        upsampled_logits_student_2d = F.interpolate(
            student_logits_2d, size=(H, W), mode="bilinear", align_corners=False
        )
        return (
            student_logits_3d,
            upsampled_logits_student_2d,
            output_dict_3d_student,
            output_dict_2d_student,
        )

    def _run_teacher_models_and_process(
        self,
        model_3d: nn.Module,
        model_2d: nn.Module,
        input_dict: Dict[str, Any],
        cfg: Any,
        H: int,
        W: int,
    ) -> Dict[str, Any]:
        """
        Keep the threshold-based logic for teacher_weight_xd and teacher_hard_label_xd,
        so your supervised loss_fn_xd can still use them. We'll also store
        teacher_prob_xd and teacher_pred_xd for the new confidence-based UCDC.
        Now also includes sky teacher processing.
        """
        with torch.no_grad():
            # ---------------------------
            # Teacher 3D
            # ---------------------------
            output_dict_3d_teacher = model_3d(input_dict)
            teacher_logits_3d = output_dict_3d_teacher["seg_logits"]
            softmax_teacher_3d = F.softmax(teacher_logits_3d, dim=1)
            teacher_prob_3d, teacher_pred_3d = softmax_teacher_3d.max(dim=1)

            # threshold-based for supervised portion
            teacher_hard_label_3d = teacher_prob_3d.ge(
                cfg.model_3d.teacher_conf_threshold
            )
            teacher_weight_3d = teacher_hard_label_3d.float().mean()

            # ---------------------------
            # Teacher 2D
            # ---------------------------
            output_dict_2d_teacher = model_2d(input_dict)
            teacher_logits_2d = output_dict_2d_teacher["seg_logits"]
            upsampled_logits_teacher_2d = F.interpolate(
                teacher_logits_2d, size=(H, W), mode="bilinear", align_corners=False
            )
            softmax_teacher_2d = F.softmax(upsampled_logits_teacher_2d, dim=1)
            teacher_prob_2d, teacher_pred_2d = softmax_teacher_2d.max(dim=1)

            # threshold-based for supervised portion
            teacher_hard_label_2d = teacher_prob_2d.ge(
                cfg.model_2d.teacher_conf_threshold
            )
            teacher_weight_2d = teacher_hard_label_2d.float().mean()

            teacher_info = {
                # Full logits
                "logits_3d": teacher_logits_3d,
                "upsampled_logits_2d": upsampled_logits_teacher_2d,
                # Hard labels & scalar weights for supervised portion
                "hard_label_3d": teacher_hard_label_3d,
                "hard_label_2d": teacher_hard_label_2d,
                "weight_3d": teacher_weight_3d,
                "weight_2d": teacher_weight_2d,
                # Also store prob & pred so we can do confidence-based UCDC
                "prob_3d": teacher_prob_3d,
                "pred_3d": teacher_pred_3d,
                "prob_2d": teacher_prob_2d,
                "pred_2d": teacher_pred_2d,
            }

        return teacher_info

    def _calculate_supervised_loss(
        self,
        cfg: Any,
        loss_fn_3d: callable,
        loss_fn_2d: callable,
        student_logits_3d: torch.Tensor,
        upsampled_logits_student_2d: torch.Tensor,
        labels_3d: torch.Tensor,
        labels_2d: torch.Tensor,
        teacher_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        You mentioned your existing loss functions require `teacher_weight` and
        `teacher_hard_label`, so we continue to provide those to them.
        """
        loss_3d_total = torch.tensor(0.0, device=student_logits_3d.device)
        loss_2d_total = torch.tensor(0.0, device=upsampled_logits_student_2d.device)

        # 3D Loss
        if "Fake" in cfg.model_3d.model_class:
            loss_3d_total = (student_logits_3d**2).mean() * 0.0
        elif teacher_info:
            # e.g. your custom teacher-based 3D loss
            loss_3d = loss_fn_3d(
                student_logits_3d,
                teacher_info["logits_3d"],
                labels_3d,
                teacher_info["weight_3d"],  # keep
                teacher_info["hard_label_3d"],  # keep
            )
            loss_3d_total += loss_3d
        else:
            loss_3d_total = loss_fn_3d(student_logits_3d, labels_3d)

        # 2D Loss
        if "Fake" in cfg.model_2d.model_class:
            loss_2d_total = (upsampled_logits_student_2d**2).mean() * 0.0
        elif teacher_info:
            # e.g. your custom teacher-based 2D loss
            loss_2d = loss_fn_2d(
                upsampled_logits_student_2d,
                teacher_info["upsampled_logits_2d"],
                labels_2d,
                teacher_info["weight_2d"],  # keep
                teacher_info["hard_label_2d"],  # keep
            )
            loss_2d_total += loss_2d
        else:
            loss_2d_total = loss_fn_2d(upsampled_logits_student_2d, labels_2d)

        return loss_3d_total, loss_2d_total

    # --------------------------------------------------------------------------
    # CMC portion using teacher's confidence instead of threshold-based masks
    # --------------------------------------------------------------------------
    def _calculate_cmc_loss(
        self,
        cfg: Any,
        epoch: int,
        data: Dict[str, Any],
        student_logits_3d: torch.Tensor,
        upsampled_logits_student_2d: torch.Tensor,
        teacher_info: Dict[str, Any],
        output_dict_3d_student: Dict[str, Any],
        output_dict_2d_student: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1) Confidence-based cross-entropy for CMC:
        - 2D->3D using teacher_pred_2d * teacher_prob_2d * recon_conf_3d
        - 3D->2D using teacher_pred_3d * teacher_prob_3d * recon_conf_3d (projected to 2D).
        """
        device = data["device"]
        cmc_loss_3d_val = torch.tensor(0.0, device=device)
        cmc_loss_2d_val = torch.tensor(0.0, device=device)

        # Check if CMC is active
        cmc_active = cfg.use_cmc_loss and epoch >= cfg.cmc_epoch
        correspondences_available = (
            data["pixel_coords_array"] is not None
            and data["point_indices_array"] is not None
        )
        if not (cmc_active and correspondences_available):
            return cmc_loss_3d_val, cmc_loss_2d_val

        # Weights for the overall CMC objective
        cmc_weight_2d = get_cmc_weight(cfg, epoch, which="2d")
        cmc_weight_3d = get_cmc_weight(cfg, epoch, which="3d")

        pixel_coords_array = data["pixel_coords_array"]  # [B, M, 2]
        point_indices_array = data["point_indices_array"]  # [B, M]
        B, M, _ = pixel_coords_array.shape

        batch_indices = (
            torch.arange(B, device=device).unsqueeze(1).expand(-1, M).reshape(-1)
        )
        pixel_coords_flat = pixel_coords_array.view(-1, 2)
        point_indices_flat = point_indices_array.view(-1)

        valid_corr_mask = (
            (pixel_coords_flat[:, 0] != -1)
            & (pixel_coords_flat[:, 1] != -1)
            & (point_indices_flat != -1)
        )
        valid_pixel_coords = pixel_coords_flat[valid_corr_mask]
        valid_batch_indices = batch_indices[valid_corr_mask]
        valid_point_indices = point_indices_flat[valid_corr_mask]
        # Convert to "global" point indices:
        global_point_indices = (
            valid_point_indices + data["starting_indices"][valid_batch_indices]
        )

        # -------------------------------------------------------------------------
        # 2D -> 3D
        # -------------------------------------------------------------------------
        teacher_pred_2d = teacher_info["pred_2d"]  # [B, H, W]
        teacher_prob_2d = teacher_info["prob_2d"]  # [B, H, W]

        # Unproject teacher's labels
        point_labels_from_2d = unproject_points(
            valid_point_indices,
            valid_pixel_coords,
            teacher_pred_2d.long(),
            valid_batch_indices,
            data["starting_indices"],
            data["total_points"],
            output_dtype=torch.long,
        )
        # Unproject teacher's per-pixel confidence
        point_conf_from_2d = unproject_points(
            valid_point_indices,
            valid_pixel_coords,
            teacher_prob_2d.float(),
            valid_batch_indices,
            data["starting_indices"],
            data["total_points"],
            output_dtype=torch.float,
        )

        # Also retrieve the 3D reconstruction confidence: shape [total_points]
        recon_conf_3d = data["conf"]  # e.g. [total_points] float

        # Combine teacher’s confidence (unprojected) with reconstruction confidence
        combined_conf_3d = point_conf_from_2d * recon_conf_3d

        # (Optional) ignore certain classes
        point_labels_from_2d[point_labels_from_2d == cfg.sky_label] = 255

        # Weighted CE in 3D
        cmc_loss_3d_raw = cross_entropy_with_confidence(
            logits=student_logits_3d,
            labels=point_labels_from_2d,
            confidence_weights=combined_conf_3d,
            ignore_index=255,
        )
        cmc_loss_3d_val = cmc_loss_3d_raw * cmc_weight_3d
        output_dict_3d_student["unprojected_labels"] = point_labels_from_2d

        # -------------------------------------------------------------------------
        # 3D -> 2D
        # -------------------------------------------------------------------------
        teacher_pred_3d = teacher_info["pred_3d"]  # [total_points]
        teacher_prob_3d = teacher_info["prob_3d"]  # [total_points]

        # Project teacher's labels into 2D
        projected_labels_from_3d = project_points(
            global_point_indices,
            valid_pixel_coords,
            teacher_pred_3d.long(),
            valid_batch_indices,
            B,
            data["H"],
            data["W"],
            output_dtype=torch.long,
        )
        # Project teacher's 3D confidence into 2D
        projected_conf_from_3d = project_points(
            global_point_indices,
            valid_pixel_coords,
            teacher_prob_3d.float(),
            valid_batch_indices,
            B,
            data["H"],
            data["W"],
            output_dtype=torch.float,
        )

        # Also project the 3D reconstruction confidence into 2D
        projected_recon_conf_3d = project_points(
            global_point_indices,
            valid_pixel_coords,
            recon_conf_3d,  # [total_points]
            valid_batch_indices,
            B,
            data["H"],
            data["W"],
            output_dtype=torch.float,
        )

        # Combine teacher’s projected confidence with reconstruction confidence
        combined_conf_2d_from_3d = projected_conf_from_3d * projected_recon_conf_3d

        sky_teacher_info = teacher_info.get("sky_teacher", None)

        if sky_teacher_info is not None:
            sky_teacher_pred_2d = sky_teacher_info["pred_2d"]  # [B, H, W]
            sky_teacher_prob_2d = sky_teacher_info["prob_2d"]  # [B, H, W]

            # Calculate density ratio for weight balancing
            total_pixels = B * data["H"] * data["W"]
            projected_pixels = (
                (projected_labels_from_3d != 255).sum().item()
            )  # Non-ignore pixels
            density_ratio = projected_pixels / total_pixels if total_pixels > 0 else 0.0

            self.logger.info(
                f"3D->2D projection density: {density_ratio:.4f} ({projected_pixels}/{total_pixels})"
            )

            # Create combined labels and confidence maps
            combined_labels_2d = projected_labels_from_3d.clone()
            combined_conf_2d = combined_conf_2d_from_3d.clone()

            # Create sky mask where we want to add sky teacher predictions
            sky_mask = sky_teacher_pred_2d == cfg.sky_label

            # For sky pixels, use sky teacher predictions with density-weighted confidence
            sky_confidence_weighted = sky_teacher_prob_2d * density_ratio

            # Only apply sky teacher where 3D projections don't exist (are 255/ignore)
            sky_apply_mask = sky_mask & (projected_labels_from_3d == 255)

            combined_labels_2d[sky_apply_mask] = cfg.sky_label
            combined_conf_2d[sky_apply_mask] = sky_confidence_weighted[sky_apply_mask]

        else:
            # Fallback to original method if no sky teacher
            combined_labels_2d = projected_labels_from_3d
            combined_conf_2d = combined_conf_2d_from_3d

        # Weighted CE in 2D with combined predictions
        cmc_loss_2d_raw = cross_entropy_with_confidence(
            logits=upsampled_logits_student_2d,
            labels=combined_labels_2d,
            confidence_weights=combined_conf_2d,
            ignore_index=255,
        )
        cmc_loss_2d_val = cmc_loss_2d_raw * cmc_weight_2d
        output_dict_2d_student["projected_image"] = combined_labels_2d

        return cmc_loss_3d_val, cmc_loss_2d_val

    # --------------------------------------------------------------------------
    # Main Forward Method
    # --------------------------------------------------------------------------

    def forward(
        self,
        cfg: Any,
        model_3d: List[nn.Module],
        model_2d: List[nn.Module],
        input_dict: Dict[str, Any],
        epoch: int,
        loss_fn_3d: callable,
        loss_fn_2d: callable,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """
        1) Supervised losses:
        - If teacher present => threshold-based teacher info (weight/hard_label).
        - Else => ground-truth labels.
        2) CMC (if teacher present, epoch >= cmc_epoch):
        - Confidence-based cross-entropy with teacher_pred_xd & teacher_prob_xd.
        - Sky teacher integration for sky class predictions.
        """
        # 1. Unpack inputs and basic info
        data = self._unpack_inputs(input_dict)
        H, W = data["H"], data["W"]

        # 2. Run student models
        (
            student_logits_3d,
            upsampled_logits_student_2d,
            output_dict_3d_student,
            output_dict_2d_student,
        ) = self._run_student_models(model_3d[0], model_2d[0], input_dict, H, W)

        # 3. Initialize total losses
        loss_3d_total = torch.tensor(0.0, device=data["device"])
        loss_2d_total = torch.tensor(0.0, device=data["device"])

        # 4. Teacher logic
        has_3d_teacher = len(model_3d) > 1
        has_2d_teacher = len(model_2d) > 1
        has_sky_teacher = len(model_2d) > 2  # Check for sky teacher
        teacher_info = None

        sup_weight_2d = 1.0
        sup_weight_3d = 1.0
        if "sup_weight_2d" in cfg.model_2d.model_class:
            sup_weight_2d = cfg.model_2d.model_class["sup_weight_2d"]
        if "sup_weight_3d" in cfg.model_3d.model_class:
            sup_weight_3d = cfg.model_3d.model_class["sup_weight_3d"]

        if has_3d_teacher and has_2d_teacher:
            # Run teachers and store teacher info (including threshold-based & confidence)
            teacher_info = self._run_teacher_models_and_process(
                model_3d[1], model_2d[1], input_dict, cfg, H, W
            )

            # Add sky teacher info if available
            if has_sky_teacher:
                with torch.no_grad():
                    sky_output_dict = model_2d[2](input_dict)
                    sky_logits_2d = sky_output_dict["seg_logits"]
                    sky_upsampled_logits = F.interpolate(
                        sky_logits_2d, size=(H, W), mode="bilinear", align_corners=False
                    )
                    sky_softmax = F.softmax(sky_upsampled_logits, dim=1)
                    sky_prob, sky_pred = sky_softmax.max(dim=1)

                    teacher_info["sky_teacher"] = {
                        "pred_2d": sky_pred,
                        "prob_2d": sky_prob,
                        "logits_2d": sky_upsampled_logits,
                    }

            # (Optional) store teacher info in student dict if needed
            output_dict_2d_student["teacher_upsampled_logits_2d"] = teacher_info[
                "upsampled_logits_2d"
            ]

            # 4a. Supervised loss using teacher weighting
            sup_loss_3d, sup_loss_2d = self._calculate_supervised_loss(
                cfg,
                loss_fn_3d,
                loss_fn_2d,
                student_logits_3d,
                upsampled_logits_student_2d,
                data["labels_3d"],
                data["labels_2d"],
                teacher_info=teacher_info,
            )
            loss_3d_total += sup_loss_3d * sup_weight_3d
            loss_2d_total += sup_loss_2d * sup_weight_2d

        else:
            # 4b. Supervised loss using ground-truth labels only
            sup_loss_3d, sup_loss_2d = self._calculate_supervised_loss(
                cfg,
                loss_fn_3d,
                loss_fn_2d,
                student_logits_3d,
                upsampled_logits_student_2d,
                data["labels_3d"],
                data["labels_2d"],
                teacher_info=None,
            )
            loss_3d_total += sup_loss_3d * sup_weight_3d
            loss_2d_total += sup_loss_2d * sup_weight_2d

        # 5. CMC (only if teacher_info is present and epoch >= cfg.cmc_epoch)
        if teacher_info and cfg.use_cmc_loss and epoch >= cfg.cmc_epoch:
            cmc_loss_3d, cmc_loss_2d = self._calculate_cmc_loss(
                cfg,
                epoch,
                data,
                student_logits_3d,
                upsampled_logits_student_2d,
                teacher_info,
                output_dict_3d_student,
                output_dict_2d_student,
            )
            loss_3d_total += cmc_loss_3d
            loss_2d_total += cmc_loss_2d

        # 6. Return final losses and output dicts
        return (
            loss_2d_total,
            loss_3d_total,
            output_dict_2d_student,
            output_dict_3d_student,
        )
