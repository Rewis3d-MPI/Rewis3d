# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
import torch.distributed as dist
from torchmetrics.classification import MulticlassConfusionMatrix
import pointops
from uuid import uuid4
import wandb
import matplotlib.pyplot as plt

from PIL import Image
from torch import nn

import torch.nn.functional as F

import pointcept.utils.comm as comm
from pointcept.utils.misc import (
    intersection_and_union_gpu,
    error_map,
    intersection_and_union_2d_gpu,
)

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class SemanticEvaluator(HookBase):
    """
    Evaluator for Semantic Segmentation.
    Handles 2D and 3D evaluation, metric computation, and visualization logging to WandB.
    """

    def __init__(self):
        self.steps = 0

    def after_step(self):
        self.steps += 1

    def after_epoch(self):
        if "eval_epochs" in self.trainer.cfg:
            current_epoch = self.trainer.epoch + 1
            if current_epoch not in self.trainer.cfg.eval_epochs:
                return
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        """
        Run evaluation on the validation set.
        """
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        # put all nets in eval mode ------------------------------------------------------
        self.trainer.model_2d[0].eval()
        self.trainer.model_3d[0].eval()
        self.trainer.model_2d[1].eval()
        self.trainer.model_3d[1].eval()

        # run over the val-loader --------------------------------------------------------
        visualization_indices = self.trainer.cfg.get(
            "visualization_indices", {1, 5, 10, 15, 100, 150, 300, 500, 700, 900, 1000}
        )

        for i, input_dict in enumerate(self.trainer.val_loader):
            # move tensors to GPU -------------------------------------------------------
            for key, val in input_dict.items():
                if isinstance(val, torch.Tensor):
                    input_dict[key] = val.cuda(non_blocking=True)

            with torch.no_grad():
                # ---------------------------------------------------------------------
                # ----------- 2-D forward pass (student / teacher) --------------------
                # ---------------------------------------------------------------------
                images_student = input_dict["student_pixel_values_1"]
                masks_student = input_dict["student_labels_1"]
                has_depth = "student_depth" in input_dict

                # crop input into 5 overlapping stripes so it fits into GPU memory ----
                image_slices = self.trainer.cfg.image_slices
                logit_slices = self.trainer.cfg.logit_slices

                downsample = 4
                H_out = images_student.shape[-2] // downsample
                W_out = images_student.shape[-1] // downsample

                logits_student = torch.zeros(
                    1,
                    self.trainer.cfg.model_2d.num_classes,
                    H_out,
                    W_out,
                    device=images_student.device,
                )
                logits_teacher = torch.zeros_like(logits_student)

                for (img_s, img_e), (log_s, log_e) in zip(image_slices, logit_slices):
                    crop_dict = {k: v for k, v in input_dict.items()}
                    crop_dict["student_pixel_values_1"] = images_student[
                        ..., img_s:img_e
                    ]
                    crop_dict["student_labels_1"] = masks_student[..., img_s:img_e]
                    crop_dict["teacher_pixel_values_1"] = crop_dict[
                        "student_pixel_values_1"
                    ]
                    crop_dict["teacher_labels_1"] = crop_dict["student_labels_1"]

                    if has_depth:
                        crop_dict["student_depth"] = input_dict["student_depth"][
                            ..., img_s:img_e
                        ]
                        crop_dict["teacher_depth"] = crop_dict["student_depth"]

                    # teacher 2-D ------------------------------------------------------
                    out_t = self.trainer.model_2d[1](crop_dict)
                    logits_teacher[..., log_s:log_e] += out_t["seg_logits"]
                    crop_dict.update(out_t)  # pass teacher logits to student

                    # student 2-D ------------------------------------------------------
                    out_s = self.trainer.model_2d[0](crop_dict)
                    logits_student[..., log_s:log_e] += out_s["seg_logits"]

                # ---------------------------------------------------------------------
                # ----------------------- 3-D forward pass ----------------------------
                # ---------------------------------------------------------------------
                out_3d_student = self.trainer.model_3d[0](input_dict)["seg_logits"]
                pred_3d_student = out_3d_student.argmax(1)

                out_3d_teacher = self.trainer.model_3d[1](input_dict)["seg_logits"]
                pred_3d_teacher = out_3d_teacher.argmax(1)

            # -------------------------------------------------------------------------
            # ---------------------   METRIC COMPUTATION   ----------------------------
            # -------------------------------------------------------------------------
            # ---------- 3-D IoU / Acc -----------------------------------------------
            if "original_segment" in input_dict:
                original_segment = input_dict["original_segment"]
                intersection_3d, union_3d, target_3d = intersection_and_union_gpu(
                    pred_3d_student,
                    original_segment,
                    self.trainer.cfg.model_3d.num_classes,
                    self.trainer.cfg.data.ignore_index,
                )
                if comm.get_world_size() > 1:
                    dist.all_reduce(intersection_3d)
                    dist.all_reduce(union_3d)
                    dist.all_reduce(target_3d)
                self.trainer.storage.put_scalar(
                    "val_intersection", intersection_3d.cpu().numpy()
                )
                self.trainer.storage.put_scalar("val_union", union_3d.cpu().numpy())
                self.trainer.storage.put_scalar("val_target", target_3d.cpu().numpy())

            # ---------- 2-D upsample + metrics ---------------------------------------
            label_tensor = input_dict["original_mask_1"]
            up_logits = nn.functional.interpolate(
                logits_student,
                size=label_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred_2d_student = up_logits.argmax(1)

            # ordinary 2-D metrics ----------------------------------------------------
            inter_2d, union_2d, targ_2d = intersection_and_union_2d_gpu(
                pred_2d_student.squeeze(0).flatten(),
                label_tensor.flatten(),
                self.trainer.cfg.model_2d.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(inter_2d)
                dist.all_reduce(union_2d)
                dist.all_reduce(targ_2d)
            self.trainer.storage.put_scalar(
                "val_intersection_2d", inter_2d.cpu().numpy()
            )
            self.trainer.storage.put_scalar("val_union_2d", union_2d.cpu().numpy())
            self.trainer.storage.put_scalar("val_target_2d", targ_2d.cpu().numpy())

            # -------------------------------------------------------------------------
            # --------------------   VISUALISATION (optional)   -----------------------
            # -------------------------------------------------------------------------
            if i in visualization_indices:
                self.visualize(
                    rank,
                    input_dict,
                    label_tensor,
                    pred_2d_student,
                    pred_3d_student,
                    pred_3d_teacher,
                )

            # progress bar / logger ----------------------------------------------------
            info = f"Test: [{i + 1}/{len(self.trainer.val_loader)}] "
            if "origin_coord" in input_dict:
                info = "Interp. " + info
            self.trainer.logger.info(info)

        # -------------------------------------------------------------------------
        # -------------------   REDUCE & LOG GLOBAL METRICS   ---------------------
        # -------------------------------------------------------------------------
        inter2d = self.trainer.storage.history("val_intersection_2d").total
        union2d = self.trainer.storage.history("val_union_2d").total
        targ2d = self.trainer.storage.history("val_target_2d").total
        iou2d = inter2d / (union2d + 1e-10)
        acc2d = inter2d / (targ2d + 1e-10)
        m_iou2d, m_acc2d = np.mean(iou2d), np.mean(acc2d)
        all_acc2d = inter2d.sum() / (targ2d.sum() + 1e-10)

        inter3d = self.trainer.storage.history("val_intersection").total
        union3d = self.trainer.storage.history("val_union").total
        targ3d = self.trainer.storage.history("val_target").total
        iou3d = inter3d / (union3d + 1e-10)
        acc3d = inter3d / (targ3d + 1e-10)
        m_iou3d, m_acc3d = np.mean(iou3d), np.mean(acc3d)
        all_acc3d = inter3d.sum() / (targ3d.sum() + 1e-10)

        if rank == 0:
            wandb.log(
                {
                    "mIoU_2D": m_iou2d,
                    "mAcc_2D": m_acc2d,
                    "allAcc_2D": all_acc2d,
                    "mIoU_3D": m_iou3d,
                    "mAcc_3D": m_acc3d,
                    "allAcc_3D": all_acc3d,
                }
            )

        # console output -------------------------------------------------------------
        self.trainer.logger.info("############ 2D Validation ############")
        self.trainer.logger.info(
            f"Val 2D: mIoU/mAcc/allAcc {m_iou2d:.4f}/{m_acc2d:.4f}/{all_acc2d:.4f}"
        )
        for cid in range(self.trainer.cfg.model_2d.num_classes):
            self.trainer.logger.info(
                f"Class_{cid}-{self.trainer.cfg.class_labels_2d[cid]}: "
                f"iou/acc {iou2d[cid]:.4f}/{acc2d[cid]:.4f}"
            )
            if rank == 0:
                wandb.log(
                    {f"mIoU_2D_{self.trainer.cfg.class_labels_2d[cid]}": iou2d[cid]}
                )

        self.trainer.logger.info("############ 3D Validation ############")
        self.trainer.logger.info(
            f"Val 3D: mIoU/mAcc/allAcc {m_iou3d:.4f}/{m_acc3d:.4f}/{all_acc3d:.4f}"
        )
        for cid in range(self.trainer.cfg.model_3d.num_classes):
            self.trainer.logger.info(
                f"Class_{cid}-{self.trainer.cfg.class_labels_3d[cid]}: "
                f"iou/acc {iou3d[cid]:.4f}/{acc3d[cid]:.4f}"
            )
            if rank == 0:
                wandb.log(
                    {f"mIoU_3D_{self.trainer.cfg.class_labels_3d[cid]}": iou3d[cid]}
                )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/mIoU", m_iou3d, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc3d, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc3d, current_epoch)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou2d
        self.trainer.comm_info["current_metric_name"] = "mIoU"

    def visualize(
        self,
        rank,
        input_dict,
        label_tensor,
        pred_2d_student,
        pred_3d_student,
        pred_3d_teacher,
    ):
        student_coord = input_dict["student_coord"].float().cpu().numpy()

        # 2-D masks & RGB image ---------------------------------------------
        input_img = input_dict["student_pixel_values_1"].cpu().numpy()[0]
        input_img = np.moveaxis(input_img, 0, -1)  # C,H,W ➜ H,W,C
        label_mask = label_tensor.cpu().numpy().squeeze()
        seg_mask = pred_2d_student.cpu().numpy().squeeze()

        height, width = label_mask.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        for tid, col in self.trainer.cfg.colormap_2d.items():
            color_image[label_mask == tid] = col

        color_image_pred = np.zeros((height, width, 3), dtype=np.uint8)
        for tid, col in self.trainer.cfg.colormap_2d.items():
            color_image_pred[seg_mask == tid] = col

        # 3-D coloured point-clouds & error-maps ----------------------------
        student_coord = input_dict["student_coord"].float().cpu().numpy()

        # coloured ground-truth
        colors = np.zeros((student_coord.shape[0], 3), np.uint8)
        segment_cpu = input_dict["student_segment"].cpu()
        for tid, col in self.trainer.cfg.colormap_2d.items():
            colors[segment_cpu == tid] = col
        pcd_label = np.hstack((student_coord, colors))

        # coloured prediction
        colors.fill(0)
        for tid, col in self.trainer.cfg.colormap_2d.items():
            colors[pred_3d_student.cpu() == tid] = col

        pcd_pred = np.hstack((student_coord, colors))

        # error-maps ---------------------------------------------------------
        pcd_error_map_original_segment = None
        pcd_original = None
        if "original_segment" in input_dict:
            original_segment = input_dict["original_segment"]
            # vs original 3-D mask
            err_map_orig = error_map(original_segment.cpu(), pred_3d_student.cpu())
            colors = np.zeros((err_map_orig.shape[0], 3), np.uint8)
            colors[err_map_orig == 1] = [0, 255, 0]
            colors[err_map_orig == 0] = [255, 0, 0]
            colors[err_map_orig == 255] = [0, 0, 0]
            pcd_error_map_original_segment = np.hstack((student_coord, colors))

            # coloured original mask
            colors = np.zeros_like(colors)
            for tid, col in self.trainer.cfg.colormap_2d.items():
                colors[original_segment.cpu() == tid] = col
            pcd_original = np.hstack((student_coord, colors))

        # error-map vs student segment (always present)
        err_map_seg = error_map(
            pred_3d_student.cpu(), input_dict["student_segment"].cpu()
        )
        colors = np.zeros((err_map_seg.shape[0], 3), np.uint8)
        colors[err_map_seg == 1] = [0, 255, 0]
        colors[err_map_seg == 0] = [255, 0, 0]
        colors[err_map_seg == 255] = [0, 0, 0]
        pcd_error_map_segment = np.hstack((student_coord, colors))

        # teacher prediction -------------------------------------------------
        teacher_coord = input_dict["teacher_coord"].float().cpu().numpy()
        colors = np.zeros((teacher_coord.shape[0], 3), np.uint8)
        for tid, col in self.trainer.cfg.colormap_2d.items():
            colors[pred_3d_teacher.cpu() == tid] = col
        pcd_pred_teacher = np.hstack((teacher_coord, colors))

        # ------------------------- WandB logging --------------------------
        class_labels = self.trainer.cfg.class_labels_2d
        image_pred = wandb.Image(
            input_img,
            masks={
                "predictions": {
                    "mask_data": seg_mask,
                    "class_labels": class_labels,
                },
                "groundtruth": {
                    "mask_data": label_mask,
                    "class_labels": class_labels,
                },
            },
        )
        if rank == 0 and self.trainer.cfg.get("wandb_log_2d", True):
            wandb.log(
                {
                    "2D": image_pred,
                    "ColoredMap": wandb.Image(color_image),
                    "ColoredMapPred": wandb.Image(color_image_pred),
                }
            )
        if rank == 0 and self.trainer.cfg.get("wandb_log_3d", False):
            wandb.log(
                {
                    "Label3D": wandb.Object3D(pcd_label),
                    "Prediction3D": wandb.Object3D(pcd_pred),
                    "ErrorMapSegment": wandb.Object3D(pcd_error_map_segment),
                }
            )
            if pcd_error_map_original_segment is not None:
                wandb.log(
                    {
                        "ErrorMapOriginalSegment": wandb.Object3D(
                            pcd_error_map_original_segment
                        ),
                        "Original3D": wandb.Object3D(pcd_original),
                    }
                )
