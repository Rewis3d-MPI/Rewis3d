"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import gc
import sys
import glob
import os
import shutil
import time

import numpy as np
from pointcept.utils.misc import get_remaining_slurm_time
import torch
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.cache import shared_dict
from pointcept.utils.scheduler import CosineScheduler
import pointcept.utils.comm as comm

import torch.distributed as dist
import wandb

from .default import HookBase
from .builder import HOOKS

from pointcept.utils.misc import TrainingStopException

from PIL import Image
import matplotlib.cm as cm


@HOOKS.register_module()
class SlurmTimeHook(HookBase):
    """
    Hook to check SLURM time limit and stop training if time is running out.
    """

    def __init__(self):
        self._start_time = None
        self._max_time_per_epoch = 0
        self._buffer = 1200

    def before_epoch(self):
        if self._start_time is not None:
            duration = time.time() - self._start_time
            if duration > self._max_time_per_epoch:
                self._max_time_per_epoch = duration
        self._start_time = time.time()

        if self._max_time_per_epoch == 0:
            return

        rem_time = get_remaining_slurm_time()
        if rem_time < self._max_time_per_epoch + self._buffer:
            self.trainer.logger.info("Remaining Time too small, restarting...")
            raise TrainingStopException("SLURM time is running out. Stopping training.")

        self.trainer.logger.info(
            "Remaining Time: {:.2f} s, Time per epoch: {:.2f} s".format(
                rem_time, self._max_time_per_epoch
            )
        )


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        self._remain_iter = self.trainer.max_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    """
    Writer for training information.
    Logs loss, learning rate, and other metrics to console and WandB.
    """

    def __init__(self):
        self.curr_iter = 0
        self.metrics_2d = []
        self.metrics_3d = []
        self.metrics_3d_keys = []
        self.metrics_2d_keys = []
        self.model_output_3d = []
        self.model_output_2d = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        self.model_output_3d = self.trainer.comm_info["model_output_dict_3d"]
        self.model_output_2d = self.trainer.comm_info["model_output_dict_2d"]
        self.metrics_2d = self.trainer.comm_info["metrics_2d"]
        self.metrics_3d = self.trainer.comm_info["metrics_3d"]

        if "metrics_3d" in self.trainer.comm_info.keys():
            self.metrics_3d_keys = [f"3D_{key}" for key in self.metrics_3d.keys()]
            for key in self.metrics_3d_keys:
                self.trainer.storage.put_scalar(
                    key, self.metrics_3d[key.replace("3D_", "")].item()
                )
        if "metrics_2d" in self.trainer.comm_info.keys():
            self.metrics_2d_keys = [f"2D_{key}" for key in self.metrics_2d.keys()]
            for key in self.metrics_2d_keys:
                self.trainer.storage.put_scalar(
                    key, self.metrics_2d[key.replace("2D_", "")].item()
                )

        for key in self.metrics_3d_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        for key in self.metrics_2d_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )

        lr_2d = self.trainer.optimizer_2d.state_dict()["param_groups"][0]["lr"]
        lr_3d = self.trainer.optimizer_3d.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr2D: {lr:.5f}".format(lr=lr_2d)
        self.trainer.comm_info["iter_info"] += "Lr3D: {lr:.5f}".format(lr=lr_3d)

        loss_2d = self.metrics_2d["loss"]
        loss_3d = self.metrics_3d["loss"]

        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if self.trainer.comm_info["iter"] % 10 == 0:
            if rank == 0:
                wandb.log(
                    {
                        "Loss2D": loss_2d,
                        "Loss3D": loss_3d,
                        "LR2D": lr_2d,
                        "LR3D": lr_3d,
                    },
                    step=self.curr_iter + 1,
                )

        train_visualization_indices = self.trainer.cfg.get(
            "train_visualization_indices",
            {1, 5, 10, 15, 100, 150, 300, 500, 700, 900, 1000},
        )

        if (
            self.trainer.cfg.log_train_image_data
            and self.trainer.comm_info["iter"] in train_visualization_indices
        ):
            class_labels = self.trainer.cfg.class_labels_2d
            iter = self.trainer.comm_info["iter"]

            upsampled_logits_2d_student = self.trainer.comm_info[
                "model_output_dict_2d"
            ]["student_upsampled_logits_2d"]

            seg_mask_student = (
                upsampled_logits_2d_student.argmax(dim=1).cpu().numpy()[0]
            )
            label_tensor_student = self.trainer.comm_info["input_dict"][
                "student_labels_1"
            ]
            label_mask = label_tensor_student.detach().cpu().numpy()[0]
            img_tensor_student = self.trainer.comm_info["input_dict"][
                "student_pixel_values_1"
            ]
            input_img = img_tensor_student.detach().cpu().numpy()[0]
            input_img = np.moveaxis(input_img, 0, -1)

            masks = {
                "predictions": {
                    "mask_data": seg_mask_student,
                    "class_labels": class_labels,
                },
                "groundtruth": {
                    "mask_data": label_mask,
                    "class_labels": class_labels,
                },
            }

            image_student = wandb.Image(input_img, masks=masks)

            if rank == 0:
                wandb.log(
                    {
                        f"{iter:04d}_student_image": image_student,
                    }
                )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.metrics_3d_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        for key in self.metrics_2d_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )

            self.trainer.logger.info("Saving checkpoint to: " + filename)
            data = {
                "epoch": self.trainer.epoch + 1,
                "student_state_dict_3d": self.trainer.model_3d[0].state_dict(),
                "student_state_dict_2d": self.trainer.model_2d[0].state_dict(),
                "optimizer_3d": self.trainer.optimizer_3d.state_dict(),
                "optimizer_2d": self.trainer.optimizer_2d.state_dict(),
                "scheduler_3d": self.trainer.scheduler_3d.state_dict(),
                "scheduler_2d": self.trainer.scheduler_2d.state_dict(),
                "scaler_3d": (
                    self.trainer.scaler_3d.state_dict()
                    if self.trainer.cfg.enable_amp
                    else None
                ),
                "scaler_2d": (
                    self.trainer.scaler_2d.state_dict()
                    if self.trainer.cfg.enable_amp
                    else None
                ),
                "best_metric_value": self.trainer.best_metric_value,
            }
            if len(self.trainer.model_3d) >= 2:
                data["teacher_state_dict_3d"] = self.trainer.model_3d[1].state_dict()
            if len(self.trainer.model_2d) >= 2:
                data["teacher_state_dict_2d"] = self.trainer.model_2d[1].state_dict()
            # Save sky teacher state if it exists (model_2d[2])
            if len(self.trainer.model_2d) >= 3:
                data["sky_teacher_state_dict_2d"] = self.trainer.model_2d[
                    2
                ].state_dict()

            torch.save(
                data,
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.trainer.epoch in self.trainer.cfg.get("save_epochs", []):
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch}.pth",
                    ),
                )

            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    """
    Loader for model checkpoints.
    Supports loading 2D and 3D models, optimizers, schedulers, and scalers.
    """

    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def smart_load_state_dict(self, model, state_dict):
        filtered_state_dict = state_dict

        is_wrapped = isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        )
        has_module_prefix = list(filtered_state_dict.keys())[0].startswith("module.")

        if is_wrapped and not has_module_prefix:
            self.trainer.logger.info("Adding 'module.' prefix to state_dict keys.")
            filtered_state_dict = {
                "module." + k: v for k, v in filtered_state_dict.items()
            }
        elif not is_wrapped and has_module_prefix:
            self.trainer.logger.info("Removing 'module.' prefix from state_dict keys.")
            filtered_state_dict = {
                k.replace("module.", "", 1): v for k, v in filtered_state_dict.items()
            }
        else:
            self.trainer.logger.info("No prefix adjustment needed for state_dict.")

        return model.load_state_dict(filtered_state_dict, strict=self.strict)

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )

            # Load 3D model weights
            skip_3d_model = self.trainer.cfg.get("skip_3d_model", False)
            if not skip_3d_model:
                self.trainer.logger.info("Loading 3D student model weights")
                state_dict_3d = checkpoint["student_state_dict_3d"]
                load_state_info_3d = self.smart_load_state_dict(
                    self.trainer.model_3d[0], state_dict_3d
                )
                self.trainer.logger.info(
                    f"Missing keys in 3D model: {load_state_info_3d.missing_keys}"
                )

                if "teacher_state_dict_3d" in checkpoint.keys():
                    self.trainer.logger.info("Loading 3D teacher model weights")
                    state_dict_teacher_3d = checkpoint["teacher_state_dict_3d"]
                    load_state_info_teacher_3d = self.smart_load_state_dict(
                        self.trainer.model_3d[1], state_dict_teacher_3d
                    )
                    self.trainer.logger.info(
                        f"Missing keys in 3D teacher model: {load_state_info_teacher_3d.missing_keys}"
                    )

            # Load 2D model weights
            self.trainer.logger.info("Loading 2D model weights")
            state_dict_2d = checkpoint["student_state_dict_2d"]
            load_state_info_2d = self.smart_load_state_dict(
                self.trainer.model_2d[0], state_dict_2d
            )
            self.trainer.logger.info(
                f"Missing keys in 2D model: {load_state_info_2d.missing_keys}"
            )

            if "teacher_state_dict_2d" in checkpoint.keys():
                self.trainer.logger.info("Loading 2D teacher model weights")
                state_dict_teacher_2d = checkpoint["teacher_state_dict_2d"]
                load_state_info_teacher_2d = self.smart_load_state_dict(
                    self.trainer.model_2d[1], state_dict_teacher_2d
                )
                self.trainer.logger.info(
                    f"Missing keys in 2D teacher model: {load_state_info_teacher_2d.missing_keys}"
                )

            # Resume training
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]

                self.trainer.optimizer_3d.load_state_dict(checkpoint["optimizer_3d"])
                self.trainer.optimizer_2d.load_state_dict(checkpoint["optimizer_2d"])

                # Load scalers if AMP is enabled
                if self.trainer.cfg.enable_amp:
                    if checkpoint.get("scaler_3d") is not None:
                        self.trainer.scaler_3d.load_state_dict(checkpoint["scaler_3d"])
                    if checkpoint.get("scaler_2d") is not None:
                        self.trainer.scaler_2d.load_state_dict(checkpoint["scaler_2d"])

                # Load schedulers

                self.trainer.scheduler_3d.load_state_dict(checkpoint["scheduler_3d"])
                self.trainer.scheduler_2d.load_state_dict(checkpoint["scheduler_2d"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class SkyCheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def smart_load_state_dict(self, model, state_dict):
        is_wrapped = isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        )
        has_module_prefix = list(state_dict.keys())[0].startswith("module.")

        if is_wrapped and not has_module_prefix:
            self.trainer.logger.info("Adding 'module.' prefix to state_dict keys.")
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not is_wrapped and has_module_prefix:
            self.trainer.logger.info("Removing 'module.' prefix from state_dict keys.")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        else:
            self.trainer.logger.info("No prefix adjustment needed for state_dict.")

        return model.load_state_dict(state_dict, strict=self.strict)

    def before_train(self):
        self.trainer.logger.info("=> Loading Sky checkpoint & weight ...")

        if len(self.trainer.model_2d) <= 2:
            self.trainer.logger.info("No Sky model found, skip loading checkpoint.")
            return

        if self.trainer.cfg.sky_weight and os.path.isfile(self.trainer.cfg.sky_weight):
            self.trainer.logger.info(
                f"Loading weight at: {self.trainer.cfg.sky_weight}"
            )
            checkpoint = torch.load(
                self.trainer.cfg.sky_weight,
                map_location=lambda storage, loc: storage.cuda(),
            )

            if "teacher_state_dict_2d" in checkpoint.keys():
                self.trainer.logger.info("Loading 2D Sky teacher model weights")
                state_dict_teacher_2d = checkpoint["teacher_state_dict_2d"]
                load_state_info_teacher_2d = self.smart_load_state_dict(
                    self.trainer.model_2d[2], state_dict_teacher_2d
                )
                self.trainer.logger.info(
                    f"Missing keys in 2D teacher model: {load_state_info_teacher_2d.missing_keys}"
                )
            else:
                self.trainer.logger.info(
                    f"No weight found at: {self.trainer.cfg.weight}"
                )


@HOOKS.register_module()
class SkyTeacherHook(HookBase):
    """
    Hook to manage the frozen sky teacher model.

    At the start of cmc_epoch:
    - Snapshots the 2D teacher model weights
    - Saves them to disk as sky_teacher.pth
    - Creates a frozen copy in memory as model_2d[2]

    On resume (if sky teacher checkpoint exists):
    - Loads the sky teacher from the saved checkpoint
    """

    def __init__(self, strict=False):
        self.strict = strict
        self._sky_teacher_initialized = False

    def _get_sky_teacher_path(self):
        """Returns the path to the sky teacher checkpoint."""
        return os.path.join(self.trainer.cfg.save_path, "model", "sky_teacher.pth")

    def _smart_load_state_dict(self, model, state_dict):
        """Load state dict with module prefix handling."""
        is_wrapped = isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        )
        has_module_prefix = list(state_dict.keys())[0].startswith("module.")

        if is_wrapped and not has_module_prefix:
            self.trainer.logger.info("Adding 'module.' prefix to state_dict keys.")
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not is_wrapped and has_module_prefix:
            self.trainer.logger.info("Removing 'module.' prefix from state_dict keys.")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        else:
            self.trainer.logger.info("No prefix adjustment needed for state_dict.")

        return model.load_state_dict(state_dict, strict=self.strict)

    def _create_sky_teacher_model(self):
        """Creates a frozen copy of the 2D teacher model architecture."""
        from pointcept.models import build_model

        # Get the 2D model config
        model_cfg = None
        for cfg in self.trainer.cfg.model_list:
            if cfg.domain == "2D":
                model_cfg = cfg
                break

        if model_cfg is None:
            self.trainer.logger.error("No 2D model config found!")
            return None

        # Build the sky teacher model
        sky_model = build_model(model_cfg)
        if self.trainer.cfg.sync_bn:
            sky_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sky_model)

        n_parameters = sum(p.numel() for p in sky_model.parameters())
        self.trainer.logger.info(f"Sky Teacher Model params: {n_parameters}")

        # Freeze all parameters
        for p in sky_model.parameters():
            p.detach_()
            p.requires_grad_(False)

        sky_model = sky_model.cuda()
        sky_model.eval()
        sky_model.set_to_teacher()

        return sky_model

    def _snapshot_and_save_sky_teacher(self):
        """Snapshot 2D teacher weights and save to disk."""
        if len(self.trainer.model_2d) < 2:
            self.trainer.logger.warning("No 2D teacher model found to snapshot!")
            return

        # Get the 2D teacher state dict
        teacher_2d_state_dict = self.trainer.model_2d[1].state_dict()

        # Save to disk (only on main process)
        if is_main_process():
            sky_teacher_path = self._get_sky_teacher_path()
            os.makedirs(os.path.dirname(sky_teacher_path), exist_ok=True)

            torch.save(
                {"sky_teacher_state_dict_2d": teacher_2d_state_dict},
                sky_teacher_path + ".tmp",
            )
            os.replace(sky_teacher_path + ".tmp", sky_teacher_path)

            self.trainer.logger.info(
                f"Saved sky teacher weights to: {sky_teacher_path}"
            )

        # Synchronize before loading
        synchronize()

        # Create and initialize the sky teacher model
        sky_model = self._create_sky_teacher_model()
        if sky_model is not None:
            self._smart_load_state_dict(sky_model, teacher_2d_state_dict)
            self.trainer.model_2d.append(sky_model)
            self._sky_teacher_initialized = True
            self.trainer.logger.info(
                f"Created frozen sky teacher model (model_2d[{len(self.trainer.model_2d) - 1}])"
            )

    def _load_sky_teacher_from_checkpoint(self):
        """Load sky teacher from checkpoint if it exists."""
        # Check for explicit sky_teacher_weight config first
        sky_weight = self.trainer.cfg.get("sky_teacher_weight", None)

        # If not specified, check default sky_teacher.pth location
        if sky_weight is None:
            sky_weight = self._get_sky_teacher_path()

        # Also check main checkpoint for sky teacher state
        main_checkpoint_path = self.trainer.cfg.weight

        # Try loading from dedicated sky teacher file first
        if sky_weight and os.path.isfile(sky_weight):
            self.trainer.logger.info(f"Loading sky teacher from: {sky_weight}")
            checkpoint = torch.load(
                sky_weight,
                map_location=lambda storage, loc: storage.cuda(),
            )
        # Fall back to main checkpoint
        elif main_checkpoint_path and os.path.isfile(main_checkpoint_path):
            self.trainer.logger.info(
                f"Checking main checkpoint for sky teacher: {main_checkpoint_path}"
            )
            checkpoint = torch.load(
                main_checkpoint_path,
                map_location=lambda storage, loc: storage.cuda(),
            )
            if "sky_teacher_state_dict_2d" not in checkpoint:
                self.trainer.logger.info(
                    "No sky teacher state dict in main checkpoint."
                )
                return False
        else:
            return False

        # Create the sky teacher model
        sky_model = self._create_sky_teacher_model()
        if sky_model is not None:
            if "sky_teacher_state_dict_2d" in checkpoint:
                state_dict = checkpoint["sky_teacher_state_dict_2d"]
            elif "teacher_state_dict_2d" in checkpoint:
                # Backwards compatibility
                state_dict = checkpoint["teacher_state_dict_2d"]
            else:
                self.trainer.logger.warning(
                    "No sky teacher state dict found in checkpoint!"
                )
                return False

            self._smart_load_state_dict(sky_model, state_dict)
            self.trainer.model_2d.append(sky_model)
            self._sky_teacher_initialized = True
            self.trainer.logger.info(
                f"Loaded sky teacher model from checkpoint (model_2d[{len(self.trainer.model_2d) - 1}])"
            )
            return True

        return False

    def before_train(self):
        """Check if we should load sky teacher from checkpoint (for resume)."""
        cmc_epoch = self.trainer.cfg.get("cmc_epoch", 15)
        use_sky_teacher = self.trainer.cfg.get("use_sky_teacher", True)

        if not use_sky_teacher:
            self.trainer.logger.info("Sky teacher disabled in config.")
            return

        # If resuming from after cmc_epoch, try to load existing sky teacher
        if self.trainer.start_epoch >= cmc_epoch:
            if self._load_sky_teacher_from_checkpoint():
                self.trainer.logger.info(
                    f"Resumed with sky teacher (start_epoch={self.trainer.start_epoch} >= cmc_epoch={cmc_epoch})"
                )
            else:
                self.trainer.logger.warning(
                    f"Resuming after cmc_epoch but no sky teacher checkpoint found! "
                    f"Will create sky teacher at epoch {cmc_epoch} if reached."
                )

    def before_epoch(self):
        """At the start of cmc_epoch, snapshot and create the sky teacher."""
        cmc_epoch = self.trainer.cfg.get("cmc_epoch", 15)
        use_sky_teacher = self.trainer.cfg.get("use_sky_teacher", True)

        if not use_sky_teacher:
            return

        # At exactly cmc_epoch, create the sky teacher if not already initialized
        if self.trainer.epoch == cmc_epoch and not self._sky_teacher_initialized:
            self.trainer.logger.info(
                f"==> Epoch {cmc_epoch}: Creating frozen sky teacher from 2D teacher snapshot"
            )
            self._snapshot_and_save_sky_teacher()


@HOOKS.register_module()
class ConcertoCheckpointLoader(HookBase):
    """
    Loader for Concerto pretrained checkpoints.

    The Concerto checkpoint format (concerto_large_outdoor.pth etc.) has keys like:
    - 'config': model configuration
    - 'state_dict': model weights with keys like 'module.xxx'

    This loader maps those weights to the 3D backbone in your student-teacher setup.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def before_train(self):
        concerto_weight = self.trainer.cfg.get("concerto_weight", None)

        if concerto_weight and os.path.isfile(concerto_weight):
            self.trainer.logger.info(
                f"=> Loading Concerto pretrained weights from: {concerto_weight}"
            )

            checkpoint = torch.load(
                concerto_weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )

            # Concerto checkpoints have 'state_dict' key with weights
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            self.trainer.logger.info(
                f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'raw state_dict'}"
            )
            self.trainer.logger.info(
                f"State dict sample keys: {list(state_dict.keys())[:5]}"
            )

            # Map Concerto weights (module.xxx) to backbone (module.backbone.xxx or backbone.xxx)
            # The Concerto checkpoint has keys like: module.enc.xxx, module.dec.xxx
            # We need to map them to the 3D model's backbone

            model_3d = self.trainer.model_3d[0]
            is_wrapped = isinstance(
                model_3d,
                (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
            )

            # Build the new state dict for backbone
            backbone_state_dict = {}
            for key, value in state_dict.items():
                # Remove 'module.' prefix from Concerto checkpoint
                new_key = (
                    key.replace("module.", "", 1) if key.startswith("module.") else key
                )

                # Add proper prefix for our model structure
                if is_wrapped:
                    # DDP model: module.backbone.xxx
                    new_key = f"module.backbone.{new_key}"
                else:
                    # Non-DDP model: backbone.xxx
                    new_key = f"backbone.{new_key}"

                backbone_state_dict[new_key] = value

            self.trainer.logger.info(
                f"Mapped state dict sample keys: {list(backbone_state_dict.keys())[:5]}"
            )

            # Load into student model
            missing_keys, unexpected_keys = model_3d.load_state_dict(
                backbone_state_dict, strict=False
            )

            self.trainer.logger.info(f"Loaded Concerto weights into 3D student model")
            self.trainer.logger.info(
                f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}..."
            )
            self.trainer.logger.info(
                f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}..."
            )

            # Also load into teacher model if exists
            if len(self.trainer.model_3d) >= 2:
                self.trainer.logger.info(
                    "Loading Concerto weights into 3D teacher model"
                )
                self.trainer.model_3d[1].load_state_dict(
                    backbone_state_dict, strict=False
                )
        else:
            self.trainer.logger.info(f"No Concerto weight found at: {concerto_weight}")


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class WeightDecaySchedular(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.scheduler = None

    def before_train(self):
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.scheduler = CosineScheduler(
            base_value=self.base_value,
            final_value=self.final_value,
            total_iters=self.trainer.cfg.scheduler.total_steps,
        )
        self.scheduler.iter = curr_step

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            param_group["weight_decay"] = wd
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)


@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()
