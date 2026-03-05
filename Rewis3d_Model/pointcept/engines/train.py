"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model, build_criteria
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry
from pointcept.utils.misc import TrainingStopException

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model_3d, self.model_2d = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building Criteria")
        self.main_loss_fn, self.loss_fn_3d, self.loss_fn_2d = self.build_criteria()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer_3d, self.optimizer_2d = self.build_optimizer()

        self.scheduler_3d, self.scheduler_2d = self.build_scheduler()

        self.scaler_3d, self.scaler_2d = self.build_scaler()

        self.logger.info("=> Building hooks ...")

        self.training_stopped = False

        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model_3d[0].train()
                self.model_2d[0].train()
                if len(self.model_3d) > 1:
                    self.model_3d[1].train()
                if len(self.model_2d) > 1:
                    self.model_2d[1].train()

                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                if self.training_stopped:
                    self.logger.info("Training stopped by hook.")
                    break

                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    if "evaluation_only" in self.cfg and self.cfg["evaluation_only"]:
                        break
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
                if "evaluation_only" in self.cfg and self.cfg["evaluation_only"]:
                    break

            # => after train
            self.after_train()

    def run_step(self):
        iteration = self.comm_info["iter"]
        with torch.no_grad():
            # 2D EMA update
            if len(self.model_2d) > 1:
                alpha = min(1 - 1 / (iteration + 1), self.cfg.teacher_alpha)
                for tp, sp in zip(
                    self.model_2d[1].parameters(), self.model_2d[0].parameters()
                ):
                    tp.data.mul_(alpha).add_(sp.data, alpha=(1 - alpha))
                for t_buffer, s_buffer in zip(
                    self.model_2d[1].buffers(), self.model_2d[0].buffers()
                ):
                    t_buffer.copy_(s_buffer)

            # 3D EMA update
            if len(self.model_3d) > 1:
                alpha = min(1 - 1 / (iteration + 1), self.cfg.teacher_alpha)
                for tp, sp in zip(
                    self.model_3d[1].parameters(), self.model_3d[0].parameters()
                ):
                    tp.data.mul_(alpha).add_(sp.data, alpha=(1 - alpha))
                for t_buffer, s_buffer in zip(
                    self.model_3d[1].buffers(), self.model_3d[0].buffers()
                ):
                    t_buffer.copy_(s_buffer)

            # Move input to GPU
            input_dict = self.comm_info["input_dict"]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            loss_2d, loss_3d, output_dict_2d, output_dict_3d = self.main_loss_fn(
                self.cfg,
                self.model_3d,
                self.model_2d,
                input_dict,
                self.epoch,
                self.loss_fn_3d,
                self.loss_fn_2d,
            )

        metrics_3d = {"loss": loss_3d}
        metrics_2d = {"loss": loss_2d}

        # Zero gradients
        self.optimizer_3d.zero_grad()
        self.optimizer_2d.zero_grad()

        if self.cfg.enable_amp:
            # Backward passes with separate scalers
            self.scaler_3d.scale(loss_3d).backward()
            self.scaler_2d.scale(loss_2d).backward()

            # Unscale gradients
            self.scaler_3d.unscale_(self.optimizer_3d)
            self.scaler_2d.unscale_(self.optimizer_2d)

            # Gradient clipping
            if self.cfg.clip_grad_2d is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model_2d[0].parameters(), self.cfg.clip_grad_2d
                )
            if self.cfg.clip_grad_3d is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model_3d[0].parameters(), self.cfg.clip_grad_3d
                )

            # Optimizer steps
            self.scaler_3d.step(self.optimizer_3d)
            self.scaler_2d.step(self.optimizer_2d)

            # Store old scales and update
            old_scale_3d = self.scaler_3d.get_scale()
            old_scale_2d = self.scaler_2d.get_scale()
            self.scaler_3d.update()
            self.scaler_2d.update()

            # Conditional scheduler stepping
            if old_scale_3d <= self.scaler_3d.get_scale():
                self.scheduler_3d.step()
            if old_scale_2d <= self.scaler_2d.get_scale():
                self.scheduler_2d.step()

        else:
            # No AMP - sequential backward passes
            loss_3d.backward()
            loss_2d.backward()

            if self.cfg.clip_grad_2d is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model_2d[0].parameters(), self.cfg.clip_grad_2d
                )
            if self.cfg.clip_grad_3d is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model_3d[0].parameters(), self.cfg.clip_grad_3d
                )

            self.optimizer_3d.step()
            self.optimizer_2d.step()
            self.scheduler_3d.step()
            self.scheduler_2d.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()

        # Store results
        self.comm_info["model_output_dict_3d"] = output_dict_3d
        self.comm_info["model_output_dict_2d"] = output_dict_2d
        self.comm_info["metrics_2d"] = metrics_2d
        self.comm_info["metrics_3d"] = metrics_3d

    def before_epoch(self):
        freeze_start = getattr(self.cfg, "freeze_encoder_start_epoch", None)
        freeze_end = getattr(self.cfg, "freeze_encoder_end_epoch", None)
        if freeze_start is not None and freeze_end is not None:
            for model in self.model_2d:
                # Check if model is SegformerInstanceSegmentation
                if hasattr(model, "backbone_encoder"):
                    should_freeze = freeze_start <= self.epoch < freeze_end
                    for param in model.backbone_encoder.parameters():
                        param.requires_grad = not should_freeze
                    if should_freeze:
                        self.logger.info(f"Freezing encoder at epoch {self.epoch}")
                    else:
                        self.logger.info(f"Unfreezing encoder at epoch {self.epoch}")

        for h in self.hooks:
            try:
                h.before_epoch()
            except TrainingStopException as e:
                self.logger.info(f"Training will be stopped: {e}")
                self.training_stopped = True
                continue

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model_list = self.cfg.model_list
        model_2d = []
        model_3d = []

        for model_cfg in model_list:
            # --------------------------
            # Student model construction
            # ----------------------------
            model = build_model(model_cfg)

            model_name = model_cfg.type

            if self.cfg.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"3D Model {model_cfg} params: {n_parameters}")

            model = model.cuda()

            if not "Fake" in model_name:
                model = create_ddp_model(
                    model,
                    broadcast_buffers=False,
                    find_unused_parameters=self.cfg.find_unused_parameters,
                )

            if model_cfg.domain == "2D":
                model_2d.append(model)
            else:
                model_3d.append(model)

            # ----------------------------
            # Teacher (EMA) model (not DDP)
            # ----------------------------
            if "student_teacher" in model_cfg.model_class:
                teacher_model = build_model(model_cfg)
                if self.cfg.sync_bn:
                    teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(
                        teacher_model
                    )
                n_parameters = sum(
                    p.numel() for p in teacher_model.parameters() if p.requires_grad
                )
                self.logger.info(
                    f"3D Model {model_cfg} params: {n_parameters} (teacher)"
                )

                for p in teacher_model.parameters():
                    p.detach_()

                teacher_model = teacher_model.cuda()
                teacher_model.set_to_teacher()

                if model_cfg.domain == "2D":
                    model_2d.append(teacher_model)
                else:
                    model_3d.append(teacher_model)

            # NOTE: Sky teacher model is now created dynamically by SkyTeacherHook
            # at cmc_epoch instead of being created upfront. This allows the sky
            # teacher to capture the 2D teacher's state at the exact epoch when
            # CMC loss begins.

        return model_3d, model_2d

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_criteria(self):
        main_criteria = build_criteria(
            self.cfg.main_criteria.criteria, self.cfg.main_criteria.model_class
        )

        criteria_3d = build_criteria(
            self.cfg.model_3d.criteria, self.cfg.model_3d.model_class
        )
        criteria_2d = build_criteria(
            self.cfg.model_2d.criteria, self.cfg.model_2d.model_class
        )

        return main_criteria, criteria_3d, criteria_2d

    def build_optimizer(self):
        optim_3d = build_optimizer(
            self.cfg.optimizer_3d, self.model_3d[0], self.cfg.param_dicts
        )
        optim_2d = build_optimizer(
            self.cfg.optimizer_2d, self.model_2d[0], self.cfg.param_dicts
        )
        return optim_3d, optim_2d

    def build_scheduler(self):
        assert hasattr(self, "optimizer_3d")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler_3d.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        self.cfg.scheduler_2d.total_steps = len(self.train_loader) * self.cfg.eval_epoch

        return build_scheduler(
            self.cfg.scheduler_3d, self.optimizer_3d
        ), build_scheduler(self.cfg.scheduler_2d, self.optimizer_2d)

    def build_scaler(self):
        if self.cfg.enable_amp:
            if version.parse(torch.__version__) >= version.parse("2.4"):
                grad_scaler = partial(torch.amp.GradScaler, device="cuda")
            else:
                grad_scaler = torch.cuda.amp.GradScaler
            scaler_3d = grad_scaler()
            scaler_2d = grad_scaler()
        else:
            scaler_3d = None
            scaler_2d = None

        return scaler_3d, scaler_2d


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
