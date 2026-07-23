"""FSDP trainer implementation."""

import dataclasses
import copy
import gc
import json
import os
import torch
import time
import logging
import contextlib
from torch.nn.parallel import DistributedDataParallel as DDP
from wall_x.distributed import (
    get_sp_world_size,
    get_sp_group,
    reduce_sp_partial_grads,
)
from typing import Optional, Any, Tuple
from wall_x.config.loader import dump_config
from wall_x.trainer.optimizer import get_optimizer
from wall_x.utils.profile_zones import zone, Z

from wall_x.trainer.fsdp_trainer.base_trainer import (
    DistributedTrainer,
    is_main_process,
    barrier,
)
from wall_x.trainer.adapters import ADAPTER_REGISTRY, format_adapter_error
from wall_x.trainer.fsdp_trainer import checkpoint_io as _ckpt_io
from wall_x.trainer.fsdp_trainer.distribution_strategy import build_strategy
from wall_x.trainer.metrics import MetricsLogger
from wall_x.trainer.utils import move_batch_to_device

from wall_x.model.core.action.normalizer import create_normalizers
from wall_x.trainer.scheduler.scheduler import get_scheduler


class FSDPTrainer(DistributedTrainer):
    """
    FSDP-based Trainer using pure PyTorch FSDP2 (``fully_shard``).

    Features:
    - No accelerate dependency
    - Direct torchrun launch
    - Mixed precision training support (bfloat16/float16)
    - Gradient accumulation
    - Full/Sharded state dict saving
    - CPU offload support
    - Activation checkpointing

    Launch:
        torchrun --nproc_per_node=8 --master_port=29500 train_fsdp.py --config config.yaml
    """

    def __init__(
        self,
        train_config,
        wandb_run: Optional[Any] = None,
    ):
        if train_config is None:
            raise ValueError("train_config is required")

        self.model_type = train_config.model_type
        super().__init__(train_config, wandb_run)

        # CUDA memory snapshot — start recording BEFORE adapter / strategy / model
        # construction so the dumped pickle covers:
        #   * adapter init (processor, tokenizer)
        #   * strategy build (FSDP2 mesh / process group)
        #   * model load + ckpt weight load
        #   * FSDP/DDP wrap (param sharding + DTensor wiring)
        #   * optimizer + scheduler creation (AdamW state ≈ 8B × #params, large)
        #   * dataset / dataloader build
        #   * the first few training steps
        # Rank 0 only — multi-rank snapshots would 8× the .pickle size for no extra
        # signal (the per-rank pattern is identical).
        if self.cfg.debug.enable_memory_snapshot and self.rank == 0:
            torch.cuda.memory._record_memory_history(
                max_entries=self.cfg.debug.memory_snapshot_max_entries
            )
            self.log(
                f"[MemSnap] recording started in __init__ (max_entries="
                f"{self.cfg.debug.memory_snapshot_max_entries})"
            )

        if self.model_type not in ADAPTER_REGISTRY:
            raise ValueError(format_adapter_error(self.model_type))
        adapter_cls = ADAPTER_REGISTRY[self.model_type]
        self.adapter = adapter_cls(
            cfg=self.cfg,
            logger=self.logger,
            model_type=self.model_type,
        )

        self.strategy = build_strategy(
            dataclasses.asdict(self.cfg.distributed),
            device=self.device,
            local_rank=self.local_rank,
            optimizer_type=self.cfg.hyperparams.optimizer.optimizer_type,
        )

        # Model config
        self.action_dim = self.cfg.action_dim
        self.use_selective_recompute = self.cfg.distributed.use_selective_recompute
        self.show_time_details = self.cfg.debug.show_time_details
        if self.show_time_details:
            self.timers._log_level = max(getattr(self.timers, "_log_level", -1), 0)
            self.log(
                "[TimeDetailsTimersEnabled] timers log_level set to 0",
                level=logging.INFO,
            )

        # num_training_steps: read from any scheduler that exposes it
        # (CosineSchedulerConfig currently; future schedulers may add it).
        # Used in train_loop to trigger loss_guard_should_stop — works with
        # constant scheduler too once the field is set.
        sched = self.cfg.hyperparams.scheduler
        self.num_training_steps = int(getattr(sched, "num_training_steps", 0) or 0)

        # ── loss_guard: optional per-step train_loss dump ─────────────
        # Set ``LOSS_GUARD_PATH`` env to opt in; trainer appends
        # ``{step, train_loss}`` each optimizer step (main rank only) and
        # writes a JSON at end-of-training or when ``num_training_steps`` is
        # reached. Used for cross-code-version loss-curve alignment (e.g.
        # acc_pi/wall-x baseline ↔ online_rl/wall-x verify).
        # Mirror of acc_pi fsdp_trainer.py:130-138, 1094-1106, 1268-1281.
        self.loss_guard_path = os.environ.get("LOSS_GUARD_PATH")
        self.loss_guard_records: list = []
        self.loss_guard_meta = {
            "id": os.environ.get("LOSS_GUARD_ID"),
            "name": os.environ.get("LOSS_GUARD_NAME"),
        }
        self.loss_guard_should_stop = False
        self.metrics_logger = MetricsLogger(
            wandb_run=self.wandb_run,
            log_interval=self.log_interval,
            smooth_window=self.cfg.logging.loss_log_smooth_window,
        )

        # Loss tracking
        self.base_l1_loss = None
        self.base_l1_loss_detail = {}

        startup_t0 = time.time()
        stage_t0 = time.time()
        self.load_normalizer()
        self.log("[StartupTiming] normalizer_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self.load_processor()
        self.log("[StartupTiming] processor_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self.load_model()
        self.log("[StartupTiming] model_s={:.3f}".format(time.time() - stage_t0))

        # DDP needs to see correct requires_grad at wrap time, so freeze first.
        stage_t0 = time.time()
        self._freeze_params_if_needed(self.model)
        self.log(
            "[StartupTiming] freeze_params_s={:.3f}".format(time.time() - stage_t0)
        )

        # Frozen-submodule prefixes for checkpoint filtering. Must be computed
        # before FSDP wrapping, while child modules still have stable names.
        stage_t0 = time.time()
        self._frozen_prefixes = self._compute_frozen_prefixes()

        # Original shapes captured before FSDP flattening.
        self.model._orig_param_shapes = {
            name: p.shape for name, p in self.model.named_parameters()
        }
        self.log(
            "[StartupTiming] param_metadata_s={:.3f}".format(time.time() - stage_t0)
        )

        if self._resume_from_single_file():
            stage_t0 = time.time()
            self.load_state_dict(
                self.model,
                {
                    "ckpt": self.cfg.checkpoint.resume_from,
                    "try_harder": self.cfg.checkpoint.try_harder,
                },
            )
            self.log(
                "[StartupTiming] single_file_resume_s={:.3f} path={}".format(
                    time.time() - stage_t0,
                    self.cfg.checkpoint.resume_from,
                )
            )

        stage_t0 = time.time()
        self._wrap_model(self.model)
        self.log("[StartupTiming] wrap_model_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self._create_optimizer()
        self.log("[StartupTiming] optimizer_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self._create_scheduler()
        self.log("[StartupTiming] scheduler_s={:.3f}".format(time.time() - stage_t0))

        if self._resume_from_training_checkpoint():
            stage_t0 = time.time()
            self.resume_from_checkpoint()
            self.log(
                "[StartupTiming] training_checkpoint_resume_s={:.3f}".format(
                    time.time() - stage_t0
                )
            )

        stage_t0 = time.time()
        self.load_dataset()
        self.log("[StartupTiming] dataset_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self.adapter.init_validation(self.normalizer_action, self.normalizer_propri)
        self.log(
            "[StartupTiming] validation_init_s={:.3f} total_s={:.3f}".format(
                time.time() - stage_t0,
                time.time() - startup_t0,
            )
        )

    def load_normalizer(self):
        norm_cfg = self.cfg.data.normalizer_config or {}
        custom_stats_path = norm_cfg.get("customized_action_statistic_dof")
        if not custom_stats_path and self.cfg.data.dataset_type == "lerobot":
            from wall_x.data.backends.lerobot.build import load_lerobot_normalizers

            loaded = load_lerobot_normalizers(self.cfg)
            if loaded is not None:
                self.normalizer_action, self.normalizer_propri = (
                    loaded[0],
                    loaded[1],
                )
                self._action_statistic_dof = None
                self.logger.info(
                    "Loaded LeRobot normalizers from %s with dataset key %s",
                    loaded[2],
                    loaded[3],
                )
                return

        # x2robot_v2 stats are nested (dataset_type/end_effector_type) and are the
        # single source of truth: wall-x's flat Normalizer cannot parse them and
        # silently collapses to identity. Build the model-side normalizer straight
        # from v2's resolver so the saved .pth carries the real statistics and is
        # bit-exact with what the v2 dataset pipeline applied during training.
        if self.cfg.data.dataset_type == "x2robot_v2" and custom_stats_path:
            from wall_x.model.core.action.normalizer import (
                V2_DEFAULT_DELTA_KEY,
                V2_DEFAULT_MIN_KEY,
                create_normalizers_from_v2_stats,
            )

            bridge_cfg = {
                "dof_config": self.cfg.task.dof_config,
                "agent_pos_config": self.cfg.task.agent_pos_config,
                "stats_file": custom_stats_path,
                "min_key": norm_cfg.get("min_key", V2_DEFAULT_MIN_KEY),
                "delta_key": norm_cfg.get("delta_key", V2_DEFAULT_DELTA_KEY),
            }
            for k in ("type", "default_range"):
                v = norm_cfg.get(k, None)
                if v is not None:
                    bridge_cfg[k] = v
            (
                self.normalizer_action,
                self.normalizer_propri,
                self._action_statistic_dof,
            ) = create_normalizers_from_v2_stats(bridge_cfg)
            self.logger.info(
                "Built model normalizers from internal_dataset_backend_v2 stats (bridge); "
                "dataset_type=x2robot_v2, stats=%s",
                custom_stats_path,
            )
            return

        merged = {
            "dof_config": self.cfg.task.dof_config,
            "agent_pos_config": self.cfg.task.agent_pos_config,
            "normalizer_config": {
                "customized_action_statistic_dof": custom_stats_path,
                "min_key": norm_cfg.get("min_key", "min"),
                "delta_key": norm_cfg.get("delta_key", "delta"),
            },
        }
        self.normalizer_action, self.normalizer_propri, self._action_statistic_dof = (
            create_normalizers(merged)
        )

    def backward(self, loss: torch.Tensor):
        """Perform backward pass."""
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        """Unscale and clip gradient norm via the active distribution strategy."""
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

        total_norm = self.strategy.clip_grad_norm(
            self.model,
            max_norm,
            optimizer=self.optimizer,
        )
        self._dedicated_param_grad_clip_stats = getattr(
            self.strategy,
            "last_grad_clip_stats",
            None,
        )
        return total_norm

    def _reduce_sp_partial_grads(self) -> None:
        """Sum the SP-sharded params' partial grads across the SP group.

        Runs once per optimizer step, after FSDP's DP reduce-scatter and before
        grad-norm/clip. The mechanism lives in ``wall_x.distributed``
        (``reduce_sp_partial_grads``); *which* params are SP-sharded is declared
        by the model adapter (``sp_sharded_param_prefixes``). Here the trainer
        only owns the *when* and a one-time sanity guard against a prefix set
        that matches nothing (SP enabled on a model that doesn't support it, or
        a drifted module FQN layout).
        """
        if get_sp_world_size() <= 1:
            return
        prefixes = self.adapter.sp_sharded_param_prefixes()
        matched = reduce_sp_partial_grads(
            self.model.named_parameters(),
            lambda name: name.startswith(prefixes),
            sp_group=get_sp_group(),
        )
        if not getattr(self, "_sp_grad_checked", False):
            self._sp_grad_checked = True
            if matched == 0:
                raise RuntimeError(
                    "sequence parallel is enabled (sp_size>1) but no params "
                    f"matched the adapter's SP-sharded prefixes {prefixes!r}; "
                    "either the model does not support SP or its module FQN "
                    "layout changed."
                )

    def load_model(self):
        """Load and prepare model, optimizer, and scheduler"""
        model_t0 = time.time()
        self.model_config = self.adapter.build_model_config()
        self.log("[StartupTiming] model_config_s={:.3f}".format(time.time() - model_t0))
        create_t0 = time.time()
        self.model = self.adapter.create_model(
            self.processor, self.tokenizer_mixin, self.model_config
        )
        self.log(
            "[StartupTiming] model_create_s={:.3f}".format(time.time() - create_t0)
        )
        type(self.adapter).log_attention_implementation(self.logger, self.model)
        weights_t0 = time.time()
        self.adapter.load_weights(
            self.model,
            self.normalizer_action,
            self.normalizer_propri,
            processor=self.processor,
        )
        self.log(
            "[StartupTiming] adapter_load_weights_s={:.3f} model_total_s={:.3f}".format(
                time.time() - weights_t0,
                time.time() - model_t0,
            )
        )

    def load_processor(self):
        """Load processor and tokenizers"""
        self.adapter.normalizer_action = self.normalizer_action
        self.adapter.normalizer_propri = self.normalizer_propri
        processors_dict = self.adapter.load_processor(self._action_statistic_dof)
        self.processor = processors_dict["processor"]
        self.train_action_tokenizer = processors_dict["train_action_tokenizer"]
        self.val_action_tokenizer = processors_dict["val_action_tokenizer"]
        self.action_mapper = processors_dict["action_mapper"]
        self.tokenizer_mixin = processors_dict.get("tokenizer_mixin")

    def _freeze_params_if_needed(self, model: torch.nn.Module):
        """Freeze non-action parameters when train_action_expert_only is set.

        Must be called BEFORE wrapping with DDP/FSDP so the wrapper sees the
        correct requires_grad flags and does not expect gradients for frozen params.
        """
        from wall_x.trainer.optimizer.utils import resolve_lr_group_configs

        opt = self.cfg.hyperparams.optimizer
        if not opt.train_action_expert_only:
            return
        lr_groups = resolve_lr_group_configs(
            opt, self.adapter.default_action_lr_keywords
        )
        if not lr_groups:
            self.log(
                "WARNING: train_action_expert_only is True but no optimizer LR "
                "group is set. No parameters will be frozen.",
                level=logging.WARNING,
            )
            return

        frozen_count = 0
        grouped_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            matches_group = any(
                any(keyword in name for keyword in group.include) for group in lr_groups
            )
            if matches_group:
                grouped_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1

        if grouped_count == 0:
            raise ValueError(
                "No grouped params found for train_action_expert_only. "
                "Please check optimizer.lr_groups or legacy action_lr_keywords."
            )

        self.log(
            f"*** train_action_expert_only: frozen {frozen_count} base params, "
            f"keeping {grouped_count} grouped params trainable ***"
        )

    def _wrap_model(self, model: torch.nn.Module):
        """Prepare model dtype/placement via adapter, then wrap via strategy.

        If the strategy created an explicit trainer process group, register
        it so trainer-level barriers and metric collectives keep all-rank
        semantics independent of the FSDP mesh topology.
        """
        self.adapter.convert_to_mix_precision_hint(
            model,
            device=self.device,
            use_fsdp=self.cfg.distributed.use_fsdp,
            backend=self.cfg.distributed.backend,
            log_fn=self.log,
        )
        from wall_x.trainer.optimizer.dmuon.utils import dmuon_owner_assignment_kwargs

        dmuon_owner_kwargs = dmuon_owner_assignment_kwargs(
            self.cfg.hyperparams.optimizer
        )
        if dmuon_owner_kwargs:
            self.log(
                "[DMuon] owner assignment config: "
                f"owner_strategy={dmuon_owner_kwargs['owner_strategy']}, "
                f"owner_cost_model={dmuon_owner_kwargs['owner_cost_model']}, "
                f"owner_pack_small_params={dmuon_owner_kwargs['pack_small_params']}, "
                f"hsdp_column_balance={dmuon_owner_kwargs['hsdp_column_balance']}"
            )
        self.model = self.strategy.wrap(
            model,
            use_dedicated_params=self.cfg.hyperparams.optimizer.use_dedicated_params,
            dmuon_owner_kwargs=dmuon_owner_kwargs,
        )

        trainer_pg = getattr(self.strategy, "trainer_process_group", None)
        if trainer_pg is not None:
            from wall_x.trainer.fsdp_trainer.base_trainer import (
                set_trainer_process_group,
            )

            set_trainer_process_group(trainer_pg)
            self.log("[FSDP2] routing trainer collectives through trainer PG")

    def _create_optimizer(self):
        """Create optimizer for FSDP wrapped model."""
        from wall_x.trainer.optimizer.utils import (
            build_lr_param_groups,
            resolve_lr_group_configs,
            uses_legacy_action_lr_groups,
        )

        opt_cfg = self.cfg.hyperparams.optimizer

        param_groups = None
        lr_group_configs = resolve_lr_group_configs(
            opt_cfg, self.adapter.default_action_lr_keywords
        )
        if lr_group_configs:
            if opt_cfg.optimizer_type not in (
                "adamw",
                "muon",
                "dmuon",
                "naive_fsdp2_muon",
            ):
                raise ValueError(
                    "optimizer.lr_groups are only supported with adamw, muon, "
                    "dmuon, or naive_fsdp2_muon"
                )
            base_group_name = (
                "base_lr_group" if uses_legacy_action_lr_groups(opt_cfg) else "base"
            )
            param_groups = build_lr_param_groups(
                self.model,
                opt_cfg,
                lr_group_configs,
                base_group_name=base_group_name,
            )
            summary = ", ".join(
                f"{group.name}={group.lr} include={group.include}"
                for group in lr_group_configs
            )
            self.log(
                f"setting optimizer LR groups: base={opt_cfg.learning_rate}; "
                f"{summary} ({opt_cfg.optimizer_type})"
            )

        optimizer_kwargs = {
            "opt_cfg": opt_cfg,
            "param_groups": param_groups,
        }
        if opt_cfg.optimizer_type in ("dmuon", "naive_fsdp2_muon"):
            optimizer_kwargs["log_fn"] = self.log

        self.optimizer = get_optimizer(
            opt_cfg.optimizer_type,
            self.model,
            **optimizer_kwargs,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        from wall_x.config.hyperparams_config import (
            ConstantSchedulerConfig,
            CosineSchedulerConfig,
        )

        sched = self.cfg.hyperparams.scheduler
        lr = self.cfg.hyperparams.optimizer.learning_rate

        if isinstance(sched, ConstantSchedulerConfig):
            self.lr_scheduler = get_scheduler(
                optimizer=self.optimizer,
                lr_scheduler_type="constant",
                num_warmup_steps=sched.num_warmup_steps,
            )
        elif isinstance(sched, CosineSchedulerConfig):
            if sched.num_training_steps <= 0:
                raise ValueError(
                    "num_training_steps must be > 0 for cosine scheduler. "
                    "Please set it explicitly in the config."
                )
            min_lr = sched.min_lr if sched.min_lr is not None else 0.1 * lr
            self.lr_scheduler = get_scheduler(
                optimizer=self.optimizer,
                lr_scheduler_type="cosine",
                num_warmup_steps=sched.num_warmup_steps,
                num_training_steps=sched.num_training_steps,
                peak_lr=lr,
                end_lr=min_lr,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {type(sched).__name__}")

    def load_dataset(self):
        """Load training and validation datasets"""
        torch.cuda.empty_cache()
        barrier()
        self._resume_checkpoint_path = None
        self._resume_loaded_global_step = None
        self._resume_next_global_step = self.global_step
        self._resume_consumed_samples = None
        self._resume_iter_offset = 0
        self._display_train_iter_total = None
        # Dispatch to adapter with unified signature (supports model-specific kwargs)
        extra_dataset_kwargs = dict(
            model_config=self.model_config,
            tokenizer_mixin=self.tokenizer_mixin,
            train_action_tokenizer=self.train_action_tokenizer,
            val_action_tokenizer=self.val_action_tokenizer,
            normalizer_action=self.normalizer_action,
            normalizer_propri=self.normalizer_propri,
        )

        resume_path = self.cfg.checkpoint.resume_from
        if resume_path is not None and os.path.isdir(resume_path):
            resume_load_start = time.time()
            self._resume_checkpoint_path = resume_path
            # Step and epoch restore are generic. Backend-owned episode
            # metadata is optional and only loaded when the adapter exposes
            # the hook.
            is_incomplete_epoch = False
            sampler_state_path = os.path.join(
                resume_path, f"sampler_state_rank_{self.rank}.pt"
            )
            if os.path.exists(sampler_state_path):
                is_incomplete_epoch = True
                extra_dataset_kwargs["sampler_state_path"] = sampler_state_path

            dataset_state = _ckpt_io.load_dataset_state(
                checkpoint_path=resume_path,
                rank=self.rank,
                current_world_size=self.world_size,
                log_fn=self.log,
            )
            if dataset_state is not None:
                is_incomplete_epoch = True
                extra_dataset_kwargs["dataset_state"] = dataset_state
                self._resume_consumed_samples = self._extract_dataset_consumed_samples(
                    dataset_state
                )

            if hasattr(self.adapter, "_load_episode_indices"):
                resume_indices = self.adapter._load_episode_indices(
                    resume_path, self.rank
                )
                if resume_indices is not None:
                    is_incomplete_epoch = bool(
                        is_incomplete_epoch or resume_indices["is_incomplete_epoch"]
                    )
                    extra_dataset_kwargs["resume_indices"] = resume_indices

            train_state = self.adapter.load_step_and_epoch(
                resume_path, is_incomplete_epoch
            )
            self.global_step = train_state["global_step"]
            self._resume_loaded_global_step = self.global_step
            self._resume_next_global_step = self.global_step + 1
            self.start_epoch = train_state["start_epoch"]
            self.log(
                "[Resume] checkpoint={} loaded_global_step={} "
                "next_global_step={} start_epoch={} dataset_state_loaded={} "
                "consumed_samples_per_rank={} load_state_s={:.1f}".format(
                    resume_path,
                    self._resume_loaded_global_step,
                    self._resume_next_global_step,
                    self.start_epoch,
                    dataset_state is not None,
                    self._resume_consumed_samples,
                    time.time() - resume_load_start,
                )
            )
            # global_step records completed steps, so resume from the next step
            # to avoid triggering _should_save_checkpoint() immediately.
            self.global_step += 1
        dataset_load_start = time.time()
        (
            self.dataset,
            self.train_dataloader,
            self.train_num,
        ) = self.adapter.load_dataset(
            self.data_config,
            self.processor,
            self.rank,
            self.world_size,
            **extra_dataset_kwargs,
        )
        self._prepare_train_iter_display(len(self.train_dataloader))
        if self._resume_checkpoint_path is not None:
            self.log(
                "[Resume] dataloader_remaining_iters={} iter_offset={} "
                "display_total_iters={} dataloader_batch_size={} "
                "dataset_build_s={:.1f}".format(
                    len(self.train_dataloader),
                    self._resume_iter_offset,
                    self._display_train_iter_total,
                    self._infer_dataloader_batch_size(),
                    time.time() - dataset_load_start,
                )
            )

    def _maybe_dump_oom_memsnap(self, exc):
        """Rescue dump of the CUDA allocator history when OOM hits before the
        normal step-3 snapshot has fired.

        Only fires when global_step < memory_snapshot_until_step, because:
        - After until_step, the normal dump already wrote a baseline pickle
          and ``_record_memory_history(enabled=None)`` stopped recording —
          a late dump here would be empty / stale.
        - And by then the allocator-history ring buffer is so long the
          memory_viz tool can't open the pickle anyway.

        Best-effort: any failure during dump is logged but swallowed, so we
        still re-raise the original OOM at the caller.
        """
        if not (
            self.cfg.debug.enable_memory_snapshot
            and self.rank == 0
            and self.global_step < self.cfg.debug.memory_snapshot_until_step
        ):
            return
        try:
            os.makedirs(self.cfg.debug.memory_snapshot_dir, exist_ok=True)
            snap_path = os.path.join(
                self.cfg.debug.memory_snapshot_dir,
                f"memory_snapshot_oom_step{self.global_step}.pickle",
            )
            torch.cuda.memory._dump_snapshot(snap_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            self.log(
                f"[MemSnap-OOM] dumped to {snap_path} after "
                f"{type(exc).__name__} at step {self.global_step}; "
                f"recording stopped"
            )
        except Exception as dump_err:
            self.log(
                f"[MemSnap-OOM] dump failed at step {self.global_step}: "
                f"{type(dump_err).__name__}: {dump_err}",
                level=logging.WARNING,
            )

    @contextlib.contextmanager
    def _oom_dump_guard(self):
        """Context manager that rescues a CUDA memsnap on OOM, then re-raises.

        Wrap the GPU-compute-heavy sections (forward / backward / optimizer
        step) so that an early-step OOM produces a viz-openable allocator
        snapshot before the process dies. See ``_maybe_dump_oom_memsnap``
        for the gate condition.
        """
        try:
            yield
        except torch.OutOfMemoryError as e:
            self._maybe_dump_oom_memsnap(e)
            raise

    def train_loop(self, epoch: int, profiler=contextlib.nullcontext()):
        """Execute training for a single epoch"""
        self.model.train()

        # VGDynamicRobotDataset (v1 path) doesn't expose set_epoch; other
        # dataset backends (v2, lerobot) do. Dispatch conditionally.
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        total = len(self.train_dataloader)
        if self._resume_checkpoint_path is not None and epoch == self.start_epoch:
            display_total = self._display_train_iter_total
        else:
            self._resume_iter_offset = 0
            self._display_train_iter_total = total
            display_total = total
        # Drop last
        stop_step = total - total % self.grad_accum_steps

        # Disable automatic GC to prevent random ~200ms stalls during compute
        gc_interval = self.cfg.logging.gc_interval_steps
        gc.disable()

        t0 = time.time()
        pending_dataset_samples = 0
        first_batch_timing_logged = False
        with profiler:
            self.timers("interval-time", log_level=0).start(
                barrier=self.show_time_details
            )
            self.timers("data-load", log_level=0).start(barrier=self.show_time_details)

            for i, batch in enumerate(self.train_dataloader, self.initial_step):
                # With distributed training, a slow dataloader on one rank makes
                # faster ranks wait at the first forward/backward collective.
                # When detailed timing is enabled, charge that cross-rank skew
                # to data-load instead of compute so rank-local logs are easier
                # to interpret.
                self.timers("data-load").stop(barrier=self.show_time_details)
                if self.show_time_details and not first_batch_timing_logged:
                    first_batch_timing_logged = True
                    self.log(
                        "[DataTiming] first_batch_wait_s={:.3f} "
                        "local_iter={} display_iter={}/{} global_step={} "
                        "resume_checkpoint={}".format(
                            time.time() - t0,
                            i,
                            self._resume_iter_offset + i,
                            display_total,
                            self.global_step,
                            getattr(self, "_resume_checkpoint_path", None),
                        )
                    )
                batch_sample_count = self._batch_sample_count(batch)

                # Save first batch for offline profiling (rank 0 only)
                debug_batch_path = self.cfg.debug.save_debug_batch_path
                if debug_batch_path and i == self.initial_step and self.rank == 0:
                    os.makedirs(os.path.dirname(debug_batch_path) or ".", exist_ok=True)
                    torch.save({"batch": batch}, debug_batch_path)
                    self.log(f"Saved debug batch to {debug_batch_path}")

                # Move batch to device
                with zone(Z.DATA_TO_DEVICE):
                    self.timers("batch-to-device", log_level=0).start(barrier=False)
                    batch = move_batch_to_device(batch, self.device)
                    self.timers("batch-to-device").stop(sync=True)

                if self.cfg.debug.enable_mfu_profile and self.global_step == 0:
                    self.log(
                        "[MFU] Forward FLOPs profiling is not available in the public package.",
                        level=logging.INFO,
                    )

                # Forward pass with autocast
                self.timers("forward-compute", log_level=0).start(
                    barrier=self.show_time_details
                )
                with zone(Z.FORWARD), self._oom_dump_guard(), self.autocast_context():
                    outputs = self.adapter.forward(
                        self.model, batch, global_step=self.global_step
                    )
                self.timers("forward-compute").stop(sync=True)
                if self.show_time_details:
                    try:
                        _free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
                        _alloc = torch.cuda.memory_allocated(self.device) / (1024**3)
                        _reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                        _peak_alloc = torch.cuda.max_memory_allocated(self.device) / (
                            1024**3
                        )
                        _peak_reserved = torch.cuda.max_memory_reserved(self.device) / (
                            1024**3
                        )
                        self.log(
                            "[MemoryDetails] "
                            f"step={self.global_step} rank={self.rank} "
                            f"after_forward_allocated_gib={_alloc:.3f} "
                            f"after_forward_reserved_gib={_reserved:.3f} "
                            f"peak_allocated_gib={_peak_alloc:.3f} "
                            f"peak_reserved_gib={_peak_reserved:.3f} "
                            f"free_gib={_free_bytes / (1024 ** 3):.3f} "
                            f"total_gib={_total_bytes / (1024 ** 3):.3f}",
                            level=logging.INFO,
                        )
                    except Exception as _mem_exc:
                        self.log(
                            f"[MemoryDetails] failed: {type(_mem_exc).__name__}: {_mem_exc}",
                            level=logging.WARNING,
                        )

                loss = self.adapter.extract_loss(outputs)

                # Check for NaN loss -- use ``nan_to_num`` instead of
                # ``zeros_like`` or ``loss * 0``:
                #  - ``zeros_like(loss)`` returns a fresh leaf tensor with no
                #    grad_fn → backward crashes with "element 0 of tensors
                #    does not require grad and does not have a grad_fn".
                #  - ``loss * 0`` preserves grad_fn but ``NaN * 0 = NaN`` per
                #    IEEE 754, so grads still propagate as NaN.
                # ``torch.nan_to_num`` is differentiable: its forward replaces
                # NaN→0, its backward zeros the upstream grad at NaN positions
                # → clean grads, no NCCL deadlock, no NaN-poisoning of params.
                # Skipping the step is the caller's responsibility (e.g., via
                # GradScaler).
                nan_loss = torch.isnan(loss)
                if nan_loss:
                    self.log(
                        f"Warning: nan in loss at epoch: {epoch}, step: {i}",
                        level=logging.WARNING,
                    )
                    loss = torch.nan_to_num(loss, nan=0.0)

                # Backward pass
                self.timers("backward-compute", log_level=0).start(
                    barrier=self.show_time_details
                )

                context = (
                    contextlib.nullcontext()
                    if self.sync_gradients()
                    else self.strategy.no_sync(self.model)
                )
                with zone(Z.BACKWARD), context, self._oom_dump_guard():
                    scaled_loss = loss / self.grad_accum_steps
                    self.backward(scaled_loss)
                self.timers("backward-compute").stop()
                pending_dataset_samples += batch_sample_count

                # Capture sync flag before the if/else below mutates micro_step /
                # global_step. Used at the end of the loop to gate profiler.step()
                # on the gbs boundary so 1 ProfilerStep = grad_accum_steps × (fwd+bwd)
                # + reduce-scatter + optim, comparable across grad_accum settings.
                did_optim_step = self.sync_gradients()

                # Gradient sync and optimizer step
                if did_optim_step:
                    self.timers("optimizer", log_level=0).start(
                        barrier=self.show_time_details
                    )

                    # Sequence parallel: sum the SP-sharded DiT block params'
                    # partial grads across the SP group, before grad-norm/clip.
                    self._reduce_sp_partial_grads()

                    # Per-component grad norms (before clipping)
                    self.timers("optimizer-grad-norms", log_level=0).start(
                        barrier=False
                    )
                    self._component_grad_norms = {}
                    with zone(Z.GRAD_NORMS):
                        self.adapter.collect_grad_norms(
                            self.model,
                            self._component_grad_norms,
                            device=self.device,
                            reduce_tensor_fn=self.reduce_tensor,
                            params_sharded=self.strategy.params_sharded,
                        )
                    self.timers("optimizer-grad-norms").stop()

                    # Clip gradients
                    self.timers("optimizer-clip", log_level=0).start(barrier=False)
                    with zone(Z.GRAD_CLIP):
                        if self.cfg.hyperparams.optimizer.enable_grad_clip:
                            total_norm = self.clip_grad_norm(self.max_grad_norm)
                        else:
                            self._dedicated_param_grad_clip_stats = None
                            total_norm = 0
                    self.timers("optimizer-clip").stop()

                    self.timers("optimizer-step", log_level=0).start(barrier=False)
                    with zone(Z.OPTIMIZER_STEP), self._oom_dump_guard():
                        self.optimizer_step()
                    self.timers("optimizer-step").stop()

                    self.timers("optimizer-zero-grad", log_level=0).start(barrier=False)
                    with zone(Z.ZERO_GRAD):
                        self.optimizer_zero_grad()
                    self.timers("optimizer-zero-grad").stop()
                    self.timers("optimizer").stop()

                    # Scheduler step
                    self.lr_scheduler_step()

                    # Logging
                    self.timers("logging", log_level=0).start(
                        barrier=self.show_time_details
                    )
                    with zone(Z.LOGGING):
                        _t_metrics_start = time.time()
                        self._log_training_metrics(
                            epoch, i, total, loss, total_norm, outputs
                        )
                        _t_metrics_ms = (time.time() - _t_metrics_start) * 1000

                    t1 = time.time()
                    self.training_log(
                        epoch,
                        self.num_epoch,
                        self._resume_iter_offset + i,
                        display_total,
                        loss,
                        self.get_lr(),
                        t1 - t0,
                        self.show_time_details,
                    )
                    self.timers("logging").stop()
                    t0 = time.time()

                    # Optional: log per-step breakdown to pinpoint spikes (e.g. param_norms every 100 steps)
                    if (
                        self.show_time_details
                        and _t_metrics_ms > 5000
                        and is_main_process()
                    ):
                        self.log(
                            f"[Step time breakdown] _log_training_metrics took {_t_metrics_ms:.0f} ms at global_step={self.global_step}",
                            level=logging.INFO,
                        )

                    # The optimizer/scheduler state now includes all pending
                    # micro-batches, so advance the dataset cursor before any
                    # checkpoint for this committed step is written.
                    self._advance_dataset(pending_dataset_samples)
                    pending_dataset_samples = 0

                    # Checkpoint saving (FSDP full state_dict can take 10-20s every save_interval steps)
                    if self._should_save_checkpoint():
                        _t_save_start = time.time()
                        with zone(Z.SAVE_CHECKPOINT):
                            self.save_checkpoint(epoch, self.global_step)
                        if is_main_process():
                            self.log(
                                f"[Step time breakdown] save_checkpoint took {(time.time() - _t_save_start):.1f} s at global_step={self.global_step}",
                                level=logging.INFO,
                            )

                    # Validation
                    if self._should_validate():
                        self.val_loop()

                    self.global_step += 1
                    self.micro_step = 0

                    # Dump memory snapshot at the configured step (rank 0).
                    if (
                        self.cfg.debug.enable_memory_snapshot
                        and self.rank == 0
                        and self.global_step
                        == self.cfg.debug.memory_snapshot_until_step
                    ):
                        os.makedirs(self.cfg.debug.memory_snapshot_dir, exist_ok=True)
                        snap_path = os.path.join(
                            self.cfg.debug.memory_snapshot_dir,
                            "memory_snapshot.pickle",
                        )
                        torch.cuda.memory._dump_snapshot(snap_path)
                        torch.cuda.memory._record_memory_history(enabled=None)
                        self.log(
                            f"[MemSnap] dumped to {snap_path} at step "
                            f"{self.global_step}; recording stopped"
                        )

                    # Manual GC outside timing window to avoid random stalls (can add 1-5s every gc_interval_steps)
                    if self.global_step % gc_interval == 0:
                        _t_gc_start = time.time()
                        gc.collect()
                        if is_main_process() and self.show_time_details:
                            self.log(
                                f"[Step time breakdown] gc.collect took {(time.time() - _t_gc_start):.1f} s at global_step={self.global_step}",
                                level=logging.INFO,
                            )
                else:
                    self.micro_step += 1

                del batch
                self.timers("interval-time").stop()

                # Drop last
                if i == stop_step:
                    break

                # Fire profiler.step() per optim cycle (gbs boundary). With
                # grad_accum_steps > 1 this groups all micro fwd+bwd + the
                # cross-node reduce-scatter + optim into one ProfilerStep.
                # Without this guard, ProfilerStep tracked micro-batches and
                # traces from different grad_accum settings were not comparable.
                if did_optim_step and not isinstance(profiler, contextlib.nullcontext):
                    profiler.step()

                if (
                    self.num_training_steps > 0
                    and self.global_step >= self.num_training_steps
                ):
                    # Set the flag so base_trainer.train() exits the outer
                    # epoch loop instead of starting another epoch.
                    self.loss_guard_should_stop = True
                    break

                # Setup timers for next iteration
                if i < total - 1:
                    self.timers("interval-time", log_level=0).start(
                        barrier=self.show_time_details
                    )
                    self.timers("data-load", log_level=0).start(
                        barrier=self.show_time_details
                    )

        # Re-enable automatic GC after training loop
        gc.enable()

        # loss_guard JSON dump on main rank (only when the LOSS_GUARD_PATH env
        # is set). Writes when training is ending — either we hit
        # num_training_steps this epoch or this is the last configured epoch.
        if (
            is_main_process()
            and self.loss_guard_path
            and (self.loss_guard_should_stop or epoch == self.num_epoch - 1)
        ):
            self._write_loss_guard()

        # Reset dataloader for next epoch
        self.train_dataloader = self.dataset.get_train_dataloader()

    def _write_loss_guard(self):
        """Dump collected per-step train_loss records to LOSS_GUARD_PATH.

        Idempotent: re-callable across epochs without harm. Layout matches
        acc_pi/wall-x ``fsdp_trainer._write_loss_guard`` so the two JSONs are
        diffable directly.
        """
        if not self.loss_guard_path:
            return
        dirname = os.path.dirname(self.loss_guard_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        payload = {
            "id": self.loss_guard_meta.get("id"),
            "name": self.loss_guard_meta.get("name"),
            "seed": getattr(self, "seed", None),
            "num_epochs": self.num_epoch,
            "global_step": int(self.global_step),
            "losses": [r["train_loss"] for r in self.loss_guard_records],
            "records": self.loss_guard_records,
        }
        with open(self.loss_guard_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _batch_sample_count(self, batch) -> int:
        """Best-effort per-rank sample count for dataset cursor advancement."""
        if isinstance(batch, dict):
            for value in batch.values():
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
                if isinstance(value, (list, tuple)):
                    return len(value)
            return int(getattr(self.train_dataloader, "batch_size", 1) or 1)
        if torch.is_tensor(batch) and batch.ndim > 0:
            return int(batch.shape[0])
        if isinstance(batch, (list, tuple)):
            return len(batch)
        return int(getattr(self.train_dataloader, "batch_size", 1) or 1)

    @staticmethod
    def _extract_dataset_consumed_samples(dataset_state) -> Optional[int]:
        """Return per-rank consumed samples from backend checkpoint state."""
        if not isinstance(dataset_state, dict):
            return None

        candidates = [dataset_state.get("consumed")]
        train_sampler_state = dataset_state.get("train_sampler")
        if isinstance(train_sampler_state, dict):
            candidates.append(train_sampler_state.get("consumed"))
        sampler_state = dataset_state.get("sampler")
        if isinstance(sampler_state, dict):
            candidates.append(sampler_state.get("consumed"))

        for value in candidates:
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _infer_dataloader_batch_size(self) -> Optional[int]:
        """Best-effort batch-size inference for resume progress display."""
        cfg_data = getattr(self.cfg, "data", None)
        for value in (
            getattr(self.train_dataloader, "batch_size", None),
            getattr(self.cfg.hyperparams, "batch_size_per_gpu", None),
            getattr(cfg_data, "batch_size", None),
        ):
            if value is None:
                continue
            try:
                batch_size = int(value)
            except (TypeError, ValueError):
                continue
            if batch_size > 0:
                return batch_size
        return None

    def _prepare_train_iter_display(self, remaining_iters: int) -> None:
        """Compute display-only epoch iter offsets after dataset resume."""
        self._resume_iter_offset = 0
        consumed_samples = getattr(self, "_resume_consumed_samples", None)
        batch_size = self._infer_dataloader_batch_size()
        if consumed_samples is not None and batch_size:
            self._resume_iter_offset = consumed_samples // batch_size
            remainder = consumed_samples % batch_size
            if remainder:
                self.log(
                    "[Resume] consumed_samples_per_rank={} is not divisible by "
                    "dataloader_batch_size={}; display iter offset is rounded down".format(
                        consumed_samples,
                        batch_size,
                    ),
                    level=logging.WARNING,
                )
        self._display_train_iter_total = self._resume_iter_offset + int(remaining_iters)

    def _advance_dataset(self, n: int) -> None:
        """Advance backend-owned dataset cursors after a committed train step."""
        if n <= 0:
            return
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "advance"):
            dataset.advance(n)
        elif dataset is not None and hasattr(dataset, "advance_train_sampler"):
            dataset.advance_train_sampler(n)

    def _log_training_metrics(self, epoch, step, total, loss, total_norm, outputs):
        """Collect per-step stats, delegate buffering + wandb emission."""
        lr = self.get_lr()
        train_loss = self.reduce_tensor(loss.detach()).item()

        # ── loss_guard: append per-step train_loss on main rank ──
        # Stops appending once we've recorded num_training_steps points so the
        # JSON payload is bounded even if extra epochs slip through.
        if is_main_process() and self.loss_guard_path:
            if (
                self.num_training_steps <= 0
                or len(self.loss_guard_records) < self.num_training_steps
            ):
                self.loss_guard_records.append(
                    {"step": int(self.global_step), "train_loss": float(train_loss)}
                )

        step_stats = {
            "lr": lr,
            "train_loss": train_loss,
            "grad_norm": (
                total_norm.item() if torch.is_tensor(total_norm) else float(total_norm)
            ),
        }
        for idx, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("group_name", f"group_{idx}")
            step_stats[f"lr_group/{group_name}"] = float(group["lr"])

        # Model-family-specific auxiliary losses / accuracies.
        self.adapter.collect_output_stats(
            outputs,
            step_stats,
            reduce_tensor_fn=self.reduce_tensor,
            true_gather_fn=self.true_gather,
            tokenizer_mixin=self.tokenizer_mixin,
        )

        # Per-component grad norms (captured before clip & optimizer.zero_grad).
        if hasattr(self, "_component_grad_norms"):
            step_stats.update(self._component_grad_norms)

        dedicated_clip_stats = getattr(self, "_dedicated_param_grad_clip_stats", None)
        if dedicated_clip_stats is not None:
            step_stats.update(
                {
                    "muon_grad_norm": dedicated_clip_stats["total_norm"],
                    "muon_grad_clip_coef": dedicated_clip_stats["clip_coef"],
                    "muon_grad_clipped": float(dedicated_clip_stats["clipped"]),
                }
            )
            if "adamw_total_norm" in dedicated_clip_stats:
                step_stats.update(
                    {
                        "dmuon_adamw_grad_norm": dedicated_clip_stats[
                            "adamw_total_norm"
                        ],
                        "dmuon_adamw_grad_clip_coef": dedicated_clip_stats[
                            "adamw_clip_coef"
                        ],
                        "dmuon_adamw_grad_clipped": float(
                            dedicated_clip_stats["adamw_clipped"]
                        ),
                        "dmuon_adamw_grad_found_inf": float(
                            dedicated_clip_stats["adamw_found_inf"]
                        ),
                    }
                )

        param_norm_interval = int(
            getattr(self.cfg.logging, "param_norm_interval", 100) or 0
        )
        if param_norm_interval > 0 and self.global_step % param_norm_interval == 0:
            self.adapter.collect_param_norms(
                self.model,
                step_stats,
                device=self.device,
                reduce_tensor_fn=self.reduce_tensor,
                params_sharded=self.strategy.params_sharded,
            )

        self._current_step_raw_stats = step_stats

        # Display-smoothing rolling window (DZ-style)
        self._current_step_stats = self.metrics_logger.smooth(step_stats)

    def training_log(
        self,
        current_epoch,
        total_epoch,
        current_train_iter,
        total_train_iter,
        loss,
        lr,
        time_per_step,
        show_time_details=False,
    ):
        timer_names = [
            "interval-time",
            "data-load",
            "batch-to-device",
            "forward-compute",
            "backward-compute",
            "optimizer",
            "optimizer-grad-norms",
            "optimizer-clip",
            "optimizer-step",
            "optimizer-zero-grad",
            "logging",
        ]
        timer_detail_stats = {}
        # timers.log() contains all_gather — must run on ALL ranks before the
        # is_main_process() guard to avoid NCCL deadlock.
        if show_time_details:
            timer_detail_stats = self._collect_timer_detail_stats(
                timer_names, time_per_step
            )
            self.timers.log(timer_names, normalizer=1)

        main = is_main_process()
        stats = getattr(self, "_current_step_stats", None) or {}
        time_stats = {"time_per_iter": float(time_per_step)}
        if timer_detail_stats:
            time_stats.update(timer_detail_stats)
        stats.update(time_stats)
        raw_stats = getattr(self, "_current_step_raw_stats", None)
        if raw_stats is not None:
            raw_stats.update(time_stats)

        # MFU is computed after step time is known, then merged into both the
        # console stats and the MetricsLogger buffer so wandb records it.
        mfu_info = None
        if self.cfg.debug.enable_mfu and hasattr(self.adapter, "compute_mfu"):
            unwrapped = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            mfu_info = self.adapter.compute_mfu(unwrapped, time_per_step)
            # Per-rank [MFU-rank] line is verbose (N_rank × N_step entries).
            # Gate it on log_interval so file logs match wandb emission cadence.
            log_mfu_line = (
                mfu_info is not None
                and self.global_step % self.cfg.logging.log_interval == 0
            )
            if log_mfu_line:
                self.log(
                    "[MFU-rank] rank={} mfu={:.2f}% step={:.3f}s "
                    "flops_step={:.3f}T flops_fwd={:.3f}T "
                    "seq={} latent={} source={}".format(
                        getattr(self, "rank", 0),
                        mfu_info["mfu"] * 100.0,
                        time_per_step,
                        mfu_info["flops_per_step_T"],
                        mfu_info.get("flops_forward_T", 0.0) or 0.0,
                        mfu_info.get("seq_dims"),
                        mfu_info.get("latent_dims"),
                        mfu_info.get("latent_source"),
                    ),
                    main_process_only=False,
                )
            if main and mfu_info is not None and stats is not None:
                mfu_stats = {
                    "mfu": mfu_info["mfu"],
                    "flops_per_step_T": mfu_info["flops_per_step_T"],
                }
                flops_forward = mfu_info.get("flops_forward_T")
                if flops_forward is not None:
                    mfu_stats["flops_forward_T"] = flops_forward
                profile_fwd_flops = mfu_info.get("profile_fwd_flops_T")
                if profile_fwd_flops is not None:
                    mfu_stats["profile_fwd_flops_T"] = profile_fwd_flops
                stats.update(mfu_stats)
                raw_stats = getattr(self, "_current_step_raw_stats", None)
                if raw_stats is not None:
                    raw_stats.update(mfu_stats)

        if main:
            # Expose per-step wall time on the same wandb track as loss/lr/mfu.
            # samples_per_sec 是全局吞吐（所有 rank 求和），跟 micro/accum 切分无关，
            # 只看 global batch / step time。
            inv_time = 1.0 / max(float(time_per_step), 1e-9)
            samples_per_step = (
                self.cfg.hyperparams.batch_size_per_gpu
                * self.world_size
                * self.grad_accum_steps
            )
            step_time_stats = {
                "step_time": float(time_per_step),
                "samples_per_sec": samples_per_step * inv_time,
            }
            if stats is not None:
                stats.update(step_time_stats)
            raw_stats = getattr(self, "_current_step_raw_stats", None)
            if raw_stats is not None:
                # 跳过 warm-up：前 N 步 step_time / samples_per_sec 不进 wandb，
                # 否则 step 0 ~85s 把 y 轴拉爆后续看不清。控制台 line（stats）
                # 仍然显示原值。其他 key（loss / lr / mfu / grad_norm）照常写。
                if self.global_step >= self.cfg.logging.perf_log_skip_steps:
                    raw_stats.update(step_time_stats)
                self.metrics_logger.record_step(raw_stats, is_main=True)
                avg_stats = self.metrics_logger.flush_if_due(self.global_step)
                if avg_stats is not None:
                    self.logger.info(
                        f"[FSDP Train] Step {self.global_step}: {avg_stats}"
                    )

        if not main:
            return

        loss_to_print = stats.get(
            "train_loss", loss.item() if torch.is_tensor(loss) else float(loss)
        )
        fields = self.adapter.console_fields(tokenizer_mixin=self.tokenizer_mixin)

        line = self.metrics_logger.format_training_line(
            epoch=current_epoch,
            total_epoch=total_epoch,
            current_iter=current_train_iter,
            total_iter=total_train_iter,
            loss=loss_to_print,
            lr=lr,
            time_per_step=time_per_step,
            stats=stats,
            fields=fields,
            mfu_info=mfu_info,
        )
        self.log(line)

    def _collect_timer_detail_stats(self, timer_names, time_per_step):
        """Collect per-step timer details before ``timers.log`` resets them."""
        try:
            elapsed = self.timers._get_global_min_max_time(
                timer_names,
                reset=False,
                barrier=False,
                normalizer=1.0,
            )
        except Exception as exc:
            self.log(
                f"[TimeDetails] failed to collect timer stats: {type(exc).__name__}: {exc}",
                level=logging.WARNING,
            )
            return {}

        step_time = max(float(time_per_step), 1e-9)
        stats = {}
        for name, (min_time, max_time) in elapsed.items():
            key = name.replace("-", "_")
            stats[f"time/{key}_min_s"] = float(min_time)
            stats[f"time/{key}_max_s"] = float(max_time)
            stats[f"time/{key}_max_pct"] = float(max_time) / step_time
        return stats

    def _should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved"""
        log = self.cfg.logging
        return (
            self.global_step >= log.ignore_until_interval
            and self.global_step % log.save_interval == 0
            and self.global_step != 0
        )

    def _should_validate(self) -> bool:
        """Check if validation should be performed"""
        log = self.cfg.logging
        return (
            log.enable_validation
            and self.global_step >= log.ignore_until_interval
            and self.global_step % log.val_interval == 0
            and self.global_step > 0
        )

    def _checkpoint_config_dict(self) -> dict:
        # dump_config (not asdict): persist the effective config in the same
        # typed yaml schema as hand-written configs.
        config = dump_config(self.cfg)
        effective_data_config = getattr(
            getattr(self, "dataset", None), "effective_data_config", None
        )
        if effective_data_config is None:
            return config

        config.setdefault("data", {})["dataset_config_effective"] = (
            self._dataset_config_params_only(effective_data_config)
        )
        return config

    @staticmethod
    def _dataset_config_params_only(effective_data_config: dict) -> dict:
        data_config = copy.deepcopy(effective_data_config)
        data_config.pop("tasks", None)

        dataset_config = data_config.get("dataset")
        if not isinstance(dataset_config, dict):
            return data_config

        dataset_config.pop("sources", None)
        dataset_config.pop("sources_include", None)

        sampler = dataset_config.get("sampler")
        if not isinstance(sampler, dict):
            return data_config

        task_balance = sampler.get("task_balance")
        if not isinstance(task_balance, dict):
            return data_config

        task_balance.pop("counts_per_task", None)
        task_balance.pop("report_path", None)
        params = task_balance.get("params")
        if isinstance(params, dict):
            params.pop("counts_per_task", None)
            if not params:
                task_balance.pop("params", None)
        return data_config

    @torch.no_grad()
    def val_loop(self):
        """Delegate the full validation run to the adapter.

        Skips silently when the dataset has no val split — adapter's
        run_validation iterates ``val_dataloader`` with ``tqdm(...,
        total=len(...))`` and cannot accept None. v2 returns None here
        when the YAML only declares a train split.
        """
        self.val_dataloader = self.dataset.get_val_dataloader()
        if self.val_dataloader is None:
            if is_main_process():
                self.logger.info("No val split configured, skipping validation.")
            return
        save_path = self.cfg.checkpoint.save_path
        self.adapter.run_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            autocast_context=self.autocast_context,
            reduce_fn=self.reduce_tensor,
            gather_fn=self.true_gather,
            logger=self.wandb_run if is_main_process() else None,
            global_step=self.global_step,
            output_path=os.path.join(save_path, f"val_rank_{self.rank}"),
            tokenizer_mixin=self.tokenizer_mixin,
        )
        barrier()

    def _compute_frozen_prefixes(self) -> Optional[Tuple[str, ...]]:
        """Return state-dict key prefixes for submodules that are fully frozen.

        For models with a ``pipe`` container, any child not present in
        ``cfg.model.trainable_models`` is treated as frozen. The matching key
        prefixes let checkpoint_io drop those entries from the saved file.
        Empty / missing ``trainable_models`` disables filtering and preserves
        "save everything" behavior.
        """
        if not hasattr(self.model, "pipe"):
            return None
        trainable_models = getattr(self.cfg.model, "trainable_models", None)
        if trainable_models is None:
            return None
        if isinstance(trainable_models, str):
            trainable_set = {
                s.strip() for s in trainable_models.split(",") if s.strip()
            }
        else:
            trainable_set = set(trainable_models)
        if not trainable_set:
            self.log(
                "WARNING: trainable_models is empty; not filtering frozen "
                "entries from checkpoint to avoid saving an empty file."
            )
            return None
        frozen_prefixes = tuple(
            f"pipe.{name}."
            for name, _ in self.model.pipe.named_children()
            if name not in trainable_set
        )
        if frozen_prefixes:
            self.log(
                f"Frozen submodule prefixes excluded from checkpoints: "
                f"{list(frozen_prefixes)}"
            )
            return frozen_prefixes
        return None

    def save_checkpoint(self, epoch: int, step: int = 0):
        """Save model checkpoint via checkpoint_io (dispatches on model wrapper type)."""
        save_path = self.cfg.checkpoint.save_path
        ckpt_path = f"{save_path}/{epoch}_{step}" if step else f"{save_path}/{epoch}"
        _ckpt_io.save_checkpoint(
            ckpt_path=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=self._checkpoint_config_dict(),
            rank=self.rank,
            is_main=is_main_process(),
            epoch=epoch,
            global_step=self.global_step,
            seed=self.seed,
            normalizer_action=self.normalizer_action,
            normalizer_propri=self.normalizer_propri,
            dataset=getattr(self, "dataset", None) if step != 0 else None,
            grad_scaler=self.grad_scaler,
            log_fn=self.log,
            frozen_prefixes=self._frozen_prefixes,
        )
        _ckpt_io.finalize_save()
        self.log(f"Saved checkpoint to {ckpt_path}")

    def load_state_dict(self, model, resume_config):
        """Load state dict with fused-weight conversion + try_harder support."""
        return _ckpt_io.load_weights(
            model=model,
            resume_config=resume_config,
            model_class=self._checkpoint_model_class(),
            log_fn=self.log,
        )

    def _checkpoint_model_class(self):
        model_class = getattr(self, "ModelClass", None)
        if model_class is None and hasattr(type(self.adapter), "model_class"):
            model_class = type(self.adapter).model_class()
        return model_class

    def _resume_from_single_file(self) -> bool:
        path = self.cfg.checkpoint.resume_from
        return bool(path) and str(path).endswith((".safetensors", ".pth"))

    def _resume_from_training_checkpoint(self) -> bool:
        path = self.cfg.checkpoint.resume_from
        return bool(path) and not self._resume_from_single_file()

    def resume_from_checkpoint(self):
        """Resume training from checkpoint via checkpoint_io."""
        _ckpt_io.resume_from_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            resume_config={
                "ckpt": self.cfg.checkpoint.resume_from,
                "try_harder": self.cfg.checkpoint.try_harder,
            },
            rank=self.rank,
            grad_scaler=self.grad_scaler,
            model_class=self._checkpoint_model_class(),
            log_fn=self.log,
        )
        barrier()

    def predict_action_loop(
        self,
        current_step=0,
        max_iteration=None,
        prediction_type="flow_action",
        mode="acc",
    ):
        """Delegate action prediction to the adapter."""
        del mode  # unused in current dispatch; kept for signature compatibility
        return self.adapter.predict(
            prediction_type,
            model=self._unwrapped_model_for_inference(),
            val_dataloader=self.dataset.get_val_dataloader(),
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            processor=self.processor,
            tokenizer_mixin=self.tokenizer_mixin,
            logger=self.wandb_run,
            current_step=current_step,
            max_iteration=max_iteration,
        )

    def predict_text_loop(self, current_step=0, max_samples=None, save_dir="./"):
        """Delegate text prediction to the adapter."""
        return self.adapter.predict(
            "text",
            model=self._unwrapped_model_for_inference(),
            val_dataloader=self.dataset.get_val_dataloader(),
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            processor=self.processor,
            tokenizer_mixin=self.tokenizer_mixin,
            logger=self.wandb_run,
            current_step=current_step,
            max_samples=max_samples,
            save_dir=save_dir,
        )

    def _unwrapped_model_for_inference(self):
        """Return the underlying module for inference: unwrap DDP, pass FSDP through."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
