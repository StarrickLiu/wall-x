"""Load checkpoint-side train config and construct runtime config objects.

Inference adapters use this module to load the train config through wall_x's
single inference-side loader (typed schema only; migrate legacy files with
``scripts/migrate_legacy_yml.py``), build the HF model config, and build the
data config required by preprocessing.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Task keys mirrored into the data: section, matching the training-side dict
# shape (``TrainConfig.build_data_loader_dict``). 
_TASK_DATA_SECTION_KEYS = (
    "dof_config",
    "agent_pos_config",
    "action_horizon",
    "action_horizon_flow",
    "ar_dof_config",
    "use_state_string_representation",
)

_QWEN_IMAGE_FACTOR = 28
_QWEN_MIN_PIXELS = 3136
_QWEN_MAX_PIXELS = 12845056


@dataclass
class _LegacyVisionParams:
    """Compatibility projection for images without x2robot_utils >= 0.3.0."""

    resolution: dict
    image_factor: int = _QWEN_IMAGE_FACTOR
    min_pixels: int = _QWEN_MIN_PIXELS
    max_pixels: int = _QWEN_MAX_PIXELS
    use_relative_action: bool = False


def _build_legacy_vision_params(train_config: dict) -> _LegacyVisionParams:
    data = train_config.get("data", {}) or {}
    return _LegacyVisionParams(
        resolution=dict(data.get("resolution", {})),
        image_factor=int(data.get("image_factor", _QWEN_IMAGE_FACTOR)),
        min_pixels=int(data.get("min_pixels", _QWEN_MIN_PIXELS)),
        max_pixels=int(data.get("max_pixels", _QWEN_MAX_PIXELS)),
        use_relative_action=bool(data.get("use_relative_action", False)),
    )


# Virtual (padding) dof keys that must sort AFTER the real LeRobot dims so the
# normalizer's virtual-tail padding lines up. Public checkpoints save
# ``action_padding`` at the head of the layout; inference must move it to the
# tail or ``normalize_data_with_virtual_tail`` falls back to whole-tensor
# normalization and mismatches the real-dim stats.
_VIRTUAL_TAIL_KEYS = frozenset(("action_padding",))


def _move_virtual_keys_to_tail(layout: dict | None) -> dict | None:
    """Keep real LeRobot dims before virtual padding dims in inference layouts."""
    if not isinstance(layout, dict):
        return layout
    head = {k: v for k, v in layout.items() if k not in _VIRTUAL_TAIL_KEYS}
    tail = {k: v for k, v in layout.items() if k in _VIRTUAL_TAIL_KEYS}
    if not tail:
        return layout
    return {**head, **tail}


def _canonicalize_task_layouts(target: dict) -> None:
    """In-place: move virtual padding dims to the tail across every layout view."""
    if os.environ.get("WALLX_CANONICALIZE_VIRTUAL_DOF_ORDER", "1") == "0":
        return
    task = target.get("task")
    data = target.get("data")
    for key in ("dof_config", "agent_pos_config", "ar_dof_config"):
        value = _move_virtual_keys_to_tail(target.get(key))
        if value is not None:
            target[key] = value
        if isinstance(task, dict):
            task_value = _move_virtual_keys_to_tail(task.get(key))
            if task_value is not None:
                task[key] = task_value
        if isinstance(data, dict):
            data_value = _move_virtual_keys_to_tail(data.get(key))
            if data_value is not None:
                data[key] = data_value


def load_train_config_with_ckpt_overlay(
    train_config_path: str,
    checkpoint_path: str,
):
    """Load the checkpoint train config through the single wall_x loader.

    Returns ``(cfg, train_config)``: the typed ``TrainConfig`` (ckpt overlays
    applied; the single source consumers should read) plus its dict view
    shaped like the training-side ``TrainConfig.build_data_loader_dict``
    output -- kept solely for the dict-contract consumers listed above.
    """
    from wall_x.config.loader import dump_config, load_train_config_from_checkpoint

    cfg = load_train_config_from_checkpoint(checkpoint_path, train_config_path)
    train_config = dump_config(cfg)
    task_dict = dataclasses.asdict(cfg.task)
    train_config.update(task_dict)
    data_section = train_config.setdefault("data", {})
    for key in _TASK_DATA_SECTION_KEYS:
        value = task_dict.get(key)
        if value is not None:
            data_section.setdefault(key, value)
    # Public checkpoints save the virtual ``action_padding`` dim at the head of
    # the layout; move it to the tail so the normalizer's virtual-tail masking
    # lines up. Canonicalize BOTH the typed ``cfg.task`` (what the normalizer and
    # preprocessing read) and the dict view (kept for dict-contract callers).
    if os.environ.get("WALLX_CANONICALIZE_VIRTUAL_DOF_ORDER", "1") != "0":
        for _key in ("dof_config", "agent_pos_config", "ar_dof_config"):
            _canon = _move_virtual_keys_to_tail(getattr(cfg.task, _key, None))
            if _canon is not None:
                setattr(cfg.task, _key, _canon)
    _canonicalize_task_layouts(train_config)
    return cfg, train_config


def build_model_config(config_class, checkpoint_path: str, cfg, train_config: dict):
    """Build the HF model config and inject train_config-derived fields.

    ``config_class`` is supplied by the variant adapter. A checkpoint-local
    ``config.json`` wins; otherwise ``cfg.model.config_path`` is used.
    """
    ckpt_config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(ckpt_config_path):
        resolved = ckpt_config_path
    else:
        resolved = getattr(cfg.model, "config_path", None)
        if not resolved or not os.path.exists(resolved):
            raise ValueError(
                f"cannot load model config: checkpoint file {ckpt_config_path} "
                f"does not exist, and fallback cfg.model.config_path={resolved!r} "
                "does not exist either"
            )

    if resolved.endswith(".json"):
        model_config = config_class.from_json_file(resolved)
    else:
        model_config = config_class.from_pretrained(resolved)

    model_config.update_model_config(train_config)
    model_config._attn_implementation = "sdpa"
    model_config.vision_config._attn_implementation = "flash_attention_2"

    return model_config


def build_data_config(cfg, train_config: dict):
    """Build the data config used by image resizing and preprocessing.

    x2robot_v2 / lerobot: ``load_x2r_data_config`` returns None for these
    backends, so image params + resolution come from the train/infer-shared
    projection in x2robot_utils (>= 0.3.0). The ``load_x2r_data_config`` call
    sites are rewritten to ``load_trainer_data_config`` at OSS export time --
    see ``scripts/export_opensource.py::patch_vendored_harrix``.
    """
    if cfg.data.dataset_type in ("x2robot_v2", "lerobot"):
        try:
            from wall_x._vendor.x2robot_utils.vision_params import build_vision_params
        except ModuleNotFoundError as error:
            if error.name != "x2robot_utils.vision_params":
                raise
            logger.warning(
                "x2robot_utils.vision_params is unavailable; using Harrix's "
                "equivalent inference-only vision parameter projection. "
                "Upgrade x2robot_utils to >= 0.3.0 to use the shared source."
            )
            return _build_legacy_vision_params(train_config)

        return build_vision_params(train_config)

    from wall_x.trainer.adapters.base_adapter import load_x2r_data_config

    return load_x2r_data_config(cfg)
