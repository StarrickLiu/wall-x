"""Build action/proprio normalizers and resolve the effective norm key.

Priority:
  1. ``norm_stats.json`` for LeRobot checkpoints.
  2. Checkpoint-side ``normalizer_action.pth`` / ``normalizer_propri.pth``.
  3. An explicit ``customized_action_statistic_dof`` path.

Public inference artifacts must carry their own normalization data. Unlike the
internal build, this module does NOT fall back to a bundled default
``action_statistic_dof`` table (it is not shipped in the OSS package). If the
requested norm key is not present, a prefix match or the only available key is
used as a compatibility fallback.
"""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn as nn

from wall_x.data.backends.lerobot.utils import NormStats
from wall_x.model.core.action.normalizer import Normalizer

logger = logging.getLogger(__name__)


def _load_norm_stats(norm_stats_path: str, action_key: str) -> NormStats:
    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)
    q01 = torch.tensor(norm_stats["norm_stats"][action_key]["q01"])
    q99 = torch.tensor(norm_stats["norm_stats"][action_key]["q99"])
    return NormStats(min=q01, max=q99, delta=q99 - q01)


def _pad_normalizer_to_dim(normalizer, target_dim, label):
    """Tail-pad every dataset entry to ``target_dim`` (virtual ``action_padding`` tail).

    Vendored locally so the public normalizer does not depend on a helper that is
    absent from the OSS ``wall_x.model.core.action.normalizer`` module.
    """
    target_dim = int(target_dim)
    for name in list(normalizer.min.keys()):
        current_dim = int(normalizer.min[name].numel())
        if current_dim == target_dim:
            continue
        if current_dim > target_dim:
            raise ValueError(
                f"Normalizer {label}[{name!r}] dim ({current_dim}) exceeds configured "
                f"{label} dim ({target_dim})"
            )
        pad_dim = target_dim - current_dim
        dtype = normalizer.min[name].dtype
        normalizer.min[name] = nn.Parameter(
            torch.cat(
                [normalizer.min[name].flatten(), torch.zeros(pad_dim, dtype=dtype)]
            ),
            requires_grad=False,
        )
        normalizer.delta[name] = nn.Parameter(
            torch.cat(
                [normalizer.delta[name].flatten(), torch.ones(pad_dim, dtype=dtype)]
            ),
            requires_grad=False,
        )
        logger.info(
            "Padded %s normalizer[%r] from %d to %d dims",
            label,
            name,
            current_dim,
            target_dim,
        )


def _missing_default_stats_error(checkpoint_path: str, custom) -> FileNotFoundError:
    return FileNotFoundError(
        "Public inference requires per-dataset normalization data in the "
        "checkpoint; the internal default action_statistic_dof table is not "
        "shipped in the OSS build. Provide one of: "
        f"{os.path.join(checkpoint_path, 'norm_stats.json')}; checkpoint-side "
        "normalizer_action.pth / normalizer_propri.pth; or an explicit "
        "data.normalizer_config.customized_action_statistic_dof path "
        f"(current customized_action_statistic_dof={custom!r})."
    )


def build_normalizers(
    checkpoint_path: str,
    cfg,
    norm_key: str,
) -> tuple[Normalizer, Normalizer, str]:
    """Return action/proprio normalizers and the resolved norm key.

    ``cfg`` is the typed ``TrainConfig``. The returned key may differ from
    the requested key if fallback resolution was needed.
    """
    # Stats options live under data.normalizer_config; migrated configs may
    # carry data-level copies.
    norm_cfg = getattr(cfg.data, "normalizer_config", None) or {}

    def _norm_opt(key, default=None):
        value = norm_cfg.get(key)
        if value is None:
            value = getattr(cfg.data, key, None)
        return default if value is None else value

    # Custom action statistics are the only supported default source in the OSS
    # build; there is no bundled internal action_statistic_dof fallback.
    custom = _norm_opt("customized_action_statistic_dof")
    if custom:
        with open(custom, "r") as f:
            action_statistic_dof = json.load(f)
    else:
        action_statistic_dof = None

    # LeRobot norm_stats.json takes precedence.
    norm_stats_path = os.path.join(checkpoint_path, "norm_stats.json")
    if os.path.exists(norm_stats_path):
        propri_stats = _load_norm_stats(norm_stats_path, "observation.state")
        action_stats = _load_norm_stats(norm_stats_path, "action")
        normalizer_propri = Normalizer.from_lerobot_norm_stats(propri_stats, norm_key)
        normalizer_action = Normalizer.from_lerobot_norm_stats(action_stats, norm_key)
    else:
        # Checkpoint-side .pth normalizers take precedence over defaults.
        action_pth = os.path.join(checkpoint_path, "normalizer_action.pth")
        propri_pth = os.path.join(checkpoint_path, "normalizer_propri.pth")
        min_key = _norm_opt("min_key", "min")
        delta_key = _norm_opt("delta_key", "delta")
        if os.path.exists(action_pth):
            normalizer_action = Normalizer.from_ckpt(action_pth)
        else:
            if action_statistic_dof is None:
                raise _missing_default_stats_error(checkpoint_path, custom)
            normalizer_action = Normalizer(
                action_statistic_dof,
                cfg.task.dof_config,
                min_key=min_key,
                delta_key=delta_key,
            )
        if os.path.exists(propri_pth):
            normalizer_propri = Normalizer.from_ckpt(propri_pth)
        else:
            if action_statistic_dof is None:
                raise _missing_default_stats_error(checkpoint_path, custom)
            normalizer_propri = Normalizer(
                action_statistic_dof,
                cfg.task.agent_pos_config,
                min_key=min_key,
                delta_key=delta_key,
            )

    # Public checkpoints carry real-dim-only stats (e.g. libero's 7-dim action /
    # 8-dim proprio) while the model tensors include a virtual action_padding
    # tail. Tail-pad the stats to the full configured dim so virtual-tail
    # normalization lines up. Order-independent: only the total dim matters.
    action_dim = sum(cfg.task.dof_config.values())
    propri_dim = sum(cfg.task.agent_pos_config.values())
    _pad_normalizer_to_dim(normalizer_action, action_dim, "action")
    _pad_normalizer_to_dim(normalizer_propri, propri_dim, "propri")

    resolved = _resolve_norm_key(norm_key, normalizer_action, normalizer_propri)
    return normalizer_action, normalizer_propri, resolved


def _resolve_norm_key(
    norm_key: str,
    normalizer_action: Normalizer,
    normalizer_propri: Normalizer,
) -> str:
    """Resolve a requested norm key against normalizer keys.

    For example, ``libero_all`` can resolve to ``libero_all_v3``.
    """
    available = sorted(
        set(normalizer_action.min.keys()) & set(normalizer_propri.min.keys())
    )
    if norm_key in available:
        return norm_key
    if not available:
        return norm_key

    prefix_matches = [k for k in available if k.startswith(f"{norm_key}_")]
    if len(prefix_matches) == 1:
        logger.warning(
            "norm_key=%r not found; using prefix fallback %r",
            norm_key,
            prefix_matches[0],
        )
        return prefix_matches[0]
    if len(available) == 1:
        logger.warning(
            "norm_key=%r not found; using the only available key %r",
            norm_key,
            available[0],
        )
        return available[0]

    logger.warning(
        "norm_key=%r not found; available=%s; returning the requested key unchanged",
        norm_key,
        available,
    )
    return norm_key
