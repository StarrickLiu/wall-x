"""Assembling the RoboDojo training pipeline from a config.

The pieces exist on their own -- the export reader, the annotation binding, the
sample index, the collator bridge -- and this is the one place that puts them
together the way a config describes. It is also where two things are taken from
the trainer rather than re-derived, because re-deriving them is how the data
and the model come to disagree:

- the **normalization statistics** are read off the model's own normalizers, so
  the numbers the batch is scaled by are the numbers the checkpoint will carry;
- the **processor** is the trainer's, which knows the selection vocabulary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader

from wall_x.data._bundle import DataBundle
# The collator of the backend this one sits inside: RoboDojo episodes are a
# LeRobot export, and normalizing them, sampling the flow time and writing
# the discretized pose into the prompt is the same work for both.
from wall_x.data.backends.lerobot.loader import DataCollator
from wall_x.data.backends.lerobot.robodojo.batches import SameKindOrder
from wall_x.data.backends.lerobot.robodojo.collate import RoboDojoCollator
from wall_x.data.backends.lerobot.robodojo.dataset import RoboDojoDataset
from wall_x.data.backends.lerobot.robodojo.annotations import recorded_frames
from wall_x.data.backends.lerobot.robodojo.episodes import EpisodeIndex
from wall_x.data.backends.lerobot.robodojo.frames import Geometry
from wall_x.data.backends.lerobot.robodojo.export import (
    LeRobotFrameSource,
    episodes_of,
    open_export,
)
from wall_x.data.state_noise import StateNoise

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "RoboDojo-Benchmark/Robodojo"
# The key the statistics are stored under, and the name a batch is normalized
# by. One embodiment, one key.
STATS_KEY = "robodojo"


class _RoboDojoData:
    """What the trainer expects to hold: the loaders and the epoch hook."""

    def __init__(self, dataset: RoboDojoDataset, loader: DataLoader, sampler) -> None:
        self.dataset = dataset
        self._loader = loader
        self._sampler = sampler
        self.effective_data_config = {
            "dataset_type": "robodojo",
            "samples": len(dataset),
            "selection_samples": dataset.selection_samples,
            "geometry": vars(dataset.geometry).copy(),
        }

    def __len__(self) -> int:
        return len(self._loader)

    def get_train_dataloader(self) -> DataLoader:
        return self._loader

    def get_val_dataloader(self) -> None:
        """No validation split: RoboDojo is scored by rollout, not by held-out loss."""
        return None

    def set_epoch(self, epoch: int) -> None:
        self._sampler.set_epoch(epoch)


def _width(layout: Optional[Dict[str, int]], *, skip: Tuple[str, ...] = ()) -> int:
    return sum(int(v) for k, v in (layout or {}).items() if k not in skip)


def _flat_stats(normalizer: Any, width: int, name: str, wanted: Optional[str] = None):
    """The ``(min, delta)`` the collator needs, out of a model-side normalizer.

    A normalizer holds one flat vector per embodiment, laid out in config
    order, so the leading ``width`` entries are the dimensions the dataset
    provides and the rest is the padding the model carries. Slicing here rather
    than letting the collator align the widths keeps the intent visible.

    ``wanted`` names the embodiment. A statistics file written for one dataset
    holds a single key and needs no naming; one that travels with a checkpoint
    trained on many holds several, and then the run has to say which of them it
    is continuing -- picking one would be picking a scale at random.
    """
    if normalizer is None:
        raise ValueError(
            f"the {name} normalizer is missing. The trainer creates it from "
            "data.normalizer_config before the dataset is built; without it a "
            "batch would be scaled by numbers nobody chose.")
    keys = list(normalizer.min.keys())
    if wanted:
        if wanted not in keys:
            raise ValueError(
                f"data.robodojo.stats_key is {wanted!r}, which the {name} "
                f"statistics do not hold; they have {keys}")
        key = wanted
    elif STATS_KEY in keys:
        key = STATS_KEY
    elif len(keys) == 1:
        key = keys[0]
    else:
        raise ValueError(
            f"the {name} statistics hold {keys}; set data.robodojo.stats_key to "
            "the embodiment this run trains")
    minimum = normalizer.min[key].detach()
    delta = normalizer.delta[key].detach()
    if minimum.shape[-1] < width:
        raise ValueError(
            f"the {name} statistics are {minimum.shape[-1]} wide but the dataset "
            f"provides {width} dimensions; the statistics describe other data")
    return minimum[:width].clone(), delta[:width].clone(), key


class _Stats:
    """The two vectors ``DataCollator`` reads off a stats object."""

    def __init__(self, minimum, delta) -> None:
        self.min = minimum
        self.delta = delta


def _geometry(data: Any) -> Geometry:
    memory = dict(getattr(data, "memory", None) or {})
    unknown = set(memory) - {
        "capacity", "window", "candidates", "short_term", "short_term_stride",
        "box_crop", "box_jitter",
    }
    if unknown:
        raise ValueError(f"unknown data.memory keys: {sorted(unknown)}")
    resolution = (data.resolution or {}).get("face_view", 448)
    return Geometry(
        capacity=int(memory.get("capacity", 22)),
        window=int(memory.get("window", 48)),
        candidates=int(memory.get("candidates", 16)),
        short_term=int(memory.get("short_term", 0)),
        short_term_stride=int(memory.get("short_term_stride", 4)),
        resolution=int(resolution),
        factor=int(data.image_factor),
        min_pixels=int(data.min_pixels),
        max_pixels=int(data.max_pixels),
        box_crop=bool(memory.get("box_crop", True)),
        box_jitter=bool(memory.get("box_jitter", True)),
    )


def _collator_config(cfg: Any, ctx: Any, dof_total: int, propri_total: int) -> Dict[str, Any]:
    data = cfg.data
    noise_scheduler = getattr(data, "noise_scheduler", None)
    if not noise_scheduler:
        # The flow time is sampled from the same distribution the model
        # integrates over, so take it from the model rather than defaulting to
        # a second copy of it.
        noise_scheduler = getattr(ctx.model_config, "noise_scheduler", None) or {}
    return {
        "processor_path": cfg.model.processor_path,
        "model_type": cfg.model_type,
        "padding_side": data.padding_side,
        "use_fast_tokenizer": False,
        "use_state_string_representation": bool(
            cfg.task.use_state_string_representation),
        "state_bins": int(data.state_bins),
        "noise_scheduler": dict(noise_scheduler),
        "dof_total_dim": dof_total,
        "agent_pos_total_dim": propri_total,
    }


def build(cfg: Any, ctx: Any) -> DataBundle:
    """The RoboDojo backend's ``build``: config in, loaders out."""
    data = cfg.data
    robodojo = dict(getattr(data, "robodojo", None) or {})
    unknown = set(robodojo) - {
        "repo_id", "root", "annotations", "selection", "stride", "frame_stamp",
        "stats_key",
    }
    if unknown:
        raise ValueError(f"unknown data.robodojo keys: {sorted(unknown)}")
    root = robodojo.get("root")
    if not root:
        raise ValueError(
            "data.robodojo.root must point at the downloaded end-effector export "
            "(the directory holding meta/, data/ and videos/)")

    export = open_export(root, repo_id=robodojo.get("repo_id", DEFAULT_REPO_ID))
    source = LeRobotFrameSource(export)
    episodes = [(e.index, e.instruction, e.length) for e in episodes_of(export.meta)]
    index = EpisodeIndex.from_release(robodojo.get("annotations"), episodes)
    # The guard the annotation release exists to make possible: an episode bound
    # to the wrong annotation recalls the wrong frames, and training proceeds.
    index.check_lengths(recorded_frames(robodojo.get("annotations")))
    logger.info(
        "RoboDojo: %d episodes, %d with an annotation, tasks %s",
        len(episodes), len(index), list(index.annotated_tasks))

    geometry = _geometry(data)
    dataset = RoboDojoDataset(
        source,
        index,
        geometry=geometry,
        action_horizon=int(cfg.task.action_horizon),
        state_bins=int(data.state_bins),
        selection=bool(robodojo.get("selection", True)),
        stride=int(robodojo.get("stride", 1)),
        state_noise=StateNoise.from_config(getattr(data, "state_noise", None)),
        frame_stamp=bool(robodojo.get("frame_stamp", False)),
    )
    if not len(dataset):
        raise ValueError(
            f"no samples came out of {root}: the export has {len(episodes)} episodes")
    # The counts, and what they mean for the two objectives' balance: one
    # update averages each micro-batch's own loss, so it weights the objectives
    # by their share of the samples, where packing them into one sequence --
    # which the published run did -- weights them ar_loss_weight to one.
    selection = int(dataset.selection_samples)
    balance = ""
    if selection and len(dataset) > selection:
        balance = (f"; ar_loss_weight x {(len(dataset) - selection) / selection:.2f} "
                   "weighs them as a packed step would")
    logger.info("RoboDojo: %d samples, %d of them selection samples%s",
                len(dataset), selection, balance)

    # The dataset's own widths; the model's are wider and the collator pads.
    propri_width = _width(cfg.task.agent_pos_config)
    action_width = _width(cfg.task.dof_config, skip=("action_padding",))
    wanted = robodojo.get("stats_key")
    action_min, action_delta, key = _flat_stats(
        ctx.normalizer_action, action_width, "action", wanted)
    propri_min, propri_delta, _ = _flat_stats(
        ctx.normalizer_propri, propri_width, "proprioception", wanted)

    inner = DataCollator(
        _collator_config(cfg, ctx, _width(cfg.task.dof_config), propri_width),
        {"max_length": int(data.max_length)},
        {"action": _Stats(action_min, action_delta),
         "state": _Stats(propri_min, propri_delta)},
        # The collator stamps this on every batch as the embodiment to
        # normalize by, which has to be the key the statistics are held under.
        {"repo_id": key},
    )
    collator = RoboDojoCollator(
        inner,
        processor=ctx.processor,
        propri_width=propri_width,
        action_width=action_width,
        action_horizon=int(cfg.task.action_horizon),
        max_length=int(data.max_length),
        padding_side=data.padding_side,
        candidates=geometry.candidates,
    )

    # Batches hold one kind of sample each, and the sampler also does the
    # per-rank split, so there is no DistributedSampler here.
    sampler = SameKindOrder(
        dataset.kinds,
        batch_size=int(cfg.hyperparams.batch_size_per_gpu),
        rank=ctx.rank,
        world_size=ctx.world_size,
        seed=int(cfg.hyperparams.seed),
    )
    if not sampler.batch_count:
        raise ValueError(
            f"{len(dataset)} samples make no full batch of "
            f"{cfg.hyperparams.batch_size_per_gpu} across {ctx.world_size} ranks; "
            "lower hyperparams.batch_size_per_gpu")
    workers = int(data.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=sampler.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    wrapper = _RoboDojoData(dataset, loader, sampler)
    return DataBundle(
        dataset=wrapper,
        train_loader=loader,
        val_loader=None,
        train_iters=len(loader),
        val_iters=0,
        set_epoch=wrapper.set_epoch,
    )


__all__ = ["build"]
