"""RoboDojo training samples: one episode, one anchor frame, one sample.

Two kinds of sample come out of the same episodes. An *action* sample shows the
keyframes recorded so far and asks for the next chunk of motion. A *selection*
sample shows the recent frames and asks which of them recorded progress worth
remembering. The second is what teaches the model to write its own memory, so
the first can rely on one at inference.

Frames are read through :class:`FrameSource`, which keeps the sample
construction testable without a dataset and lets the same code read either of
the official exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from wall_x.data.backends.lerobot.robodojo import frames as frames_module
from wall_x.data.backends.lerobot.robodojo import poses
from wall_x.data.backends.lerobot.robodojo.annotations import AnnotationStore, EMPTY, EpisodeAnnotation
from wall_x.data.backends.lerobot.robodojo.episodes import EpisodeIndex
from wall_x.data.backends.lerobot.robodojo.frames import Geometry
from wall_x.data.backends.lerobot.robodojo.prompts import action_prompt
from wall_x.data.state_noise import StateNoise, perturb_state
from wall_x.model.core.keyframe_memory import normalize_box, selection_prompt

ACTION = "action"
SELECTION = "selection"


@dataclass(frozen=True)
class Episode:
    """One episode of the dataset, as the sample index needs to see it."""

    index: int
    instruction: str
    length: int


class FrameSource(Protocol):
    """What a RoboDojo sample needs to read out of a dataset."""

    def episodes(self) -> Sequence[Episode]:
        ...

    def head_image(self, episode: int, frame: int) -> Any:
        """The head camera at one frame -- what a recalled frame is."""

    def camera_images(self, episode: int, frame: int) -> Sequence[Any]:
        """Every camera at one frame, in the order the prompt names them."""

    def state(self, episode: int, frame: int) -> Sequence[float]:
        """The observed pose at one frame."""

    def commands(self, episode: int, frame: int, count: int) -> Sequence[Sequence[float]]:
        """The commanded poses from ``frame`` on, at most ``count`` of them."""


@dataclass(frozen=True)
class Sample:
    """Where one sample comes from."""

    kind: str
    episode: int
    anchor: int


class RoboDojoDataset:
    """Action and selection samples over a RoboDojo dataset.

    ``selection`` turns the selection samples on; they need annotations, so an
    unannotated dataset yields action samples only. ``stride`` thins the
    anchors, which is useful for a smoke run and changes what the model sees,
    so it is not something to set casually.
    """

    def __init__(
        self,
        source: FrameSource,
        annotations: EpisodeIndex,
        *,
        geometry: Optional[Geometry] = None,
        action_horizon: int = 32,
        state_bins: int = 255,
        camera_labels: Sequence[str] = ("front view", "left wrist view", "right wrist view"),
        instruction_label: str = "Instruction",
        selection: bool = True,
        stride: int = 1,
        state_noise: Optional[StateNoise] = None,
        frame_stamp: bool = False,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be at least 1")
        self.source = source
        self.annotations = annotations
        self.geometry = geometry or Geometry()
        self.action_horizon = action_horizon
        self.state_bins = state_bins
        self.camera_labels = tuple(camera_labels)
        self.instruction_label = instruction_label
        self.state_noise = state_noise or StateNoise()
        self.frame_stamp = frame_stamp
        self._store = AnnotationStore(annotations.root)
        self._episodes = {episode.index: episode for episode in source.episodes()}
        self._samples: List[Sample] = []
        for episode in source.episodes():
            # The last frame has no command after it, so there is no motion to
            # ask for; a selection sample needs no future and can use it.
            self._samples += [Sample(ACTION, episode.index, a)
                              for a in range(0, max(0, episode.length - 1), stride)]
            if selection and annotations.binding(episode.index) is not None:
                self._samples += [Sample(SELECTION, episode.index, a)
                                  for a in range(0, episode.length, stride)]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def selection_samples(self) -> int:
        return sum(1 for sample in self._samples if sample.kind == SELECTION)

    @property
    def kinds(self) -> Tuple[str, ...]:
        """Each sample's kind, in index order, for batching by kind."""
        return tuple(sample.kind for sample in self._samples)

    def annotation(self, episode_index: int) -> EpisodeAnnotation:
        """The annotation bound to an episode, empty when it has none."""
        binding = self.annotations.binding(episode_index)
        if binding is None:
            return EMPTY
        return self._store.for_episode(binding.task, binding.episode)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        if sample.kind == ACTION:
            return self._action_sample(sample, index)
        return self._selection_sample(sample, index)

    # ------------------------------------------------------------------
    # Samples
    # ------------------------------------------------------------------
    def _read_head(self, episode: int, frames: Sequence[int]) -> Dict[int, Any]:
        # One read per distinct frame: a recalled frame is often recalled by
        # many anchors in a row.
        return {frame: self.source.head_image(episode, frame) for frame in set(frames)}

    def _action_sample(self, sample: Sample, index: int) -> Dict[str, Any]:
        annotation = self.annotation(sample.episode)
        request = frames_module.action_request(annotation, sample.anchor, self.geometry)
        images = frames_module.prepare_images(
            request,
            self._read_head(sample.episode, request.recalled),
            self.source.camera_images(sample.episode, request.current),
            self.geometry,
            raw_size=self._raw_size(sample.episode, request.current),
            rng=np.random.default_rng(index),
        )
        state = np.asarray(
            self.source.state(sample.episode, sample.anchor), dtype=np.float64)
        commands = self.source.commands(
            sample.episode, sample.anchor, self.action_horizon + 1)
        chunk, valid = poses.action_chunk(state, commands, self.action_horizon)
        pose = self._perturb_pose(poses.observed_pose(state), index)
        prefix, postfix = action_prompt(
            self._episodes[sample.episode].instruction,
            self.camera_labels,
            action_horizon=self.action_horizon,
            num_memory=len(request.recalled),
            frame=sample.anchor if self.frame_stamp else None,
            instruction_label=self.instruction_label,
        )
        # An invalid step is padding, and padding is NaN: the collator turns
        # NaN into zero and records a mask from it, which is how a dimension or
        # a step the dataset cannot provide is excluded from the loss.
        chunk = np.where(valid[:, None], chunk, np.nan)
        return {
            "kind": ACTION,
            "episode": sample.episode,
            # A task is identified by its instruction: that is what the export
            # carries for every episode, annotated or not, and what the
            # annotations are bound by.
            "task": self._episodes[sample.episode].instruction,
            "frame": sample.anchor,
            "frame_index": sample.anchor,
            "images": images,
            "text": prefix + postfix,
            "agent_pos": pose,
            "action": chunk,
            "action_valid_mask": valid,
            "frames_read": request.all_frames,
        }

    def _selection_sample(self, sample: Sample, index: int) -> Dict[str, Any]:
        annotation = self.annotation(sample.episode)
        episode = self._episodes[sample.episode]
        request = frames_module.selection_request(
            annotation, sample.anchor, self.geometry, last_frame=episode.length - 1)
        selection = annotation.selection(
            sample.anchor, window=self.geometry.window,
            candidates=self.geometry.candidates, capacity=self.geometry.capacity)
        images = frames_module.prepare_images(
            request,
            self._read_head(sample.episode, request.head_frames),
            (),
            self.geometry,
            raw_size=self._raw_size(sample.episode, request.current),
            rng=np.random.default_rng(index),
        )
        prompt = selection_prompt(
            episode.instruction, len(request.recalled), len(request.candidates))
        return {
            "kind": SELECTION,
            "episode": sample.episode,
            "task": episode.instruction,
            "frame": sample.anchor,
            "frame_index": sample.anchor,
            "images": images,
            "text": prompt + self._answer(selection, sample.episode, request)
            + "<|im_end|>\n",
            "slot_ranges": [list(pair) for pair in selection.slot_ranges],
            "frames_read": request.all_frames,
        }

    def _answer(self, selection, episode: int, request) -> str:
        """The answer a selection sample is trained on."""
        if not selection.writes:
            return "<|kf_skip|>"
        width, height = self._raw_size(episode, request.current)
        pieces = ["<|kf_write|>"]
        for slot, box in zip(selection.targets, selection.boxes):
            pieces.append(f"<|R_{slot}|>")
            if box is not None:
                x1, y1, x2, y2 = normalize_box(box, width, height)
                pieces.append(f" ({x1},{y1}),({x2},{y2})")
        pieces.append("<|kf_end|>")
        return "".join(pieces)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _raw_size(self, episode: int, frame: int) -> Tuple[int, int]:
        """``(width, height)`` of the dataset's own frames, which boxes refer to.

        Note a numpy array also has ``size``, as a count of elements, so the
        image kind is tested rather than duck-typed.
        """
        from PIL import Image

        image = self.source.head_image(episode, frame)
        if isinstance(image, Image.Image):
            return image.size
        array = np.asarray(image)
        return int(array.shape[1]), int(array.shape[0])

    # The pose as the model reads it, which is the layout the perturbation
    # understands: the export's quaternions have already become 6D here.
    POSE_LAYOUT = {
        "left_ee_cartesian_pos": 3, "left_ee_rotation_6D": 6, "left_gripper": 1,
        "right_ee_cartesian_pos": 3, "right_ee_rotation_6D": 6, "right_gripper": 1,
    }

    def _perturb_pose(self, pose: np.ndarray, index: int) -> np.ndarray:
        """The observed pose, perturbed when training asks for it.

        Only what the model is shown moves; the actions stay relative to the
        true pose, which is what teaches it to read the pose as approximate.
        The generator is seeded from the sample index, so a run is reproducible.
        """
        return perturb_state(
            pose, self.POSE_LAYOUT, self.state_noise, np.random.default_rng(index))


__all__ = ["ACTION", "SELECTION", "Episode", "FrameSource", "RoboDojoDataset", "Sample"]
