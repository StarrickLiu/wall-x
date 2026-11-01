"""Reading RoboDojo frames out of a LeRobot dataset.

RoboDojo is published in several forms; the end-effector LeRobot export carries
exactly what this model reads -- a position, a ``wxyz`` quaternion and a gripper
value per arm -- so it is read directly.

One thing to know about cost. A recalled frame is an arbitrary earlier frame,
so reading it is a seek into a compressed video rather than a sequential read.
A sample recalling twenty frames pays twenty seeks. That is inherent to asking
for distant frames, not to this code; a dataset stored all-intra pays far less
for it.

The export nests inside a larger repository and the repository carries no
version tag, so it cannot be opened by name: download the export's directory
and point ``root`` at it, which is what the official download script produces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wall_x.data.backends.lerobot.robodojo.dataset import Episode

logger = logging.getLogger(__name__)

# What the export calls its cameras, head first: a recalled frame is a frame of
# the scene, so the head camera is the one that carries it.
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _instruction(value: Any) -> str:
    """The task string of an episode row, however the table happens to store it.

    An episode carries a list of tasks; these datasets give one task per
    episode, and the table may hand it back as a Python list, as a string, or
    as an Arrow scalar wrapping either.
    """
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        return value
    if hasattr(value, "__len__"):
        if not len(value):
            raise ValueError("an episode of this dataset carries no task string")
        return str(value[0])
    return str(value)


def episodes_of(meta: Any) -> Tuple[Episode, ...]:
    """The episodes of a dataset's metadata, in the order they were published.

    Takes a ``LeRobotDatasetMetadata``, so the episode list can be read without
    opening the videos -- which is what binding the annotations needs.
    """
    table = meta.episodes
    return tuple(
        Episode(index=int(table["episode_index"][row]),
                instruction=_instruction(table["tasks"][row]),
                length=int(table["length"][row]))
        for row in range(len(table))
    )


def _to_hwc_uint8(frame: Any) -> np.ndarray:
    """A LeRobot image as an ``HxWx3`` uint8 array.

    The dataset hands back a float tensor in ``[0, 1]`` with channels first.
    """
    array = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[0] < array.shape[-1]:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return array


class LeRobotFrameSource:
    """A :class:`~wall_x.data.backends.lerobot.robodojo.dataset.FrameSource` over LeRobot.

    ``dataset`` is an opened ``LeRobotDataset``. Rows are addressed by the
    episode's first row plus the frame, which the dataset's own episode table
    records, so this holds whatever order the episodes were written in.
    """

    def __init__(self, dataset: Any, camera_keys: Sequence[str] = CAMERA_KEYS) -> None:
        self.dataset = dataset
        self.camera_keys = tuple(camera_keys)
        table = dataset.meta.episodes
        missing = [key for key in self.camera_keys if key not in dataset.meta.features]
        if missing:
            raise KeyError(
                f"this dataset has no {missing}; its cameras are "
                f"{sorted(k for k in dataset.meta.features if k.startswith('observation.images'))}")
        for key in (STATE_KEY, ACTION_KEY):
            if key not in dataset.meta.features:
                raise KeyError(f"this dataset has no {key!r}")
        self._episodes: List[Episode] = []
        self._first_row: Dict[int, int] = {}
        for row in range(len(table)):
            entry = {name: table[name][row] for name in table.column_names}
            index = int(entry["episode_index"])
            instruction = _instruction(entry["tasks"])
            length = int(entry["length"])
            start = int(entry["dataset_from_index"])
            end = int(entry.get("dataset_to_index", start + length))
            if end - start != length:
                raise ValueError(
                    f"episode {index} spans rows {start}..{end} but reports {length} "
                    "frames; the dataset's episode table is inconsistent")
            self._episodes.append(Episode(index, instruction, length))
            self._first_row[index] = start
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._cache_order: List[int] = []

    # ------------------------------------------------------------------
    def episodes(self) -> Sequence[Episode]:
        return tuple(self._episodes)

    def row(self, episode: int, frame: int) -> int:
        """The dataset row of one frame of one episode."""
        try:
            start = self._first_row[episode]
        except KeyError:
            raise KeyError(f"this dataset has no episode {episode}") from None
        length = next(e.length for e in self._episodes if e.index == episode)
        if not 0 <= frame < length:
            raise IndexError(
                f"episode {episode} has {length} frames; frame {frame} is outside it")
        return start + frame

    def _read(self, episode: int, frame: int) -> Dict[str, Any]:
        row = self.row(episode, frame)
        hit = self._cache.get(row)
        if hit is not None:
            return hit
        item = self.dataset[row]
        # A small cache: an action sample reads the anchor for its cameras and
        # again for its state, and consecutive anchors share recalled frames.
        self._cache[row] = item
        self._cache_order.append(row)
        if len(self._cache_order) > 64:
            del self._cache[self._cache_order.pop(0)]
        return item

    def head_image(self, episode: int, frame: int) -> np.ndarray:
        return _to_hwc_uint8(self._read(episode, frame)[self.camera_keys[0]])

    def camera_images(self, episode: int, frame: int) -> List[np.ndarray]:
        item = self._read(episode, frame)
        return [_to_hwc_uint8(item[key]) for key in self.camera_keys]

    def state(self, episode: int, frame: int) -> np.ndarray:
        item = self._read(episode, frame)[STATE_KEY]
        return np.asarray(item.numpy() if hasattr(item, "numpy") else item, dtype=np.float64)

    def commands(self, episode: int, frame: int, count: int) -> List[np.ndarray]:
        """The commanded poses from ``frame`` on, stopping at the episode's end."""
        length = next(e.length for e in self._episodes if e.index == episode)
        out = []
        for offset in range(count):
            if frame + offset >= length:
                break
            item = self._read(episode, frame + offset)[ACTION_KEY]
            out.append(np.asarray(
                item.numpy() if hasattr(item, "numpy") else item, dtype=np.float64))
        return out


def open_export(root: str, repo_id: str = "RoboDojo-Benchmark/Robodojo", **kwargs):
    """Open a downloaded RoboDojo LeRobot export.

    ``root`` is the directory holding ``meta/``, ``data/`` and ``videos/``.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=root, **kwargs)


__all__ = ["ACTION_KEY", "CAMERA_KEYS", "STATE_KEY", "LeRobotFrameSource",
           "episodes_of", "open_export"]
