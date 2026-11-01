"""Progress annotations of the RoboDojo memory tasks.

One file per episode, ``progress.json`` beside the episode, listing the frames
at which the task made progress that has to be remembered -- and, where the
task needs it, the object each frame is about and the interval in which writing
that frame still counts as correct. Episodes without a file simply train with
an empty memory.

The annotation is what makes the two passes learnable: the action pass is shown
the keyframes recorded so far, and the selection pass is asked which of the
recent frames are keyframes.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from wall_x.model.core.keyframe_memory import candidate_positions

logger = logging.getLogger(__name__)

FILENAME = "progress.json"

Box = Tuple[int, int, int, int]
Interval = Tuple[int, int]


@dataclass(frozen=True)
class Selection:
    """What one selection sample asks about, and the answer it is trained on.

    ``memory`` and ``candidates`` are frame indices, in prompt order.
    ``targets`` are the candidate slots whose frames are keyframes; for each of
    them, ``boxes`` holds the object box when the task annotates one, and
    ``slot_ranges`` the inclusive slot range that also counts as correct.
    """

    memory: Tuple[int, ...]
    candidates: Tuple[int, ...]
    targets: Tuple[int, ...]
    boxes: Tuple[Optional[Box], ...]
    slot_ranges: Tuple[Interval, ...]

    @property
    def writes(self) -> bool:
        return bool(self.targets)


@dataclass(frozen=True)
class EpisodeAnnotation:
    """The progress keyframes of one episode."""

    frames: Tuple[int, ...]
    boxes: Dict[int, Box]
    windows: Dict[int, Interval]

    @property
    def empty(self) -> bool:
        return not self.frames

    def memory_before(self, frame: int, capacity: int) -> Tuple[int, ...]:
        """Keyframes already recorded at ``frame``, newest ``capacity`` kept."""
        recorded = [f for f in self.frames if f < frame]
        if capacity > 0:
            recorded = recorded[-capacity:]
        return tuple(recorded)

    def selection(self, anchor: int, *, window: int, candidates: int,
                  capacity: int, with_boxes: bool = True) -> Selection:
        """The selection sample at ``anchor``.

        Memory is what was recorded before the candidate window, so a frame is
        never both a memory entry and a candidate. Each keyframe inside the
        window is asked about at its nearest candidate; when two land on the
        same candidate their acceptance ranges are intersected, and the later
        one's box wins.
        """
        start = max(0, anchor - window)
        positions = candidate_positions(anchor, window=window, count=candidates)

        def slot_of(frame: int) -> int:
            return min(range(candidates), key=lambda i: abs(positions[i] - frame))

        ranges: Dict[int, List[int]] = {}
        boxes: Dict[int, Box] = {}
        for frame in self.frames:
            if not start <= frame < anchor:
                continue
            slot = slot_of(frame)
            lo = hi = slot
            interval = self.windows.get(frame)
            if interval is not None:
                inside = [i for i, p in enumerate(positions) if interval[0] <= p <= interval[1]]
                if inside:
                    lo, hi = min(min(inside), slot), max(max(inside), slot)
            prior = ranges.get(slot)
            if prior is None:
                ranges[slot] = [lo, hi]
            else:
                merged = [max(prior[0], lo), min(prior[1], hi)]
                ranges[slot] = merged if merged[0] <= merged[1] else [slot, slot]
            if with_boxes and frame in self.boxes:
                boxes[slot] = self.boxes[frame]  # ascending frames: the later one wins

        targets = tuple(sorted(ranges))
        return Selection(
            memory=self.memory_before(start, capacity),
            candidates=tuple(positions),
            targets=targets,
            boxes=tuple(boxes.get(slot) for slot in targets),
            slot_ranges=tuple((ranges[slot][0], ranges[slot][1]) for slot in targets),
        )


EMPTY = EpisodeAnnotation(frames=(), boxes={}, windows={})


def recorded_frames(root: Optional[str]) -> Dict[str, int]:
    """How long each annotated episode is, as its own file records it.

    Keyed ``"<task>/<episode>"``, which is how the binding names an episode, so
    the caller can check that what it bound describes the episode it bound it
    to. An absent root annotates nothing and needs no check.
    """
    out: Dict[str, int] = {}
    if not root:
        return out
    for path in glob.glob(os.path.join(root, "*", "*", FILENAME)):
        task, episode = path.split(os.sep)[-3:-1]
        try:
            with open(path, encoding="utf-8") as stream:
                total = json.load(stream).get("total_frames")
        except (OSError, json.JSONDecodeError):
            continue
        if total is not None:
            out[f"{task}/{episode}"] = int(total)
    return out


def _parse(record: Dict, where: str) -> Optional[EpisodeAnnotation]:
    frames = record.get("keyframes", [])
    try:
        frames = tuple(sorted({int(f) for f in frames}))
    except (TypeError, ValueError):
        logger.warning("malformed keyframes in %s; training with empty memory", where)
        return None
    boxes: Dict[int, Box] = {}
    windows: Dict[int, Interval] = {}
    for entry in record.get("meta") or []:
        if not isinstance(entry, dict):
            continue
        try:
            frame = int(entry["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        box = entry.get("box")
        if box is not None:
            try:
                x1, y1, x2, y2 = (int(v) for v in box)
            except (TypeError, ValueError):
                logger.warning("malformed box %r in %s; ignoring it", box, where)
            else:
                if x1 < x2 and y1 < y2:
                    boxes[frame] = (x1, y1, x2, y2)
                else:
                    logger.warning("degenerate box %r in %s; ignoring it", box, where)
        interval = entry.get("window")
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            try:
                a, b = int(interval[0]), int(interval[1])
            except (TypeError, ValueError):
                continue
            if a <= b:
                windows[frame] = (a, b)
    return EpisodeAnnotation(frames=frames, boxes=boxes, windows=windows)


def load(episode_dir: str) -> Optional[EpisodeAnnotation]:
    """Read one episode's annotation, or ``None`` when it has none.

    A missing or unreadable file is not an error: those episodes train as plain
    action imitation with an empty memory.
    """
    path = os.path.join(episode_dir, FILENAME)
    try:
        with open(path, encoding="utf-8") as stream:
            record = json.load(stream)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("unreadable %s (%s); training with empty memory", path, exc)
        return None
    return _parse(record, path)


class AnnotationStore:
    """Annotations of a dataset, read once per episode."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root
        self._cache: Dict[str, EpisodeAnnotation] = {}

    def for_episode(self, task: str, episode: str) -> EpisodeAnnotation:
        """The annotation of ``task``/``episode``, empty when there is none."""
        key = f"{task}/{episode}"
        hit = self._cache.get(key)
        if hit is None:
            hit = EMPTY
            if self.root is not None:
                hit = load(os.path.join(self.root, task, episode)) or EMPTY
            self._cache[key] = hit
        return hit


__all__ = [
    "recorded_frames",
    "AnnotationStore",
    "EMPTY",
    "EpisodeAnnotation",
    "Selection",
    "load",
]
