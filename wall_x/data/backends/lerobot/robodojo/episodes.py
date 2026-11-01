"""Binding the progress annotations to the episodes of a RoboDojo dataset.

The annotations are published per task, as ``<task>/trajectory_<n>``; a LeRobot
dataset identifies an episode by an index and the instruction of its task. The
index file of the annotation release maps the two, and the binding is checked
against episode lengths, because the failure this guards against is silent: an
episode bound to the wrong annotation trains the model to recall the wrong
frames, and an episode bound to nothing trains it with an empty memory. Both
look like ordinary training.

Most tasks have no annotation at all -- only the memory tasks do -- so "no
annotation" is a normal answer, but "the annotation root is configured and
nothing matched" is an error.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
EPISODE_PREFIX = "trajectory_"
# What the fields of a ``progress.json`` mean -- in particular that ``window``
# is the interval in which writing that keyframe still counts as correct. The
# release states it once here rather than once per episode file, and a release
# that states something else is refused: the same field has carried another
# meaning before, and reading it as an acceptance interval would train the
# selector against intervals nobody annotated.
SCHEMA = "robodojo_progress_annotations_v1"


@dataclass(frozen=True)
class Task:
    """An annotated task: its directory, the instruction identifying it, its size."""

    task: str
    instruction: str
    episodes: int


@dataclass(frozen=True)
class Binding:
    """Where one episode's annotation lives."""

    task: str
    episode: str
    frames: int


class EpisodeIndex:
    """Which annotation belongs to which episode of a dataset.

    ``episodes`` is the dataset's episode list: ``(episode_index, instruction,
    length)`` per episode, as a LeRobot dataset's metadata provides it. An
    annotated task's episodes are a contiguous block, so the position inside
    that block names the annotation directory.
    """

    def __init__(
        self,
        tasks: Sequence[Task],
        episodes: Iterable[Tuple[int, str, int]],
        root: Optional[str] = None,
    ) -> None:
        self.root = root
        by_instruction = {task.instruction: task for task in tasks}
        first: Dict[str, int] = {}
        counted: Dict[str, int] = {}
        for index, instruction, _ in episodes:
            task = by_instruction.get(instruction)
            if task is None:
                continue
            first[task.task] = min(first.get(task.task, index), index)
            counted[task.task] = counted.get(task.task, 0) + 1
        self._tasks = by_instruction
        self._first = first
        self._bindings: Dict[int, Binding] = {}

        missing = [task.task for task in tasks if task.task not in first]
        if tasks and not first:
            raise ValueError(
                "no episode of this dataset matches any annotated task. The "
                "annotations were published for a different dataset, or its "
                "task instructions have changed since: "
                f"{[task.instruction for task in tasks][:2]}"
            )
        if missing:
            logger.warning(
                "annotated tasks absent from this dataset: %s", sorted(missing))
        for task in tasks:
            seen = counted.get(task.task)
            if seen is not None and seen != task.episodes:
                raise ValueError(
                    f"task {task.task!r} is published with {task.episodes} annotated "
                    f"episodes but this dataset has {seen}; the annotations do not "
                    "belong to this dataset")

        for index, instruction, length in episodes:
            task = by_instruction.get(instruction)
            if task is None:
                continue
            number = index - first[task.task]
            self._bindings[index] = Binding(
                task=task.task, episode=f"{EPISODE_PREFIX}{number}", frames=length)

    @classmethod
    def from_release(
        cls,
        root: Optional[str],
        episodes: Iterable[Tuple[int, str, int]],
    ) -> "EpisodeIndex":
        """Read the annotation release at ``root``; no root means no annotations."""
        if not root:
            return cls(tasks=(), episodes=episodes, root=None)
        path = os.path.join(root, INDEX_FILENAME)
        try:
            with open(path, encoding="utf-8") as stream:
                record = json.load(stream)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{path} is missing: point data.robodojo.annotations at the "
                "directory of the annotation release, which carries it") from exc
        schema = record.get("schema")
        if schema != SCHEMA:
            raise ValueError(
                f"{path} declares schema {schema!r}; this reads {SCHEMA!r}")
        tasks = tuple(
            Task(task=str(entry["task"]), instruction=str(entry["instruction"]),
                 episodes=int(entry["episodes"]))
            for entry in record.get("tasks", ())
        )
        return cls(tasks=tasks, episodes=episodes, root=root)

    def __len__(self) -> int:
        return len(self._bindings)

    @property
    def annotated_tasks(self) -> Tuple[str, ...]:
        return tuple(sorted(self._first))

    def binding(self, episode_index: int) -> Optional[Binding]:
        """Where episode ``episode_index``'s annotation lives, if it has one."""
        return self._bindings.get(episode_index)

    def directory(self, episode_index: int) -> Optional[str]:
        """The annotation directory of an episode, if it has one."""
        found = self._bindings.get(episode_index)
        if found is None or self.root is None:
            return None
        return os.path.join(self.root, found.task, found.episode)

    def check_lengths(self, frames_of: Mapping[str, int]) -> None:
        """Fail if a bound annotation describes an episode of another length.

        ``frames_of`` maps ``"<task>/<episode>"`` to the frame count the
        annotation records. A mismatch means the episode numbering of the
        annotations and of this dataset have drifted apart, which would recall
        frames from the wrong place in the episode.
        """
        wrong = []
        for index, binding in sorted(self._bindings.items()):
            recorded = frames_of.get(f"{binding.task}/{binding.episode}")
            if recorded is not None and recorded != binding.frames:
                wrong.append((index, binding.task, binding.episode, recorded, binding.frames))
        if wrong:
            first = wrong[0]
            raise ValueError(
                f"{len(wrong)} episodes are bound to an annotation of a different "
                f"length, e.g. episode {first[0]} -> {first[1]}/{first[2]}: the "
                f"annotation describes {first[3]} frames, the dataset has {first[4]}")


__all__ = ["Binding", "EpisodeIndex", "Task"]
