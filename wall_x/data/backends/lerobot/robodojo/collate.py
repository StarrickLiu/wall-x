"""Handing RoboDojo samples to the collator the LeRobot path already uses.

That collator already does what a RoboDojo batch needs: it normalizes the pose
and the action chunk, samples the flow time, writes the discretized pose into
the prompt, and turns NaN into zero while recording a mask from it. What it
does not know is this dataset's two kinds of sample, so this translates them
into the keys it reads.

A selection sample has no action to predict. It arrives with an all-NaN pose
and chunk, which the collator reads as "nothing provided": the mask is zero
everywhere and no flow loss can come from it. Its prompt contains no action and
no pose placeholder either, so the action branch sees none of its tokens -- the
sample trains the selector through the autoregressive loss on its answer, and
nothing else.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from wall_x.data.backends.lerobot.robodojo.dataset import ACTION, SELECTION
from wall_x.model.core.selection_loss import (
    NO_TOKEN,
    accepted_token_ids,
    answer_slots,
)

logger = logging.getLogger(__name__)


def collator_item(
    sample: Dict[str, Any],
    *,
    propri_width: int,
    action_width: int,
    action_horizon: int,
) -> Dict[str, Any]:
    """One sample in the keys the collator reads.

    Widths are the dataset's, not the model's: the collator pads a narrower
    vector out to the model's width and masks what it added, which is how the
    dimensions this embodiment does not have stay out of the loss.
    """
    kind = sample.get("kind")
    if kind == ACTION:
        pose = np.asarray(sample["agent_pos"], dtype=np.float32)
        chunk = np.asarray(sample["action"], dtype=np.float32)
        if pose.shape != (propri_width,):
            raise ValueError(
                f"an action sample's pose is {pose.shape}, expected ({propri_width},): "
                "agent_pos_config and the dataset's layout disagree")
        if chunk.shape != (action_horizon, action_width):
            raise ValueError(
                f"an action sample's chunk is {chunk.shape}, expected "
                f"({action_horizon}, {action_width}): action_horizon or dof_config "
                "and the dataset's layout disagree")
    elif kind == SELECTION:
        pose = np.full(propri_width, np.nan, dtype=np.float32)
        chunk = np.full((action_horizon, action_width), np.nan, dtype=np.float32)
    else:
        raise ValueError(f"unknown sample kind {kind!r}")

    return {
        "image_inputs": sample["images"],
        "text": sample["text"],
        "agent_pos": torch.from_numpy(pose),
        "action": torch.from_numpy(chunk),
        "frame_index": torch.tensor(int(sample["frame_index"]), dtype=torch.long),
    }


class RoboDojoCollator:
    """The LeRobot collator, fed RoboDojo samples.

    ``inner`` is a ``DataCollator``. ``processor`` replaces the one it would
    load for itself: the selection answer is written with tokens that exist
    only after the model's vocabulary was extended (``<|kf_write|>``,
    ``<|R_0|>``, ...). The trainer's processor knows them; a freshly loaded one
    would split them into ordinary text and the answer would never be learned.
    """

    def __init__(
        self,
        inner: Any,
        *,
        processor: Any,
        propri_width: int,
        action_width: int,
        action_horizon: int,
        max_length: int,
        padding_side: str = "right",
        candidates: int = 0,
    ) -> None:
        self.inner = inner
        self.candidates = int(candidates)
        self.propri_width = int(propri_width)
        self.action_width = int(action_width)
        self.action_horizon = int(action_horizon)
        self.max_length = int(max_length)
        self._write_id = None
        self._slot_ids = ()
        if processor is not None:
            self.inner.processor = processor
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.padding_side = padding_side
                # The selector's own vocabulary, needed to find the answer's
                # slots among the candidates the prompt also names.
                self._write_id = tokenizer.convert_tokens_to_ids("<|kf_write|>")
                self._slot_ids = tuple(
                    tokenizer.convert_tokens_to_ids(f"<|R_{i}|>")
                    for i in range(self.candidates))

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = [
            collator_item(
                sample,
                propri_width=self.propri_width,
                action_width=self.action_width,
                action_horizon=self.action_horizon,
            )
            for sample in batch
        ]
        inputs = self.inner(items)
        self._refuse_truncation(inputs)
        self._mark_accepted_intervals(inputs, batch)
        # Which task each sample came from. The refiner is scored against
        # per-task margins, so a batch that cannot say this can only be
        # trained unfiltered -- which is what the margins exist to prevent.
        inputs["task_names"] = [sample.get("task") for sample in batch]
        if all(sample.get("kind") == SELECTION for sample in batch):
            # Nothing in this batch has an action to predict, and the model
            # pairs the action rows it is given with the action-token positions
            # in the prompts. There are none, so the rows have to go too --
            # otherwise the model refuses the batch. The loss it computes is
            # then the autoregressive one alone, which is the point of a
            # selection sample.
            for key in ("action_chunk", "dof_mask", "sample_time"):
                inputs.pop(key, None)
        return inputs

    def _mark_accepted_intervals(self, inputs: Any, batch: Sequence[Dict[str, Any]]) -> None:
        """Say which answer tokens are scored against a set rather than one id.

        The annotation gives each written keyframe an interval of candidates
        that count, so the loss at those positions is the probability on the
        whole interval. Marking them here keeps the model's loss free of
        anything RoboDojo-specific: it is handed positions and token ids.
        """
        if not self._slot_ids or self._write_id is None:
            return
        rows, accepted = [], []
        marks = torch.full_like(inputs["input_ids"], NO_TOKEN)
        for index, sample in enumerate(batch):
            ranges = sample.get("slot_ranges")
            if not ranges:
                continue
            positions = answer_slots(
                inputs["input_ids"][index].tolist(), self._write_id, self._slot_ids)
            if len(positions) != len(ranges):
                raise ValueError(
                    f"the answer writes {len(positions)} slots but the annotation "
                    f"gives {len(ranges)} accepted intervals; they must describe "
                    "the same answer")
            ids = accepted_token_ids(self._slot_ids, [tuple(r) for r in ranges])
            for position, row in zip(positions, range(len(ranges))):
                marks[index, position] = len(rows) + row
            rows.extend(range(len(ranges)))
            accepted.append(ids)
        if not accepted:
            return
        width = max(a.shape[1] for a in accepted)
        padded = [torch.nn.functional.pad(a, (0, width - a.shape[1]), value=NO_TOKEN)
                  for a in accepted]
        inputs["interval_rows"] = marks
        inputs["interval_accepted"] = torch.cat(padded, dim=0)

    def _refuse_truncation(self, inputs: Any) -> None:
        """A truncated sample trains on less than it was built to teach.

        The answer is at the end of a selection prompt, so truncation removes
        exactly the thing being learned while training carries on and the loss
        keeps falling. ``max_length`` is generous enough that reaching it means
        the config is wrong, not that one episode was unusual.
        """
        tokens = inputs["input_ids"].shape[1]
        if tokens >= self.max_length:
            raise ValueError(
                f"a batch reached the {self.max_length}-token limit ({tokens} tokens), "
                "so it was truncated and the end of the prompt -- a selection "
                "answer, or the action tokens -- is gone. Raise data.max_length, or "
                "lower data.memory.capacity / data.memory.candidates.")


__all__ = ["RoboDojoCollator", "collator_item"]
