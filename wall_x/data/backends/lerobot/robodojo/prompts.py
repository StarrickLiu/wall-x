"""The prompt of one RoboDojo training sample, as the model reads it.

The LeRobot path writes a prompt with two placeholders and fills them when
samples are collated: ``<|propri|>`` becomes the discretized pose and
``<|action_fast|><|im_end|>`` is dropped when there is no autoregressive action
to supervise. Neither placeholder ever reaches the model. This builds the
filled form directly, so there is one string to compare against what the
released checkpoint was trained on.

Memory is inserted by ``wall_x.model.core.keyframe_memory.prompts``, the same
code serving uses, so a recalled frame is described identically in both.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from wall_x.model.core.keyframe_memory import inject_memory

IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
ACTION = "<|action|>"
# The collator fills this from the pose it normalized.
PROPRIO = "<|propri|>"
SYSTEM_PROMPT = "You are a helpful assistant."
MEMORY_LABEL = "progress memory"


def state_string(pose: Sequence[float], bins: int) -> str:
    """The pose as bin indices, which is how it enters the prompt.

    ``pose`` is normalized to ``[-1, 1]``. Each dimension becomes the index of
    the bin it falls in, so the model reads the pose as text rather than as a
    vector. The lower edge of a bin belongs to it -- the boundary case here is
    exactly ``1.0``, which the clip keeps in the last bin instead of letting it
    spill into one that does not exist.
    """
    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    indices = np.clip(np.floor((values + 1.0) / 2.0 * bins), 0, bins - 1)
    return " ".join(str(int(i)) for i in indices)


def action_prompt(
    instruction: str,
    camera_labels: Sequence[str],
    *,
    action_horizon: int,
    pose: Optional[Sequence[float]] = None,
    state_bins: int = 255,
    num_memory: int = 0,
    num_short_term: int = 0,
    frame: Optional[int] = None,
    instruction_label: str = "Instruction",
    system_prompt: str = SYSTEM_PROMPT,
    memory_label: str = MEMORY_LABEL,
) -> tuple:
    """The prompt and the action slots of one action sample.

    Returns ``(prefix, postfix)``: the text up to where the action chunk
    begins, and the chunk's slots. Recalled frames are described before the
    cameras and their images are prepended in the same order.

    The pose is left as ``<|propri|>`` unless one is given. The collator fills
    it from the pose it normalized, so the normalization and the text it is
    written into cannot disagree; passing a ``pose`` here writes the filled
    form instead, which is what the parity checks compare.
    """
    observation = "".join(f" {label}: {IMAGE}" for label in camera_labels)
    written = PROPRIO if pose is None else state_string(pose, state_bins)
    prefix = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nObservation:{observation}\n"
        f"{instruction_label}: {instruction}\n"
        "Predict the next action in robot action.\n"
        f"Proprioception: {written}\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    prefix = inject_memory(
        prefix,
        num_memory=num_memory,
        num_short_term=num_short_term,
        frame=frame,
        label=memory_label,
    )
    return prefix, ACTION * action_horizon


__all__ = ["ACTION", "IMAGE", "MEMORY_LABEL", "PROPRIO", "SYSTEM_PROMPT",
           "action_prompt", "state_string"]
