"""The pose vectors of the RoboDojo end-effector export, as the model reads them.

The export gives, per arm, a position, a ``wxyz`` quaternion and a gripper
value -- ``observation.state`` for where the arm is, ``action`` for where it was
told to go. The model reads something else: an orientation as the first two
rows of its rotation matrix, and actions as a motion relative to the observed
pose, expressed in that pose's own frame.

Two details here are not guessable from the data format and were read off the
released checkpoint's own training samples:

- The motion is relative in the *body* frame: the displacement is rotated into
  the observed pose's frame, and the orientation delta is ``R_stateᵀ R_target``,
  not ``R_target R_stateᵀ``.
- Step ``k`` of a chunk targets the command one frame *after* ``k``. Getting
  this wrong leaves most frames right -- the arm is often slow -- and silently
  biases the fast ones.

The gripper is absolute throughout: it is an opening, not a motion.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# One arm of the export: position (3) + quaternion wxyz (4) + gripper (1).
ARM_WIDTH = 8
ARMS = 2
# One arm as the model reads it: position (3) + rotation 6D (6) + gripper (1).
MODEL_ARM_WIDTH = 10


def rotation_from_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """The rotation matrix of a ``wxyz`` quaternion."""
    w, x, y, z = (float(v) for v in quaternion)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("a quaternion of zero length is not a rotation")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    """The 6D encoding of a rotation: its first two rows.

    The third row is recoverable as their cross product, which is what the
    reverse conversion does.
    """
    return np.asarray(rotation)[:2, :].reshape(6)


def split_arm(vector: Sequence[float], arm: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """One arm's ``(position, quaternion, gripper)`` out of an export vector."""
    vector = np.asarray(vector, dtype=np.float64)
    base = arm * ARM_WIDTH
    return vector[base:base + 3], vector[base + 3:base + 7], float(vector[base + 7])


def observed_pose(state: Sequence[float]) -> np.ndarray:
    """The observed pose as the model reads it: ``[pos, rot6d, gripper]`` per arm."""
    blocks = []
    for arm in range(ARMS):
        position, quaternion, gripper = split_arm(state, arm)
        blocks.append(np.concatenate(
            [position, rotation_to_6d(rotation_from_quaternion(quaternion)), [gripper]]))
    return np.concatenate(blocks)


def relative_action(state: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """One action: ``target`` as a motion from ``state``, in ``state``'s own frame.

    Position and orientation are relative; the gripper is the absolute opening
    being commanded.
    """
    blocks = []
    for arm in range(ARMS):
        p_state, q_state, _ = split_arm(state, arm)
        p_target, q_target, gripper = split_arm(target, arm)
        r_state = rotation_from_quaternion(q_state)
        r_target = rotation_from_quaternion(q_target)
        blocks.append(np.concatenate([
            r_state.T @ (p_target - p_state),
            rotation_to_6d(r_state.T @ r_target),
            [gripper],
        ]))
    return np.concatenate(blocks)


def action_chunk(
    state: Sequence[float],
    commands: Sequence[Sequence[float]],
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """``horizon`` actions from the observed pose, and which of them are real.

    ``commands`` are the export's ``action`` rows from the anchor frame onward,
    so ``commands[k + 1]`` is what step ``k`` targets. A chunk running past the
    end of the episode repeats its last action and marks the rest invalid, the
    way a right-padded window does.
    """
    chunk = np.zeros((horizon, ARMS * MODEL_ARM_WIDTH), dtype=np.float64)
    valid = np.zeros(horizon, dtype=bool)
    last = None
    for step in range(horizon):
        index = step + 1
        if index < len(commands):
            last = relative_action(state, commands[index])
            valid[step] = True
        if last is None:
            raise ValueError("an action chunk needs at least one command after the anchor")
        chunk[step] = last
    return chunk, valid


__all__ = [
    "ARMS",
    "ARM_WIDTH",
    "MODEL_ARM_WIDTH",
    "action_chunk",
    "observed_pose",
    "relative_action",
    "rotation_from_quaternion",
    "rotation_to_6d",
    "split_arm",
]
