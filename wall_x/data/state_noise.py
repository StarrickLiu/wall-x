"""Random perturbation of the observed pose, for training.

The actions this model predicts are relative to the pose it is shown, so the
pose is a strong input and the model can come to depend on its exact numbers --
including on the normalization statistics it was trained with. Perturbing the
pose during training teaches it to read the pose as approximate, which is what
you want from a controller whose state estimate is never exact, and makes the
particular statistics behind the normalizer much less load-bearing.

The perturbation is physical, applied to the unnormalized pose: metres of
position, radians of orientation, gripper units. It therefore has one meaning
regardless of the normalizer, and the discretized ``<|propri|>`` prompt slot
follows automatically because it is rendered from the normalized pose.

Off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from wall_x._vendor.x2robot_utils.geometry import (
    canonicalize_euler_zyx_batch_nb,
    euler_to_matrix_zyx_batch_nb,
    matrix_to_euler_zyx_batch_nb,
    so3_to_matrix_batch_nb,
)

# A layout entry is perturbed according to what it holds. Anything else -- the
# padding dimensions a dataset does not provide, base height, head angles -- is
# left exactly as it was, so masked dimensions stay at their padding value.
ROTATION_MARK = "rotation"
GRIPPER_MARK = "gripper"
POSITION_MARKS = ("cartesian_pos", "_pos", "_position")


def kind_of(name: str) -> Optional[str]:
    """What a layout entry holds: ``position``, ``rotation``, ``gripper`` or None.

    Matching is deliberately narrow: ``velocity_decomposed`` contains "pos" as
    a substring but is not a position, and must not be perturbed as one.
    """
    if ROTATION_MARK in name:
        return "rotation"
    if GRIPPER_MARK in name:
        return "gripper"
    if any(name.endswith(mark) or mark == "cartesian_pos" and mark in name
           for mark in POSITION_MARKS):
        return "position"
    return None


@dataclass(frozen=True)
class StateNoise:
    """How far the observed pose may be off, per sample.

    ``position`` and ``gripper`` are standard deviations of additive noise, in
    metres and in gripper units. ``rotation`` is the standard deviation of the
    rotation angle, in radians, about a uniformly random axis. ``probability``
    is the fraction of samples perturbed at all, so the model still sees exact
    poses.
    """

    position: float = 0.0
    rotation: float = 0.0
    gripper: float = 0.0
    probability: float = 1.0

    def __post_init__(self) -> None:
        for field in ("position", "rotation", "gripper"):
            if getattr(self, field) < 0:
                raise ValueError(f"state_noise.{field} must be >= 0")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("state_noise.probability must be in [0, 1]")

    @property
    def enabled(self) -> bool:
        return self.probability > 0 and (
            self.position > 0 or self.rotation > 0 or self.gripper > 0
        )

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "StateNoise":
        """Read ``data.state_noise``; an absent block means no perturbation."""
        if not config:
            return cls(position=0.0, rotation=0.0, gripper=0.0)
        unknown = set(config) - {"position", "rotation", "gripper", "probability"}
        if unknown:
            raise ValueError(f"unknown state_noise keys: {sorted(unknown)}")
        return cls(
            position=float(config.get("position", 0.0)),
            rotation=float(config.get("rotation", 0.0)),
            gripper=float(config.get("gripper", 0.0)),
            probability=float(config.get("probability", 1.0)),
        )


def _random_rotation(rng: np.random.Generator, sigma: float) -> np.ndarray:
    """A rotation of ``N(0, sigma)`` radians about a uniformly random axis."""
    axis = rng.normal(size=3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    angle = float(rng.normal(0.0, sigma))
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _perturb_rotation(block: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Rotate an orientation by a small random rotation, keeping its encoding.

    Accepts the 6D encoding (the first two rows of the rotation matrix) or ZYX
    Euler angles, and returns the same one. The perturbation is applied in the
    frame the orientation is expressed in, which is how a measurement error of
    the observed pose shows up.
    """
    width = block.shape[-1]
    if width == 6:
        matrix = so3_to_matrix_batch_nb(np.ascontiguousarray(block[None, :], dtype=np.float64))[0]
        rotated = _random_rotation(rng, sigma) @ matrix
        return rotated[:2, :].reshape(6).astype(block.dtype, copy=False)
    if width == 3:
        matrix = euler_to_matrix_zyx_batch_nb(
            np.ascontiguousarray(block[None, :], dtype=np.float64))[0]
        rotated = _random_rotation(rng, sigma) @ matrix
        euler = matrix_to_euler_zyx_batch_nb(np.ascontiguousarray(rotated[None, ...]))
        return canonicalize_euler_zyx_batch_nb(euler)[0].astype(block.dtype, copy=False)
    raise ValueError(
        f"an orientation must be 6D or ZYX Euler, got {width} dimensions")


def perturb_state(
    state: np.ndarray,
    layout: Mapping[str, int],
    noise: StateNoise,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb one unnormalized pose, laid out as ``layout`` describes it.

    ``state`` is ``[..., D]`` with ``D`` the sum of ``layout``'s dimensions, in
    its order. Returns the input unchanged -- the same object -- when the
    perturbation is off, so a run without it is bit-identical to one from
    before this existed.
    """
    if not noise.enabled:
        return state
    if rng.random() >= noise.probability:
        return state
    width = sum(int(dim) for dim in layout.values())
    if state.shape[-1] != width:
        raise ValueError(
            f"state has {state.shape[-1]} dimensions but the layout describes {width}")
    out = np.array(state, copy=True)
    cursor = 0
    for name, dim in layout.items():
        dim = int(dim)
        kind = kind_of(name)
        block = out[..., cursor:cursor + dim]
        if kind == "position" and noise.position > 0:
            block += rng.normal(0.0, noise.position, size=block.shape)
        elif kind == "gripper" and noise.gripper > 0:
            block += rng.normal(0.0, noise.gripper, size=block.shape)
        elif kind == "rotation" and noise.rotation > 0:
            flat = block.reshape(-1, dim)
            for row in range(flat.shape[0]):
                flat[row] = _perturb_rotation(flat[row], rng, noise.rotation)
            out[..., cursor:cursor + dim] = flat.reshape(block.shape)
        cursor += dim
    return out


__all__ = ["StateNoise", "kind_of", "perturb_state"]
