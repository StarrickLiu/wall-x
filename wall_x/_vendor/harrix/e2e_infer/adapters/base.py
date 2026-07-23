"""Thin environment-to-executor adapter abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from wall_x._vendor.harrix.e2e_infer.config import EvalConfig


class BaseInferAdapter(ABC):
    @abstractmethod
    def __init__(self, cfg: EvalConfig) -> None:
        """Build the owned executor and environment conversion state."""

    @property
    @abstractmethod
    def chunk_horizon(self) -> int:
        """Number of action steps returned by each ``predict_batch`` call."""

    @property
    @abstractmethod
    def action_mode(self) -> str:
        """Configured inference algorithm, fixed at construction time."""

    @abstractmethod
    def predict_batch(self, payloads: list[dict]) -> list[np.ndarray]:
        """Run one batched inference call and return one chunk per payload.

        ``payloads[i]`` schema is defined by the env-adapter pair:
            {
                "observation": dict[str, np.ndarray],
                    # Env-defined ndarray bundle, for example LIBERO:
                    # {"eef_pos":(3,), "eef_axisangle":(3,), "gripper":(1,),
                    #  "face_view":(H,W,3), "wrist_view":(H,W,3)}
                "instruction": str,
                "noise":       np.ndarray | None,
                    # Flow may pass (chunk_horizon, action_dim); other modes
                    # may leave this as None.
            }
        """
