"""Vanilla (open-source) Wall-X serving policy.

This is the opt-engine-free counterpart to :class:`WallXPolicy`. It runs flow
inference through the same vanilla model path the LIBERO evaluation adapter uses
(``QwenVLActInferAdapter``), so it has no dependency on the closed
``harrix.model_executors`` inference engine or on internal robot preprocessors.

Contract (single observation):

    obs = {
        "state": {                       # proprioception, one entry per model dof group
            "eef_pos":        [x, y, z],          # end-effector position (3)
            "eef_axisangle":  [rx, ry, rz],       # end-effector rotation, axis-angle (3)
            "gripper":        [g0, ...],          # gripper qpos (1 or 2 dims)
        },
        "views": {                        # camera images, base64-encoded or raw ndarray
            "face_view":        <img>,
            "right_wrist_view": <img>,
        },
        "instruction": "pick up the red block",
    }

``infer`` returns the raw model action chunk::

    {"action": np.ndarray of shape (action_horizon, action_dim)}

Downstream robot-specific serialization/interpolation is intentionally left to
the client, since real-robot preprocessors are not part of the open-source cut.
"""

import base64
import logging
from typing import Any, Dict

import numpy as np

from wall_x._vendor.harrix.serving.handlers.base import BasePolicy

logger = logging.getLogger(__name__)

INFER_MODE_FLOW = "flow"


class WallXVanillaPolicy(BasePolicy):
    """Opt-free Wall-X policy: vanilla flow inference, raw action chunks."""

    def __init__(
        self,
        eval_config,
        image_passing_mode: str = "base64",
    ):
        """Build the vanilla inference adapter from a harrix ``EvalConfig``.

        Args:
            eval_config: ``harrix.e2e_infer.config.EvalConfig`` with
                ``model.checkpoint_path``
                (and optionally ``model.train_config_path`` / ``model.action_horizon``)
                populated.
            image_passing_mode: ``"base64"`` (client sends JPEG/PNG bytes, base64) or
                ``"numpy"`` (client sends raw H×W×C uint8 arrays).
        """
        import wall_x._vendor.harrix.e2e_infer.adapters  # noqa: F401  registers model adapters
        from wall_x._vendor.harrix.e2e_infer.adapters.registry import build_adapter

        if image_passing_mode not in ("base64", "numpy"):
            raise ValueError(
                f"image_passing_mode must be 'base64' or 'numpy', got "
                f"{image_passing_mode!r}"
            )
        self.image_passing_mode = image_passing_mode
        self._adapter = build_adapter(eval_config)
        self._action_horizon = int(self._adapter.chunk_horizon)
        self._action_dim = int(self._adapter.action_dim)
        logger.info(
            "WallXVanillaPolicy ready: action_horizon=%d action_dim=%d image_mode=%s",
            self._action_horizon,
            self._action_dim,
            self.image_passing_mode,
        )

    # ── BasePolicy API ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Stateless policy: nothing to reset."""

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "engine": "vanilla",
            "infer_mode": INFER_MODE_FLOW,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "image_passing_mode": self.image_passing_mode,
        }

    def infer(self, obs: Dict) -> Dict:
        """Run one flow-inference pass; return the raw action chunk.

        Delegates to the adapter's validated ``predict_batch`` (the same path the
        LIBERO evaluation uses), which handles proprio encoding, prompt
        construction, and flow sampling internally.
        """
        payload = {
            "observation": self._assemble_observation(obs),
            "instruction": obs.get("instruction", ""),
            "noise": obs.get("noise"),
        }
        actions = self._adapter.predict_batch([payload])
        return {"action": np.asarray(actions[0])}

    # ── observation assembly ─────────────────────────────────────────────

    def _decode_view(self, value) -> np.ndarray:
        """Decode one camera view into an (H, W, 3) RGB uint8 array."""
        if self.image_passing_mode == "base64":
            import cv2

            img_bytes = base64.b64decode(value)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("failed to decode base64 image (cv2.imdecode)")
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        arr = np.asarray(value, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"numpy image must be (H, W, 3), got shape {arr.shape}"
            )
        return arr

    def _assemble_observation(self, obs: Dict) -> Dict:
        """Build the compact ndarray payload the adapter's encoder expects.

        The adapter's ``predict_batch`` runs ``encode_proprio`` on this dict, so
        the keys mirror the LIBERO evaluation payload
        (see ``harrix.e2e_infer.envs.libero_common``).
        """
        state = obs["state"]
        views = obs["views"]
        return {
            "eef_pos": np.asarray(state["eef_pos"], dtype=np.float32),
            "eef_axisangle": np.asarray(state["eef_axisangle"], dtype=np.float32),
            "gripper": np.asarray(state["gripper"], dtype=np.float32),
            "face_view": self._decode_view(views["face_view"]),
            "wrist_view": self._decode_view(views["right_wrist_view"]),
        }
