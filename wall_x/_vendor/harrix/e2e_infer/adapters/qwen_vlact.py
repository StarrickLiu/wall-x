"""Thin environment adapter for the Qwen2.5 evaluation executor."""

import numpy as np

from wall_x._vendor.harrix.e2e_infer.adapters.base import BaseInferAdapter
from wall_x._vendor.harrix.e2e_infer.envs.libero_common import decode_chunk, encode_proprio
from wall_x._vendor.harrix.model_executors.qwen2_5_eval import Qwen2_5EvalExecutor


class QwenVLActInferAdapter(BaseInferAdapter):
    """Select the model variant and delegate execution to model_executors."""

    @classmethod
    def _training_adapter(cls):
        raise NotImplementedError(
            f"{cls.__name__} must override _training_adapter()"
        )

    def __init__(self, cfg) -> None:
        self._executor = Qwen2_5EvalExecutor(cfg, self._training_adapter())

    @property
    def chunk_horizon(self) -> int:
        return self._executor.chunk_horizon

    @property
    def action_mode(self) -> str:
        return self._executor.action_mode

    @property
    def action_dim(self) -> int:
        return self._executor.action_dim

    def predict_batch(self, payloads):
        task = self._executor.task_config
        requests = [
            {
                **payload,
                "observation": encode_proprio(
                    payload["observation"], task, self.chunk_horizon
                ),
            }
            for payload in payloads
        ]
        outputs = self._executor.infer_batch(requests)
        return [
            np.asarray(decode_chunk(output, task), dtype=np.float32)
            for output in outputs
        ]
