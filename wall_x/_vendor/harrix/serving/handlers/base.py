"""Serving request-handler contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

from wall_x._vendor.harrix.model_executors.base import ServingExecutor


class ServingHandler:
    """Translate serving requests into inference results.

    ``infer``/``infer_batch`` remain the compatibility ABI during migration.
    """

    def handle(self, request: Dict, session: Any = None) -> Dict:
        """Canonical request entry point; ``session`` is reserved for migration."""

        return self.infer(request)

    def handle_batch(self, requests: List[Dict]) -> List[Dict]:
        return self.infer_batch(requests)

    def infer(self, obs: Dict) -> Dict:
        raise NotImplementedError

    def infer_batch(self, obs_list: List[Dict]) -> List[Dict]:
        return [self.infer(obs) for obs in obs_list]

    def reset(self) -> None:
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}


class ExecutorServingHandler(ServingHandler):
    """Expose an executor through the serving lifecycle contract."""

    def __init__(self, executor: ServingExecutor) -> None:
        missing = [
            name
            for name in ("infer", "infer_batch", "reset")
            if not callable(getattr(executor, name, None))
        ]
        if not hasattr(executor, "metadata"):
            missing.append("metadata")
        if missing:
            raise TypeError(
                "ExecutorServingHandler requires a ServingExecutor; "
                f"missing capabilities: {', '.join(missing)}"
            )
        if not isinstance(executor.metadata, Mapping):
            raise TypeError(
                "ExecutorServingHandler requires ServingExecutor.metadata "
                f"to be a mapping property, got {type(executor.metadata).__name__}"
            )
        self.executor = executor

    def infer(self, obs: Dict) -> Dict:
        return self.executor.infer(obs)

    def infer_batch(self, obs_list: List[Dict]) -> List[Dict]:
        return self.executor.infer_batch(obs_list)

    def reset(self) -> None:
        self.executor.reset()

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.executor.metadata


BasePolicy = ServingHandler

__all__ = ["BasePolicy", "ExecutorServingHandler", "ServingHandler"]
