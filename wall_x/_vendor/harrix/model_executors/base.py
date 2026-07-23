"""Minimal model-executor contract."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class ModelExecutor(Protocol):
    """Model loading and batched execution capability."""

    def infer_batch(self, requests: Sequence[Any]) -> list[Any]:
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        ...


class ServingExecutor(ModelExecutor, Protocol):
    """Executor capability required by a stateful serving handler."""

    def infer(self, request: Any) -> Any:
        ...

    def reset(self) -> None:
        ...
