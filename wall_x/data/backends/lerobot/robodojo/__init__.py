"""RoboDojo backend registration.

A backend of its own, inside the LeRobot one: the episodes are a LeRobot
export read with that backend's collator, and what differs is how a sample
is built -- a memory of recalled keyframes, and the selection samples that
teach the model to write one.
"""

from __future__ import annotations

import sys

try:
    from wall_x.data._registry import register_module
    from wall_x.data.backends.lerobot.robodojo.build import build

    register_module("robodojo", sys.modules[__name__])
except ImportError as _e:  # pragma: no cover - exercised by a missing dependency
    from wall_x.data._registry import record_import_error

    record_import_error("robodojo", _e)


__all__ = ["build"]
