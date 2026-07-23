"""Adapter package entry point.

Importing ``variants`` triggers variant registration side effects.
"""

import logging

from wall_x._vendor.harrix.e2e_infer.adapters import variants  # noqa: F401

logger = logging.getLogger(__name__)

