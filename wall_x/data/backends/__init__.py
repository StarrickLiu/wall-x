"""Data backend registration.

Importing this package registers the default shipped backends. Additional
backends are loaded by plugin modules so their names and dependencies do not
have to appear in the default import path.
"""

import importlib

from wall_x.data._registry import record_import_error

# ``robodojo`` sits inside ``lerobot``: its episodes *are* a LeRobot export and
# it reads them with that backend's collator. What differs is how a sample is
# built, not the corpus format, so it registers as a backend of its own while
# living in the family it belongs to.
for _name, _module in (("lerobot", "lerobot"), ("robodojo", "lerobot.robodojo")):
    try:
        importlib.import_module(f"wall_x.data.backends.{_module}")
    except ImportError as _e:
        record_import_error(_name, _e)
