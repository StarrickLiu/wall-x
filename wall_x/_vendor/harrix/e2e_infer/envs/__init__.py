"""Environment registrations for harrix."""

_LIBERO_IMPORT_ERROR = None

try:
    from wall_x._vendor.harrix.e2e_infer.envs import libero  # noqa: F401
except ImportError as exc:
    _LIBERO_IMPORT_ERROR = exc
