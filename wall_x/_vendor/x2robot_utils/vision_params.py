"""Vision preprocessing parameters -- the single source of truth shared by
training and inference.

These are the parameters the model-side image preprocessing (Qwen
``smart_resize``) needs: per-camera resolution, the resize constants, and the
relative-action marker. The training side (when each data backend builds
samples) and the inference side (when the harrix adapter preprocesses camera
frames) must use the same semantics, otherwise the train/infer distributions
diverge.

The constants are pinned as literals here rather than forwarded from
``qwen_vl_utils``: a third-party library's defaults drift across versions
(``qwen_vl_utils`` 0.0.14 removed ``IMAGE_FACTOR``), whereas these values are a
reproducibility anchor consistent with the Qwen2.5-VL processor spec. A
checkpoint's ``train_config`` may override any field.
"""

from __future__ import annotations

from dataclasses import dataclass

# Qwen2.5-VL smart_resize standard constants (image_factor=28,
# min_pixels=28**2*4, max_pixels=28**2*16384).
QWEN_IMAGE_FACTOR = 28
QWEN_MIN_PIXELS = 3136
QWEN_MAX_PIXELS = 12845056


@dataclass
class VisionParams:
    """Projection of the image preprocessing parameters.

    ``resolution`` maps each camera name to its edge length; ``use_relative_action``
    is the action-space marker, consumed by the inference adapter alongside the
    preprocessing config, so it is folded into this projection.
    """

    resolution: dict
    image_factor: int = QWEN_IMAGE_FACTOR
    min_pixels: int = QWEN_MIN_PIXELS
    max_pixels: int = QWEN_MAX_PIXELS
    use_relative_action: bool = False


def build_vision_params(train_config: dict) -> VisionParams:
    """Build a :class:`VisionParams` from a checkpoint's (flat) ``train_config``.

    Reads the same-named fields under ``train_config["data"]``, falling back to
    the Qwen standard constants when absent.
    """
    data = train_config.get("data", {}) or {}
    return VisionParams(
        resolution=dict(data.get("resolution", {})),
        image_factor=int(data.get("image_factor", QWEN_IMAGE_FACTOR)),
        min_pixels=int(data.get("min_pixels", QWEN_MIN_PIXELS)),
        max_pixels=int(data.get("max_pixels", QWEN_MAX_PIXELS)),
        use_relative_action=bool(data.get("use_relative_action", False)),
    )
