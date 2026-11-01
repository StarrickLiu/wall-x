"""Which frames a RoboDojo sample needs, and the images it hands the model.

A sample's images and its prompt are written from one source of truth, because
they are read together: the n-th image block of the prompt is filled by the
n-th image. Describing one recalled frame and passing another scatters the
image features onto the wrong tokens, which trains without any error.

So the frames are named once, in prompt order, and the images are prepared in
that same order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from wall_x._vendor.harrix.utils.dataset_v2 import dataset_v2_smart_resize
from wall_x.model.core.keyframe_memory import crop_to_box, normalize_box, short_term_positions

# The head camera is the one a recalled frame comes from: keyframes are
# milestones of the scene, not of a wrist.
HEAD_CAMERA = 0


@dataclass(frozen=True)
class Geometry:
    """How a sample is cut out of an episode.

    ``capacity`` caps how many recalled frames a prompt carries, ``window`` and
    ``candidates`` describe the selection pass, and ``short_term`` adds recent
    head-camera frames before the current one. ``resolution`` with ``factor``
    and the pixel bounds decide the size every image is resized to.
    """

    capacity: int = 22
    window: int = 48
    candidates: int = 16
    short_term: int = 0
    short_term_stride: int = 4
    resolution: int = 448
    factor: int = 28
    min_pixels: int = 3136
    max_pixels: int = 1003520
    box_crop: bool = False
    # Training varies the framing of a cropped memory frame; inference uses the
    # fixed one. Turn this off to see the inference framing while training.
    box_jitter: bool = True

    def image_size(self, height: int, width: int) -> Tuple[int, int]:
        """The ``(height, width)`` an image of this shape is resized to.

        The long side goes to ``resolution`` first, then the sizing rule of the
        vision processor rounds both sides onto its grid. The released
        checkpoint used a factor of 28 while its processor aligns to 32, so a
        second resize happens inside the processor -- reproduced here rather
        than corrected, since the checkpoint was trained through it.
        """
        if height < 1 or width < 1:
            raise ValueError(f"an image cannot be {height}x{width}")
        scale = self.resolution / max(height, width)
        return dataset_v2_smart_resize(
            round(height * scale), round(width * scale),
            factor=self.factor, min_pixels=self.min_pixels, max_pixels=self.max_pixels)


@dataclass(frozen=True)
class FrameRequest:
    """The frames one sample reads, in the order the prompt describes them.

    ``recalled`` and ``candidates`` are head-camera frames; ``current`` is the
    anchor, read from every camera. ``boxes`` holds the object box of a recalled
    frame, where the annotation has one.
    """

    recalled: Tuple[int, ...] = ()
    candidates: Tuple[int, ...] = ()
    current: int = 0
    boxes: Mapping[int, Tuple[int, int, int, int]] = field(default_factory=dict)

    @property
    def head_frames(self) -> Tuple[int, ...]:
        """Head-camera frames, in prompt order."""
        return self.recalled + self.candidates

    @property
    def all_frames(self) -> Tuple[int, ...]:
        return self.head_frames + (self.current,)


def action_request(annotation, anchor: int, geometry: Geometry) -> FrameRequest:
    """The frames of an action sample at ``anchor``.

    Recalled frames are the keyframes already recorded, followed by the recent
    head-camera frames; both are described as memory entries, which is how the
    released checkpoint was trained.
    """
    recalled = list(annotation.memory_before(anchor, geometry.capacity))
    boxes = {frame: annotation.boxes[frame] for frame in recalled
             if frame in annotation.boxes} if geometry.box_crop else {}
    recalled += short_term_positions(
        anchor, count=geometry.short_term, stride=geometry.short_term_stride)
    return FrameRequest(recalled=tuple(recalled), current=anchor, boxes=boxes)


def selection_request(annotation, anchor: int, geometry: Geometry,
                      last_frame: Optional[int] = None) -> FrameRequest:
    """The frames of a selection sample at ``anchor``.

    Memory first, then the candidates being asked about. Frames are clamped
    into the episode, which matters at the very start where the window is
    shorter than it asks for.
    """
    selection = annotation.selection(
        anchor, window=geometry.window, candidates=geometry.candidates,
        capacity=geometry.capacity)

    def clamp(frame: int) -> int:
        return frame if last_frame is None else min(frame, last_frame)

    recalled = tuple(clamp(f) for f in selection.memory)
    boxes = {frame: annotation.boxes[frame] for frame in recalled
             if frame in annotation.boxes} if geometry.box_crop else {}
    return FrameRequest(
        recalled=recalled,
        candidates=tuple(clamp(f) for f in selection.candidates),
        current=clamp(anchor),
        boxes=boxes,
    )


def prepare_images(
    request: FrameRequest,
    head: Mapping[int, object],
    cameras: Sequence[object],
    geometry: Geometry,
    raw_size: Optional[Tuple[int, int]] = None,
    rng=None,
) -> list:
    """The images of one sample, in the order its prompt describes them.

    ``head`` maps a head-camera frame index to its image, ``cameras`` holds the
    anchor's image from each camera. A recalled frame with an object box is
    cropped to it, so the memory keeps the object and drops the rest of the
    scene; candidates and the current observation are never cropped.

    ``rng`` varies the framing of those crops, which is what training does;
    without one the framing is the fixed one inference uses.
    """
    from PIL import Image

    def resized(image):
        pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
        height, width = geometry.image_size(pil.height, pil.width)
        if (pil.height, pil.width) == (height, width):
            return pil
        return pil.resize((width, height), Image.BICUBIC)

    out = []
    for frame in request.recalled:
        if frame not in head:
            raise KeyError(f"frame {frame} is described in the prompt but was not read")
        image = resized(head[frame])
        box = request.boxes.get(frame)
        if box is not None:
            width, height = (raw_size if raw_size else (image.width, image.height))
            image = crop_to_box(
                image, normalize_box(box, width, height),
                rng if geometry.box_jitter else None)
        out.append(image)
    for frame in request.candidates:
        if frame not in head:
            raise KeyError(f"frame {frame} is described in the prompt but was not read")
        out.append(resized(head[frame]))
    out.extend(resized(image) for image in cameras)
    return out


__all__ = [
    "FrameRequest",
    "Geometry",
    "HEAD_CAMERA",
    "action_request",
    "prepare_images",
    "selection_request",
]
