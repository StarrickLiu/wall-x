"""The keyframe-memory contract shared by dataset construction and serving."""

import numpy as np
import pytest
from PIL import Image

from wall_x.model.core.keyframe_memory import (
    BOX_SCALE,
    candidate_positions,
    crop_rect,
    crop_to_box,
    inject_memory,
    normalize_box,
    selection_prompt,
    short_term_positions,
)
from wall_x.model.core.keyframe_memory.prompts import IMAGE


# --- candidate positions ---------------------------------------------------


def test_candidates_split_the_window_into_equal_spans():
    # 48 frames before frame 100, 8 candidates: one per 6-frame span, taken
    # from the middle of each span.
    assert candidate_positions(100, window=48, count=8) == [55, 61, 67, 73, 79, 85, 91, 97]


def test_candidates_are_phase_free():
    # Shifting the anchor shifts every candidate by the same amount, so the
    # pass does not depend on where in the episode it falls.
    a = candidate_positions(100, window=48, count=8)
    b = candidate_positions(107, window=48, count=8)
    assert [y - x for x, y in zip(a, b)] == [7] * 8


def test_candidates_never_reach_before_the_episode_start():
    positions = candidate_positions(5, window=48, count=8)
    assert min(positions) >= 0
    assert max(positions) < 5
    assert len(positions) == 8  # repeats rather than dropping candidates


def test_candidates_stay_inside_the_window_and_before_now():
    for now in (1, 7, 48, 49, 200):
        positions = candidate_positions(now, window=48, count=16)
        assert all(max(0, now - 48) <= p < now for p in positions)
        assert positions == sorted(positions)


# --- short-term positions --------------------------------------------------


def test_short_term_taps_are_oldest_first():
    assert short_term_positions(100, count=3, stride=4) == [88, 92, 96]


def test_short_term_keeps_its_count_near_the_episode_start():
    # Fixed image count per sample: taps repeat the oldest frame instead of
    # being dropped.
    assert short_term_positions(2, count=3, stride=4) == [0, 0, 0]
    assert short_term_positions(2, count=3, stride=4, oldest=1) == [1, 1, 1]


def test_short_term_off():
    assert short_term_positions(100, count=0, stride=4) == []


# --- box coordinates -------------------------------------------------------


def test_normalize_box_is_resolution_independent():
    # The same object on a 640x480 and a 1280x960 frame gets the same box.
    assert normalize_box((160, 120, 480, 360), 640, 480) == normalize_box(
        (320, 240, 960, 720), 1280, 960
    )


def test_normalize_box_spans_the_full_scale():
    assert normalize_box((0, 0, 640, 480), 640, 480) == (0, 0, BOX_SCALE, BOX_SCALE)


def test_normalize_box_clamps_out_of_frame_corners():
    x1, y1, x2, y2 = normalize_box((-20, -20, 900, 700), 640, 480)
    assert (x1, y1) == (0, 0)
    assert (x2, y2) == (BOX_SCALE, BOX_SCALE)


def test_crop_rect_inverts_normalize_box():
    box = normalize_box((160, 120, 480, 360), 640, 480)
    x1, y1, x2, y2 = crop_rect(box, 640, 480)
    # crop_rect expands each edge by EXPAND of the box side, so the rectangle
    # contains the original box and is larger by that margin.
    assert x1 < 160 and y1 < 120 and x2 > 480 and y2 > 360


def test_crop_rect_stays_inside_the_frame():
    for box in ((0, 0, 20, 20), (980, 980, 1000, 1000), (0, 400, 1000, 600)):
        x1, y1, x2, y2 = crop_rect(box, 448, 336)
        assert 0 <= x1 < x2 <= 448
        assert 0 <= y1 < y2 <= 336


def test_crop_to_box_keeps_the_image_size():
    # The image-token count of the prompt must not change.
    image = Image.fromarray(np.zeros((336, 448, 3), dtype=np.uint8))
    assert crop_to_box(image, (300, 300, 400, 400)).size == image.size
    array = np.zeros((336, 448, 3), dtype=np.uint8)
    assert crop_to_box(array, (300, 300, 400, 400)).shape == array.shape


# --- prompts ---------------------------------------------------------------


def test_selection_prompt_counts_its_images():
    prompt = selection_prompt("do the thing", 3, 8)
    assert prompt.count(IMAGE) == 11
    assert "Progress memory: [M_0]: " in prompt
    assert "[M_2]: " in prompt and "[M_3]: " not in prompt
    assert "<|R_7|>: " in prompt and "<|R_8|>: " not in prompt
    # Ends where the decision token is generated.
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_selection_prompt_with_empty_memory():
    prompt = selection_prompt("do the thing", 0, 8)
    assert "Progress memory: (empty)" in prompt
    assert prompt.count(IMAGE) == 8


def test_inject_memory_puts_blocks_after_the_observation_tag():
    prefix = "Observation: front view: " + IMAGE + "\nInstruction: go\n"
    out = inject_memory(prefix, num_memory=2, num_short_term=1, frame=96, label="progress memory")
    assert out.count(IMAGE) == 4  # 2 memory + 1 recent view + the camera
    assert out.index("[M_0]") < out.index("[M_1]") < out.index("Recent view:") < out.index("now[f96]:")
    assert out.index("now[f96]:") < out.index("front view:")
    assert out.startswith("Observation: progress memory:")


def test_inject_memory_stamps_the_frame_without_memory():
    prefix = "Observation: front view: " + IMAGE
    assert inject_memory(prefix, num_memory=0, num_short_term=0, frame=7, label="m") == (
        "Observation: now[f7]: front view: " + IMAGE
    )


def test_inject_memory_off_leaves_the_prompt_alone():
    prefix = "Observation: front view: " + IMAGE
    assert (
        inject_memory(
            prefix, num_memory=2, num_short_term=0, frame=None, label="m", memory_in_prompt=False
        )
        == prefix
    )


def test_inject_memory_touches_only_the_first_observation_tag():
    prefix = "Observation: a Observation: b"
    out = inject_memory(prefix, num_memory=1, num_short_term=0, frame=1, label="m")
    assert out.count("Observation:") == 2
    assert out.endswith("Observation: b")


@pytest.mark.parametrize("num_memory,num_short_term", [(0, 0), (1, 0), (0, 2), (3, 2)])
def test_inject_memory_image_count_matches_the_frames_it_describes(num_memory, num_short_term):
    prefix = "Observation: front view: " + IMAGE
    out = inject_memory(
        prefix, num_memory=num_memory, num_short_term=num_short_term, frame=0, label="m"
    )
    assert out.count(IMAGE) == 1 + num_memory + num_short_term


# --- the varied framing training uses -------------------------------------


def test_the_framing_is_fixed_without_a_generator():
    box = (250, 250, 750, 750)
    assert crop_rect(box, 448, 336) == crop_rect(box, 448, 336)


def test_a_generator_varies_the_framing():
    box = (250, 250, 750, 750)
    rng = np.random.default_rng(0)
    rects = {crop_rect(box, 448, 336, rng) for _ in range(20)}
    assert len(rects) > 10                      # a family, not one framing


def test_the_varied_framing_still_contains_the_object():
    # Every framing must show the object; the expansion only ever grows the box
    # before the shift, which is bounded by a tenth of its side.
    box = (250, 250, 750, 750)
    rng = np.random.default_rng(1)
    left, top = 250 / 1000 * 448, 250 / 1000 * 336
    right, bottom = 750 / 1000 * 448, 750 / 1000 * 336
    for _ in range(200):
        x1, y1, x2, y2 = crop_rect(box, 448, 336, rng)
        assert x1 <= left and y1 <= top and x2 >= right and y2 >= bottom


def test_the_varied_framing_stays_inside_the_image():
    rng = np.random.default_rng(2)
    for box in ((0, 0, 60, 60), (940, 940, 1000, 1000), (250, 250, 750, 750)):
        for _ in range(100):
            x1, y1, x2, y2 = crop_rect(box, 448, 336, rng)
            assert 0 <= x1 < x2 <= 448 and 0 <= y1 < y2 <= 336


def test_the_same_generator_state_gives_the_same_framing():
    box = (250, 250, 750, 750)
    a = crop_rect(box, 448, 336, np.random.default_rng(7))
    b = crop_rect(box, 448, 336, np.random.default_rng(7))
    assert a == b
