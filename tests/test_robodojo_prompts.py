"""The prompt of a RoboDojo action sample."""

import numpy as np
import pytest

from wall_x.data.backends.lerobot.robodojo.prompts import ACTION, IMAGE, action_prompt, state_string

LABELS = ("front view", "left wrist view", "right wrist view")
POSE = [0.0] * 20


def prompt(**kwargs):
    kwargs.setdefault("action_horizon", 4)
    kwargs.setdefault("pose", POSE)
    return action_prompt("do the thing", LABELS, **kwargs)


# --- the pose as text ------------------------------------------------------


def test_the_pose_is_written_as_bin_indices():
    # Normalized to [-1, 1]; 255 bins, so the middle is 127 and the top 254.
    assert state_string([-1.0, 0.0, 1.0], bins=255) == "0 127 254"


def test_the_last_bin_absorbs_the_upper_edge():
    # Without the clip, exactly 1.0 would index a bin that does not exist.
    assert state_string([1.0], bins=255) == "254"
    assert state_string([0.9999], bins=255) == "254"


def test_values_outside_the_range_are_clipped():
    assert state_string([-3.0, 3.0], bins=255) == "0 254"


def test_a_value_lands_in_the_bin_it_falls_in_not_the_nearest():
    # floor, not round: this is what the released checkpoint was trained with,
    # and rounding would shift about one value in ten into its neighbour.
    # 1.9 bins along: inside bin 1, but rounding would call it 2.
    assert state_string([-1.0 + 2.0 * 1.9 / 255], bins=255) == "1"
    assert state_string([-1.0 + 2.0 * 0.5 / 255], bins=255) == "0"
    assert state_string([-1.0 + 2.0 * 1.5 / 255], bins=255) == "1"


def test_every_dimension_appears():
    assert len(state_string(POSE, bins=255).split()) == 20


# --- the prompt ------------------------------------------------------------


def test_the_prompt_names_every_camera_and_gives_it_an_image():
    prefix, _ = prompt()
    for label in LABELS:
        assert f"{label}: {IMAGE}" in prefix
    assert prefix.count(IMAGE) == 3


def test_the_action_slots_are_the_horizon():
    _, postfix = prompt(action_horizon=32)
    assert postfix == ACTION * 32
    assert postfix.count(ACTION) == 32


def test_a_pose_can_be_written_in_for_comparison():
    # The filled form is what the parity checks compare against the gold.
    prefix, _ = prompt(pose=[1.0] * 20)
    assert "<|propri|>" not in prefix
    assert "Proprioception: " + " ".join(["254"] * 20) in prefix


def test_without_a_pose_the_slot_is_left_for_the_collator():
    prefix, _ = prompt(pose=None)
    assert "Proprioception: <|propri|>" in prefix


def test_the_prompt_ends_where_the_action_chunk_begins():
    prefix, _ = prompt()
    assert prefix.endswith("<|im_start|>assistant\n")
    assert "<|action_fast|>" not in prefix and "<|action_ar|>" not in prefix


def test_the_instruction_label_is_configurable():
    prefix, _ = prompt(instruction_label="Task")
    assert "Task: do the thing" in prefix


# --- recalled frames -------------------------------------------------------


def test_memory_blocks_come_before_the_cameras():
    prefix, _ = prompt(num_memory=2)
    assert prefix.index("[M_0]") < prefix.index("[M_1]") < prefix.index("front view:")
    assert prefix.count(IMAGE) == 5     # two recalled frames plus three cameras


def test_no_memory_leaves_the_prompt_as_it_was():
    with_memory, _ = prompt(num_memory=0)
    assert "progress memory" not in with_memory
    assert with_memory.count(IMAGE) == 3


def test_the_frame_number_is_only_stamped_when_asked():
    # The released recipe left it off.
    assert "now[" not in prompt(num_memory=2)[0]
    assert "now[f96]:" in prompt(num_memory=2, frame=96)[0]


def test_recent_view_frames_are_described_separately_when_used():
    prefix, _ = prompt(num_memory=1, num_short_term=2)
    assert prefix.index("[M_0]") < prefix.index("Recent view:") < prefix.index("front view:")
    assert prefix.count(IMAGE) == 6


@pytest.mark.parametrize("memory,short_term", [(0, 0), (1, 0), (3, 0), (1, 2), (22, 0)])
def test_the_image_count_matches_the_frames_described(memory, short_term):
    # A mismatch here scatters the image features onto the wrong tokens.
    prefix, _ = prompt(num_memory=memory, num_short_term=short_term)
    assert prefix.count(IMAGE) == memory + short_term + len(LABELS)
