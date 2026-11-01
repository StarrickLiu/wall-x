"""Perturbation of the observed pose."""

import numpy as np
import pytest

from wall_x.data.state_noise import StateNoise, kind_of, perturb_state

# The released checkpoint's proprioception layout: 20 dimensions.
LAYOUT = {
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation_6D": 6,
    "follow_left_gripper": 1,
    "follow_right_ee_cartesian_pos": 3,
    "follow_right_ee_rotation_6D": 6,
    "follow_right_gripper": 1,
}
IDENTITY_6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def pose(seed=0):
    rng = np.random.default_rng(seed)
    left = list(rng.normal(size=3)) + IDENTITY_6D + [0.5]
    right = list(rng.normal(size=3)) + IDENTITY_6D + [0.2]
    return np.array([left + right], dtype=np.float32)


def matrix_of(six):
    """The rotation matrix a 6D block encodes: two rows, third by cross product."""
    r1 = np.asarray(six[:3], dtype=np.float64)
    r2 = np.asarray(six[3:], dtype=np.float64)
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - r1 * (r1 @ r2)
    r2 = r2 / np.linalg.norm(r2)
    return np.stack([r1, r2, np.cross(r1, r2)])


# --- which dimensions are what ---------------------------------------------


@pytest.mark.parametrize("name,kind", [
    ("follow_left_ee_cartesian_pos", "position"),
    ("follow_right_ee_rotation_6D", "rotation"),
    ("master_left_ee_rotation_6D_relative", "rotation"),
    ("follow_left_gripper", "gripper"),
    ("follow_left_ee_position", "position"),
    # "decomposed" contains "pos": it is not a position and must not be moved.
    ("velocity_decomposed", None),
    ("height", None),
    ("head_actions", None),
])
def test_layout_entries_are_classified_by_what_they_hold(name, kind):
    assert kind_of(name) == kind


# --- off means untouched ---------------------------------------------------


def test_disabled_returns_the_very_same_array():
    # A run without the perturbation has to be bit-identical to one from
    # before it existed, so every parity result stays valid.
    state = pose()
    assert perturb_state(state, LAYOUT, StateNoise(), np.random.default_rng(0)) is state


@pytest.mark.parametrize("noise", [
    StateNoise(position=0.0, rotation=0.0, gripper=0.0),
    StateNoise(position=0.01, probability=0.0),
])
def test_nothing_configured_means_nothing_happens(noise):
    assert not noise.enabled
    state = pose()
    assert perturb_state(state, LAYOUT, noise, np.random.default_rng(0)) is state


def test_probability_skips_samples():
    noise = StateNoise(position=1.0, probability=0.5)
    state = pose()
    touched = sum(
        perturb_state(state, LAYOUT, noise, np.random.default_rng(seed)) is not state
        for seed in range(400)
    )
    assert 150 < touched < 250


# --- what gets moved, and what does not ------------------------------------


def test_only_the_configured_quantity_moves():
    state = pose()
    out = perturb_state(state, LAYOUT, StateNoise(position=0.01), np.random.default_rng(1))
    assert not np.allclose(out[:, 0:3], state[:, 0:3])      # left position
    assert np.array_equal(out[:, 3:9], state[:, 3:9])        # left orientation
    assert np.array_equal(out[:, 9:10], state[:, 9:10])      # left gripper


def test_padding_dimensions_are_never_touched():
    layout = dict(LAYOUT, velocity_decomposed=3, height=1, head_actions=2)
    state = np.concatenate([pose(), np.zeros((1, 6), dtype=np.float32)], axis=-1)
    out = perturb_state(
        state, layout, StateNoise(position=0.05, rotation=0.1, gripper=0.05),
        np.random.default_rng(2))
    assert np.array_equal(out[:, 20:], np.zeros((1, 6), dtype=np.float32))


def test_the_input_is_not_modified_in_place():
    state = pose()
    before = state.copy()
    perturb_state(state, LAYOUT, StateNoise(position=0.05), np.random.default_rng(3))
    assert np.array_equal(state, before)


def test_dtype_survives():
    state = pose()
    out = perturb_state(state, LAYOUT, StateNoise(position=0.01, rotation=0.02, gripper=0.01),
                        np.random.default_rng(4))
    assert out.dtype == state.dtype


# --- orientation stays an orientation --------------------------------------


def test_a_perturbed_orientation_is_still_a_rotation():
    state = pose()
    for seed in range(20):
        out = perturb_state(state, LAYOUT, StateNoise(rotation=0.2),
                            np.random.default_rng(seed))
        for start in (3, 13):
            matrix = matrix_of(out[0, start:start + 6])
            assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
            assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6)


def test_the_rotation_angle_follows_the_configured_size():
    state = pose()
    base = matrix_of(state[0, 3:9])
    angles = []
    for seed in range(200):
        out = perturb_state(state, LAYOUT, StateNoise(rotation=0.1),
                            np.random.default_rng(seed))
        delta = matrix_of(out[0, 3:9]) @ base.T
        angles.append(abs(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
    # Half-normal with sigma=0.1: mean about 0.08 rad, nothing beyond ~4 sigma.
    assert 0.05 < float(np.mean(angles)) < 0.11
    assert max(angles) < 0.5


def test_both_arms_are_perturbed_independently():
    state = pose()
    out = perturb_state(state, LAYOUT, StateNoise(rotation=0.2), np.random.default_rng(5))
    left = matrix_of(out[0, 3:9]) @ matrix_of(state[0, 3:9]).T
    right = matrix_of(out[0, 13:19]) @ matrix_of(state[0, 13:19]).T
    assert not np.allclose(left, right, atol=1e-3)


def test_euler_orientations_are_supported():
    layout = {"follow_left_ee_cartesian_pos": 3, "follow_left_ee_rotation_euler": 3}
    state = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)
    out = perturb_state(state, layout, StateNoise(rotation=0.1), np.random.default_rng(6))
    assert out.shape == state.shape
    assert not np.array_equal(out[:, 3:], state[:, 3:])


def test_an_unsupported_orientation_width_is_refused():
    layout = {"follow_left_ee_rotation_quat": 4}
    state = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="6D or ZYX Euler"):
        perturb_state(state, layout, StateNoise(rotation=0.1), np.random.default_rng(7))


# --- determinism and configuration -----------------------------------------


def test_the_same_seed_gives_the_same_perturbation():
    state = pose()
    noise = StateNoise(position=0.01, rotation=0.05, gripper=0.02)
    a = perturb_state(state, LAYOUT, noise, np.random.default_rng(8))
    b = perturb_state(state, LAYOUT, noise, np.random.default_rng(8))
    assert np.array_equal(a, b)


def test_a_mismatched_layout_is_refused():
    with pytest.raises(ValueError, match="layout describes"):
        perturb_state(np.zeros((1, 7), dtype=np.float32), LAYOUT,
                      StateNoise(position=0.01), np.random.default_rng(9))


def test_config_defaults_to_off():
    assert not StateNoise.from_config(None).enabled
    assert not StateNoise.from_config({}).enabled


def test_config_is_read_and_checked():
    noise = StateNoise.from_config(
        {"position": 0.005, "rotation": 0.02, "gripper": 0.01, "probability": 0.5})
    assert (noise.position, noise.rotation, noise.gripper, noise.probability) == (
        0.005, 0.02, 0.01, 0.5)
    with pytest.raises(ValueError, match="unknown state_noise keys"):
        StateNoise.from_config({"positon": 0.01})
    with pytest.raises(ValueError, match="must be >= 0"):
        StateNoise.from_config({"position": -1})
    with pytest.raises(ValueError, match="probability"):
        StateNoise.from_config({"position": 0.01, "probability": 2})
