"""Pose vectors of the RoboDojo end-effector export."""

import numpy as np
import pytest

from wall_x.data.backends.lerobot.robodojo import poses

IDENTITY = [1.0, 0.0, 0.0, 0.0]            # wxyz
QUARTER_Z = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]   # +90 degrees about z


def arm(position, quaternion, gripper):
    return list(position) + list(quaternion) + [gripper]


def export(left, right):
    return left + right


def at_rest():
    return export(arm([0.1, 0.2, 0.3], IDENTITY, 1.0), arm([-0.1, 0.2, 0.3], IDENTITY, 0.0))


# --- rotations -------------------------------------------------------------


def test_identity_quaternion_is_the_identity_rotation():
    assert np.allclose(poses.rotation_from_quaternion(IDENTITY), np.eye(3))


def test_quarter_turn_about_z():
    assert np.allclose(poses.rotation_from_quaternion(QUARTER_Z),
                       [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)


def test_a_quaternion_and_its_negation_are_the_same_rotation():
    a = poses.rotation_from_quaternion(QUARTER_Z)
    b = poses.rotation_from_quaternion([-v for v in QUARTER_Z])
    assert np.allclose(a, b)


def test_an_unnormalized_quaternion_is_normalized_first():
    scaled = [2 * v for v in QUARTER_Z]
    assert np.allclose(poses.rotation_from_quaternion(scaled),
                       poses.rotation_from_quaternion(QUARTER_Z))


def test_a_zero_quaternion_is_refused():
    with pytest.raises(ValueError, match="zero length"):
        poses.rotation_from_quaternion([0.0, 0.0, 0.0, 0.0])


def test_the_6d_encoding_is_the_first_two_rows():
    rotation = poses.rotation_from_quaternion(QUARTER_Z)
    six = poses.rotation_to_6d(rotation)
    assert np.allclose(six, rotation[:2, :].reshape(6))
    # The third row is their cross product, so nothing is lost.
    assert np.allclose(np.cross(six[:3], six[3:]), rotation[2, :], atol=1e-9)


# --- the observed pose -----------------------------------------------------


def test_observed_pose_layout():
    pose = poses.observed_pose(at_rest())
    assert pose.shape == (20,)
    assert np.allclose(pose[0:3], [0.1, 0.2, 0.3])        # left position
    assert np.allclose(pose[3:9], [1, 0, 0, 0, 1, 0])      # left orientation
    assert pose[9] == 1.0                                   # left gripper
    assert np.allclose(pose[10:13], [-0.1, 0.2, 0.3])      # right position
    assert pose[19] == 0.0                                  # right gripper


# --- relative actions ------------------------------------------------------


def test_no_motion_is_a_zero_action_with_an_identity_rotation():
    action = poses.relative_action(at_rest(), at_rest())
    assert np.allclose(action[0:3], 0)
    assert np.allclose(action[3:9], [1, 0, 0, 0, 1, 0])


def test_the_displacement_is_expressed_in_the_observed_frame():
    # Observed facing +90 degrees about z; a target one metre along world +x
    # is one metre along the body -y.
    state = export(arm([0, 0, 0], QUARTER_Z, 0.0), arm([0, 0, 0], IDENTITY, 0.0))
    target = export(arm([1, 0, 0], QUARTER_Z, 0.0), arm([1, 0, 0], IDENTITY, 0.0))
    action = poses.relative_action(state, target)
    assert np.allclose(action[0:3], [0, -1, 0], atol=1e-9)   # rotated into the body frame
    assert np.allclose(action[10:13], [1, 0, 0], atol=1e-9)  # the arm at rest sees world +x


def test_the_orientation_delta_is_left_multiplied_by_the_inverse_state():
    # R_delta = R_stateᵀ R_target, not R_target R_stateᵀ. The two differ unless
    # the rotations commute, so the test uses axes that do not.
    state = export(arm([0, 0, 0], QUARTER_Z, 0.0), arm([0, 0, 0], IDENTITY, 0.0))
    quarter_x = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    target = export(arm([0, 0, 0], quarter_x, 0.0), arm([0, 0, 0], IDENTITY, 0.0))
    r_state = poses.rotation_from_quaternion(QUARTER_Z)
    r_target = poses.rotation_from_quaternion(quarter_x)
    got = poses.relative_action(state, target)[3:9]
    assert np.allclose(got, poses.rotation_to_6d(r_state.T @ r_target), atol=1e-9)
    assert not np.allclose(got, poses.rotation_to_6d(r_target @ r_state.T), atol=1e-3)


def test_the_gripper_is_absolute_not_relative():
    state = export(arm([0, 0, 0], IDENTITY, 1.0), arm([0, 0, 0], IDENTITY, 1.0))
    target = export(arm([0, 0, 0], IDENTITY, 0.25), arm([0, 0, 0], IDENTITY, 0.75))
    action = poses.relative_action(state, target)
    assert action[9] == 0.25 and action[19] == 0.75


def test_a_relative_action_is_invariant_to_where_the_pair_sits_in_the_world():
    offset = np.array([3.0, -2.0, 1.0])
    state = export(arm([0, 0, 0], QUARTER_Z, 0.0), arm([0, 0, 0], IDENTITY, 0.0))
    target = export(arm([0.1, 0.2, 0.3], QUARTER_Z, 0.0), arm([0, 0, 0], IDENTITY, 0.0))
    moved_state = export(arm(offset, QUARTER_Z, 0.0), arm(offset, IDENTITY, 0.0))
    moved_target = export(arm(offset + [0.1, 0.2, 0.3], QUARTER_Z, 0.0),
                          arm(offset, IDENTITY, 0.0))
    assert np.allclose(poses.relative_action(state, target),
                       poses.relative_action(moved_state, moved_target))


# --- chunks ----------------------------------------------------------------


def test_a_chunk_step_targets_the_command_one_frame_later():
    # The alignment that matters: step 0 targets commands[1], not commands[0].
    state = at_rest()
    commands = [export(arm([float(i), 0, 0], IDENTITY, 0.0), arm([0, 0, 0], IDENTITY, 0.0))
                for i in range(4)]
    chunk, valid = poses.action_chunk(state, commands, horizon=3)
    assert valid.all()
    assert [round(float(chunk[k][0]) - (-0.1), 6) for k in range(3)] == [1.0, 2.0, 3.0]


def test_a_chunk_past_the_episode_end_repeats_and_marks_the_rest_invalid():
    state = at_rest()
    commands = [export(arm([float(i), 0, 0], IDENTITY, 0.0), arm([0, 0, 0], IDENTITY, 0.0))
                for i in range(3)]
    chunk, valid = poses.action_chunk(state, commands, horizon=5)
    assert valid.tolist() == [True, True, False, False, False]
    assert np.allclose(chunk[2], chunk[1]) and np.allclose(chunk[4], chunk[1])


def test_a_chunk_needs_at_least_one_command_after_the_anchor():
    with pytest.raises(ValueError, match="at least one command"):
        poses.action_chunk(at_rest(), [at_rest()], horizon=4)


def test_chunk_width_is_both_arms():
    chunk, _ = poses.action_chunk(at_rest(), [at_rest(), at_rest()], horizon=2)
    assert chunk.shape == (2, poses.ARMS * poses.MODEL_ARM_WIDTH) == (2, 20)
