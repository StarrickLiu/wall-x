"""Training the refiner: the candidate filter and the loss."""

import numpy as np
import pytest
import torch

from wall_x.model.core.refiner.train import (
    ARM_WIDTH,
    Deviation,
    Margins,
    load_margins,
    margins_for,
    measure,
    refinement_loss,
    within,
    within_each,
)

STEPS = 4
IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def arm(position=(0.0, 0.0, 0.0), rotation=IDENTITY, gripper=0.0):
    return list(position) + list(rotation) + [gripper]


def chunk(left=None, right=None, steps=STEPS):
    """A ``[1, steps, 20]`` action chunk with both arms the same at every step."""
    left = left or arm()
    right = right or arm()
    row = torch.tensor(left + right, dtype=torch.float32)
    return row.repeat(steps, 1).unsqueeze(0)


def rotated(angle, axis=2):
    """The 6D encoding of a rotation of ``angle`` about one axis."""
    matrix = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    other = [i for i in range(3) if i != axis]
    matrix[other[0], other[0]] = c
    matrix[other[0], other[1]] = -s
    matrix[other[1], other[0]] = s
    matrix[other[1], other[1]] = c
    return matrix[:2].reshape(6).tolist()


# --- measuring the distance to the demonstration ---------------------------


def test_an_identical_candidate_deviates_by_nothing():
    deviation = measure(chunk(), chunk())
    assert float(deviation.position) == 0.0
    assert float(deviation.rotation) == pytest.approx(0.0, abs=1e-6)
    assert int(deviation.gripper_steps) == 0


def test_the_position_deviation_is_a_distance_not_an_axis():
    candidate = chunk(left=arm(position=(0.03, 0.04, 0.0)))
    assert float(measure(candidate, chunk()).position) == pytest.approx(0.05, abs=1e-6)


def test_the_worst_step_and_arm_decide():
    # One bad step on one arm is enough to describe the candidate.
    candidate = chunk()
    candidate[0, 2, ARM_WIDTH:ARM_WIDTH + 3] = torch.tensor([0.0, 0.0, 0.2])
    assert float(measure(candidate, chunk()).position) == pytest.approx(0.2, abs=1e-6)


def test_the_rotation_deviation_is_an_angle():
    candidate = chunk(left=arm(rotation=rotated(0.3)))
    assert float(measure(candidate, chunk()).rotation) == pytest.approx(0.3, abs=1e-5)


def test_a_half_turn_is_the_largest_angle():
    candidate = chunk(left=arm(rotation=rotated(np.pi)))
    assert float(measure(candidate, chunk()).rotation) == pytest.approx(np.pi, abs=1e-4)


def test_the_gripper_counts_the_steps_that_disagree():
    candidate = chunk(left=arm(gripper=0.0))
    demonstration = chunk(left=arm(gripper=1.0))
    assert int(measure(candidate, demonstration).gripper_steps) == STEPS


def test_a_small_gripper_difference_is_not_a_disagreement():
    candidate = chunk(left=arm(gripper=0.0))
    demonstration = chunk(left=arm(gripper=0.2))
    assert int(measure(candidate, demonstration).gripper_steps) == 0


def test_padding_steps_are_not_measured():
    candidate = chunk()
    candidate[0, 3, 0:3] = torch.tensor([9.0, 0.0, 0.0])      # a padded step
    valid = torch.tensor([[True, True, True, False]])
    assert float(measure(candidate, chunk(), valid=valid).position) == 0.0
    assert float(measure(candidate, chunk()).position) == pytest.approx(9.0)


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="but the demonstration is"):
        measure(chunk(steps=4), chunk(steps=5))


# --- which candidates are scored -------------------------------------------


def test_a_close_candidate_is_kept():
    deviation = measure(chunk(left=arm(position=(0.001, 0, 0))), chunk())
    assert bool(within(deviation, Margins(position=0.01, rotation=0.1, gripper_steps=2)))


def test_a_candidate_beyond_any_margin_is_dropped():
    far = measure(chunk(left=arm(position=(0.5, 0, 0))), chunk())
    turned = measure(chunk(left=arm(rotation=rotated(1.0))), chunk())
    opposite = measure(chunk(left=arm(gripper=0.0)), chunk(left=arm(gripper=1.0)))
    margins = Margins(position=0.05, rotation=0.3, gripper_steps=1)
    assert not bool(within(far, margins))
    assert not bool(within(turned, margins))
    assert not bool(within(opposite, margins))


def test_a_margin_left_out_is_not_checked():
    far = measure(chunk(left=arm(position=(9.0, 0, 0))), chunk())
    assert bool(within(far, Margins(position=None, rotation=None, gripper_steps=None)))


def test_margins_are_looked_up_per_task():
    table = {"swap_T": Margins(position=0.01), "cover_blocks": Margins(position=0.02)}
    got = margins_for(["cover_blocks", "swap_T"], table)
    assert [m.position for m in got] == [0.02, 0.01]


def test_an_uncalibrated_task_is_refused_rather_than_left_unfiltered():
    # Training it unfiltered would use the candidates the margins exist to drop.
    with pytest.raises(KeyError, match="calibrate it"):
        margins_for(["make_toast"], {"swap_T": Margins()})
    assert margins_for(["make_toast"], {}, default=Margins(position=0.1))


# --- the loss --------------------------------------------------------------


def test_a_perfect_refinement_costs_nothing():
    assert float(refinement_loss(chunk(), chunk())) == 0.0


def test_the_loss_is_the_mean_absolute_error():
    refined = chunk(left=arm(position=(0.1, 0.0, 0.0)))
    # One of twenty dimensions is off by 0.1, at every step.
    assert float(refinement_loss(refined, chunk())) == pytest.approx(0.1 / 20, abs=1e-6)


def test_dimensions_this_embodiment_lacks_do_not_count():
    refined = chunk()
    demonstration = chunk()
    refined[..., 19] = 5.0                       # a dimension that is padding
    dof_mask = torch.ones_like(refined)
    dof_mask[..., 19] = 0.0
    assert float(refinement_loss(refined, demonstration, dof_mask=dof_mask)) == 0.0
    assert float(refinement_loss(refined, demonstration)) > 0.0


def test_padded_steps_do_not_count():
    refined = chunk()
    refined[0, 3] = 5.0
    valid = torch.tensor([[True, True, True, False]])
    assert float(refinement_loss(refined, chunk(), valid=valid)) == 0.0


def test_a_dropped_candidate_does_not_count():
    refined = torch.cat([chunk(), chunk(left=arm(position=(1.0, 0, 0)))], dim=0)
    demonstration = torch.cat([chunk(), chunk()], dim=0)
    keep = torch.tensor([True, False])
    assert float(refinement_loss(refined, demonstration, keep=keep)) == 0.0


def test_a_batch_with_nothing_left_to_score_still_has_a_gradient_path():
    refined = chunk().requires_grad_(True)
    loss = refinement_loss(refined, chunk(), keep=torch.tensor([False]))
    assert float(loss) == 0.0
    loss.backward()
    assert refined.grad is not None


def test_a_zero_orientation_is_refused_rather_than_mismeasured():
    # Normalizing it would quietly give 120 degrees, enough to drop a candidate.
    candidate = chunk()
    candidate[..., 3:9] = 0.0
    with pytest.raises(ValueError, match="zero length"):
        measure(candidate, chunk())


def test_a_zero_orientation_on_a_padded_step_is_not_refused():
    """Padding has no orientation, and padded steps are not measured.

    Refusing the batch over a step that does not count would stop a run on
    data that is perfectly ordinary -- an episode ending before the horizon.
    """
    candidate = chunk()
    candidate[0, 3, 3:9] = 0.0
    valid = torch.tensor([[True, True, True, False]])
    assert float(measure(candidate, chunk(), valid=valid).rotation) == 0.0


# --- one margin per candidate ----------------------------------------------
def deviation(position, rotation=0.0, gripper_steps=0):
    """A deviation for a batch, from lists or scalars."""
    def column(value):
        return torch.as_tensor(value if isinstance(value, (list, tuple))
                               else [value], dtype=torch.float32)
    return Deviation(position=column(position), rotation=column(rotation),
                     gripper_steps=column(gripper_steps))


def test_each_candidate_is_judged_by_its_own_tasks_margin():
    # A batch mixes tasks, and a task the margins were calibrated loosely for
    # should not drop a candidate that a stricter task would.
    measured = deviation([0.05, 0.05])
    keep = within_each(measured, [Margins(position=0.10), Margins(position=0.01)])
    assert keep.tolist() == [True, False]


def test_a_margin_of_none_constrains_nothing():
    measured = deviation([1e6], rotation=[3.0], gripper_steps=[99])
    assert within_each(measured, [Margins()]).tolist() == [True]


def test_every_margin_has_to_be_met():
    measured = deviation([0.01], rotation=[1.0], gripper_steps=[0])
    margins = [Margins(position=0.10, rotation=0.1, gripper_steps=4)]
    assert within_each(measured, margins).tolist() == [False]


def test_the_gripper_count_is_compared_as_a_count():
    measured = deviation([0.0], gripper_steps=[4])
    assert within_each(measured, [Margins(gripper_steps=4)]).tolist() == [True]
    assert within_each(measured, [Margins(gripper_steps=3)]).tolist() == [False]


def test_a_margin_per_candidate_is_required():
    with pytest.raises(ValueError, match="1 margins for 2 candidates"):
        within_each(deviation([0.0, 0.0]), [Margins()])


# --- reading a calibrated margins file --------------------------------------
def margins_file(tmp_path, tasks, schema="refiner_margins_v1"):
    import json

    path = tmp_path / "margins.json"
    path.write_text(json.dumps({"schema": schema, "keep": 0.99, "tasks": tasks}),
                    encoding="utf-8")
    return str(path)


def test_a_measured_task_keeps_its_three_margins(tmp_path):
    path = margins_file(tmp_path, {
        "Cover the blocks.": {"position": 0.08, "rotation": 0.4, "gripper_steps": 3},
    })
    table, default = load_margins(path)
    assert default is None
    assert table["Cover the blocks."] == Margins(
        position=0.08, rotation=0.4, gripper=0.5, gripper_steps=3)


def test_a_default_entry_becomes_the_fallback(tmp_path):
    path = margins_file(tmp_path, {
        "Sort them.": {"position": 0.05, "rotation": 0.2, "gripper_steps": 1},
        "default": {"position": 0.2, "rotation": 1.0, "gripper_steps": 8},
    })
    table, default = load_margins(path)
    assert set(table) == {"Sort them."}
    assert default.position == 0.2


def test_a_task_with_no_measured_margin_is_an_error_without_a_default(tmp_path):
    table, default = load_margins(margins_file(tmp_path, {"a": {"position": 0.1}}))
    with pytest.raises(KeyError, match="no margins calibrated"):
        margins_for(["b"], table, default)


def test_a_file_of_another_schema_is_refused(tmp_path):
    path = margins_file(tmp_path, {"a": {"position": 0.1}}, schema="something_else")
    with pytest.raises(ValueError, match="refiner_margins_v1"):
        load_margins(path)


def test_a_file_with_no_tasks_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no measured tasks"):
        load_margins(margins_file(tmp_path, {}))


def test_an_absent_margin_stays_unconstrained(tmp_path):
    table, _ = load_margins(margins_file(tmp_path, {"a": {"position": 0.1}}))
    assert table["a"].rotation is None
    assert table["a"].gripper_steps is None
