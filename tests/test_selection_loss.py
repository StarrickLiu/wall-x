"""Scoring a selection answer against every answer that is right.

Three pieces: the data side finds the slots an answer writes and turns each
one's accepted interval into token ids; the model side replaces the per-token
cross-entropy at exactly those positions.
"""

import math

import pytest
import torch

from wall_x.model.core.selection_loss import (
    NO_TOKEN,
    accepted_token_ids,
    answer_slots,
    interval_cross_entropy,
)
from wall_x.model.core.vla_mixin import _replace_interval_scored

WRITE = 900
SLOTS = (910, 911, 912, 913)        # <|R_0|> .. <|R_3|>
SKIP = 901
END = 902


# --- finding the answer's slots --------------------------------------------
def test_only_the_slots_after_the_decision_count():
    # The prompt names every candidate before it asks the question, so the
    # slots before the decision token are not the answer.
    ids = [5, 910, 911, 912, 913, 7, WRITE, 912, END]
    assert answer_slots(ids, WRITE, SLOTS) == (7,)


def test_several_written_slots_come_back_in_order():
    ids = [910, 911, WRITE, 913, 4, 910, END]
    assert answer_slots(ids, WRITE, SLOTS) == (3, 5)


def test_an_answer_that_skips_writes_nothing():
    assert answer_slots([910, 911, SKIP, END], WRITE, SLOTS) == ()


def test_a_prompt_without_the_decision_token_writes_nothing():
    assert answer_slots([1, 2, 3], WRITE, SLOTS) == ()


def test_tokens_that_are_not_slots_are_ignored():
    ids = [WRITE, 912, 40, 41, 913, END]
    assert answer_slots(ids, WRITE, SLOTS) == (1, 4)


# --- the accepted ids ------------------------------------------------------
def test_an_exact_annotation_accepts_one_token():
    ids = accepted_token_ids(SLOTS, [(2, 2)])
    assert ids.tolist() == [[912]]


def test_an_interval_accepts_every_slot_inside_it():
    ids = accepted_token_ids(SLOTS, [(1, 3)])
    assert ids.tolist() == [[911, 912, 913]]


def test_rows_are_padded_to_the_widest_interval():
    ids = accepted_token_ids(SLOTS, [(0, 0), (1, 3)])
    assert ids.tolist() == [[910, NO_TOKEN, NO_TOKEN], [911, 912, 913]]


@pytest.mark.parametrize("bad", [(-1, 0), (0, 4), (3, 2), (4, 4)])
def test_a_range_outside_the_candidates_is_refused(bad):
    with pytest.raises(ValueError, match="outside the 4 candidates"):
        accepted_token_ids(SLOTS, [bad])


# --- the loss --------------------------------------------------------------
def logits_for(probabilities):
    return torch.log(torch.tensor([probabilities], dtype=torch.float32))


def test_one_accepted_token_is_plain_cross_entropy():
    logits = logits_for([0.1, 0.2, 0.3, 0.4])
    accepted = torch.tensor([[2]])
    assert float(interval_cross_entropy(logits, accepted)) == pytest.approx(
        -math.log(0.3), abs=1e-6)


def test_accepting_more_can_only_lower_the_loss():
    logits = logits_for([0.1, 0.2, 0.3, 0.4])
    one = float(interval_cross_entropy(logits, torch.tensor([[2]])))
    two = float(interval_cross_entropy(logits, torch.tensor([[2, 3]])))
    assert two < one
    assert two == pytest.approx(-math.log(0.3 + 0.4), abs=1e-6)


def test_padding_is_not_accepted():
    logits = logits_for([0.1, 0.2, 0.3, 0.4])
    padded = torch.tensor([[2, NO_TOKEN, NO_TOKEN]])
    assert float(interval_cross_entropy(logits, padded)) == pytest.approx(
        -math.log(0.3), abs=1e-6)


def test_accepting_everything_costs_nothing():
    logits = logits_for([0.1, 0.2, 0.3, 0.4])
    whole = torch.tensor([[0, 1, 2, 3]])
    assert float(interval_cross_entropy(logits, whole)) == pytest.approx(0.0, abs=1e-6)


def test_a_row_that_accepts_nothing_is_refused():
    with pytest.raises(ValueError, match="accepts no token"):
        interval_cross_entropy(logits_for([0.5, 0.5]), torch.tensor([[NO_TOKEN]]))


def test_a_mismatched_number_of_rows_is_refused():
    with pytest.raises(ValueError, match="rows of logits"):
        interval_cross_entropy(torch.zeros(2, 4), torch.tensor([[1]]))


# --- the substitution in the loss ------------------------------------------
def substitution(marks, accepted, per_token, logits):
    """``_replace_interval_scored`` with the shift and mask the trainer uses."""
    mask = torch.ones((1, marks.shape[1] - 1), dtype=torch.bool)
    return _replace_interval_scored(per_token, logits, mask, marks, accepted)


def test_only_the_marked_positions_change():
    logits = torch.log(torch.tensor(
        [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]))
    per_token = torch.tensor([1.0, 2.0, 3.0])
    # labels [x, a, b, c]: after the shift, rows 0..2 are positions 1..3.
    marks = torch.tensor([[NO_TOKEN, NO_TOKEN, 0, NO_TOKEN]])
    accepted = torch.tensor([[2, 3]])
    out = substitution(marks, accepted, per_token, logits)
    assert float(out[0]) == 1.0 and float(out[2]) == 3.0      # untouched
    assert float(out[1]) == pytest.approx(-math.log(0.3 + 0.4), abs=1e-6)


def test_nothing_marked_leaves_the_loss_alone():
    per_token = torch.tensor([1.0, 2.0])
    marks = torch.full((1, 3), NO_TOKEN)
    out = substitution(marks, torch.tensor([[0]]), per_token, torch.zeros(2, 4))
    assert torch.equal(out, per_token)


def test_a_batch_without_the_marks_at_all_is_untouched():
    per_token = torch.tensor([1.0, 2.0])
    assert torch.equal(
        _replace_interval_scored(per_token, torch.zeros(2, 4),
                                 torch.ones((1, 2), dtype=torch.bool), None, None),
        per_token)


def test_the_substitution_keeps_the_gradient_path():
    logits = torch.zeros(1, 4, requires_grad=True)
    per_token = torch.tensor([5.0])
    marks = torch.tensor([[NO_TOKEN, 0]])
    out = substitution(marks, torch.tensor([[1, 2]]), per_token, logits)
    out.sum().backward()
    assert logits.grad is not None
