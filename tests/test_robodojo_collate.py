"""Handing RoboDojo samples to the collator the LeRobot path already uses."""

import numpy as np
import pytest
import torch

from wall_x.data.backends.lerobot.robodojo.collate import RoboDojoCollator, collator_item
from wall_x.data.backends.lerobot.robodojo.dataset import ACTION, SELECTION

WIDTHS = {"propri_width": 20, "action_width": 20, "action_horizon": 4}


def action_sample():
    return {
        "kind": ACTION,
        "frame_index": 7,
        "images": ["an image"],
        "text": "Observation: ...",
        "agent_pos": np.arange(20, dtype=np.float64),
        "action": np.zeros((4, 20)),
    }


def selection_sample():
    return {
        "kind": SELECTION,
        "frame_index": 7,
        "images": ["an image"],
        "text": "Which recent frames ...",
    }


def test_an_action_sample_keeps_its_pose_and_chunk():
    item = collator_item(action_sample(), **WIDTHS)
    assert item["agent_pos"].dtype == torch.float32
    assert item["action"].shape == (4, 20)
    assert torch.equal(item["agent_pos"], torch.arange(20, dtype=torch.float32))


def test_a_selection_sample_arrives_as_nothing_provided():
    # All NaN is how the collator is told a dimension is absent: it zeroes the
    # value and records a zero mask, so no flow loss can come from this sample.
    item = collator_item(selection_sample(), **WIDTHS)
    assert torch.isnan(item["action"]).all()
    assert torch.isnan(item["agent_pos"]).all()
    assert item["action"].shape == (4, 20)


def test_both_kinds_carry_the_keys_the_collator_reads():
    for sample in (action_sample(), selection_sample()):
        item = collator_item(sample, **WIDTHS)
        assert set(item) == {"image_inputs", "text", "agent_pos", "action", "frame_index"}


def test_a_pose_of_the_wrong_width_is_refused():
    sample = action_sample()
    sample["agent_pos"] = np.zeros(14)
    with pytest.raises(ValueError, match="agent_pos_config"):
        collator_item(sample, **WIDTHS)


def test_a_chunk_of_the_wrong_horizon_is_refused():
    sample = action_sample()
    sample["action"] = np.zeros((8, 20))
    with pytest.raises(ValueError, match="action_horizon"):
        collator_item(sample, **WIDTHS)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown sample kind"):
        collator_item({"kind": "something else"}, **WIDTHS)


class Inner:
    """A stand-in ``DataCollator`` that reports the length it produced."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.processor = "the one it would have loaded itself"
        self.seen = None

    def __call__(self, items):
        self.seen = items
        return {"input_ids": torch.zeros((len(items), self.tokens), dtype=torch.long),
                "action_chunk": torch.zeros((len(items), 4, 26)),
                "dof_mask": torch.zeros((len(items), 4, 26)),
                "sample_time": torch.zeros(len(items))}


class Tokenizer:
    """Enough of a tokenizer for the collator: the selector's own vocabulary."""

    padding_side = "left"
    vocabulary = {"<|kf_write|>": 900,
                  **{f"<|R_{i}|>": 910 + i for i in range(4)}}

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.get(token, -1)


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()


def test_the_trainers_processor_replaces_the_one_the_collator_would_load():
    # The selection answer is written with tokens that exist only after the
    # model's vocabulary was extended; a freshly loaded processor splits them
    # into ordinary text and the answer is never learned.
    inner = Inner(tokens=16)
    processor = Processor()
    RoboDojoCollator(inner, processor=processor, max_length=32, padding_side="right",
                     candidates=4, **WIDTHS)
    assert inner.processor is processor
    assert processor.tokenizer.padding_side == "right"


def test_a_batch_that_fits_is_passed_through():
    inner = Inner(tokens=16)
    collate = RoboDojoCollator(inner, processor=None, max_length=32, **WIDTHS)
    out = collate([action_sample(), selection_sample()])
    assert out["input_ids"].shape == (2, 16)
    assert [item["text"] for item in inner.seen] == [
        "Observation: ...", "Which recent frames ..."]


def test_a_truncated_batch_is_refused_rather_than_trained_on():
    inner = Inner(tokens=32)
    collate = RoboDojoCollator(inner, processor=None, max_length=32, **WIDTHS)
    with pytest.raises(ValueError, match="truncated"):
        collate([selection_sample()])


def test_a_selection_only_batch_arrives_without_action_rows():
    # The model pairs action rows with action-token positions; a selection
    # prompt has none, so rows would be rows it cannot place.
    inner = Inner(tokens=16)
    collate = RoboDojoCollator(inner, processor=None, max_length=32, **WIDTHS)
    out = collate([selection_sample(), selection_sample()])
    assert "action_chunk" not in out
    assert "dof_mask" not in out
    assert "sample_time" not in out


def test_a_batch_with_an_action_sample_keeps_them():
    inner = Inner(tokens=16)
    collate = RoboDojoCollator(inner, processor=None, max_length=32, **WIDTHS)
    out = collate([action_sample()])
    assert "action_chunk" in out
