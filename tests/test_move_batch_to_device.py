"""Moving a batch to the device, whatever shape of mapping it arrives in."""

import torch
from transformers.feature_extraction_utils import BatchFeature

from wall_x.trainer.utils.data import move_batch_to_device

CPU = torch.device("cpu")


def test_a_plain_dict_is_moved():
    moved = move_batch_to_device({"x": torch.zeros(2)}, CPU)
    assert moved["x"].device == CPU


def test_a_processor_batch_is_moved_too():
    # A processor returns a BatchFeature, which is a UserDict: it fails an
    # isinstance(x, dict) test, and a batch that is not recognised here is
    # passed through on the host, where the first embedding lookup fails.
    batch = BatchFeature({"input_ids": torch.zeros(2, dtype=torch.long)})
    moved = move_batch_to_device(batch, CPU)
    assert set(moved) == {"input_ids"}
    assert moved["input_ids"].device == CPU


def test_nested_containers_are_followed():
    batch = BatchFeature({"images": [torch.zeros(1), torch.zeros(1)],
                          "nested": {"x": torch.zeros(1)}})
    moved = move_batch_to_device(batch, CPU)
    assert isinstance(moved["images"], list)
    assert moved["nested"]["x"].device == CPU


def test_values_that_are_not_tensors_are_left_alone():
    batch = BatchFeature({"dataset_names": ["robodojo"], "steps": 4})
    moved = move_batch_to_device(batch, CPU)
    assert moved["dataset_names"] == ["robodojo"]
    assert moved["steps"] == 4


def test_the_result_can_be_splatted_into_a_forward():
    batch = BatchFeature({"input_ids": torch.zeros(1, dtype=torch.long)})
    moved = move_batch_to_device(batch, CPU)

    def forward(**kwargs):
        return sorted(kwargs)

    assert forward(**moved) == ["input_ids"]
