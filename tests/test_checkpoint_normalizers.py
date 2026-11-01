"""Where a run's normalization statistics come from.

The statistics are not a preference: the weights were fitted reading them, and
the same weights read a different scale if the numbers change under them. So a
run inherits them from the checkpoint it starts from, and a config that names a
file of its own overrides that deliberately.
"""

from types import SimpleNamespace

import pytest
import torch

from wall_x.model.core.action.normalizer import (
    load_checkpoint_normalizers,
    normalizers_from_config,
)


def checkpoint(path, action_width=26, propri_width=20, key="ex_normal"):
    """A checkpoint directory that ships a normalizer pair, as training saves."""
    for name, width in (("normalizer_action.pth", action_width),
                        ("normalizer_propri.pth", propri_width)):
        torch.save({f"min.{key}": torch.full((width,), -1.0),
                    f"delta.{key}": torch.full((width,), 2.0)}, path / name)
    return path


def config(pretrained_path, normalizer_config=None):
    return SimpleNamespace(
        model=SimpleNamespace(pretrained_path=str(pretrained_path)),
        data=SimpleNamespace(normalizer_config=normalizer_config),
        task=SimpleNamespace(dof_config={"a": 26}, agent_pos_config={"a": 20}),
    )


def test_a_checkpoints_own_pair_is_what_a_run_reads(tmp_path):
    action, propri = load_checkpoint_normalizers(checkpoint(tmp_path))
    assert list(action.min.keys()) == ["ex_normal"]
    assert action.min["ex_normal"].shape == (26,)
    assert propri.delta["ex_normal"].shape == (20,)


def test_a_checkpoint_without_them_is_not_an_error(tmp_path):
    # A plain VLM ships no statistics, which is a fact about it, not a failure.
    assert load_checkpoint_normalizers(tmp_path) is None


def test_the_statistics_follow_the_weights_by_default(tmp_path):
    action, _, statistics = normalizers_from_config(config(checkpoint(tmp_path)))
    assert float(action.min["ex_normal"][0]) == -1.0
    # Nothing was read from a file of per-dof statistics, so there are none.
    assert statistics is None


def test_another_pair_can_be_named_instead(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    checkpoint(elsewhere, action_width=26, propri_width=20, key="measured_here")
    action, _, _ = normalizers_from_config(
        config(tmp_path, {"from_checkpoint": str(elsewhere)}))
    assert list(action.min.keys()) == ["measured_here"]


def test_a_run_with_no_statistics_anywhere_is_refused(tmp_path):
    # Training would otherwise scale actions by numbers nobody chose.
    with pytest.raises(ValueError, match="requires action statistics"):
        normalizers_from_config(config(tmp_path))
