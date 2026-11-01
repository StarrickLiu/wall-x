"""The refiner's training step, against a stand-in model.

The step is where the pieces meet: it samples from the base model, keeps the
closure that sampled it, scores a correction, and drops the candidates the
margins reject. The model here is a toy one -- what is under test is that
wiring, not the decoder.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from wall_x.model.core.action.flow_sample import CachedFlowSample
from wall_x.model.core.refiner import install_twins
from wall_x.model.core.refiner.step import RefinerTrainingMixin
from wall_x.model.core.refiner.train import Margins

HORIZON, WIDTH = 4, 26
IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def chunk(position=0.0, batch=1):
    """An action chunk with a real orientation, so rotations can be compared."""
    out = torch.zeros(batch, HORIZON, WIDTH)
    for arm in (0, 10):
        out[..., arm + 3:arm + 9] = torch.tensor(IDENTITY)
        out[..., arm] = position
    return out


class Normalizer(nn.Module):
    """A normalizer that does nothing, so the units stay readable."""

    def unnormalize_data(self, values, dataset_names=None, dof_mask=None):
        return values

    def normalize_data(self, values, dataset_names=None):
        return values


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(WIDTH, 8)
        self.action_proj_back = nn.Linear(8, WIDTH)
        self.normalizer_action = Normalizer()

    last_input = None

    def step(self, timestep, noisy_action, dof_mask=None):
        Processor.last_input = noisy_action.detach().clone()
        return self.w(noisy_action), None

    def forward(self, action_chunk, dataset_names, sample_time, dof_mask=None,
                mode=None):
        if mode == "embed":
            return self.step(sample_time, action_chunk, dof_mask)
        if mode == "project":
            return self.action_proj_back(action_chunk)
        raise AssertionError(f"the refiner only embeds and projects, got {mode!r}")


class Inner(nn.Module):
    """The module that owns the action branch, as the refiner expects it."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vla_config=SimpleNamespace(dim_inputs=(8, 8)))
        self.action_processor = Processor()
        self.body_action = nn.Linear(8, 8)


class Model(RefinerTrainingMixin, nn.Module):
    """A stand-in for the model: it samples a fixed candidate and refines it."""

    def __init__(self, candidate, refiner=None):
        super().__init__()
        self.model = Inner()
        self.config = SimpleNamespace(vla_config=SimpleNamespace(
            refiner=refiner or {"enabled": True, "num_steps": 10,
                                "refine_loops": 1, "refine_step_scale": 1.0}))
        install_twins(self.model)
        self._candidate = candidate
        self.sampled_in_eval = None
        self.sampled_with = None

    @property
    def action_preprocessor(self):
        return self.model.action_processor

    def sample_flow_with_cache(self, **kwargs):
        self.sampled_in_eval = not self.training
        self.sampled_with = kwargs
        return CachedFlowSample(
            candidate=self._candidate,
            run_lm=lambda embeds, conds: SimpleNamespace(action_embeds=embeds),
            embed_dtype=torch.float32,
            dof_mask=kwargs.get("dof_mask"),
            num_steps=kwargs.get("num_inference_timesteps"),
        )


def model_for(candidate, **kwargs):
    return Model(candidate, **kwargs)


def test_a_step_scores_a_correction_and_carries_a_gradient():
    model = model_for(chunk(position=0.1))
    output = model.refiner_train_forward(action_chunk=chunk(position=0.0))
    assert output.loss.requires_grad
    output.loss.backward()
    head = model.model.action_processor_refiner.action_proj_back
    assert head.weight.grad is not None


def test_the_base_model_is_not_asked_for_a_gradient():
    # The refiner corrects the policy; a gradient reaching the policy would
    # mean it is being changed instead.
    model = model_for(chunk(position=0.1))
    output = model.refiner_train_forward(action_chunk=chunk(position=0.0))
    output.loss.backward()
    assert model.model.action_processor.w.weight.grad is None
    assert model.model.body_action.weight.grad is None


def test_the_candidate_is_sampled_with_the_budget_the_config_declares():
    # The refiner learns the residual of a particular candidate distribution.
    model = model_for(chunk(), refiner={"enabled": True, "num_steps": 7,
                                        "refine_loops": 1, "refine_step_scale": 1.0})
    model.refiner_train_forward(action_chunk=chunk())
    assert model.sampled_with["num_inference_timesteps"] == 7


def test_the_sample_is_drawn_outside_training_mode():
    # Gradient checkpointing refuses to build a KV cache while training, and
    # without the cache the postfix attends to nothing.
    model = model_for(chunk())
    model.train()
    model.refiner_train_forward(action_chunk=chunk())
    assert model.sampled_in_eval is True
    assert model.training is True          # and the mode is put back


def test_training_mode_is_restored_even_when_the_step_fails():
    model = model_for(chunk())
    model.train()
    with pytest.raises(ValueError):
        model.refiner_train_forward(action_chunk=chunk(), refiner_margins={},
                                    task_names=None)
    assert model.training is True


def test_a_batch_with_no_demonstration_is_refused():
    model = model_for(chunk())
    with pytest.raises(ValueError, match="Selection samples"):
        model.refiner_train_forward(action_chunk=None)


def test_a_model_without_the_refiner_enabled_is_refused():
    model = model_for(chunk(), refiner={"enabled": False})
    with pytest.raises(ValueError, match="no twin weights"):
        model.refiner_train_forward(action_chunk=chunk())


def test_a_candidate_beyond_its_margin_is_dropped():
    # Scoring a sample the base model deliberately put elsewhere teaches the
    # refiner to undo a decision.
    model = model_for(chunk(position=1.0))
    output = model.refiner_train_forward(
        action_chunk=chunk(position=0.0),
        refiner_margins={"far": Margins(position=0.01)},
        task_names=["far"])
    assert output.kept == 0
    assert output.dropped == 1
    assert float(output.loss.detach()) == 0.0   # and it still has a gradient path
    assert output.loss.requires_grad


def test_a_candidate_within_its_margin_is_kept():
    model = model_for(chunk(position=0.005))
    output = model.refiner_train_forward(
        action_chunk=chunk(position=0.0),
        refiner_margins={"near": Margins(position=0.01)},
        task_names=["near"])
    assert output.kept == 1


def test_margins_without_task_names_are_refused():
    model = model_for(chunk())
    with pytest.raises(ValueError, match="which task"):
        model.refiner_train_forward(action_chunk=chunk(), refiner_margins={},
                                    task_names=None)


def test_one_task_name_per_sample_is_required():
    model = model_for(chunk(batch=2))
    with pytest.raises(ValueError, match="1 task names for 2 samples"):
        model.refiner_train_forward(action_chunk=chunk(batch=2),
                                    refiner_margins={"a": Margins()},
                                    task_names=["a"])


def test_without_margins_every_candidate_is_scored():
    model = model_for(chunk(position=99.0))
    output = model.refiner_train_forward(action_chunk=chunk(position=0.0))
    assert output.kept == 1
    assert output.dropped == 0


def test_padding_steps_are_not_scored():
    # A step the episode did not have is not a correction anyone can learn.
    model = model_for(chunk())
    mask = torch.ones(1, HORIZON, WIDTH)
    mask[:, 2:] = 0.0
    output = model.refiner_train_forward(action_chunk=chunk(), dof_mask=mask)
    assert output.loss.requires_grad


def test_the_step_reports_how_far_it_moved_the_candidate():
    model = model_for(chunk(position=0.1))
    output = model.refiner_train_forward(action_chunk=chunk())
    assert output.correction >= 0.0


def test_only_the_last_of_several_passes_is_trained():
    """Chained passes share weights, so the chain is not backpropagated.

    The optimizer that owns these parameters reshards a layer after its first
    backward edge, so a layer used four times in one graph meets an empty
    parameter on the second. The last pass is also the one that matters: it is
    the correction the refiner applies to an estimate it produced itself.
    """
    refiner = {"enabled": True, "num_steps": 10, "refine_loops": 4,
               "refine_step_scale": 0.25}
    model = model_for(chunk(position=0.1), refiner=refiner)
    output = model.refiner_train_forward(action_chunk=chunk())
    output.loss.backward()
    head = model.model.action_processor_refiner.action_proj_back
    assert head.weight.grad is not None

    # One pass through the projection, not four: the graph does not grow.
    passes = []
    original = type(model.model.action_processor_refiner).forward

    def counting(self, *args, **kwargs):
        passes.append(torch.is_grad_enabled())
        return original(self, *args, **kwargs)

    type(model.model.action_processor_refiner).forward = counting
    try:
        model.refiner_train_forward(action_chunk=chunk())
    finally:
        type(model.model.action_processor_refiner).forward = original
    # Two calls per pass (embed, project); grad on for the last pass only.
    assert len(passes) == 8
    assert passes.count(True) == 2
    assert passes.count(False) == 6


def test_a_single_pass_still_carries_its_gradient():
    refiner = {"enabled": True, "num_steps": 10, "refine_loops": 1,
               "refine_step_scale": 1.0}
    model = model_for(chunk(position=0.1), refiner=refiner)
    output = model.refiner_train_forward(action_chunk=chunk())
    output.loss.backward()
    assert model.model.action_processor_refiner.action_proj_back.weight.grad is not None


def test_dimensions_this_embodiment_lacks_are_not_shown_to_the_refiner():
    # The sampler drives the padded dimensions towards a constant but does not
    # guarantee one; the refiner should not be conditioned on the remainder.
    candidate = chunk(position=0.1)
    candidate[..., 19] = 7.0                 # a dimension the mask excludes
    model = model_for(candidate)
    mask = torch.ones(1, HORIZON, WIDTH)
    mask[..., 19] = 0.0
    model.refiner_train_forward(action_chunk=chunk(), dof_mask=mask)
    shown = model.model.action_processor_refiner.last_input
    assert float(shown[..., 19].abs().max()) == 0.0
    assert float(shown[..., 0].abs().max()) > 0.0
