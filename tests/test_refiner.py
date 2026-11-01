"""Model-free checks of the residual refiner twins and load guards."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from wall_x.model.core.refiner import (
    RefinerConfig,
    install_twins,
    is_twin_key,
    refine,
    seed_twins,
    swapped_in,
    verify_loaded,
)


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 3)
        self.action_proj_back = nn.Linear(3, 2)
        self.normalizer_action = nn.Module()

    def step(self, timestep, noisy_action, dof_mask=None):
        return self.w(noisy_action), timestep

    def forward(self, action_chunk, dataset_names, sample_time, dof_mask=None,
                mode=None):
        # The refiner reaches these weights through forward, so that a sharded
        # parameter is materialized by the wrapper that owns it.
        if mode == "embed":
            return self.step(sample_time, action_chunk, dof_mask)
        if mode == "project":
            return self.action_proj_back(action_chunk)
        raise AssertionError(f"the refiner only embeds and projects, got {mode!r}")


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3, 3)
        self.q_proj_action = nn.Linear(3, 3)
        self.mlp_action_expert = nn.Linear(3, 3)

    def forward(self, x):
        return self.mlp_action_expert(self.q_proj_action(x))


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vla_config=SimpleNamespace(dim_inputs=(3, 3)))
        self.action_processor = Processor()
        self.language_model = nn.ModuleList([Layer(), Layer()])

    def run(self, embeds):
        for layer in self.language_model:
            embeds = layer(embeds)
        return SimpleNamespace(action_embeds=embeds)


def test_install_twins_names_and_identity_head():
    model = Inner()
    created = install_twins(model)
    assert "action_processor_refiner" in created
    assert "language_model.0.q_proj_action_refiner" in created
    assert not any("q_proj_refiner" in c or "normalizer" in c for c in created)
    assert install_twins(model) == []  # idempotent
    assert all(is_twin_key(k) for k in model.state_dict() if "_refiner" in k)
    # zeroed head: an untrained refiner returns the candidate unchanged
    candidate = torch.randn(1, 4, 2)
    out = refine(model, candidate, dof_mask=None, embed_dtype=torch.float32,
                 run_lm=lambda e, a: model.run(e), config=RefinerConfig(True, 10))
    assert torch.equal(out, candidate)


def test_swap_uses_twins_and_restores_on_error():
    model = Inner()
    install_twins(model)
    layer = model.language_model[0]
    original = layer.q_proj_action
    with swapped_in(model) as n:
        assert n == 4 and layer.q_proj_action is layer.q_proj_action_refiner
        with pytest.raises(RuntimeError):
            with swapped_in(model):
                pass
    assert layer.q_proj_action is original
    with pytest.raises(ValueError):
        with swapped_in(model):
            raise ValueError("boom")
    assert layer.q_proj_action is original


def test_verify_loaded_guards():
    on, off = RefinerConfig(True, 10), RefinerConfig()
    model = Inner()
    install_twins(model)
    keys = set(model.state_dict())
    assert verify_loaded(model, keys, [], on) == sum(map(is_twin_key, keys))
    with pytest.raises(RuntimeError, match="disabled"):
        verify_loaded(Inner(), keys, [], off)
    with pytest.raises(RuntimeError, match="no refiner weights"):
        verify_loaded(model, {k for k in keys if not is_twin_key(k)}, [], on)
    missing = ["language_model.0.q_proj_action_refiner.weight"]
    with pytest.raises(RuntimeError, match="not loaded"):
        verify_loaded(model, keys - set(missing), missing, on)
    # a twin of a module the checkpoint never had is not an error
    ghost = ["language_model.0.mlp_action_expert_refiner.bias"]
    assert verify_loaded(
        model, keys - set(ghost) - {"language_model.0.mlp_action_expert.bias"}, ghost, on
    )


def test_config_parsing():
    assert not RefinerConfig.from_vla_config(SimpleNamespace()).enabled
    cfg = RefinerConfig.from_vla_config(
        SimpleNamespace(refiner={"enabled": True, "num_steps": 10, "timestep": 1.0})
    )
    assert cfg == RefinerConfig(True, 10, 1.0)
    with pytest.raises(ValueError):
        RefinerConfig.from_vla_config(SimpleNamespace(refiner={"enabled": True}))


# ----------------------------------------------------------------------
# Loop checkpoints: chained passes with a step scale
# ----------------------------------------------------------------------
def _trained_model(seed: int = 0):
    """Twins installed, output head non-zero so the residual actually moves."""
    torch.manual_seed(seed)
    model = Inner()
    install_twins(model)
    head = model.action_processor_refiner.action_proj_back
    with torch.no_grad():
        head.weight.copy_(torch.randn_like(head.weight))
        head.bias.copy_(torch.randn_like(head.bias))
    return model


def _run(model, candidate, config):
    return refine(
        model,
        candidate,
        dof_mask=None,
        embed_dtype=torch.float32,
        run_lm=lambda embeds, adarms: model.run(embeds),
        config=config,
    )


def test_refine_loops_chain_the_estimate_and_scale_each_step():
    model = _trained_model()
    candidate = torch.randn(2, 4, 2)
    single = RefinerConfig(True, 10, 1.0, refine_loops=1)
    one_pass = _run(model, candidate, single)
    assert not torch.equal(one_pass, candidate)

    calls = []

    def run_lm(embeds, adarms):
        calls.append(1)
        return model.run(embeds)

    loops = RefinerConfig(True, 10, 1.0, refine_loops=2, refine_step_scale=0.5)
    refined = refine(
        model,
        candidate,
        dof_mask=None,
        embed_dtype=torch.float32,
        run_lm=run_lm,
        config=loops,
    )
    assert len(calls) == 2
    # Each pass conditions on the current estimate and contributes 0.5 of its
    # residual: mid = candidate + 0.5*d1, out = mid + 0.5*d2.
    delta1 = one_pass - candidate
    mid = candidate + 0.5 * delta1
    delta2 = _run(model, mid, single) - mid
    assert torch.allclose(refined, mid + 0.5 * delta2, atol=1e-6)


def test_default_loops_are_bitwise_the_single_pass_release():
    model = _trained_model()
    candidate = torch.randn(2, 4, 2)
    out = _run(model, candidate, RefinerConfig(True, 10))
    processor = model.action_processor_refiner
    head = processor.action_proj_back
    with swapped_in(model):
        embeds, _ = processor.step(
            timestep=torch.full((2,), 1.0), noisy_action=candidate, dof_mask=None
        )
        output = model.run(embeds)
    expected = candidate + head(output.action_embeds.to(head.weight.dtype)).float().reshape(
        candidate.shape
    )
    assert torch.equal(out, expected)


def test_refine_config_parses_loops_and_scale():
    cfg = RefinerConfig.from_vla_config(
        SimpleNamespace(
            refiner={
                "enabled": True,
                "num_steps": 10,
                "refine_loops": 4,
                "refine_step_scale": 0.25,
            }
        )
    )
    assert (cfg.refine_loops, cfg.refine_step_scale) == (4, 0.25)
    default = RefinerConfig()
    assert (default.refine_loops, default.refine_step_scale) == (1, 1.0)


@pytest.mark.parametrize("bad", [{"refine_loops": 0}, {"refine_step_scale": 0.0}])
def test_refine_config_rejects_bad_loop_settings(bad):
    with pytest.raises(ValueError):
        RefinerConfig.from_vla_config(
            SimpleNamespace(refiner={"enabled": True, "num_steps": 10, **bad})
        )


# ----------------------------------------------------------------------
# Flow dispatch: RTC requests take the uncached path, the cached path refuses them
# ----------------------------------------------------------------------
def dispatch():
    """A stand-in model carrying only the flow dispatch, built on demand.

    The import is deferred so the twin, config and refinement checks above run
    anywhere; importing the model package needs the transformers version the
    training image pins.
    """
    mixin = pytest.importorskip(
        "wall_x.model.qact.qwen3_5.qwen3_5_inference_mixin",
        reason="importing the model package needs the pinned transformers version")
    Qwen3_5VLAInferenceMixin = mixin.Qwen3_5VLAInferenceMixin

    class _Dispatch(Qwen3_5VLAInferenceMixin):
        def __init__(self):
            self.config = SimpleNamespace(
                _attn_implementation="sdpa",
                vla_config=SimpleNamespace(action_linear_attention=False,
                                           causal_action_attention_mask=False),
            )

        def generate_flow_action_no_cache(self, **kwargs):
            return "no_cache"

    return _Dispatch()


def test_rtc_requests_use_the_uncached_path():
    model = dispatch()
    model.generate_flow_action_with_cache = lambda **kwargs: "with_cache"
    prefix = torch.zeros(1, 4, 2)
    assert model.generate_flow_action() == "with_cache"
    assert model.generate_flow_action(prefix_actions=prefix, inference_delay=0) == "with_cache"
    assert model.generate_flow_action(prefix_actions=prefix, inference_delay=2) == "no_cache"


def test_cached_path_refuses_rtc():
    model = dispatch()
    with pytest.raises(ValueError, match="RTC"):
        type(model).generate_flow_action_with_cache(
            model, input_ids=None, action_horizon=4, action_dim=2,
            prefix_actions=torch.zeros(1, 4, 2), inference_delay=2,
        )


# ----------------------------------------------------------------------
# Starting a refiner off: the twins are built before any checkpoint is loaded
# ----------------------------------------------------------------------
def test_the_twins_are_refilled_from_the_weights_beside_them():
    # Built with the model, the twins hold the construction-time values; a run
    # that starts from a policy checkpoint has to refill them or it trains a
    # correction on top of random features.
    model = Inner()
    install_twins(model)
    with torch.no_grad():                      # the policy arrives afterwards
        for layer in model.language_model:
            layer.q_proj_action.weight.fill_(0.25)
        model.action_processor.w.weight.fill_(0.5)

    assert not torch.equal(model.language_model[0].q_proj_action_refiner.weight,
                           model.language_model[0].q_proj_action.weight)
    filled = seed_twins(model)
    assert filled >= 5
    assert torch.equal(model.language_model[0].q_proj_action_refiner.weight,
                       model.language_model[0].q_proj_action.weight)
    assert torch.equal(model.action_processor_refiner.w.weight,
                       model.action_processor.w.weight)


def test_refilling_leaves_the_output_head_at_zero():
    # A refiner that starts by changing nothing is a refiner that starts from
    # the policy's own behaviour.
    model = Inner()
    install_twins(model)
    with torch.no_grad():
        model.action_processor.action_proj_back.weight.fill_(0.3)
        model.action_processor.action_proj_back.bias.fill_(0.7)
    seed_twins(model)
    head = model.action_processor_refiner.action_proj_back
    assert float(head.weight.abs().sum()) == 0.0
    assert float(head.bias.abs().sum()) == 0.0


def test_refilling_a_model_without_twins_does_nothing():
    assert seed_twins(Inner()) == 0
