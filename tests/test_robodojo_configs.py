"""The RoboDojo example configs, against the code that reads them.

A config and the code can drift apart silently -- a renamed key is simply not
read, and training proceeds with a default nobody chose. These tests read the
shipped configs through the same types the pipeline uses.
"""

from pathlib import Path

import pytest
import yaml

from wall_x.data.backends.lerobot.robodojo.frames import Geometry
from wall_x.data.state_noise import StateNoise
from wall_x.model.core.refiner import RefinerConfig

BASE_MODEL = "x-square-robot/wall-oss-1.0-lite-robodojo-base"

CONFIGS = sorted((Path(__file__).resolve().parents[1]
                  / "workspace" / "example" / "robodojo").glob("*.yml"))


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def geometry_of(config):
    data = config["data"]
    memory = data["memory"]
    return Geometry(
        capacity=memory["capacity"], window=memory["window"],
        candidates=memory["candidates"], short_term=memory["short_term"],
        short_term_stride=memory["short_term_stride"],
        resolution=data["resolution"]["face_view"], factor=data["image_factor"],
        min_pixels=data["min_pixels"], max_pixels=data["max_pixels"],
        box_crop=memory["box_crop"], box_jitter=memory["box_jitter"])


def test_both_stages_are_shipped():
    assert [path.name for path in CONFIGS] == ["stage1_memory.yml", "stage2_refiner.yml"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_the_config_parses_into_what_the_code_reads(path):
    config = load(path)
    geometry = geometry_of(config)
    assert geometry.image_size(480, 640) == (336, 448)
    StateNoise.from_config(config["data"].get("state_noise"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_the_action_layout_leaves_room_for_the_dimensions_it_pads(path):
    config = load(path)["task"]
    # The embodiment provides 20; the model's vector is 26, the rest padded.
    provided = sum(v for k, v in config["dof_config"].items() if k != "action_padding")
    assert provided == 20
    assert sum(config["dof_config"].values()) == 26
    assert sum(config["agent_pos_config"].values()) == 20


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_there_is_one_slot_token_per_candidate(path):
    # One short, and the selector cannot name the last candidate; one too many,
    # and it can answer with a slot that does not exist.
    config = load(path)
    slots = [t for t in config["model"]["new_special_tokens"] if t.startswith("<|R_")]
    assert len(slots) == config["data"]["memory"]["candidates"]
    assert slots == [f"<|R_{i}|>" for i in range(len(slots))]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_the_selector_can_say_write_and_skip(path):
    tokens = load(path)["model"]["new_special_tokens"]
    assert {"<|kf_write|>", "<|kf_skip|>", "<|kf_end|>"} <= set(tokens)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_no_config_points_at_weights_nobody_can_fetch(path):
    # Naming a checkpoint that is not published would make the recipe
    # unrunnable. The published base model and this recipe's own stage-1
    # output are the only two starting points these configs may name.
    pretrained = str(load(path)["model"]["pretrained_path"])
    assert pretrained == BASE_MODEL or pretrained.startswith("./checkpoints/")


def test_stage_one_learns_to_write_its_own_memory():
    config = load(CONFIGS[0])
    assert config["data"]["robodojo"]["selection"] is True
    # The selection answer is text, so it needs the autoregressive loss.
    assert config["model"]["ar_loss_weight"] > 0


def test_stage_one_starts_from_the_published_base_model():
    config = load(CONFIGS[0])
    assert config["model"]["pretrained_path"] == BASE_MODEL
    # That checkpoint carries weights and statistics, not a vocabulary: the
    # tokenizer and the image processor come from the VLM it was built on.
    assert config["model"]["processor_path"].startswith("Qwen/")


def test_stage_two_refines_a_policy_rather_than_the_public_model():
    # Refining the public VLM would be refining nothing: there is no policy yet.
    config = load(CONFIGS[1])
    assert config["model"]["pretrained_path"].startswith("./checkpoints/")
    assert config["model"]["refiner"]["enabled"] is True
    assert config["model"]["refiner"]["freeze_base"] is True
    assert config["data"]["robodojo"]["selection"] is False


def test_the_refiner_passes_sum_to_one_residual():
    refiner = load(CONFIGS[1])["model"]["refiner"]
    parsed = RefinerConfig.from_vla_config(type("V", (), {"refiner": refiner})())
    assert parsed.refine_loops * parsed.refine_step_scale == pytest.approx(1.0)


def test_the_refiner_declares_the_budget_its_candidates_were_drawn_with():
    # Serving with another budget is refused, so the config has to say it.
    assert load(CONFIGS[1])["model"]["refiner"]["num_steps"] >= 1


def test_stage_one_leaves_the_refiner_alone():
    # Enabling it would build a second action branch with nothing to correct.
    assert "refiner" not in load(CONFIGS[0])["model"]


def test_the_refiner_section_reaches_the_model_that_reads_it():
    """The config has to change the model, not just parse.

    ``vla_config.refiner`` is what decides whether the twin action branch is
    built at all, and it is read from the model json. A refiner section in a
    training config that never reaches there would leave the run training a
    policy it believes it is refining.
    """
    import logging

    from wall_x.config.loader import load_config
    from wall_x.trainer.adapters import ADAPTER_IMPORT_ERRORS, ADAPTER_REGISTRY

    config = load_config(str(CONFIGS[1]))
    if config.model_type not in ADAPTER_REGISTRY:
        pytest.skip(f"adapter unavailable here: "
                    f"{ADAPTER_IMPORT_ERRORS.get(config.model_type)}")
    config.model.config_path = "workspace/models_config/qwen3_5_2B_mot.json"
    adapter = ADAPTER_REGISTRY[config.model_type](
        cfg=config, logger=logging.getLogger(__name__))
    parsed = RefinerConfig.from_vla_config(adapter.build_model_config().vla_config)
    assert parsed.enabled is True
    assert parsed.refine_loops == 4
    assert parsed.num_steps == config.model.refiner["num_steps"]


def test_neither_stage_carries_statistics_of_its_own():
    """Statistics come with the weights, so no config names a file of them.

    The refiner corrects a policy trained through one scale, and stage 1 trains
    a base model fitted through one. A run that measured its own would hand
    those weights different numbers, which no loss would report as wrong.
    """
    for path in CONFIGS:
        assert load(path)["data"].get("normalizer_config") is None


def test_both_stages_see_the_same_memory():
    first, second = (load(path)["data"]["memory"] for path in CONFIGS)
    assert first == second


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_the_margins_are_not_shipped_as_constants(path):
    # They describe the policy being refined, so they are calibrated, not given.
    config = load(path)
    margins = config["data"].get("refiner", {}).get("margins")
    assert margins is None or str(margins).endswith(".json")


def test_the_configs_load_through_the_trainers_own_loader():
    """The loader drops keys it does not know, so this is the real check.

    A section the typed config does not declare is silently ignored, and
    training proceeds with a default nobody chose -- reading the yaml directly
    cannot see that happen. Loading through ``load_config`` can.
    """
    from wall_x.config.loader import load_config

    for path in CONFIGS:
        config = load_config(str(path))
        assert config.data.dataset_type == "robodojo"
        # Every section the configs set has to survive the round trip.
        assert config.data.robodojo["root"]
        assert config.data.memory["capacity"] == 22
        assert config.data.max_length > 4096
        assert config.task.action_horizon == 32


def test_the_backend_the_configs_name_is_registered():
    # Naming a backend nobody registered fails at startup, after the model has
    # been built; this fails here instead.
    from wall_x.data import available_backends

    assert "robodojo" in available_backends()


def test_the_token_limit_leaves_room_for_the_longest_prompt():
    # A memory of 22 frames plus 16 candidates at 448 is the longest prompt the
    # configs can produce; each image is about 256 tokens after merging.
    for path in CONFIGS:
        memory = load(path)["data"]["memory"]
        images = memory["capacity"] + memory["candidates"] + 3
        assert load(path)["data"]["max_length"] > images * 256 * 2


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_no_config_asks_for_an_attention_kernel_this_package_lacks(path):
    # The published run used a masked-attention kernel that is not one of this
    # package's dependencies, so a config naming it cannot be installed and
    # run. Attention implementations that ship with the dependencies only.
    root = Path(__file__).resolve().parents[1]
    declared = (root / "requirements.txt").read_text(encoding="utf-8")
    attention = load(path)["model"]["attn_implementation"]
    assert attention in {"sdpa", "eager"} or attention in declared


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_muon_and_adamw_run_at_the_same_rate(path):
    """This recipe gives Muon the same rate as AdamW, which is unusual.

    Muon is normally run orders of magnitude higher, so the natural thing to
    write here is a rate this recipe was not trained at -- and nothing about a
    run at the wrong rate looks wrong until the numbers come out.
    """
    optimizer = load(path)["hyperparams"]["optimizer"]
    assert float(optimizer["muon_lr"]) == float(optimizer["adamw_lr"])
    assert float(optimizer["learning_rate"]) == float(optimizer["muon_lr"])


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_the_floor_of_the_schedule_is_a_tenth_of_its_rate(path):
    config = load(path)["hyperparams"]
    assert (float(config["scheduler"]["min_lr"])
            == pytest.approx(float(config["optimizer"]["learning_rate"]) / 10))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_both_stages_train_with_the_optimizer_this_repository_defaults_to(path):
    # The install instructions name DMuon as what the default configs use, and
    # the backend and the optimizer have to agree: the loader rejects every
    # other pairing.
    config = load(path)
    assert config["distributed"]["backend"] == "dmuon"
    assert config["hyperparams"]["optimizer"]["optimizer_type"] == "dmuon"
