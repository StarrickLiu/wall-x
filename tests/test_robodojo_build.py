"""Assembling the pipeline from the shipped configs.

Built against the real typed config loader and the real sample index, with the
export reader and the inner collator stood in for -- those need a downloaded
dataset and a downloaded processor. What is under test is the wiring: which
widths, which statistics, which embodiment key, which loader.
"""

import json
from importlib import import_module

import numpy as np
import pytest
import torch

from wall_x.config.loader import load_config
from wall_x.data._protocol import BuildContext
# The package re-exports ``build`` as a name, which shadows the submodule,
# so reach the module itself rather than the function.
build_module = import_module("wall_x.data.backends.lerobot.robodojo.build")
from wall_x.data.backends.lerobot.robodojo.dataset import Episode
from wall_x.data.backends.lerobot.robodojo.episodes import SCHEMA

CONFIG = "workspace/example/robodojo/stage1_memory.yml"
COVER = "Cover the blocks, remember their colors, then uncover them."
RAW = (640, 480)


class Source:
    """Two episodes of a stand-in export, one of them annotated."""

    def __init__(self, length=80):
        self.length = length

    def episodes(self):
        return [Episode(0, COVER, self.length), Episode(1, "Sort them.", self.length)]

    def _pose(self, frame):
        arm = [0.001 * frame, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return arm + arm

    def head_image(self, episode, frame):
        return np.full((RAW[1], RAW[0], 3), frame % 256, dtype=np.uint8)

    def camera_images(self, episode, frame):
        return [np.zeros((RAW[1], RAW[0], 3), dtype=np.uint8) for _ in range(3)]

    def state(self, episode, frame):
        return self._pose(frame)

    def commands(self, episode, frame, count):
        return [self._pose(frame + k) for k in range(count) if frame + k < self.length]


class Table:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, column):
        return [row[column] for row in self._rows]


class Meta:
    def __init__(self):
        self.episodes = Table([
            {"episode_index": 0, "tasks": [COVER], "length": 80},
            {"episode_index": 1, "tasks": ["Sort them."], "length": 80},
        ])


class RealisticExport:
    def __init__(self):
        self.meta = Meta()


class Collator:
    """A stand-in ``DataCollator`` that keeps what it was configured with."""

    instances = []

    def __init__(self, config, dataload_config, stats, lerobot_config):
        self.config = config
        self.dataload_config = dataload_config
        self.stats = stats
        self.lerobot_config = lerobot_config
        self.processor = None
        Collator.instances.append(self)

    def __call__(self, items):  # pragma: no cover - the loader is not iterated
        return {"input_ids": torch.zeros((len(items), 8), dtype=torch.long)}


class Normalizer:
    """A model-side normalizer, as ``build`` reads it: flat vectors per key."""

    def __init__(self, width, key="robodojo"):
        self.min = {key: torch.arange(width, dtype=torch.float32)}
        self.delta = {key: torch.ones(width, dtype=torch.float32)}


@pytest.fixture
def release(tmp_path):
    directory = tmp_path / "cover_blocks" / "trajectory_0"
    directory.mkdir(parents=True)
    (directory / "progress.json").write_text(json.dumps(
        {"version": 14, "keyframes": [10, 40], "meta": [], "total_frames": 80}),
        encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps(
        {"schema": SCHEMA,
         "tasks": [{"task": "cover_blocks", "instruction": COVER,
                    "episodes": 1}]}),
        encoding="utf-8")
    return tmp_path


@pytest.fixture
def wired(monkeypatch, release):
    """``build`` with the export reader and the inner collator stood in."""
    Collator.instances.clear()
    monkeypatch.setattr(build_module, "open_export",
                        lambda root, repo_id=None, **kw: RealisticExport())
    monkeypatch.setattr(build_module, "LeRobotFrameSource", lambda export: Source())
    monkeypatch.setattr(build_module, "DataCollator", Collator)

    def make(**overrides):
        cfg = load_config(CONFIG)
        cfg.data.robodojo = dict(cfg.data.robodojo or {})
        cfg.data.robodojo["root"] = "/stand-in/export"
        cfg.data.robodojo["annotations"] = str(release)
        cfg.data.robodojo.update(overrides.pop("robodojo", {}))
        cfg.data.num_workers = 0
        for key, value in overrides.items():
            setattr(cfg.data, key, value)
        return cfg

    return make


def context(action_width=26, propri_width=20, **kwargs):
    return BuildContext(
        rank=kwargs.pop("rank", 0),
        world_size=kwargs.pop("world_size", 1),
        processor=kwargs.pop("processor", None),
        normalizer_action=kwargs.pop("normalizer_action", Normalizer(action_width)),
        normalizer_propri=kwargs.pop("normalizer_propri", Normalizer(propri_width)),
        model_config=kwargs.pop("model_config", None),
        **kwargs,
    )


def test_the_shipped_config_builds_a_loader(wired):
    bundle = build_module.build(wired(), context())
    assert bundle.train_iters == len(bundle.train_loader)
    assert bundle.train_iters > 0
    assert bundle.val_loader is None
    assert bundle.dataset.get_val_dataloader() is None
    assert bundle.dataset.get_train_dataloader() is bundle.train_loader


def test_stage_one_builds_selection_samples_too(wired):
    bundle = build_module.build(wired(), context())
    assert bundle.dataset.effective_data_config["selection_samples"] > 0


def test_turning_selection_off_leaves_only_action_samples(wired):
    cfg = wired(robodojo={"selection": False})
    bundle = build_module.build(cfg, context())
    assert bundle.dataset.effective_data_config["selection_samples"] == 0


def test_the_collator_normalizes_with_the_models_own_statistics(wired):
    # Two sets of numbers for one run is how the data and the checkpoint come
    # to disagree, so the collator is handed the model's.
    build_module.build(wired(), context())
    collator = Collator.instances[-1]
    assert torch.equal(collator.stats["action"].min, torch.arange(20, dtype=torch.float32))
    assert torch.equal(collator.stats["state"].min, torch.arange(20, dtype=torch.float32))


def test_the_action_statistics_are_cut_to_the_width_the_dataset_provides(wired):
    # The model's action vector is 26 wide; six of those dimensions this
    # embodiment does not have, and the collator pads and masks them itself.
    build_module.build(wired(), context())
    collator = Collator.instances[-1]
    assert collator.stats["action"].min.shape == (20,)
    assert collator.config["dof_total_dim"] == 26
    assert collator.config["agent_pos_total_dim"] == 20


def test_the_batch_is_normalized_by_the_key_the_statistics_use(wired):
    build_module.build(wired(), context())
    assert Collator.instances[-1].lerobot_config["repo_id"] == "robodojo"


def test_statistics_under_another_single_name_are_still_used(wired):
    # One name, no ambiguity: the config does not have to repeat it.
    build_module.build(wired(robodojo={"stats_key": None}), context(
        normalizer_action=Normalizer(26, key="x2"),
        normalizer_propri=Normalizer(20, key="x2")))
    assert Collator.instances[-1].lerobot_config["repo_id"] == "x2"


def test_statistics_under_several_names_have_to_say_which(wired):
    normalizer = Normalizer(26, key="one")
    normalizer.min["two"] = torch.zeros(26)
    normalizer.delta["two"] = torch.ones(26)
    with pytest.raises(ValueError, match="set data.robodojo.stats_key"):
        build_module.build(wired(robodojo={"stats_key": None}),
                           context(normalizer_action=normalizer))


def test_missing_statistics_are_refused_rather_than_defaulted(wired):
    with pytest.raises(ValueError, match="normalizer is missing"):
        build_module.build(wired(), context(normalizer_action=None))


def test_statistics_narrower_than_the_data_are_refused(wired):
    with pytest.raises(ValueError, match="describe other data"):
        build_module.build(wired(), context(normalizer_action=Normalizer(10)))


def test_the_token_limit_reaches_the_collator(wired):
    build_module.build(wired(max_length=12345), context())
    assert Collator.instances[-1].dataload_config["max_length"] == 12345


def test_the_flow_time_comes_from_the_model_when_the_config_is_silent(wired):
    # The data samples the flow time the model integrates over; a second
    # default would be a second distribution.
    class ModelConfig:
        noise_scheduler = {"beta_alpha": 2.0, "beta_beta": 3.0}

    build_module.build(wired(), context(model_config=ModelConfig()))
    assert Collator.instances[-1].config["noise_scheduler"] == {
        "beta_alpha": 2.0, "beta_beta": 3.0}


def test_an_export_without_a_root_says_so(wired):
    cfg = wired()
    cfg.data.robodojo["root"] = None
    with pytest.raises(ValueError, match="end-effector export"):
        build_module.build(cfg, context())


def test_a_misspelled_memory_key_is_refused(wired):
    with pytest.raises(ValueError, match="unknown data.memory keys"):
        build_module.build(wired(memory={"capacity": 4, "capcity": 8}), context())


def test_a_misspelled_robodojo_key_is_refused(wired):
    with pytest.raises(ValueError, match="unknown data.robodojo keys"):
        build_module.build(wired(robodojo={"anotations": "nope"}), context())


def test_the_geometry_of_the_config_reaches_the_dataset(wired):
    bundle = build_module.build(wired(), context())
    geometry = bundle.dataset.effective_data_config["geometry"]
    assert geometry["capacity"] == 22
    assert geometry["candidates"] == 16
    assert geometry["box_jitter"] is True


def test_one_rank_needs_no_distributed_sampler(wired):
    bundle = build_module.build(wired(), context())
    assert bundle.train_loader.sampler is not None       # the default shuffler
    bundle.set_epoch(3)                                   # and nothing to seed


def test_several_ranks_each_see_their_own_share(wired):
    whole = build_module.build(wired(), context(world_size=1))
    part = build_module.build(wired(), BuildContext(
        rank=1, world_size=4,
        normalizer_action=Normalizer(26), normalizer_propri=Normalizer(20)))
    assert part.train_iters < whole.train_iters
    part.set_epoch(2)


def test_the_run_can_name_the_embodiment_its_statistics_are_under(wired):
    # A statistics file that travels with a checkpoint holds one key per
    # embodiment it was trained on; the run has to say which it continues.
    normalizer = Normalizer(26, key="another_embodiment")
    normalizer.min["something_else"] = torch.zeros(26)
    normalizer.delta["something_else"] = torch.ones(26)
    propri = Normalizer(20, key="another_embodiment")
    propri.min["something_else"] = torch.zeros(20)
    propri.delta["something_else"] = torch.ones(20)
    build_module.build(
        wired(robodojo={"stats_key": "another_embodiment"}),
        context(normalizer_action=normalizer, normalizer_propri=propri))
    assert Collator.instances[-1].lerobot_config["repo_id"] == "another_embodiment"


def test_naming_an_embodiment_the_statistics_lack_says_so(wired):
    with pytest.raises(ValueError, match="statistics do not hold"):
        build_module.build(wired(robodojo={"stats_key": "nope"}), context())


def test_the_loader_keeps_a_fixed_batch_size(wired):
    # The trainer reads it off the loader to advance its data cursor; a custom
    # batch sampler would leave it None and the run would refuse to start.
    bundle = build_module.build(wired(), context())
    assert bundle.train_loader.batch_size == 1


def test_a_batch_bigger_than_the_data_says_so(wired):
    cfg = wired()
    cfg.hyperparams.batch_size_per_gpu = 10_000
    with pytest.raises(ValueError, match="make no full batch"):
        build_module.build(cfg, context())
