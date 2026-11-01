"""A self-contained RoboDojo setup, small enough to check a machine with.

The real export is 200 GB and the real run is days long, so a mistake in the
wiring -- a path, a processor without the selection vocabulary, a normalizer of
the wrong width -- is expensive to discover. This writes a miniature stand-in
of the whole setup:

    <out>/export/        two episodes in the end-effector LeRobot layout
    <out>/annotations/   one of them annotated, boxes and windows included
    <out>/normalizer/      a pair that leaves the stand-in data as it is
    <out>/margins.json   a stand-in, wide enough to accept everything
    <out>/stage1_smoke.yml, <out>/stage2_smoke.yml

``--check-data`` then builds the loader those configs describe and pushes a few
batches through the real processor, which is where the two kinds of sample, the
memory geometry and the token budget all have to agree. Training itself is the
ordinary command:

    python scripts/robodojo_smoke.py --out /tmp/robodojo_smoke \
        --processor /path/to/Qwen3.5-2B --check-data
    torchrun --nproc_per_node=1 wall_x/trainer/fsdp_trainer/train_fsdp.py \
        --config /tmp/robodojo_smoke/stage1_smoke.yml

Nothing here is a substitute for the data: two episodes of a moving arm teach
nothing. It answers "does this machine run the pipeline", not "does the policy
learn".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"
HEIGHT, WIDTH = 120, 160
TASKS = (
    "Cover the blocks from left to right, remember their colors, then uncover them.",
    "Sort the objects into the baskets.",
)
LENGTHS = (60, 48)
EXAMPLES = PROJECT_ROOT / "workspace" / "example" / "robodojo"


# ----------------------------------------------------------------------
# The stand-in data
# ----------------------------------------------------------------------
def pose(episode: int, frame: int) -> np.ndarray:
    """A pose that moves steadily, with a real (unit) quaternion."""
    x = 0.30 + 0.002 * frame + 0.01 * episode
    angle = 0.01 * frame
    quaternion = [float(np.cos(angle / 2)), float(np.sin(angle / 2)), 0.0, 0.0]
    left = [x, 0.10, 0.25] + quaternion + [0.5 + 0.004 * frame]
    right = [x, -0.10, 0.25] + quaternion + [0.2]
    return np.asarray(left + right, dtype=np.float32)


def picture(episode: int, frame: int, camera: int, rng) -> np.ndarray:
    """A frame with a patch that says which frame and camera it is."""
    image = rng.integers(0, 40, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:20, :20] = (frame * 4 % 256, camera * 80, episode * 100)
    return image


def write_export(root: Path, use_videos: bool) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        STATE_KEY: {"dtype": "float32", "shape": (16,), "names": None},
        ACTION_KEY: {"dtype": "float32", "shape": (16,), "names": None},
    }
    for key in CAMERA_KEYS:
        features[key] = {
            "dtype": "video" if use_videos else "image",
            "shape": (HEIGHT, WIDTH, 3),
            "names": ["height", "width", "channels"],
        }
    data = LeRobotDataset.create(
        repo_id="RoboDojo-Benchmark/Robodojo", fps=25, features=features,
        root=root, use_videos=use_videos)
    rng = np.random.default_rng(0)
    for episode, length in enumerate(LENGTHS):
        for frame in range(length):
            row = {STATE_KEY: pose(episode, frame),
                   ACTION_KEY: pose(episode, frame + 1),
                   "task": TASKS[episode]}
            for camera, key in enumerate(CAMERA_KEYS):
                row[key] = picture(episode, frame, camera, rng)
            data.add_frame(row)
        data.save_episode()
    # Without this the episode table stays in memory and opening the dataset
    # later asks the hub for metadata it should find on disk.
    data.finalize()
    logger.info("export: %s", root)


def write_annotations(root: Path) -> None:
    """Annotate the first episode: three keyframes, two of them with a box."""
    directory = root / "cover_blocks" / "trajectory_0"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "progress.json").write_text(json.dumps({
        "version": 14,
        "task": "cover_blocks",
        "episode": "trajectory_0",
        "total_frames": LENGTHS[0],
        "frame_coord": [WIDTH, HEIGHT],
        "keyframes": [8, 24, 40],
        "meta": [
            {"frame": 8, "box": [20, 20, 60, 70], "window": [6, 10]},
            {"frame": 24, "window": [22, 27]},
            {"frame": 40, "box": [40, 30, 90, 80], "window": [38, 43]},
        ],
    }, indent=1), encoding="utf-8")
    from wall_x.data.backends.lerobot.robodojo.episodes import SCHEMA

    (root / "index.json").write_text(json.dumps({
        "schema": SCHEMA,
        "describes": "RoboDojo-Benchmark/Robodojo",
        "tasks": [{"task": "cover_blocks", "instruction": TASKS[0], "episodes": 1}],
    }, indent=1), encoding="utf-8")
    logger.info("annotations: %s", root)


def write_normalizer(out: Path, action_width: int, propri_width: int) -> None:
    """A normalizer for data that was made up.

    A run reads the statistics its checkpoint ships; this one starts from a
    plain VLM, which ships none. The pair written here spans [-1, 1] in every
    dimension, except that the two orientation blocks are placed so a
    zero-centred sample lands on the identity orientation: an untrained model
    samples near zero, and a 6D orientation of zeros is not a pose -- the
    refiner's own arithmetic refuses it, for good reason.
    """
    import torch

    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    out.mkdir(parents=True, exist_ok=True)
    for name, width in (("normalizer_action.pth", action_width),
                        ("normalizer_propri.pth", propri_width)):
        minimum = torch.full((width,), -1.0)
        delta = torch.full((width,), 2.0)
        for arm in (3, 13):          # position, orientation, gripper per arm
            minimum[arm:arm + 6] = identity - 1.0
        torch.save({"min.ex_normal": minimum, "delta.ex_normal": delta}, out / name)
    logger.info("normalizer: %s", out)


def write_margins(out: Path, tasks=("cover_blocks", "default")) -> None:
    """Margins wide enough to accept every candidate.

    Not a calibration: the real ones are measured from the policy being
    refined, and refusing nothing is the one choice that cannot be mistaken for
    a measurement.
    """
    out.write_text(json.dumps({
        "schema": "refiner_margins_v1",
        "keep": 1.0,
        "note": "a stand-in for a smoke run; calibrate_refiner_margins.py measures the real ones",
        "tasks": {task: {"position": 1e3, "rotation": 3.2, "gripper_steps": 999}
                  for task in tasks},
    }, indent=1) + "\n", encoding="utf-8")
    logger.info("margins: %s", out)


# ----------------------------------------------------------------------
# The configs
# ----------------------------------------------------------------------
def shrink(config: Dict, args, out: Path, stage: str) -> Dict:
    """The shipped config, pointed at the stand-in data and cut down to size."""
    data = config["data"]
    data["robodojo"] = dict(data["robodojo"])
    data["robodojo"]["root"] = str(out / "export")
    data["robodojo"]["annotations"] = str(out / "annotations")
    data["num_workers"] = args.workers
    # The shipped configs take the statistics from the base model they start
    # from; this one starts from a plain VLM and reads the pair written beside
    # the stand-in data instead.
    data["normalizer_config"] = {"from_checkpoint": str(out / "normalizer")}
    for view in data["resolution"]:
        data["resolution"][view] = args.resolution
    # Two short episodes cannot fill the released geometry, and a smoke run
    # wants a short prompt anyway.
    data["memory"].update(capacity=args.capacity, window=args.window,
                          candidates=args.candidates)
    if "refiner" in data:
        data["refiner"]["margins"] = str(out / "margins.json")

    # One slot token per candidate, or the selector can name a candidate that
    # does not exist.
    model = config["model"]
    model["new_special_tokens"] = [
        token for token in model["new_special_tokens"] if not token.startswith("<|R_")
    ] + [f"<|R_{i}|>" for i in range(args.candidates)]
    model["processor_path"] = args.processor
    model["pretrained_path"] = (
        args.pretrained if stage == "memory" else stage1_checkpoint(out))
    if args.model_config:
        model["config_path"] = args.model_config

    config["hyperparams"]["batch_size_per_gpu"] = 1
    config["hyperparams"]["num_epoch"] = 1
    config["hyperparams"]["scheduler"]["num_warmup_steps"] = 1
    config["hyperparams"]["scheduler"]["num_training_steps"] = args.steps
    config["logging"].update(log_interval=1, save_interval=args.steps,
                             use_wandb=False, log_name=f"robodojo_smoke_{stage}")
    config["checkpoint"]["save_path"] = str(out / "checkpoints" / stage)
    config["debug"]["max_steps"] = args.steps
    return config


def stage1_checkpoint(out: Path) -> str:
    """The newest stage-1 checkpoint under this run, if it has saved one.

    Training writes one directory per saved step, so the path stage 2 starts
    from only exists once stage 1 has run. Re-run with ``--only-configs``
    afterwards to fill it in; until then the config carries a path that fails
    loudly rather than one that looks plausible.
    """
    saved = sorted((out / "checkpoints" / "memory").glob("*/model.safetensors"),
                   key=lambda path: path.stat().st_mtime)
    if saved:
        return str(saved[-1].parent)
    return str(out / "checkpoints" / "memory" / "<run stage 1 first>")


def write_configs(args, out: Path) -> Dict[str, Path]:
    written = {}
    for stage, name in (("memory", "stage1_memory.yml"), ("refiner", "stage2_refiner.yml")):
        config = yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
        target = out / f"stage{'1' if stage == 'memory' else '2'}_smoke.yml"
        target.write_text(
            "# Written by scripts/robodojo_smoke.py: a miniature stand-in run.\n"
            + yaml.safe_dump(shrink(config, args, out, stage),
                             allow_unicode=True, sort_keys=False),
            encoding="utf-8")
        written[stage] = target
        logger.info("config: %s", target)
    return written


# ----------------------------------------------------------------------
# The data path
# ----------------------------------------------------------------------
def check_data(config_path: Path, batches: int) -> int:
    """Build the loader the config describes and read a few batches.

    This runs the real processor over real prompts, which is where the parts
    that have to agree -- the selection vocabulary, the image count, the token
    budget -- stop agreeing if anything is wrong.
    """
    from transformers import AutoProcessor

    from wall_x.config.loader import load_config
    from wall_x.data import BuildContext, build_data
    from wall_x.model.core.action.normalizer import normalizers_from_config

    cfg = load_config(str(config_path))
    processor = AutoProcessor.from_pretrained(cfg.model.processor_path, use_fast=True)
    added = processor.tokenizer.add_tokens(
        ["<|propri|>", "<|action|>"] + list(cfg.model.new_special_tokens or ()))
    logger.info("processor: %s (+%d tokens)", cfg.model.processor_path, added)

    normalizer_action, normalizer_propri, _ = normalizers_from_config(cfg)
    bundle = build_data(cfg, BuildContext(
        rank=0, world_size=1, processor=processor,
        normalizer_action=normalizer_action, normalizer_propri=normalizer_propri,
        model_config=None))

    report = bundle.dataset.effective_data_config
    print(f"samples: {report['samples']} "
          f"({report['selection_samples']} of them selection samples)")
    print(f"batches in an epoch: {bundle.train_iters}")

    kinds = {"with actions": 0, "selection only": 0}
    longest = 0
    for index, batch in enumerate(bundle.train_loader):
        if index >= batches:
            break
        tokens = batch["input_ids"].shape[1]
        longest = max(longest, tokens)
        images = int(batch["image_grid_thw"].shape[0]) if "image_grid_thw" in batch else 0
        acts = int(batch["moe_token_types"].sum())
        kinds["with actions" if acts else "selection only"] += 1
        # A selection batch carries no action rows at all; the model pairs
        # rows with action-token positions and there are none.
        if "dof_mask" in batch:
            supervised = f"{float(batch['dof_mask'].sum()):.0f} supervised action dims"
        else:
            supervised = "no action rows"
        print(f"  batch {index}: {tokens:6d} tokens, {images:2d} images, "
              f"{acts:3d} action tokens, {supervised}")
    print(f"kinds seen: {kinds}")
    print(f"longest batch: {longest} tokens of the {cfg.data.max_length} allowed")
    if not kinds["selection only"]:
        print("NOTE: no selection batch was read; raise --batches to see one")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processor", type=str, default="Qwen/Qwen3.5-4B",
                        help="a local directory or a hub id; the tokenizer and "
                             "image processor come from it")
    parser.add_argument("--pretrained", type=str, default="",
                        help="weights to start from; empty trains from noise, "
                             "which is enough to check the plumbing")
    parser.add_argument("--model-config", type=str, default="",
                        help="override model.config_path (e.g. the 2B layout)")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--videos", action="store_true",
                        help="store the frames as video, as the real export does")
    parser.add_argument("--check-data", action="store_true",
                        help="build the loader and read a few batches")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--only-check", action="store_true",
                        help="skip writing, check what is already there")
    parser.add_argument("--only-configs", action="store_true",
                        help="rewrite just the configs, keeping the data; this is "
                             "how stage 2 learns where stage 1 saved")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stage1 = out / "stage1_smoke.yml"
    if args.only_configs:
        write_configs(args, out)
        print(f"stage 2 starts from {stage1_checkpoint(out)}")
        return 0
    if not args.only_check:
        write_export(out / "export", use_videos=args.videos)
        write_annotations(out / "annotations")
        write_normalizer(out / "normalizer", action_width=26, propri_width=20)
        write_margins(out / "margins.json")
        write_configs(args, out)
        print(f"\nwrote a stand-in setup in {out}")
        print("train stage 1 with:")
        print(f"  torchrun --nproc_per_node=1 "
              f"wall_x/trainer/fsdp_trainer/train_fsdp.py --config {stage1}")
    if args.check_data or args.only_check:
        return check_data(stage1, args.batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
