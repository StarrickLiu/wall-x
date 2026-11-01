"""Calibrate the margins that decide which candidates the refiner is scored on.

The refiner is trained to correct its base model's own samples. When a sample
is a different reasonable way to do the task, or misses badly, scoring it
against this particular demonstration teaches the refiner to undo a decision
rather than correct an error -- so those samples are dropped by a per-task
margin on how far a candidate may be.

Those margins describe the base model being refined, not the task, so they
cannot be shipped with the code: a different base model misses differently.
This measures your base model on your data and writes the margins out.

    python scripts/calibrate_refiner_margins.py \
        --train-config workspace/example/robodojo/stage2_refiner.yml \
        --out margins.json [--episodes 40] [--keep 0.99]

``--keep`` is the fraction of candidates to keep per task: the margin lands on
that quantile of the measured deviations. The default keeps almost everything,
dropping only the clear misses.

Measuring every task takes a pass over the data. A shortened run measures
whichever tasks its batches held, so the file also carries a margin pooled over
every candidate measured, which a task of its own falls back to -- training
refuses a task with no margin at all, and refusing it after the run has started
is worse than filtering it by the pooled one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wall_x.model.core.refiner.train import measure  # noqa: E402

# The key a task with no name of its own is measured under.
DEFAULT_TASK = "default"


def quantile(values: List[float], keep: float) -> float:
    """The deviation that ``keep`` of the candidates stay under."""
    if not values:
        raise ValueError("no candidates were measured")
    return float(np.quantile(np.asarray(values, dtype=np.float64), keep))


def entry(measured: Dict[str, List[float]], keep: float) -> Dict:
    """One margin, and what was measured to arrive at it."""
    return {
        "position": round(quantile(measured["position"], keep), 6),
        "rotation": round(quantile(measured["rotation"], keep), 6),
        "gripper_steps": int(np.ceil(quantile(measured["gripper_steps"], keep))),
        "candidates": len(measured["position"]),
        "position_median": round(float(np.median(measured["position"])), 6),
        "rotation_median": round(float(np.median(measured["rotation"])), 6),
    }


def summarize(samples: Dict[str, Dict[str, List[float]]], keep: float) -> Dict[str, Dict]:
    """Per-task margins, plus the pooled one a task falls back to.

    Measuring every task means a full pass over the data; a run shortened with
    ``--episodes`` covers whichever tasks its batches happened to hold. A task
    with no margin of its own is refused during training -- which would stop a
    run after it had started -- so the pooled margin, measured over every
    candidate seen here, is written as the fallback.
    """
    out = {}
    for task, measured in sorted(samples.items()):
        out[task] = entry(measured, keep)
    pooled = {field: [value for measured in samples.values() for value in measured[field]]
              for field in ("position", "rotation", "gripper_steps")}
    out.setdefault(DEFAULT_TASK, entry(pooled, keep))
    return out


def collect(sampler, episodes: int, horizon: int) -> Dict[str, Dict[str, List[float]]]:
    """Measure how far the base model's candidates fall from the demonstrations.

    ``sampler`` yields ``(task, candidate, demonstration, valid)`` in
    unnormalized units, which is what the margins are expressed in.
    """
    samples: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"position": [], "rotation": [], "gripper_steps": []})
    seen = 0
    for task, candidate, demonstration, valid in sampler:
        deviation = measure(candidate, demonstration, valid=valid)
        for row in range(candidate.shape[0]):
            samples[task]["position"].append(float(deviation.position[row]))
            samples[task]["rotation"].append(float(deviation.rotation[row]))
            samples[task]["gripper_steps"].append(float(deviation.gripper_steps[row]))
        seen += 1
        if seen % 20 == 0:
            print(f"  measured {seen} batches", flush=True)
        if episodes and seen >= episodes:
            break
    if not samples:
        raise RuntimeError("the sampler produced nothing to measure")
    return samples


def build_model_and_loader(train_config: Path, batch_size: int):
    """The policy the config points at, unsharded, and its dataloader.

    Unsharded on purpose: this samples by calling the flow sampler directly,
    and under FSDP a module's weights only exist while its own forward runs.
    One card, one process, no distributed setup.
    """
    import logging

    import torch

    from wall_x.config.loader import load_config
    from wall_x.data import BuildContext, build_data
    from wall_x.model.core.action.normalizer import normalizers_from_config
    from wall_x.trainer.adapters import ADAPTER_REGISTRY

    cfg = load_config(str(train_config))
    if batch_size:
        cfg.hyperparams.batch_size_per_gpu = batch_size
    # Candidates are what the refiner will be trained on, so they come from
    # action samples; a selection sample has no chunk to correct.
    cfg.data.robodojo = dict(cfg.data.robodojo or {})
    cfg.data.robodojo["selection"] = False

    adapter = ADAPTER_REGISTRY[cfg.model_type](
        cfg=cfg, logger=logging.getLogger("calibrate"))
    # The same statistics the run uses: the ones the checkpoint ships, or the
    # file its config names. Measuring candidates against another scale would
    # measure the scale instead of the policy.
    action, propri, stats = normalizers_from_config(cfg)
    adapter.normalizer_action, adapter.normalizer_propri = action, propri
    processors = adapter.load_processor(action_statistic_dof=stats)
    processor = processors["processor"]

    model_config = adapter.build_model_config()
    model = adapter.create_model(
        processor, processors.get("tokenizer_mixin"), model_config)
    model = adapter.load_weights(model, action, propri, processor=processor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.float32).eval()

    bundle = build_data(cfg, BuildContext(
        rank=0, world_size=1, processor=processor,
        normalizer_action=action, normalizer_propri=propri,
        model_config=model_config))
    return cfg, model, bundle, device


def build_sampler(train_config: Path, batch_size: int, horizon: int):
    """Sample candidates from the base model the config points at.

    Yields ``(task, candidate, demonstration, valid)`` in unnormalized units,
    which is what the margins are expressed in. The candidates come from the
    same integration budget serving uses, because that is the distribution the
    refiner will be asked to correct.

    Imported lazily so ``--self-check`` needs no model, dataset or GPU.
    """
    import torch

    from wall_x.model.core.refiner import RefinerConfig

    cfg, model, bundle, device = build_model_and_loader(train_config, batch_size)
    refiner = RefinerConfig.from_vla_config(model.config.vla_config)
    steps = refiner.num_steps or 10
    print(f"sampling with {steps} integration steps on {device}", flush=True)

    normalizer = model.action_preprocessor.normalizer_action
    for batch in bundle.train_loader:
        chunk = batch.get("action_chunk")
        if chunk is None:
            continue
        observation = {k: (v.to(device) if torch.is_tensor(v) else v)
                       for k, v in batch.items()
                       if k not in ("action_chunk", "sample_time", "moe_token_types",
                                    "frame_index", "task_names", "labels")}
        chunk = chunk.to(device)
        dof_mask = observation.get("dof_mask")
        with torch.no_grad():
            sample = model.sample_flow_with_cache(
                action_horizon=int(chunk.shape[1]),
                action_dim=int(chunk.shape[2]),
                num_inference_timesteps=steps,
                **observation,
            )
            candidate = normalizer.unnormalize_data(
                sample.candidate, observation.get("dataset_names"))
            demonstration = normalizer.unnormalize_data(
                chunk, observation.get("dataset_names"))
        valid = None
        if dof_mask is not None:
            valid = dof_mask[..., :20].amax(dim=-1) > 0
        tasks = batch.get("task_names") or [DEFAULT_TASK]
        # One task per batch: same-kind batching already groups by sample kind,
        # not by task, so measure each row under its own task.
        for row in range(candidate.shape[0]):
            yield (str(tasks[row]),
                   candidate[row:row + 1, :, :20].float().cpu(),
                   demonstration[row:row + 1, :, :20].float().cpu(),
                   None if valid is None else valid[row:row + 1].cpu())


def from_dump(path: Path):
    """Candidates saved earlier: a list of dicts with the four measured fields.

    Lets the calibration run anywhere, including where the base model does not
    fit, and makes it reproducible from the same dump.
    """
    saved = torch.load(path, weights_only=False)
    for entry in saved:
        yield (entry["task"], entry["candidate"], entry["demonstration"],
               entry.get("valid"))


def self_check() -> int:
    """Check the arithmetic on made-up deviations, with no model involved."""
    rows = []
    for task, spread in (("near", 0.01), ("far", 0.10)):
        for index in range(200):
            candidate = torch.zeros(1, 4, 20)
            demonstration = torch.zeros(1, 4, 20)
            for arm in (0, 10):      # both arms need a real orientation
                identity = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
                demonstration[..., arm + 3:arm + 9] = identity
                candidate[..., arm + 3:arm + 9] = identity
            candidate[..., 0] = spread * (index / 199.0)
            rows.append({"task": task, "candidate": candidate,
                         "demonstration": demonstration, "valid": None})
    samples = collect(iter([(r["task"], r["candidate"], r["demonstration"], r["valid"])
                            for r in rows]), episodes=0, horizon=4)
    margins = summarize(samples, keep=0.9)
    print(json.dumps(margins, indent=1))
    ok = (abs(margins["near"]["position"] - 0.009) < 0.002
          and abs(margins["far"]["position"] - 0.09) < 0.02
          and margins["near"]["rotation"] < 1e-5
          and margins["near"]["candidates"] == 200
          # The fallback is measured over both tasks' candidates together.
          and margins[DEFAULT_TASK]["candidates"] == 400)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--from-dump", type=Path,
                        help="candidates saved earlier, instead of sampling now")
    parser.add_argument("--episodes", type=int, default=0,
                        help="stop after this many batches; 0 measures every "
                             "task, which is what a run should be trained with")
    parser.add_argument("--keep", type=float, default=0.99,
                        help="fraction of candidates to keep per task")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--self-check", action="store_true",
                        help="check the arithmetic without a model")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    if not 0.0 < args.keep <= 1.0:
        parser.error("--keep must be in (0, 1]")
    if args.out is None:
        parser.error("--out is required")

    if args.from_dump:
        sampler = from_dump(args.from_dump)
    elif args.train_config:
        sampler = build_sampler(args.train_config, args.batch_size, args.horizon)
    else:
        parser.error("pass --train-config or --from-dump")

    samples = collect(sampler, args.episodes, args.horizon)
    margins = summarize(samples, args.keep)
    record = {"schema": "refiner_margins_v1", "keep": args.keep, "tasks": margins}
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    measured = [task for task in margins if task != DEFAULT_TASK]
    kept = margins[DEFAULT_TASK]["candidates"]
    print(f"wrote {args.out}: {len(measured)} tasks from {kept} candidates")
    for task in measured:
        print(f"  {margins[task]['candidates']:6d}  {task}")
    print("A task not listed above falls back to the pooled margin.")
    print("Point data.refiner.margins at it in the stage-2 config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
