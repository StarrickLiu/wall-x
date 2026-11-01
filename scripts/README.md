# Scripts

This directory contains the public Wall-X command-line helpers. Run commands
from the repository root unless noted otherwise.

## Inference smoke test

Use `fake_inference.py` to verify that a checkpoint can be loaded and can
produce one action chunk from a synthetic LIBERO-style observation.

```bash
python scripts/fake_inference.py \
  --checkpoint-path /path/to/checkpoint
```

If the training config is not stored next to the checkpoint as `config.yml` or
`config.yaml`, pass it explicitly:

```bash
python scripts/fake_inference.py \
  --checkpoint-path /path/to/checkpoint \
  --train-config-path /path/to/config.yml
```

## LIBERO evaluation

`run_libero.sh` is a small shell wrapper around `infer_libero.py`.

```bash
bash scripts/run_libero.sh /path/to/checkpoint
```

Useful environment variables:

```bash
CHECKPOINT_PATH=/path/to/checkpoint
TRAIN_CONFIG_PATH=/path/to/config.yml
TASK_SUITE_NAME=libero_spatial
TASK_INDICES=0,1,2
NUM_TRIALS_PER_TASK=50
CUDA_ID=0
SMOKE=1
MAX_INFER_TIMES=52
```

`MAX_INFER_TIMES` is optional. When omitted, the launcher uses suite-specific
defaults aligned with the LIBERO evaluator: spatial 22, object 28, goal 30,
libero_10 52, and libero_90 40 action chunks.

For full control, call the Python entry directly:

```bash
python scripts/infer_libero.py \
  --checkpoint-path /path/to/checkpoint \
  --task-suite-name libero_spatial \
  --num-trials-per-task 50 \
  --driver-mode in_process
```

You can also pass a complete harrix eval config:

```bash
python scripts/infer_libero.py --config /path/to/eval_config.yml
```

## RoboDojo serving

`run_robodojo_serving.sh` serves a flow checkpoint over the RoboDojo
(XPolicyLab) WebSocket protocol, including batched requests from parallel
simulator environments.

```bash
CUDA_ID=0 bash scripts/run_robodojo_serving.sh \
  --checkpoint-path /path/to/checkpoint \
  --norm-key ex_normal \
  --port 19000 \
  --replan-steps 16
```

Each request returns `--replan-steps` actions as
`{left,right}_ee_pose` (`xyz + quaternion wxyz`) and
`{left,right}_ee_joint_state`. Relative actions are composed with the observed
end-effector pose in the frame recorded in the checkpoint config. The first
request for each new batch size is slower while kernels autotune.

### Keyframe memory

Checkpoints trained with the `keyframe_memory_packing` epilogue are detected
from their config (`--memory auto`, the default) and served in two passes per
request: every `--memory-select-stride` frames (default 48) a selection pass
decides which recent head-camera frames to keep, and the action pass sees the
kept frames in its prompt. The memory is held by the client: each observation
carries the state returned by the previous response (`kfmem_client_state_v1`,
documented in `wall_x/serving/robodojo/memory_protocol.py`), so the server
keeps no per-episode state and does not batch requests. Observations need a
`task_id`; `--memory-tasks` restricts memory to a list of tasks.

### Residual refiner

A checkpoint whose `vla_config.refiner` is enabled carries a second set of
action-branch weights (`*_refiner`) that run once after flow sampling and add a
residual to the sampled chunk. Nothing needs to be configured at serving time,
but `--num-inference-timesteps` must equal the refiner's `num_steps`; any other
value, an enabled refiner without weights, or refiner weights in a checkpoint
whose config has it disabled, is refused at load or first request.

## Dataset and checkpoint utilities

- `compute_norm_stats.py`: compute action normalization statistics for a
  LeRobot dataset.
- `merge_sharded_weights.py`: merge FSDP sharded checkpoint files into a single
  checkpoint directory.
- `normalize.py`: helper for generating normalization files from dataset
  statistics.

Most scripts support `--help` for their command-line options.

## RoboDojo

`robodojo_smoke.py` writes a two-episode stand-in of a whole RoboDojo setup —
export, memory annotations, a normalizer and both stage configs — and can push
a few batches through the real processor. It answers "does this
machine run the pipeline" before 200 GB is downloaded.

```bash
python scripts/robodojo_smoke.py --out /tmp/robodojo_smoke \
    --processor /path/to/Qwen3.5 --check-data
torchrun --nproc_per_node=1 wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /tmp/robodojo_smoke/stage1_smoke.yml
```

`calibrate_refiner_margins.py` measures how far the base model's own samples
fall from the demonstrations, per task, and writes the margins that decide
which candidates the residual refiner is trained on. They describe the policy
being refined, so they are measured rather than carried around as constants.

```bash
python scripts/calibrate_refiner_margins.py \
    --train-config workspace/example/robodojo/stage2_refiner.yml \
    --out margins.json
python scripts/calibrate_refiner_margins.py --self-check   # arithmetic only
```

The run measures every task it is given, which takes a pass over the data;
`--episodes N` stops early and measures whichever tasks those batches held.
Either way the file carries a margin pooled over every candidate measured, and
a task without one of its own is filtered by that instead of stopping the run.

Normalization statistics are not measured here: they come with the weights. The
base model ships the pair it was pretrained with, and a stage-2 run inherits
them from the stage-1 checkpoint it refines — the same weights read a
different scale if the numbers change under them.
