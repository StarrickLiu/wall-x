#!/usr/bin/env bash
# Launch the open-source (vanilla) Wall-X websocket policy server.
#
# Serves a checkpoint over a websocket so a client can stream observations and
# receive action chunks. It uses the vanilla model path (identical to the LIBERO
# evaluation), with no closed-source inference engine.
#
# The vanilla cut returns RAW model action chunks (shape action_horizon x
# action_dim); robot-side serialization/interpolation is left to the client,
# because the real-robot preprocessors are not part of the open-source release.
# Flags that require the closed inference engine (CUDA graph / experimental
# engine) or the excluded robot preprocessors (action serialization / robot
# types / non-LIBERO envs) are recognized and rejected with a clear message, so
# a command copied from the full (internal) serving stack fails loudly instead
# of silently behaving differently.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_serving.sh --checkpoint-path /path/to/checkpoint [options]

Required:
  --checkpoint-path PATH     Checkpoint directory or checkpoint file.

Options (vanilla serving):
  --train-config-path PATH   Training config used by the checkpoint
                             (default: <ckpt>/config.yml).
  --norm-key KEY             Normalizer key. Default: libero_all.
  --cam-names "V1 V2 ..."    Camera view names in model input order.
                             Default: "face_view right_wrist_view".
  --action-horizon N         Model action horizon (default: from train config).
  --host HOST                Bind host. Default: 0.0.0.0.
  --port PORT                WebSocket port. Default: 32195.
  --cuda-id ID               Sets CUDA_VISIBLE_DEVICES. Default: 0.
  --image-passing-mode MODE  base64 or numpy. Default: base64.
  --max-batch-size N         Enable server-side dynamic batching (0 = disabled).
  --env ENV                  Serving environment. Only LIBERO is supported by the
                             vanilla obs assembly. Default: LIBERO.
  --debug                    Enable debug logging.
  --dry-run                  Print the command without running it.

Not available in the open-source (vanilla) cut — rejected with guidance:
  --serialize-actions / --robot-type   (need the excluded robot preprocessors;
                                         vanilla returns raw action chunks)
  --enable-cuda-graph / --enable-experimental-engine  (closed opt inference engine)

The server returns raw model action chunks (shape action_horizon x action_dim);
robot-side action serialization is left to the client.
EOF
}

# Reject a flag that only the full (internal) serving stack supports, pointing at
# the reason, instead of accepting it as a silent no-op.
reject_full_stack_flag() {
    echo "error: '$1' is not available in the open-source (vanilla) serving." >&2
    echo "       $2" >&2
    echo "       The vanilla server returns raw model action chunks; serialize" >&2
    echo "       and post-process on the client side." >&2
    exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -d "${SOURCE_ROOT}/wall_x" ]]; then
    export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-}"
NORM_KEY="${NORM_KEY:-libero_all}"
CAM_NAMES="${CAM_NAMES:-face_view right_wrist_view}"
ACTION_HORIZON="${ACTION_HORIZON:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-32195}"
CUDA_ID="${CUDA_ID:-0}"
IMAGE_PASSING_MODE="${IMAGE_PASSING_MODE:-base64}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
DEBUG=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --checkpoint-path) CHECKPOINT_PATH="${2:?}"; shift 2 ;;
        --checkpoint-path=*) CHECKPOINT_PATH="${1#*=}"; shift ;;
        --train-config-path) TRAIN_CONFIG_PATH="${2:?}"; shift 2 ;;
        --train-config-path=*) TRAIN_CONFIG_PATH="${1#*=}"; shift ;;
        --norm-key) NORM_KEY="${2:?}"; shift 2 ;;
        --norm-key=*) NORM_KEY="${1#*=}"; shift ;;
        --cam-names) CAM_NAMES="${2:?}"; shift 2 ;;
        --cam-names=*) CAM_NAMES="${1#*=}"; shift ;;
        --action-horizon) ACTION_HORIZON="${2:?}"; shift 2 ;;
        --action-horizon=*) ACTION_HORIZON="${1#*=}"; shift ;;
        --host) HOST="${2:?}"; shift 2 ;;
        --host=*) HOST="${1#*=}"; shift ;;
        --port) PORT="${2:?}"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --cuda-id) CUDA_ID="${2:?}"; shift 2 ;;
        --cuda-id=*) CUDA_ID="${1#*=}"; shift ;;
        --image-passing-mode) IMAGE_PASSING_MODE="${2:?}"; shift 2 ;;
        --image-passing-mode=*) IMAGE_PASSING_MODE="${1#*=}"; shift ;;
        --max-batch-size) MAX_BATCH_SIZE="${2:?}"; shift 2 ;;
        --max-batch-size=*) MAX_BATCH_SIZE="${1#*=}"; shift ;;
        --env) ENV_ARG="${2:?}"; shift 2 ;;
        --env=*) ENV_ARG="${1#*=}"; shift ;;
        --raw-actions|--no-serialize-actions) shift ;;  # raw is the vanilla default
        --serialize-actions)
            reject_full_stack_flag "$1" \
                "It needs the robot preprocessors, which are not in the OSS cut." ;;
        --robot-type|--robot-type=*)
            reject_full_stack_flag "${1%%=*}" \
                "It needs the robot preprocessors, which are not in the OSS cut." ;;
        --enable-cuda-graph|--enable-experimental-engine)
            reject_full_stack_flag "$1" \
                "It needs the closed opt inference engine, not in the OSS cut." ;;
        --debug) DEBUG=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "error: unknown argument '$1'" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
done

# --env is a convenience: only the LIBERO-style observation assembly ships in the
# vanilla cut. Accept LIBERO (the default); reject anything else with guidance.
ENV_ARG="${ENV_ARG:-LIBERO}"
case "${ENV_ARG^^}" in
    LIBERO) ;;
    *)
        echo "error: --env '${ENV_ARG}' is not supported by the vanilla serving." >&2
        echo "       Only LIBERO-style observations (eef_pos / eef_axisangle /" >&2
        echo "       gripper + face_view / right_wrist_view) are assembled in the" >&2
        echo "       open-source cut." >&2
        exit 2
        ;;
esac

if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "error: --checkpoint-path is required." >&2
    echo >&2
    usage >&2
    exit 2
fi

CMD=(
    "${PYTHON_BIN}" -m wall_x._vendor.harrix.serving.launch_vanilla_serving
    --checkpoint-path "${CHECKPOINT_PATH}"
    --norm-key "${NORM_KEY}"
    --host "${HOST}"
    --port "${PORT}"
    --cuda-id "${CUDA_ID}"
    --image-passing-mode "${IMAGE_PASSING_MODE}"
)
# --cam-names is nargs="+" on the launcher; pass each token unquoted-split.
if [[ -n "${CAM_NAMES}" ]]; then
    # shellcheck disable=SC2206
    CAM_ARR=(${CAM_NAMES})
    CMD+=(--cam-names "${CAM_ARR[@]}")
fi
if [[ -n "${TRAIN_CONFIG_PATH}" ]]; then
    CMD+=(--train-config-path "${TRAIN_CONFIG_PATH}")
fi
if [[ -n "${ACTION_HORIZON}" ]]; then
    CMD+=(--action-horizon "${ACTION_HORIZON}")
fi
if [[ -n "${MAX_BATCH_SIZE}" ]]; then
    CMD+=(--max-batch-size "${MAX_BATCH_SIZE}")
fi
if [[ "${DEBUG}" == "1" ]]; then
    CMD+=(--debug)
fi

printf 'Launching Wall-X vanilla serving:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

exec "${CMD[@]}"
