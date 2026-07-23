"""Launch a vanilla (open-source) Wall-X websocket policy server.

Opt-free counterpart to ``launch_serving.py``: builds :class:`WallXVanillaPolicy`
(no ``harrix.model_executors`` engine, no CUDA-graph / flashinfer) and serves it over the
same websocket transport. Intended for the open-source LIBERO/qwen2.5 release.

Usage::

    python -m harrix.serving.launch_vanilla_serving \
        --checkpoint-path /path/to/checkpoint \
        --port 32195

The client sends observations of the form::

    {"state": {"eef_pos": [...], "eef_axisangle": [...], "gripper": [...]},
     "views": {"face_view": <img>, "right_wrist_view": <img>},
     "instruction": "..."}

and receives ``{"action": <action_horizon x action_dim array>}``.
"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vanilla Wall-X websocket policy server")
    p.add_argument("--checkpoint-path", required=True, help="Checkpoint dir or file.")
    p.add_argument(
        "--train-config-path",
        default=None,
        help="Training config used by the checkpoint (default: <ckpt>/config.yml).",
    )
    p.add_argument("--norm-key", default="libero_all", help="Normalizer key.")
    p.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Model action horizon (default: read from the train config).",
    )
    p.add_argument(
        "--cam-names",
        nargs="+",
        default=["face_view", "right_wrist_view"],
        help="Camera view names, in model input order.",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host.")
    p.add_argument("--port", type=int, default=32195, help="WebSocket port.")
    p.add_argument(
        "--image-passing-mode",
        choices=["base64", "numpy"],
        default="base64",
        help="How the client passes images.",
    )
    p.add_argument(
        "--cuda-id", default="0", help="Sets CUDA_VISIBLE_DEVICES for the server."
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="Enable dynamic batching (0 = disabled).",
    )
    p.add_argument("--max-wait-time-ms", type=int, default=10)
    p.add_argument("--max-queue-size", type=int, default=64)
    p.add_argument("--timeout-ms", type=int, default=60000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

    from wall_x._vendor.harrix.e2e_infer.config import EvalConfig, autofill_from_checkpoint
    from wall_x._vendor.harrix.serving.handlers.wall_x_vanilla_policy import WallXVanillaPolicy
    from wall_x._vendor.harrix.serving.replica.server import WebsocketPolicyServer

    cfg = EvalConfig()
    cfg.model.checkpoint_path = args.checkpoint_path
    cfg.model.train_config_path = args.train_config_path
    cfg.model.norm_key = args.norm_key
    cfg.model.cam_names = list(args.cam_names)
    cfg.model.action_mode = "flow"
    if args.action_horizon is not None:
        cfg.model.action_horizon = args.action_horizon
    # Resolve train_config_path / action_horizon from the checkpoint when unset.
    autofill_from_checkpoint(cfg)

    logger.info("Building vanilla Wall-X policy from %s", args.checkpoint_path)
    policy = WallXVanillaPolicy(cfg, image_passing_mode=args.image_passing_mode)

    logger.info("Serving at ws://%s:%d", args.host, args.port)
    logger.info(
        "Batching: %s",
        f"batch_size={args.max_batch_size}, wait={args.max_wait_time_ms}ms"
        if args.max_batch_size
        else "disabled",
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
        max_batch_size=args.max_batch_size or None,
        max_wait_time_ms=args.max_wait_time_ms,
        max_queue_size=args.max_queue_size,
        timeout_ms=args.timeout_ms,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
