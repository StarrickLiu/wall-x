"""WebSocket and health endpoints for one serving replica."""

from __future__ import annotations

import asyncio
import http
import json
import logging
import os
import time
import threading
import uuid
from contextlib import AsyncExitStack
from functools import partial
from typing import Any, Callable, Dict, List

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except ImportError:
    logging.warning(
        "msgpack-numpy not installed. Install with: pip install msgpack-numpy"
    )
    msgpack = None

import websockets.asyncio.server as _server
import websockets.frames

from wall_x._vendor.harrix.serving.monitoring.metrics import LatencyHistogram
from wall_x._vendor.harrix.serving.handlers.base import BasePolicy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol.

    Implements a websocket server that:
    1. Sends policy metadata on connection
    2. Receives observations
    3. Returns predicted actions (single or batched)
    4. Tracks timing information

    When batching parameters are provided, dynamic batching is enabled:
    requests from multiple clients are queued and processed in batches.
    """

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Dict | None = None,
        # Dynamic batching parameters (None = disabled)
        max_batch_size: int | None = None,
        max_wait_time_ms: float | None = None,
        max_queue_size: int = 100,
        timeout_ms: float = 30000,
        policy_mode_override: str | None = None,
        additional_policy_mode_ports: Dict[int, str] | None = None,
        request_transform: Callable[[Any, str | None], Any] | None = None,
        listener_metadata: Dict[str, Dict[str, Any]] | None = None,
        lifecycle_hooks: List[Any] | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._started_at = time.time()
        self._active_connections = 0
        self._active_connection_ids = set()
        self._in_flight = 0
        self._total_connections = 0
        self._total_requests = 0
        self._error_count = 0
        self._last_error = None
        self._last_request_started_at = None
        self._last_request_finished_at = None
        self._last_request_latency_ms = None
        self._ewma_latency_ms = None
        self._latency_hist = LatencyHistogram()
        self._listener_modes = {port: policy_mode_override}
        self._listener_modes.update(additional_policy_mode_ports or {})
        self._request_transform = request_transform
        self._listener_metadata = listener_metadata or {}
        self._policy_lock = threading.RLock()
        self._lifecycle_hooks = list(lifecycle_hooks or [])

        # Dynamic batching
        self._scheduler = None
        if max_batch_size is not None:
            from .scheduler import RequestScheduler

            self._scheduler = RequestScheduler(
                policy=policy,
                max_batch_size=max_batch_size,
                max_wait_time_ms=(
                    max_wait_time_ms if max_wait_time_ms is not None else 0
                ),
                max_queue_size=max_queue_size,
                timeout_ms=timeout_ms,
                policy_lock=self._policy_lock,
                request_transform=request_transform,
            )

        logging.getLogger("websockets.server").setLevel(logging.WARNING)

    @property
    def batching_enabled(self) -> bool:
        return self._scheduler is not None

    @property
    def policy_lock(self):
        return self._policy_lock

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def add_lifecycle_hook(self, hook) -> None:
        self._lifecycle_hooks.append(hook)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with AsyncExitStack() as stack:
            if self._scheduler is not None:
                await self._scheduler.start()
                stack.push_async_callback(self._scheduler.stop)
            for hook in self._lifecycle_hooks:
                hook.start()
                stack.callback(hook.stop)

            servers = []
            for port, override in self._listener_modes.items():
                server = await stack.enter_async_context(
                    _server.serve(
                        partial(self._handler, policy_mode_override=override),
                        self._host,
                        port,
                        compression=None,
                        max_size=None,
                        ping_interval=None,
                        ping_timeout=None,
                        process_request=self._process_request,
                    )
                )
                servers.append(server)
            mode_str = "batched" if self.batching_enabled else "single"
            for port, override in self._listener_modes.items():
                logger.info(
                    "Server started on %s:%s (mode=%s policy_mode_override=%s)",
                    self._host,
                    port,
                    mode_str,
                    override or "request",
                )
            await asyncio.gather(*(server.serve_forever() for server in servers))

    async def _handler(
        self,
        websocket: _server.ServerConnection,
        *,
        policy_mode_override: str | None = None,
    ):
        connection_id = uuid.uuid4().hex[:12]
        logger.info(
            "Worker connection opened connection_id=%s client=%s",
            connection_id,
            websocket.remote_address,
        )
        self._active_connections += 1
        self._active_connection_ids.add(connection_id)
        self._total_connections += 1

        try:
            if msgpack is None:
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="msgpack-numpy not installed on server",
                )
                return

            # Send metadata, refreshing the live per-module weight versions so
            # the router tracks this worker's current version after in-place
            # swaps (self._metadata is captured once at startup).
            meta = dict(self._metadata)
            meta["policy_mode_override"] = policy_mode_override
            if policy_mode_override is not None:
                meta.update(
                    self._listener_metadata.get(policy_mode_override, {})
                )
            ws_fn = getattr(self._policy, "weights_status", None)
            if callable(ws_fn):
                versions = ws_fn().get("versions")
                if versions:
                    meta["module_versions"] = versions
            await websocket.send(msgpack.packb(meta))

            prev_total_time = None
            request_index = 0
            while True:
                try:
                    packed_obs = await websocket.recv()
                except websockets.ConnectionClosed:
                    logger.info(
                        "Worker connection closed connection_id=%s client=%s",
                        connection_id,
                        websocket.remote_address,
                    )
                    break

                request_index += 1
                self._in_flight += 1
                self._total_requests += 1
                recv_ts = time.time()
                self._last_request_started_at = recv_ts
                start_time = time.monotonic()
                try:
                    infer_time = time.monotonic()

                    if self._scheduler is not None:
                        # Dynamic batching path. Queue the packed payload
                        # immediately; unpacking runs inside the scheduler's
                        # batch thread so the worker event loop can enqueue
                        # concurrent requests without serial msgpack work.
                        action = await self._scheduler.add_request(
                            packed_obs,
                            packed=True,
                            policy_mode_override=policy_mode_override,
                        )
                    else:
                        obs = msgpack.unpackb(packed_obs)
                        if self._request_transform is not None:
                            obs = self._request_transform(
                                obs,
                                policy_mode_override,
                            )
                        # Single inference path: run synchronously on the
                        # event-loop thread. torch.compile/CUDA-graph state
                        # set up by warmup is per-thread TLS — dispatching to
                        # `asyncio.to_thread` lands on a fresh worker thread
                        # whose TLS is empty, which makes the compiled
                        # sample_actions trip an internal assertion (see
                        # cudagraph_trees.get_obj). For a single-robot WS
                        # session the ~100ms block per request is fine.
                        with self._policy_lock:
                            action = self._policy.infer(obs)

                    infer_time = time.monotonic() - infer_time

                    sched_timing = (
                        action.pop("__sched_timing__", None)
                        if isinstance(action, dict)
                        else None
                    )
                    # t_recv/t_send: epoch seconds on the worker clock. Cross-
                    # machine differences vs client timestamps include clock
                    # offset — only durations (t_send - t_recv) and intra-
                    # episode trends are trustworthy (robot_monitor.md §3).
                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                        "t_recv": recv_ts,
                        "t_send": time.time(),
                    }
                    if sched_timing:
                        # queue_ms = batch-queue wait; batch_ms = model forward
                        action["server_timing"].update(sched_timing)
                    if prev_total_time is not None:
                        action["server_timing"]["prev_total_ms"] = (
                            prev_total_time * 1000
                        )

                    await websocket.send(msgpack.packb(action))
                    total_time = time.monotonic() - start_time
                    self._record_request_latency(total_time * 1000)
                    prev_total_time = total_time

                except websockets.ConnectionClosed:
                    logger.info(
                        "Worker connection closed connection_id=%s client=%s",
                        connection_id,
                        websocket.remote_address,
                    )
                    break
                except Exception as e:
                    self._record_error(e)
                    logger.error(
                        "Error handling request connection_id=%s request_index=%s: %s",
                        connection_id,
                        request_index,
                        e,
                        exc_info=True,
                    )
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error.",
                    )
                    raise
                finally:
                    self._in_flight = max(0, self._in_flight - 1)
        finally:
            self._active_connections = max(0, self._active_connections - 1)
            self._active_connection_ids.discard(connection_id)

    def _record_request_latency(self, latency_ms: float) -> None:
        self._last_request_latency_ms = latency_ms
        self._last_request_finished_at = time.time()
        self._latency_hist.observe(latency_ms / 1000.0)
        alpha = 0.2
        if self._ewma_latency_ms is None:
            self._ewma_latency_ms = latency_ms
        else:
            self._ewma_latency_ms = (
                alpha * latency_ms + (1.0 - alpha) * self._ewma_latency_ms
            )

    def _record_error(self, error: Exception) -> None:
        self._error_count += 1
        self._last_error = f"{type(error).__name__}: {error}"

    def stats_snapshot(self) -> Dict[str, Any]:
        scheduler_stats = None
        if self._scheduler is not None:
            scheduler_stats = self._scheduler.stats_snapshot()
        return {
            "worker": {
                "pid": os.getpid(),
                "host": self._host,
                "port": self._port,
                "listeners": dict(self._listener_modes),
                "metadata": self._metadata,
                "started_at": self._started_at,
                "uptime_s": time.time() - self._started_at,
                "batching_enabled": self.batching_enabled,
            },
            "requests": {
                "active_connections": self._active_connections,
                "active_connection_ids": sorted(self._active_connection_ids),
                "in_flight": self._in_flight,
                "total_connections": self._total_connections,
                "total_requests": self._total_requests,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "last_request_started_at": self._last_request_started_at,
                "last_request_finished_at": self._last_request_finished_at,
                "last_request_latency_ms": self._last_request_latency_ms,
                "ewma_latency_ms": self._ewma_latency_ms,
                "latency_histogram": self._latency_hist.snapshot(),
            },
            "scheduler": scheduler_stats,
        }

    def _process_request(
        self, connection: _server.ServerConnection, request: _server.Request
    ) -> _server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if request.path == "/v2/health/ready":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if request.path == "/metrics":
            from wall_x._vendor.harrix.serving.monitoring.metrics import (
                render_worker_metrics,
            )

            return connection.respond(
                http.HTTPStatus.OK, render_worker_metrics(self.stats_snapshot())
            )
        if request.path in ("/worker/status", "/stats"):
            return connection.respond(
                http.HTTPStatus.OK,
                json.dumps(self.stats_snapshot(), default=str, indent=2, sort_keys=True)
                + "\n",
            )
        if request.path == "/weights/status":
            ws = getattr(self._policy, "weights_status", None)
            payload = ws() if callable(ws) else {}
            return connection.respond(
                http.HTTPStatus.OK,
                json.dumps(payload, default=str) + "\n",
            )
        return None
