# SPDX-License-Identifier: Apache-2.0
"""ZeroMQ stage socket layer + ZeroMQ-only data path (engine neutral).

This is the engine-neutral port of the vLLM fork's ``ZeroMQPPCommunicator``
(integrations/vllm/zeromq_communicator.py), which ConnexTE used as its base
class: one ROUTER per stage (bound at ``router_port``, receives forward steps
and returns), DEALERs to peers, and an optional control socket to a legacy
"Front ROUTER" (REGISTER / ASSIGN json).

ZeroMQ-only fallback wire format (used when no transfer engine is available),
one multi-frame message per dict plus one per tensor::

    [!Q8 len(meta), meta=pickle({"metadata_list", "send_timestamp"}), !Q8 n]
    n x [!Q8 len(t), torch.save(t.cpu())]

``metadata_list`` is ``[(key, TensorMetadata(device, dtype, size)) | (key,
value)]``. The TensorMetadata class is pickled by reference, so peers must
agree on it: :class:`TensorDictCodec` is injectable (the vLLM adapter plugs in
vLLM's own ``TensorMetadata`` / ``_split_tensor_dict`` so the bytes are
identical to the fork's).
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import time
from collections import namedtuple
from typing import Any
from uuid import uuid4

import torch
import zmq

from connex.pp.hooks import NULL_METRICS, NetUtils

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


class TensorDictCodec:
    """Split a tensor dict into (metadata_list, tensors) and recognise tensor
    metadata entries on receive."""

    def __init__(self, meta_cls=TensorMetadata, split=None):
        self.meta_cls = meta_cls
        self._split = split

    def split(self, tensor_dict: dict):
        if self._split is not None:
            return self._split(tensor_dict)
        metadata_list, tensor_list = [], []
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                metadata_list.append(
                    (key, self.meta_cls(value.device.type, value.dtype, value.size())))
                tensor_list.append(value)
            else:
                metadata_list.append((key, value))
        return metadata_list, tensor_list

    def is_tensor_meta(self, value) -> bool:
        return isinstance(value, self.meta_cls)


DEFAULT_CODEC = TensorDictCodec()


class ZmqStageBase:
    """ROUTER/DEALER stage endpoint with the ZeroMQ-only data path."""

    # Engine adapters override these class attributes (or set them per
    # instance before __init__ runs) instead of subclassing every method.
    log: logging.Logger = logging.getLogger("connex.pp")
    net = NetUtils
    codec: TensorDictCodec = DEFAULT_CODEC
    metrics = NULL_METRICS

    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        router_port: int,
        front_router_addr: str,
        device: str = "cuda",
        timeout_ms: int = -1,
    ):
        init_start_time = time.time()

        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.device = device
        self.timeout_ms = timeout_ms

        self.is_first = stage_idx == 0
        self.is_last = stage_idx == num_stages - 1

        self.context = zmq.Context()

        self.front_router_addr = front_router_addr
        self.control_socket: zmq.Socket | None = None

        self.router_port = router_port
        self.router_endpoint = self.net.get_tcp_uri(self.net.get_ip(), router_port)
        self.router_socket: zmq.Socket | None = None
        self.forward_socket: zmq.Socket | None = None
        self.return_socket: zmq.Socket | None = None

        self._setup_control_plane()
        self._setup_router_socket()

        connection_ready_time = time.time() - init_start_time
        if self.metrics.enabled:
            self.metrics.observe("connection_ready_time", connection_ready_time)
            self.metrics.set("connection_ready_timestamp", time.time())
            self.log.info("Stage %s connection ready in %.4f seconds (backend=%s)",
                          self.stage_idx, connection_ready_time, self.backend_name)

    @property
    def backend_name(self) -> str:
        return "zeromq-naive"

    # ---- control plane (legacy Front ROUTER) -----------------------------------
    def _setup_control_plane(self) -> None:
        host, port = self.net.split_host_port(self.front_router_addr)
        endpoint = self.net.get_tcp_uri(host, port)
        self.control_socket = self.context.socket(zmq.DEALER)
        self.control_socket.setsockopt(zmq.IDENTITY, f"ctrl-{self.stage_idx}-{uuid4().hex}".encode())
        if self.timeout_ms > 0:
            self.control_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.control_socket.connect(endpoint)
        self.log.info("Stage %s connected to Front ROUTER at %s", self.stage_idx, endpoint)

    def _setup_router_socket(self) -> None:
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 1000)
        if self.timeout_ms > 0:
            self.router_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.router_socket.bind(self.router_endpoint)
        self.log.info("Stage %s bound ROUTER socket on %s (data-plane forward/return)",
                      self.stage_idx, self.router_endpoint)
        if self.metrics.enabled:
            self.metrics.set("connection_status", 1, socket_type="router")
        self._register_stage()

    def _register_stage(self) -> None:
        if self.control_socket is None:
            return
        payload = {"cmd": "REGISTER", "stage_idx": self.stage_idx, "endpoint": self.router_endpoint}
        self.control_socket.send_multipart([b"REGISTER", json.dumps(payload).encode()])
        try:
            self.control_socket.recv_multipart()
            self.log.info("Stage %s registered with Front ROUTER", self.stage_idx)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Stage %s registration ack failed: %s", self.stage_idx, exc)

    def _request_assignment(self, target_stage_idx: int, direction: str) -> tuple[str, str]:
        if self.control_socket is None:
            raise RuntimeError("Control socket not initialized")
        request = {"cmd": "ASSIGN", "requester_stage_idx": self.stage_idx,
                   "target_stage_idx": target_stage_idx, "direction": direction}
        self.control_socket.send_multipart([b"ASSIGN", json.dumps(request).encode()])
        reply = self.control_socket.recv_multipart()
        if len(reply) < 2:
            raise RuntimeError(f"Invalid ASSIGN reply from Front ROUTER: {reply}")
        try:
            payload = json.loads(reply[1].decode())
            endpoint = payload["endpoint"]
            routing_key = payload.get("routing_key") or f"rk-{uuid4().hex}"
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to parse ASSIGN reply: {reply}") from exc
        self.log.info("Stage %s assignment -> stage %s: endpoint=%s routing_key=%s",
                      self.stage_idx, target_stage_idx, endpoint, routing_key)
        return endpoint, routing_key

    def _connect_dealer(self, endpoint: str, routing_key: str) -> zmq.Socket:
        dealer = self.context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 1000)
        dealer.setsockopt(zmq.IDENTITY, routing_key.encode())
        if self.timeout_ms > 0:
            dealer.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        dealer.connect(endpoint)
        if self.metrics.enabled:
            self.metrics.set("connection_status", 1, socket_type="dealer")
        return dealer

    def _get_forward_socket(self) -> zmq.Socket:
        if self.is_last:
            raise RuntimeError("Last stage has no forward socket")
        if self.forward_socket is None:
            endpoint, routing_key = self._request_assignment(self.stage_idx + 1, "forward")
            self.forward_socket = self._connect_dealer(endpoint, routing_key)
        return self.forward_socket

    def _get_return_socket(self) -> zmq.Socket:
        if not self.is_last:
            raise RuntimeError("Return socket only on last stage")
        if self.return_socket is None:
            endpoint, routing_key = self._request_assignment(0, "return")
            self.return_socket = self._connect_dealer(endpoint, routing_key)
        return self.return_socket

    def _recv_with_identity(self, socket: zmq.Socket) -> tuple[bytes | None, list[bytes]]:
        parts = socket.recv_multipart()
        if socket.getsockopt(zmq.TYPE) == zmq.ROUTER:
            if not parts:
                raise RuntimeError("Received empty message on ROUTER socket")
            return parts[0], parts[1:]
        return None, parts

    # ---- ZeroMQ-only data path ------------------------------------------------
    def send_tensor_dict(self, tensor_dict: dict[str, torch.Tensor | Any],
                         dst: int | None = None) -> None:
        send_start = time.time()
        m = self.metrics
        if self.is_last:
            socket = self._get_return_socket()
            direction = "return"
        else:
            socket = self._get_forward_socket()
            direction = "forward"

        metadata_list, tensor_list = self.codec.split(tensor_dict)
        if m.enabled:
            m.observe("tensor_count", len(tensor_list), direction=direction)

        metadata_with_timestamp = {"metadata_list": metadata_list, "send_timestamp": None}
        metadata_bytes = pickle.dumps(metadata_with_timestamp)

        tensor_bytes_list = []
        total_tensor_bytes = 0
        for tensor in tensor_list:
            buffer = io.BytesIO()
            torch.save(tensor.cpu(), buffer)
            tensor_bytes = buffer.getvalue()
            tensor_bytes_list.append(tensor_bytes)
            total_tensor_bytes += len(tensor_bytes)
            if m.enabled:
                m.observe("tensor_size", len(tensor_bytes), direction=direction)

        if m.enabled:
            m.observe("serialization_latency", time.time() - send_start)

        # send_timestamp just before the network send (after serialization)
        metadata_with_timestamp["send_timestamp"] = time.time()
        metadata_bytes = pickle.dumps(metadata_with_timestamp)

        enqueue_start = time.time()
        socket.send_multipart([len(metadata_bytes).to_bytes(8, "big"), metadata_bytes,
                               len(tensor_bytes_list).to_bytes(8, "big")])
        for tensor_bytes in tensor_bytes_list:
            socket.send_multipart([len(tensor_bytes).to_bytes(8, "big"), tensor_bytes])
        enqueue_time = time.time() - enqueue_start

        total_send_time = time.time() - send_start
        total_bytes = len(metadata_bytes) + total_tensor_bytes + (len(tensor_bytes_list) + 1) * 8
        if m.enabled:
            m.inc("bytes_sent", total_bytes, direction=direction)
            m.inc("messages_sent", direction=direction)
            m.observe("send_latency", total_send_time, direction=direction)
            m.observe("enqueue_latency", enqueue_time, direction=direction)

    def recv_tensor_dict(self, src: int | None = None) -> dict[str, torch.Tensor | Any] | None:
        recv_start = time.time()
        m = self.metrics
        if self.router_socket is None:
            raise RuntimeError("Router socket not initialized")

        direction = "return" if self.is_first else "forward"
        try:
            _identity, parts = self._recv_with_identity(self.router_socket)
            if len(parts) < 3:
                raise RuntimeError(f"Invalid message format: expected at least 3 parts, got {len(parts)}")
            metadata_size = int.from_bytes(parts[0], "big")
            metadata_bytes = parts[1]
            num_tensors = int.from_bytes(parts[2], "big")
            if len(metadata_bytes) != metadata_size:
                raise RuntimeError(
                    f"Metadata size mismatch: expected {metadata_size}, got {len(metadata_bytes)}")
            try:
                metadata_payload = pickle.loads(metadata_bytes)
                if isinstance(metadata_payload, dict) and "metadata_list" in metadata_payload:
                    metadata_list = metadata_payload["metadata_list"]
                    send_timestamp = metadata_payload.get("send_timestamp")
                else:
                    metadata_list = metadata_payload
                    send_timestamp = None
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Stage {self.stage_idx}: Failed to deserialize metadata: {exc}") from exc

            meta_received_time = time.time()
            tensor_bytes_list: list[bytes] = []
            total_bytes_received = len(metadata_bytes) + 16
            for _ in range(num_tensors):
                _identity, tensor_parts = self._recv_with_identity(self.router_socket)
                if len(tensor_parts) < 2:
                    raise RuntimeError("Invalid tensor message format")
                tensor_size = int.from_bytes(tensor_parts[0], "big")
                tensor_bytes = tensor_parts[1]
                total_bytes_received += tensor_size + 8
                if len(tensor_bytes) != tensor_size:
                    raise RuntimeError(
                        f"Tensor size mismatch: expected {tensor_size}, got {len(tensor_bytes)}")
                tensor_bytes_list.append(tensor_bytes)
                if m.enabled:
                    m.observe("tensor_size", tensor_size, direction=direction)

            data_received_time = time.time()
            deserialization_start = time.time()
            tensor_list = []
            for tensor_bytes in tensor_bytes_list:
                buffer = io.BytesIO(tensor_bytes)
                try:
                    loaded_obj = torch.load(buffer, map_location="cpu", weights_only=True)
                except Exception:
                    buffer.seek(0)
                    loaded_obj = torch.load(buffer, map_location="cpu", weights_only=False)
                if not isinstance(loaded_obj, torch.Tensor):
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: Deserialization failed - expected torch.Tensor, "
                        f"got {type(loaded_obj)}")
                tensor_list.append(loaded_obj)
            if m.enabled:
                m.observe("deserialization_latency", time.time() - deserialization_start)

            gpu_transfer_start = time.time()
            tensor_dict: dict[str, Any] = {}
            tensor_idx = 0
            for key, value in metadata_list:
                if self.codec.is_tensor_meta(value):
                    if tensor_idx >= len(tensor_list):
                        raise RuntimeError(
                            f"Stage {self.stage_idx}: Not enough tensors for key '{key}', "
                            f"expected {tensor_idx + 1}, got {len(tensor_list)}")
                    tensor = tensor_list[tensor_idx]
                    if tensor.dtype != value.dtype:
                        tensor = tensor.to(dtype=value.dtype)
                    if self.device.startswith("cuda"):
                        copy_start = time.time()
                        tensor = tensor.to(dtype=value.dtype, device=self.device)
                        if m.enabled:
                            m.observe("buffer_copy_latency", time.time() - copy_start,
                                      operation="copy_to_gpu")
                    else:
                        tensor = tensor.to(dtype=value.dtype, device="cpu")
                    tensor_dict[key] = tensor
                    tensor_idx += 1
                else:
                    tensor_dict[key] = value

            gpu_transfer_time = time.time() - gpu_transfer_start
            if m.enabled:
                m.inc("bytes_received", total_bytes_received, direction=direction)
                m.inc("messages_received", direction=direction)
                m.observe("recv_latency", time.time() - recv_start, direction=direction)
                m.observe("ready_to_use_latency", gpu_transfer_time, direction=direction)
                m.observe("delivery_latency_local",
                          max(0.0, data_received_time - meta_received_time), direction=direction)
                if send_timestamp is not None:
                    dl = data_received_time - send_timestamp
                    if dl >= 0:
                        m.observe("delivery_latency", dl, direction=direction)
            return tensor_dict

        except zmq.Again:
            if m.enabled:
                m.inc("errors", error_type="timeout")
            self.log.error("Stage %s receive timeout after %sms.", self.stage_idx, self.timeout_ms)
            return None
        except Exception as exc:  # noqa: BLE001
            if m.enabled:
                m.inc("errors", error_type="network" if isinstance(exc, zmq.ZMQError) else "serialization")
            raise RuntimeError(f"Stage {self.stage_idx} failed to receive tensor dict: {exc}") from exc

    def close(self) -> None:
        for name in ("forward_socket", "return_socket", "router_socket", "control_socket"):
            s = getattr(self, name, None)
            if s:
                s.close()
                setattr(self, name, None)
        if self.context:
            self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
