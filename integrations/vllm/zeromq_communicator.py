# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ZeroMQ communication layer for external pipeline mode (ROUTER/DEALER).

Control-plane: a Front ROUTER assigns stage endpoints + routing_key.
Data-plane: direct DEALER -> Stage ROUTER (one hop) for both forward and
return directions. The Front ROUTER is contacted only on (re)assignment.
"""

import io
import json
import pickle
import time
from typing import Any
from uuid import uuid4

import torch
import zmq

from vllm.distributed.parallel_state import TensorMetadata, _split_tensor_dict
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_tcp_uri, split_host_port

# Import metrics (optional, gracefully handle if prometheus not available)
try:
    from vllm.distributed.device_communicators.pipeline_comm_metrics import (
        vllm_pipeline_comm_bytes_received,
        vllm_pipeline_comm_bytes_sent,
        vllm_pipeline_comm_connection_status,
        vllm_pipeline_comm_deserialization_latency,
        vllm_pipeline_comm_errors,
        vllm_pipeline_comm_messages_received,
        vllm_pipeline_comm_messages_sent,
        vllm_pipeline_comm_recv_latency,
        vllm_pipeline_comm_send_latency,
        vllm_pipeline_comm_serialization_latency,
        vllm_pipeline_comm_tensor_count,
        vllm_pipeline_comm_tensor_size,
        # Three-level latency breakdown
        vllm_pipeline_comm_enqueue_latency,
        vllm_pipeline_comm_delivery_latency,
        vllm_pipeline_comm_delivery_latency_local,
        vllm_pipeline_comm_ready_to_use_latency,
        vllm_pipeline_comm_buffer_copy_latency,
        # Connection ready time metrics
        vllm_pipeline_connection_ready_time,
        vllm_pipeline_connection_discovery_time,
        vllm_pipeline_connection_ready_timestamp,
    )
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

logger = init_logger(__name__)

if METRICS_ENABLED:
    logger.info("Pipeline communication metrics enabled for pipeline parallelism monitoring")
else:
    logger.warning("Pipeline communication metrics disabled (prometheus_client not available)")


class ZeroMQPPCommunicator:
    """ROUTER/DEALER ZeroMQ communicator for external pipeline mode."""

    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        router_port: int,
        front_router_addr: str,
        device: str = "cuda",
        timeout_ms: int = -1,
    ):
        # Record connection initialization start time
        init_start_time = time.time()
        
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.device = device
        self.timeout_ms = timeout_ms

        self.is_first = stage_idx == 0
        self.is_last = stage_idx == num_stages - 1

        self.context = zmq.Context()

        # Control-plane socket
        self.front_router_addr = front_router_addr
        self.control_socket: zmq.Socket | None = None

        # Data-plane sockets
        self.router_port = router_port
        self.router_endpoint = get_tcp_uri(get_ip(), router_port)
        self.router_socket: zmq.Socket | None = None
        self.forward_socket: zmq.Socket | None = None
        self.return_socket: zmq.Socket | None = None

        self._setup_control_plane()
        self._setup_router_socket()
        
        # Record connection ready time
        connection_ready_time = time.time() - init_start_time
        if METRICS_ENABLED:
            vllm_pipeline_connection_ready_time.labels(
                backend=self.backend_name, stage_idx=self.stage_idx
            ).observe(connection_ready_time)
            vllm_pipeline_connection_ready_timestamp.labels(
                backend=self.backend_name, stage_idx=self.stage_idx
            ).set(time.time())
            logger.info(
                "Stage %s connection ready in %.4f seconds (backend=%s)",
                self.stage_idx, connection_ready_time, self.backend_name
            )
    
    @property
    def backend_name(self) -> str:
        """Get the backend name for metrics labeling.
        
        Subclasses should override this to return their specific backend name.
        """
        # Default to class name without 'Communicator' suffix, lowercased
        class_name = self.__class__.__name__
        if class_name.endswith("Communicator"):
            backend = class_name[:-len("Communicator")].lower()
        else:
            backend = class_name.lower()
        
        # Map common names to standard backend identifiers
        backend_map = {
            "zeromqpp": "zeromq-naive",
            "ncclreconfig": "nccl-reconfig",
            "nixl": "nixl",
            "mooncake": "mooncake",
            "connex": "connex",
            "connexte": "connexte",
        }
        return backend_map.get(backend, backend)
    
    def _setup_control_plane(self) -> None:
        host, port = split_host_port(self.front_router_addr)
        endpoint = get_tcp_uri(host, port)
        self.control_socket = self.context.socket(zmq.DEALER)
        self.control_socket.setsockopt(zmq.IDENTITY, f"ctrl-{self.stage_idx}-{uuid4().hex}".encode())
        if self.timeout_ms > 0:
            self.control_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.control_socket.connect(endpoint)
        logger.info("Stage %s connected to Front ROUTER at %s", self.stage_idx, endpoint)

    def _setup_router_socket(self) -> None:
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 1000)
        if self.timeout_ms > 0:
            self.router_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.router_socket.bind(self.router_endpoint)
        logger.info(
            "Stage %s bound ROUTER socket on %s (data-plane forward/return)",
            self.stage_idx,
            self.router_endpoint,
        )
        if METRICS_ENABLED:
            vllm_pipeline_comm_connection_status.labels(
                backend=self.backend_name, stage_idx=self.stage_idx, socket_type="router"
            ).set(1)
        self._register_stage()

    def _register_stage(self) -> None:
        if self.control_socket is None:
            return
        payload = {
            "cmd": "REGISTER",
            "stage_idx": self.stage_idx,
            "endpoint": self.router_endpoint,
        }
        self.control_socket.send_multipart([b"REGISTER", json.dumps(payload).encode()])
        try:
            _reply = self.control_socket.recv_multipart()
            logger.info("Stage %s registered with Front ROUTER", self.stage_idx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage %s registration ack failed: %s", self.stage_idx, exc)

    def _request_assignment(self, target_stage_idx: int, direction: str) -> tuple[str, str]:
        if self.control_socket is None:
            raise RuntimeError("Control socket not initialized")
        request = {
            "cmd": "ASSIGN",
            "requester_stage_idx": self.stage_idx,
            "target_stage_idx": target_stage_idx,
            "direction": direction,
        }
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
        logger.info(
            "Stage %s assignment -> stage %s: endpoint=%s routing_key=%s",
            self.stage_idx,
            target_stage_idx,
            endpoint,
            routing_key,
        )
        return endpoint, routing_key

    def _connect_dealer(self, endpoint: str, routing_key: str) -> zmq.Socket:
        dealer = self.context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 1000)
        dealer.setsockopt(zmq.IDENTITY, routing_key.encode())
        if self.timeout_ms > 0:
            dealer.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        dealer.connect(endpoint)
        if METRICS_ENABLED:
            vllm_pipeline_comm_connection_status.labels(
                backend=self.backend_name, stage_idx=self.stage_idx, socket_type="dealer"
            ).set(1)
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

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
    ) -> None:
        send_start = time.time()

        if self.is_last:
            socket = self._get_return_socket()
            direction = "return"
        else:
            socket = self._get_forward_socket()
            direction = "forward"

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)

        if METRICS_ENABLED:
            vllm_pipeline_comm_tensor_count.labels(
                backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
            ).observe(len(tensor_list))

        # Note: send_timestamp will be set just before network send
        metadata_with_timestamp = {
            "metadata_list": metadata_list,
            "send_timestamp": None,  # Placeholder, will be set later
        }
        metadata_bytes = pickle.dumps(metadata_with_timestamp)

        tensor_bytes_list = []
        total_tensor_bytes = 0
        for tensor in tensor_list:
            buffer = io.BytesIO()
            torch.save(tensor.cpu(), buffer)
            tensor_bytes = buffer.getvalue()
            tensor_bytes_list.append(tensor_bytes)
            tensor_size = len(tensor_bytes)
            total_tensor_bytes += tensor_size
            if METRICS_ENABLED:
                vllm_pipeline_comm_tensor_size.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
                ).observe(tensor_size)

        serialization_time = time.time() - send_start
        if METRICS_ENABLED:
            vllm_pipeline_comm_serialization_latency.labels(
                backend=self.backend_name, stage_idx=self.stage_idx
            ).observe(serialization_time)

        # v1.5.6: Set send_timestamp just before network send (after serialization)
        # This ensures delivery_latency accurately measures network transfer time
        metadata_with_timestamp["send_timestamp"] = time.time()
        metadata_bytes = pickle.dumps(metadata_with_timestamp)
        
        # Record enqueue start time (after serialization, before network send)
        enqueue_start = time.time()
        
        socket.send_multipart(
            [
                len(metadata_bytes).to_bytes(8, "big"),
                metadata_bytes,
                len(tensor_bytes_list).to_bytes(8, "big"),
            ]
        )
        for tensor_bytes in tensor_bytes_list:
            socket.send_multipart(
                [
                    len(tensor_bytes).to_bytes(8, "big"),
                    tensor_bytes,
                ]
            )
        
        # Enqueue latency: time for all send() calls to return
        enqueue_time = time.time() - enqueue_start

        total_send_time = time.time() - send_start
        total_bytes = len(metadata_bytes) + total_tensor_bytes + (len(tensor_bytes_list) + 1) * 8
        if METRICS_ENABLED:
            vllm_pipeline_comm_bytes_sent.labels(
                backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
            ).inc(total_bytes)
            vllm_pipeline_comm_messages_sent.labels(
                backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
            ).inc()
            vllm_pipeline_comm_send_latency.labels(
                backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
            ).observe(total_send_time)
            # Record enqueue latency (for ZeroMQ, this is the time for all send() calls)
            vllm_pipeline_comm_enqueue_latency.labels(
                backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
            ).observe(enqueue_time)

    def recv_tensor_dict(
        self,
        src: int | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        recv_start = time.time()
        
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
                    f"Metadata size mismatch: expected {metadata_size}, got {len(metadata_bytes)}"
                )

            try:
                metadata_payload = pickle.loads(metadata_bytes)
                # Handle both old format (list) and new format (dict with timestamp)
                if isinstance(metadata_payload, dict) and "metadata_list" in metadata_payload:
                    metadata_list = metadata_payload["metadata_list"]
                    send_timestamp = metadata_payload.get("send_timestamp")
                else:
                    # Old format: metadata_payload is the metadata_list directly
                    metadata_list = metadata_payload
                    send_timestamp = None
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Stage {self.stage_idx}: Failed to deserialize metadata: {exc}") from exc

            # Receiver-local reference point for delivery latency (metadata received)
            meta_received_time = time.time()
            
            # Receive tensor payloads first (network time only), then deserialize.
            tensor_bytes_list: list[bytes] = []
            total_bytes_received = len(metadata_bytes) + 16  # metadata_size(8) + num_tensors(8)
            for _ in range(num_tensors):
                _identity, tensor_parts = self._recv_with_identity(self.router_socket)
                if len(tensor_parts) < 2:
                    raise RuntimeError("Invalid tensor message format")
                tensor_size = int.from_bytes(tensor_parts[0], "big")
                tensor_bytes = tensor_parts[1]
                total_bytes_received += tensor_size + 8

                if len(tensor_bytes) != tensor_size:
                    raise RuntimeError(
                        f"Tensor size mismatch: expected {tensor_size}, got {len(tensor_bytes)}"
                    )

                tensor_bytes_list.append(tensor_bytes)
                if METRICS_ENABLED:
                    vllm_pipeline_comm_tensor_size.labels(
                        backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
                    ).observe(tensor_size)

            # All network payload received (before deserialization / GPU transfer)
            data_received_time = time.time()

            # Measure deserialization time
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
                        f"Stage {self.stage_idx}: Deserialization failed - expected torch.Tensor, got {type(loaded_obj)}"
                    )

                tensor_list.append(loaded_obj)

            deserialization_time = time.time() - deserialization_start
            if METRICS_ENABLED:
                vllm_pipeline_comm_deserialization_latency.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx
                ).observe(deserialization_time)

            # v1.5.6: Measure GPU transfer time separately (ready-to-use latency)
            # This only measures device transfer, not deserialization
            gpu_transfer_start = time.time()
            tensor_dict: dict[str, Any] = {}
            tensor_idx = 0
            metadata_keys = [k for k, _ in metadata_list]
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    if tensor_idx >= len(tensor_list):
                        raise RuntimeError(
                            f"Stage {self.stage_idx}: Not enough tensors for key '{key}', expected {tensor_idx + 1}, got {len(tensor_list)}"
                        )
                    tensor = tensor_list[tensor_idx]
                    if tensor.dtype != value.dtype:
                        tensor = tensor.to(dtype=value.dtype)
                    if self.device.startswith("cuda"):
                        copy_start = time.time()
                        tensor = tensor.to(dtype=value.dtype, device=self.device)
                        if METRICS_ENABLED:
                            vllm_pipeline_comm_buffer_copy_latency.labels(
                                backend=self.backend_name, stage_idx=self.stage_idx, operation="copy_to_gpu"
                            ).observe(time.time() - copy_start)
                    else:
                        tensor = tensor.to(dtype=value.dtype, device="cpu")
                    tensor_dict[key] = tensor
                    tensor_idx += 1
                else:
                    tensor_dict[key] = value

            gpu_transfer_time = time.time() - gpu_transfer_start
            total_recv_time = time.time() - recv_start
            
            if METRICS_ENABLED:
                vllm_pipeline_comm_bytes_received.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
                ).inc(total_bytes_received)
                vllm_pipeline_comm_messages_received.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
                ).inc()
                vllm_pipeline_comm_recv_latency.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
                ).observe(total_recv_time)
                # v1.5.6: Record ready-to-use latency as GPU transfer time only
                vllm_pipeline_comm_ready_to_use_latency.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
                ).observe(gpu_transfer_time)
                # v1.5.6: Record delivery latency = network time only (data_received_time - send_timestamp)
                vllm_pipeline_comm_delivery_latency_local.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
                ).observe(max(0.0, data_received_time - meta_received_time))
                if send_timestamp is not None:
                    delivery_latency = data_received_time - send_timestamp
                    if delivery_latency >= 0:
                        vllm_pipeline_comm_delivery_latency.labels(
                            backend=self.backend_name, stage_idx=self.stage_idx, direction=direction
                        ).observe(delivery_latency)

            return tensor_dict

        except zmq.Again:
            if METRICS_ENABLED:
                vllm_pipeline_comm_errors.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx, error_type="timeout"
                ).inc()
            logger.error(
                "Stage %s receive timeout after %sms.", self.stage_idx, self.timeout_ms
            )
            return None
        except Exception as exc:  # noqa: BLE001
            error_type = "network" if isinstance(exc, zmq.ZMQError) else "serialization"
            if METRICS_ENABLED:
                vllm_pipeline_comm_errors.labels(
                    backend=self.backend_name, stage_idx=self.stage_idx, error_type=error_type
                ).inc()
            raise RuntimeError(f"Stage {self.stage_idx} failed to receive tensor dict: {exc}") from exc

    def close(self) -> None:
        if self.forward_socket:
            self.forward_socket.close()
            self.forward_socket = None
        if self.return_socket:
            self.return_socket.close()
            self.return_socket = None
        if self.router_socket:
            self.router_socket.close()
            self.router_socket = None
        if self.control_socket:
            self.control_socket.close()
            self.control_socket = None
        if self.context:
            self.context.term()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
