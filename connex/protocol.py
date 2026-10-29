# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Mobility-Aware Protocol Header and Message Format.

Defines the unified flow model and protocol headers for dynamic endpoint mobility:
- LID: Stable logical endpoint identifier (stage replica identity)
- epoch: Route version for reconfiguration/migration
- stream_id: Multiplexed stream identifier (request_id, direction, token_step)
- seqno: In-stream ordering and deduplication
- class/prio: Critical path token vs background migration
- credit: Backpressure and quota management

This converts "dynamics" from control plane events into data plane verifiable semantics:
- Stale epoch detection
- Duplicate sequence dropping
- Priority-based scheduling

"""

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, Union
import hashlib


class MessageClass(IntEnum):
    """Traffic priority classes for QoS scheduling.
    
    Higher value = higher priority.
    Critical token traffic takes precedence over background migration.
    """
    MIGRATION_BULK = 0      # Background migration/checkpoint data
    MIGRATION_STATE = 1     # Migration state transfers
    CONTROL = 2             # Control messages (heartbeat, routing, credits)
    KV_CACHE = 3            # KV cache transfer (Mooncake TE integration)
    TOKEN_NORMAL = 4        # Normal token flow
    TOKEN_CRITICAL = 5      # Critical path token (first token, low latency)


class MessageType(IntEnum):
    """Message types in the protocol."""
    # Data plane messages
    DATA = 0                # Regular data payload
    DATA_CHUNKED = 1        # Chunked tensor data
    DATA_LAST_CHUNK = 2     # Last chunk of chunked transfer
    
    # Control plane messages
    CREDIT_UPDATE = 10      # Credit update/backpressure
    CREDIT_REQUEST = 11     # Request credits from peer
    
    # Routing messages
    EPOCH_UPDATE = 20       # New epoch announcement
    STALE_EPOCH = 21        # Stale epoch feedback (fast route refresh)
    ROUTE_REQUEST = 22      # Request route for LID
    ROUTE_RESPONSE = 23     # Route response
    
    # Cutover messages
    PREPARE = 30            # Prepare for migration
    PREPARE_ACK = 31        # Preparation complete
    CUTOVER = 32            # Begin traffic switch
    CUTOVER_ACK = 33        # Cutover confirmed
    DRAIN = 34              # Drain notification
    DRAIN_COMPLETE = 35     # Drain finished
    FINALIZE = 36           # Finalize migration
    
    # Health messages
    HEARTBEAT = 40          # Keepalive
    HEARTBEAT_ACK = 41      # Keepalive response


class PDRole:
    """PD (Prefill-Decode) separation roles."""
    NONE = "none"           # Not in PD separation mode
    PREFILL = "prefill"     # Prefill instance (KV producer)
    DECODE = "decode"       # Decode instance (KV consumer)


@dataclass
class LogicalEndpointID:
    """Logical Endpoint Identifier (LID).
    
    A stable identity for stage replicas that survives migration.
    Physical addresses may change, but LID remains constant.
    
    Format: (model_name, stage_idx, shard_idx, replica_idx, pd_role)
    
    For PD separation:
    - pd_role="prefill": KV producer, sends KV cache after prefill
    - pd_role="decode": KV consumer, receives KV cache for decoding
    - pd_role="none": Not in PD mode (pipeline parallelism)
    """
    model_name: str
    stage_idx: int
    shard_idx: int = 0
    replica_idx: int = 0
    pd_role: str = PDRole.NONE  # NEW: "none", "prefill", or "decode"
    
    def __hash__(self) -> int:
        return hash(self.to_tuple())
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, LogicalEndpointID):
            return self.to_tuple() == other.to_tuple()
        if isinstance(other, tuple):
            # Support both 4-tuple and 5-tuple comparison
            my_tuple = self.to_tuple()
            if len(other) == 4:
                return my_tuple[:4] == other
            return my_tuple == other
        return False
    
    def to_tuple(self) -> Tuple[str, int, int, int, str]:
        """Convert to tuple representation (5-tuple with pd_role)."""
        return (self.model_name, self.stage_idx, self.shard_idx, self.replica_idx, self.pd_role)
    
    def to_tuple_legacy(self) -> Tuple[str, int, int, int]:
        """Convert to legacy 4-tuple (without pd_role) for backward compatibility."""
        return (self.model_name, self.stage_idx, self.shard_idx, self.replica_idx)
    
    @classmethod
    def from_tuple(cls, t: Union[Tuple[str, int, int, int], Tuple[str, int, int, int, str]]) -> "LogicalEndpointID":
        """Create from tuple (supports both 4-tuple and 5-tuple)."""
        if len(t) == 5:
            return cls(model_name=t[0], stage_idx=t[1], shard_idx=t[2], replica_idx=t[3], pd_role=t[4])
        return cls(model_name=t[0], stage_idx=t[1], shard_idx=t[2], replica_idx=t[3])
    
    @property
    def is_prefill(self) -> bool:
        """Check if this is a prefill instance."""
        return self.pd_role == PDRole.PREFILL
    
    @property
    def is_decode(self) -> bool:
        """Check if this is a decode instance."""
        return self.pd_role == PDRole.DECODE
    
    @property
    def is_pd_mode(self) -> bool:
        """Check if this endpoint is in PD separation mode."""
        return self.pd_role != PDRole.NONE
    
    def __str__(self) -> str:
        if self.pd_role == PDRole.NONE:
            return f"LID({self.model_name}:s{self.stage_idx}:sh{self.shard_idx}:r{self.replica_idx})"
        return f"LID({self.model_name}:s{self.stage_idx}:sh{self.shard_idx}:r{self.replica_idx}:{self.pd_role})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class StreamID:
    """Stream identifier for multiplexing.
    
    Each stream represents a logical flow of messages with ordering guarantees.
    Format: (request_id, direction, token_step)
    """
    request_id: str           # Unique request identifier
    direction: str            # "forward" or "backward" or "kv"
    token_step: int = 0       # Token generation step (for autoregressive)
    
    def __hash__(self) -> int:
        return hash((self.request_id, self.direction, self.token_step))
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, StreamID):
            return (self.request_id == other.request_id and 
                    self.direction == other.direction and
                    self.token_step == other.token_step)
        return False
    
    def to_str(self) -> str:
        """Serialize to string for lookup keys."""
        return f"{self.request_id}:{self.direction}:{self.token_step}"
    
    @classmethod
    def from_str(cls, s: str) -> "StreamID":
        """Parse from string."""
        parts = s.split(":", 2)
        if len(parts) == 3:
            return cls(
                request_id=parts[0],
                direction=parts[1],
                token_step=int(parts[2])
            )
        raise ValueError(f"Invalid stream ID format: {s}")
    
    def __str__(self) -> str:
        return f"Stream({self.request_id[:8]}:{self.direction}:{self.token_step})"


@dataclass
class CreditInfo:
    """Credit information for backpressure control.
    
    Receiver advertises available credits to sender.
    Sender must not exceed advertised credits.
    """
    # Per-class credits
    credits: dict[MessageClass, int] = field(default_factory=lambda: {
        MessageClass.TOKEN_CRITICAL: 100,
        MessageClass.TOKEN_NORMAL: 100,
        MessageClass.KV_CACHE: 50,
        MessageClass.CONTROL: 50,
        MessageClass.MIGRATION_STATE: 20,
        MessageClass.MIGRATION_BULK: 10,
    })
    
    # Window size for congestion control
    window_size: int = 64
    
    # RTT estimate for pacing (microseconds)
    rtt_us: int = 0
    
    def get_credit(self, msg_class: MessageClass) -> int:
        """Get available credits for message class."""
        return self.credits.get(msg_class, 0)
    
    def has_credit(self, msg_class: MessageClass, count: int = 1) -> bool:
        """Check if sufficient credits available."""
        return self.get_credit(msg_class) >= count
    
    def consume(self, msg_class: MessageClass, count: int = 1) -> bool:
        """Consume credits. Returns True if successful."""
        if self.has_credit(msg_class, count):
            self.credits[msg_class] -= count
            return True
        return False
    
    def replenish(self, msg_class: MessageClass, count: int) -> None:
        """Replenish credits (typically from receiver ack)."""
        max_credit = self._max_credit(msg_class)
        self.credits[msg_class] = min(
            self.credits.get(msg_class, 0) + count,
            max_credit
        )
    
    def _max_credit(self, msg_class: MessageClass) -> int:
        """Get max credit for message class."""
        defaults = {
            MessageClass.TOKEN_CRITICAL: 200,
            MessageClass.TOKEN_NORMAL: 200,
            MessageClass.KV_CACHE: 100,
            MessageClass.CONTROL: 100,
            MessageClass.MIGRATION_STATE: 50,
            MessageClass.MIGRATION_BULK: 20,
        }
        return defaults.get(msg_class, 100)


@dataclass
class ProtocolHeader:
    """Mobility-Aware Protocol Header.
    
    This header enables data plane verification of dynamic semantics:
    - Epoch validation: detect stale routes immediately
    - Sequence ordering: in-stream ordering and deduplication
    - Class priority: QoS scheduling at receiver
    - Credit flow: backpressure signaling
    
    Wire format (fixed 64 bytes):
    - magic (4B): Protocol magic number
    - version (2B): Protocol version
    - msg_type (2B): Message type
    - epoch (4B): Route epoch
    - seqno (8B): Stream sequence number
    - msg_class (1B): Message priority class
    - flags (1B): Message flags
    - credit (2B): Credit update (piggyback)
    - stream_hash (8B): Hash of stream_id for fast lookup
    - src_lid_hash (8B): Hash of source LID
    - dst_lid_hash (8B): Hash of destination LID
    - timestamp (4B): Send timestamp, low 32 bits of microseconds (wraps ~71 min)
    - payload_len (4B): Payload length
    - chunk_idx (2B), total_chunks (2B)
    - checksum (4B): Header checksum
    """
    # Protocol identification
    magic: int = 0x434F4E58  # "CONX"
    version: int = 0x0301    # v3.0.1
    
    # Message metadata
    msg_type: MessageType = MessageType.DATA
    epoch: int = 0
    seqno: int = 0
    msg_class: MessageClass = MessageClass.TOKEN_NORMAL
    
    # Flags
    is_retransmit: bool = False
    requires_ack: bool = False
    is_chunked: bool = False
    is_last_chunk: bool = False
    
    # Credit piggyback
    credit: int = 0
    
    # Endpoint identification (hashed for fixed size)
    stream_id: Optional[StreamID] = None
    src_lid: Optional[LogicalEndpointID] = None
    dst_lid: Optional[LogicalEndpointID] = None
    
    # Timing
    timestamp_us: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    
    # Payload
    payload_len: int = 0
    
    # For chunked transfers
    chunk_idx: int = 0
    total_chunks: int = 1
    
    # Computed fields
    _stream_hash: int = 0
    _src_lid_hash: int = 0
    _dst_lid_hash: int = 0
    
    def __post_init__(self):
        """Compute hash fields after initialization."""
        if self.stream_id:
            self._stream_hash = self._compute_hash(self.stream_id.to_str())
        if self.src_lid:
            self._src_lid_hash = self._compute_hash(str(self.src_lid.to_tuple()))
        if self.dst_lid:
            self._dst_lid_hash = self._compute_hash(str(self.dst_lid.to_tuple()))
    
    @staticmethod
    def _compute_hash(s: str) -> int:
        """Compute 8-byte hash for string."""
        h = hashlib.md5(s.encode()).digest()
        return struct.unpack("<Q", h[:8])[0]
    
    @property
    def flags_byte(self) -> int:
        """Pack flags into single byte."""
        flags = 0
        if self.is_retransmit:
            flags |= 0x01
        if self.requires_ack:
            flags |= 0x02
        if self.is_chunked:
            flags |= 0x04
        if self.is_last_chunk:
            flags |= 0x08
        return flags
    
    @flags_byte.setter
    def flags_byte(self, value: int):
        """Unpack flags from byte."""
        self.is_retransmit = bool(value & 0x01)
        self.requires_ack = bool(value & 0x02)
        self.is_chunked = bool(value & 0x04)
        self.is_last_chunk = bool(value & 0x08)
    
    def is_stale(self, current_epoch: int) -> bool:
        """Check if message is from stale epoch."""
        return self.epoch < current_epoch
    
    def is_control(self) -> bool:
        """Check if this is a control message."""
        return self.msg_type.value >= 10
    
    def serialize(self) -> bytes:
        """Serialize header to bytes (64 bytes fixed).
        
        Format:
        - magic: 4B (uint32)
        - version: 2B (uint16)
        - msg_type: 2B (uint16)
        - epoch: 4B (uint32)
        - seqno: 8B (uint64)
        - msg_class: 1B (uint8)
        - flags: 1B (uint8)
        - credit: 2B (uint16)
        - stream_hash: 8B (uint64)
        - src_lid_hash: 8B (uint64)
        - dst_lid_hash: 8B (uint64)
        - timestamp: 4B (uint32, low 32 bits of microseconds)
        - payload_len: 4B (uint32)
        - chunk_idx: 2B (uint16)
        - total_chunks: 2B (uint16)
        """
        header_data = struct.pack(
            "<IHHIQBBHQQQIIHH",
            self.magic,
            self.version,
            self.msg_type.value,
            self.epoch,
            self.seqno,
            self.msg_class.value,
            self.flags_byte,
            self.credit,
            self._stream_hash,
            self._src_lid_hash,
            self._dst_lid_hash,
            self.timestamp_us & 0xFFFFFFFF,
            self.payload_len,
            self.chunk_idx,
            self.total_chunks,
        )
        
        # Pad to 64 bytes and add checksum
        padded = header_data.ljust(60, b'\x00')
        checksum = self._compute_checksum(padded)
        return padded + struct.pack("<I", checksum)
    
    @classmethod
    def deserialize(cls, data: bytes) -> "ProtocolHeader":
        """Deserialize header from bytes."""
        if len(data) < 64:
            raise ValueError(f"Header too short: {len(data)} bytes, need 64")
        
        # Verify checksum
        header_data = data[:60]
        expected_checksum = struct.unpack("<I", data[60:64])[0]
        actual_checksum = cls._compute_checksum(header_data)
        if expected_checksum != actual_checksum:
            raise ValueError(f"Header checksum mismatch: {expected_checksum} != {actual_checksum}")
        
        # Unpack fields
        (
            magic, version, msg_type, epoch, seqno,
            msg_class, flags, credit,
            stream_hash, src_lid_hash, dst_lid_hash,
            timestamp_us, payload_len,
            chunk_idx, total_chunks
        ) = struct.unpack("<IHHIQBBHQQQIIHH", header_data)
        
        if magic != 0x434F4E58:
            raise ValueError(f"Invalid magic number: {magic:08x}")
        
        header = cls(
            magic=magic,
            version=version,
            msg_type=MessageType(msg_type),
            epoch=epoch,
            seqno=seqno,
            msg_class=MessageClass(msg_class),
            credit=credit,
            timestamp_us=timestamp_us,
            payload_len=payload_len,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
        )
        header.flags_byte = flags
        header._stream_hash = stream_hash
        header._src_lid_hash = src_lid_hash
        header._dst_lid_hash = dst_lid_hash
        
        return header
    
    @staticmethod
    def _compute_checksum(data: bytes) -> int:
        """Compute simple checksum for header validation."""
        checksum = 0
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if len(chunk) == 4:
                checksum ^= struct.unpack("<I", chunk)[0]
        return checksum
    
    @classmethod
    def header_size(cls) -> int:
        """Return fixed header size."""
        return 64


@dataclass
class MobilityMessage:
    """Complete message with header and payload.
    
    Combines protocol header with actual data payload.
    Supports both control messages and data transfers.
    """
    header: ProtocolHeader
    payload: bytes = b""
    
    # Optional metadata for internal tracking
    receive_time: float = 0.0
    local_epoch: int = 0
    
    def serialize(self) -> bytes:
        """Serialize complete message."""
        self.header.payload_len = len(self.payload)
        return self.header.serialize() + self.payload
    
    @classmethod
    def deserialize(cls, data: bytes) -> "MobilityMessage":
        """Deserialize complete message."""
        header = ProtocolHeader.deserialize(data[:64])
        payload = data[64:64 + header.payload_len]
        if len(payload) != header.payload_len:
            raise ValueError(
                f"Truncated message: payload {len(payload)} bytes, header says {header.payload_len}"
            )
        return cls(header=header, payload=payload)
    
    def is_stale(self, current_epoch: int) -> bool:
        """Check if message epoch is stale."""
        return self.header.is_stale(current_epoch)
    
    def is_duplicate(self, expected_seqno: int) -> bool:
        """Check if message is a duplicate (seqno already received)."""
        return self.header.seqno < expected_seqno


# Utility functions for creating common message types

def create_data_message(
    stream_id: StreamID,
    src_lid: LogicalEndpointID,
    dst_lid: LogicalEndpointID,
    epoch: int,
    seqno: int,
    payload: bytes,
    msg_class: MessageClass = MessageClass.TOKEN_NORMAL,
) -> MobilityMessage:
    """Create a data message."""
    header = ProtocolHeader(
        msg_type=MessageType.DATA,
        epoch=epoch,
        seqno=seqno,
        msg_class=msg_class,
        stream_id=stream_id,
        src_lid=src_lid,
        dst_lid=dst_lid,
        payload_len=len(payload),
    )
    return MobilityMessage(header=header, payload=payload)


def create_credit_update(
    src_lid: LogicalEndpointID,
    dst_lid: LogicalEndpointID,
    epoch: int,
    credits: dict[MessageClass, int],
) -> MobilityMessage:
    """Create a credit update message."""
    import msgpack
    payload = msgpack.packb({c.value: v for c, v in credits.items()})
    
    header = ProtocolHeader(
        msg_type=MessageType.CREDIT_UPDATE,
        epoch=epoch,
        seqno=0,
        msg_class=MessageClass.CONTROL,
        src_lid=src_lid,
        dst_lid=dst_lid,
        payload_len=len(payload),
    )
    return MobilityMessage(header=header, payload=payload)


def create_stale_epoch_feedback(
    src_lid: LogicalEndpointID,
    dst_lid: LogicalEndpointID,
    stale_epoch: int,
    current_epoch: int,
) -> MobilityMessage:
    """Create stale epoch feedback message.
    
    Sent by receiver when it detects a stale epoch.
    Triggers fast route refresh at sender.
    """
    import msgpack
    payload = msgpack.packb({
        "stale_epoch": stale_epoch,
        "current_epoch": current_epoch,
    })
    
    header = ProtocolHeader(
        msg_type=MessageType.STALE_EPOCH,
        epoch=current_epoch,
        seqno=0,
        msg_class=MessageClass.CONTROL,
        src_lid=src_lid,
        dst_lid=dst_lid,
        payload_len=len(payload),
    )
    return MobilityMessage(header=header, payload=payload)


def create_heartbeat(
    src_lid: LogicalEndpointID,
    dst_lid: LogicalEndpointID,
    epoch: int,
) -> MobilityMessage:
    """Create heartbeat message."""
    header = ProtocolHeader(
        msg_type=MessageType.HEARTBEAT,
        epoch=epoch,
        seqno=0,
        msg_class=MessageClass.CONTROL,
        src_lid=src_lid,
        dst_lid=dst_lid,
        payload_len=0,
    )
    return MobilityMessage(header=header, payload=b"")
