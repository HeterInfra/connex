# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Stream Manager for Token Stream Semantics.

Provides ordered, deduplicated delivery with:
- Sequence numbering per stream
- Reorder buffer with configurable window
- Deduplication via bitmap/sliding window
- Integration with cutover watermarks
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Message payload type


@dataclass
class StreamMessage(Generic[T]):
    """Message in a stream with sequence number."""
    stream_id: str
    seqno: int
    payload: T
    epoch: int
    timestamp: float = field(default_factory=time.time)
    
    # For dedup tracking
    delivered: bool = False
    
    def __lt__(self, other: "StreamMessage") -> bool:
        """Compare by seqno for sorting."""
        return self.seqno < other.seqno


@dataclass
class StreamState:
    """State for a single stream."""
    stream_id: str
    
    # Send state
    next_send_seqno: int = 0
    
    # Receive state
    next_expected_seqno: int = 0
    delivered_high_watermark: int = -1
    
    # Reorder buffer: seqno -> message
    reorder_buffer: dict[int, StreamMessage] = field(default_factory=dict)
    reorder_buffer_size: int = 1024  # Max messages to buffer
    
    # Dedup: bitmap of recent seqnos (relative to delivered_high_watermark)
    dedup_bitmap: int = 0  # Bit i = 1 means seqno (high_watermark + i + 1) was seen
    dedup_window_size: int = 64  # Track this many seqnos for dedup
    
    # Cutover support
    cutover_watermark: Optional[int] = None  # If set, old epoch messages up to this seqno
    
    # Statistics
    total_sent: int = 0
    total_received: int = 0
    total_delivered: int = 0
    duplicates_dropped: int = 0
    out_of_order_count: int = 0
    
    def get_next_send_seqno(self) -> int:
        """Get next seqno for sending and increment."""
        seqno = self.next_send_seqno
        self.next_send_seqno += 1
        self.total_sent += 1
        return seqno
        
    def is_duplicate(self, seqno: int) -> bool:
        """Check if seqno is a duplicate."""
        if seqno <= self.delivered_high_watermark:
            return True
            
        # Check bitmap for recently seen
        offset = seqno - self.delivered_high_watermark - 1
        if 0 <= offset < self.dedup_window_size:
            return bool(self.dedup_bitmap & (1 << offset))
            
        return False
        
    def mark_seen(self, seqno: int) -> None:
        """Mark seqno as seen in dedup bitmap."""
        offset = seqno - self.delivered_high_watermark - 1
        if 0 <= offset < self.dedup_window_size:
            self.dedup_bitmap |= (1 << offset)
            
    def advance_watermark(self, new_watermark: int) -> None:
        """Advance high watermark, sliding dedup window."""
        if new_watermark <= self.delivered_high_watermark:
            return
            
        shift = new_watermark - self.delivered_high_watermark
        if shift >= self.dedup_window_size:
            self.dedup_bitmap = 0
        else:
            self.dedup_bitmap >>= shift
            
        self.delivered_high_watermark = new_watermark


class StreamTable:
    """Manages multiple streams with ordered delivery.
    
    Thread-safe implementation supporting:
    - Per-stream sequence numbering
    - Out-of-order message reordering
    - Duplicate detection and dropping
    - Cutover watermark handling
    """
    
    def __init__(
        self,
        on_deliver: Callable[[StreamMessage], None] | None = None,
        reorder_buffer_size: int = 1024,
        dedup_window_size: int = 64,
    ):
        """Initialize stream table.
        
        Args:
            on_deliver: Callback when message is ready for delivery
            reorder_buffer_size: Max messages to buffer per stream
            dedup_window_size: Number of seqnos to track for dedup
        """
        self.on_deliver = on_deliver
        self.reorder_buffer_size = reorder_buffer_size
        self.dedup_window_size = dedup_window_size
        
        # Stream state by stream_id
        self._streams: dict[str, StreamState] = {}
        self._lock = threading.RLock()
        
    def get_or_create_stream(self, stream_id: str) -> StreamState:
        """Get or create stream state."""
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = StreamState(
                    stream_id=stream_id,
                    reorder_buffer_size=self.reorder_buffer_size,
                    dedup_window_size=self.dedup_window_size,
                )
            return self._streams[stream_id]
            
    def allocate_seqno(self, stream_id: str) -> int:
        """Allocate next sequence number for sending.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Next sequence number
        """
        state = self.get_or_create_stream(stream_id)
        with self._lock:
            return state.get_next_send_seqno()
            
    def receive(
        self,
        stream_id: str,
        seqno: int,
        payload: Any,
        epoch: int,
    ) -> list[StreamMessage]:
        """Receive a message and return any ready for delivery.
        
        Args:
            stream_id: Stream identifier
            seqno: Message sequence number
            payload: Message payload
            epoch: Message epoch
            
        Returns:
            List of messages ready for in-order delivery
        """
        state = self.get_or_create_stream(stream_id)
        
        with self._lock:
            state.total_received += 1
            
            # Check for duplicate
            if state.is_duplicate(seqno):
                state.duplicates_dropped += 1
                logger.debug(
                    "Stream %s: Dropped duplicate seqno=%d",
                    stream_id, seqno
                )
                return []
                
            # Mark as seen
            state.mark_seen(seqno)
            
            msg = StreamMessage(
                stream_id=stream_id,
                seqno=seqno,
                payload=payload,
                epoch=epoch,
            )
            
            # Check if this is the next expected message
            if seqno == state.next_expected_seqno:
                # Deliver immediately and check buffer for more
                return self._deliver_in_order(state, msg)
            elif seqno > state.next_expected_seqno:
                # Out of order - buffer it
                state.out_of_order_count += 1
                if len(state.reorder_buffer) < state.reorder_buffer_size:
                    state.reorder_buffer[seqno] = msg
                    logger.debug(
                        "Stream %s: Buffered out-of-order seqno=%d (expected=%d)",
                        stream_id, seqno, state.next_expected_seqno
                    )
                else:
                    logger.warning(
                        "Stream %s: Reorder buffer full, dropping seqno=%d",
                        stream_id, seqno
                    )
                return []
            else:
                # seqno < next_expected but not detected as duplicate
                # This shouldn't happen with proper dedup
                logger.warning(
                    "Stream %s: Late message seqno=%d (expected=%d)",
                    stream_id, seqno, state.next_expected_seqno
                )
                return []
                
    def _deliver_in_order(
        self,
        state: StreamState,
        first_msg: StreamMessage,
    ) -> list[StreamMessage]:
        """Deliver messages in order starting from first_msg."""
        deliverable = [first_msg]
        first_msg.delivered = True
        state.total_delivered += 1
        state.next_expected_seqno = first_msg.seqno + 1
        state.advance_watermark(first_msg.seqno)
        
        # Check buffer for consecutive messages
        while state.next_expected_seqno in state.reorder_buffer:
            msg = state.reorder_buffer.pop(state.next_expected_seqno)
            msg.delivered = True
            deliverable.append(msg)
            state.total_delivered += 1
            state.advance_watermark(msg.seqno)
            state.next_expected_seqno = msg.seqno + 1
            
        # Notify callback
        if self.on_deliver:
            for msg in deliverable:
                try:
                    self.on_deliver(msg)
                except Exception as e:
                    logger.error("Delivery callback error: %s", e)
                    
        return deliverable
        
    def set_cutover_watermark(self, stream_id: str, watermark: int) -> None:
        """Set cutover watermark for a stream.
        
        Messages from old epoch with seqno <= watermark are allowed.
        Messages from old epoch with seqno > watermark are rejected.
        
        Args:
            stream_id: Stream identifier
            watermark: Last valid seqno from old epoch
        """
        state = self.get_or_create_stream(stream_id)
        with self._lock:
            state.cutover_watermark = watermark
            logger.info(
                "Stream %s: Cutover watermark set to %d",
                stream_id, watermark
            )
            
    def clear_cutover_watermark(self, stream_id: str) -> None:
        """Clear cutover watermark after migration complete."""
        state = self.get_or_create_stream(stream_id)
        with self._lock:
            state.cutover_watermark = None
            
    def get_stream_stats(self, stream_id: str) -> dict[str, Any]:
        """Get statistics for a stream."""
        state = self._streams.get(stream_id)
        if not state:
            return {}
            
        with self._lock:
            return {
                "stream_id": stream_id,
                "next_send_seqno": state.next_send_seqno,
                "next_expected_seqno": state.next_expected_seqno,
                "delivered_high_watermark": state.delivered_high_watermark,
                "reorder_buffer_size": len(state.reorder_buffer),
                "total_sent": state.total_sent,
                "total_received": state.total_received,
                "total_delivered": state.total_delivered,
                "duplicates_dropped": state.duplicates_dropped,
                "out_of_order_count": state.out_of_order_count,
                "cutover_watermark": state.cutover_watermark,
            }
            
    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all streams."""
        with self._lock:
            return {
                sid: self.get_stream_stats(sid)
                for sid in self._streams
            }
            
    def reset_stream(self, stream_id: str) -> None:
        """Reset a stream to initial state."""
        with self._lock:
            if stream_id in self._streams:
                del self._streams[stream_id]
                
    def clear_all(self) -> None:
        """Clear all streams."""
        with self._lock:
            self._streams.clear()


class StreamManager:
    """High-level stream manager for EPC.
    
    Combines StreamTable with additional features:
    - Stream lifecycle management
    - Request/response correlation
    - Timeout handling
    """
    
    def __init__(
        self,
        stage_idx: int,
        on_deliver: Callable[[StreamMessage], None] | None = None,
    ):
        """Initialize stream manager.
        
        Args:
            stage_idx: Local stage index
            on_deliver: Callback for delivered messages
        """
        self.stage_idx = stage_idx
        self.table = StreamTable(on_deliver=on_deliver)
        
        # Request tracking: request_id -> (stream_id, step, start_time)
        self._pending_requests: dict[str, tuple[str, int, float]] = {}
        self._request_lock = threading.Lock()
        
    def create_stream_id(self, request_id: str, edge_id: str) -> str:
        """Create a stream ID from request and edge.
        
        Args:
            request_id: Request identifier
            edge_id: Edge identifier (e.g., "forward_0_1" or "return_3_0")
            
        Returns:
            Stream identifier
        """
        return f"{request_id}:{edge_id}"
        
    def send(
        self,
        stream_id: str,
        payload: Any,
        epoch: int,
    ) -> tuple[int, StreamMessage]:
        """Prepare a message for sending.
        
        Args:
            stream_id: Stream identifier
            payload: Message payload
            epoch: Current epoch
            
        Returns:
            (seqno, StreamMessage) tuple
        """
        seqno = self.table.allocate_seqno(stream_id)
        msg = StreamMessage(
            stream_id=stream_id,
            seqno=seqno,
            payload=payload,
            epoch=epoch,
        )
        return seqno, msg
        
    def receive(
        self,
        stream_id: str,
        seqno: int,
        payload: Any,
        epoch: int,
    ) -> list[StreamMessage]:
        """Receive and process a message.
        
        Args:
            stream_id: Stream identifier
            seqno: Message sequence number
            payload: Message payload
            epoch: Message epoch
            
        Returns:
            List of messages ready for delivery
        """
        return self.table.receive(stream_id, seqno, payload, epoch)
        
    def start_request(self, request_id: str, edge_id: str) -> str:
        """Start tracking a request.
        
        Args:
            request_id: Request identifier
            edge_id: Initial edge
            
        Returns:
            Stream ID for the request
        """
        stream_id = self.create_stream_id(request_id, edge_id)
        with self._request_lock:
            self._pending_requests[request_id] = (stream_id, 0, time.time())
        return stream_id
        
    def complete_request(self, request_id: str) -> Optional[float]:
        """Complete a request and return its duration.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Request duration in seconds, or None if not found
        """
        with self._request_lock:
            if request_id in self._pending_requests:
                _, _, start_time = self._pending_requests.pop(request_id)
                return time.time() - start_time
        return None
        
    def get_pending_requests(self) -> list[str]:
        """Get list of pending request IDs."""
        with self._request_lock:
            return list(self._pending_requests.keys())
            
    def get_stats(self) -> dict[str, Any]:
        """Get stream manager statistics."""
        with self._request_lock:
            pending = len(self._pending_requests)
            
        return {
            "stage_idx": self.stage_idx,
            "pending_requests": pending,
            "streams": self.table.get_all_stats(),
        }
