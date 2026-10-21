# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Mobility Data Plane: Multi-Queue Scheduling with Credit-Based Backpressure.

Implements the data plane for dynamic endpoint mobility:
- Stream multiplexing with per-class queues
- Credit-based backpressure (receiver-driven flow control)
- Dynamic congestion/queue delay sensing
- Pacing for burst suppression during churn

Key insight: Dynamic scenarios cause rapid queue state changes,
requiring adaptive scheduling in the data plane.

"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from connex.protocol import (
    CreditInfo,
    LogicalEndpointID,
    MessageClass,
    MessageType,
    MobilityMessage,
    ProtocolHeader,
    StreamID,
)
from connex.mobility_control import VersionedRoute, VersionedRouteTable

logger = logging.getLogger(__name__)


@dataclass
class QueuedMessage:
    """Message in a send queue."""
    message: MobilityMessage
    stream_id: StreamID
    priority: MessageClass
    enqueue_time: float = field(default_factory=time.time)
    
    # Scheduling metadata
    earliest_send_time: float = 0.0  # For pacing
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue."""
        return time.time() - self.enqueue_time
    
    @property
    def is_ready_to_send(self) -> bool:
        """Check if message can be sent (past pacing delay)."""
        return time.time() >= self.earliest_send_time


@dataclass
class PeerQueueState:
    """Per-peer queue and credit state.
    
    Each peer has:
    - Multiple priority queues (per MessageClass)
    - Credit tracking per class
    - Congestion metrics
    """
    peer_lid: LogicalEndpointID
    
    # Per-class message queues (higher priority = processed first)
    queues: Dict[MessageClass, Deque[QueuedMessage]] = field(
        default_factory=lambda: {cls: deque() for cls in MessageClass}
    )
    
    # Credits from peer (receiver-advertised)
    credits: CreditInfo = field(default_factory=CreditInfo)
    
    # Credit request pending
    credit_request_pending: bool = False
    last_credit_request: float = 0.0
    
    # Congestion metrics
    rtt_estimate_us: float = 0.0
    rtt_variance_us: float = 0.0
    queue_delay_us: float = 0.0
    
    # Pacing state
    bytes_in_flight: int = 0
    max_bytes_in_flight: int = 16 * 1024 * 1024  # 16MB window
    last_send_time: float = 0.0
    pacing_rate_bps: float = 10_000_000_000  # 10 Gbps default
    
    # Statistics
    total_enqueued: int = 0
    total_sent: int = 0
    total_dropped: int = 0
    total_credit_waits: int = 0
    
    def get_queue_depth(self, priority: Optional[MessageClass] = None) -> int:
        """Get total messages in queue(s)."""
        if priority is not None:
            return len(self.queues.get(priority, []))
        return sum(len(q) for q in self.queues.values())
    
    def has_credits(self, priority: MessageClass) -> bool:
        """Check if credits available for priority class."""
        return self.credits.has_credit(priority)
    
    def consume_credit(self, priority: MessageClass) -> bool:
        """Consume a credit for sending."""
        return self.credits.consume(priority)


class CongestionController:
    """Adaptive congestion control using RTT and queue delay signals.
    
    Features:
    - AIMD-style window adjustment
    - Queue delay as congestion signal
    - Burst suppression during churn
    """
    
    # Congestion control parameters
    ALPHA = 0.125  # RTT smoothing factor
    BETA = 0.25    # Variance smoothing factor
    MIN_WINDOW = 4  # Minimum credits
    MAX_WINDOW = 200  # Maximum credits
    TARGET_DELAY_US = 1000  # Target queue delay (1ms)
    INCREASE_FACTOR = 1.1  # Additive increase
    DECREASE_FACTOR = 0.5  # Multiplicative decrease
    
    def __init__(self):
        self._state_lock = threading.Lock()
    
    def update_rtt(
        self,
        peer_state: PeerQueueState,
        rtt_sample_us: float,
    ) -> None:
        """Update RTT estimate from new sample.
        
        Uses exponential weighted moving average (EWMA).
        """
        with self._state_lock:
            if peer_state.rtt_estimate_us == 0:
                peer_state.rtt_estimate_us = rtt_sample_us
                peer_state.rtt_variance_us = rtt_sample_us / 2
            else:
                diff = abs(rtt_sample_us - peer_state.rtt_estimate_us)
                peer_state.rtt_variance_us = (
                    (1 - self.BETA) * peer_state.rtt_variance_us +
                    self.BETA * diff
                )
                peer_state.rtt_estimate_us = (
                    (1 - self.ALPHA) * peer_state.rtt_estimate_us +
                    self.ALPHA * rtt_sample_us
                )
    
    def update_queue_delay(
        self,
        peer_state: PeerQueueState,
        queue_delay_us: float,
    ) -> None:
        """Update queue delay estimate."""
        with self._state_lock:
            # Use EWMA for queue delay too
            peer_state.queue_delay_us = (
                (1 - self.ALPHA) * peer_state.queue_delay_us +
                self.ALPHA * queue_delay_us
            )
    
    def compute_pacing_rate(
        self,
        peer_state: PeerQueueState,
        bandwidth_gbps: float = 100.0,
    ) -> float:
        """Compute pacing rate based on congestion state.
        
        Args:
            peer_state: Peer queue state
            bandwidth_gbps: Link bandwidth
            
        Returns:
            Pacing rate in bytes per second
        """
        # Base rate from bandwidth
        base_rate = bandwidth_gbps * 1e9 / 8  # Convert to bytes/s
        
        # Adjust based on queue delay
        if peer_state.queue_delay_us > self.TARGET_DELAY_US:
            # Congested: reduce rate
            excess = peer_state.queue_delay_us / self.TARGET_DELAY_US
            rate = base_rate / (1 + excess * 0.5)
        else:
            # Not congested: use full rate
            rate = base_rate
        
        peer_state.pacing_rate_bps = rate
        return rate
    
    def on_ack_received(
        self,
        peer_state: PeerQueueState,
        rtt_us: float,
        queue_delay_us: float,
    ) -> Dict[MessageClass, int]:
        """Handle ACK and compute credit adjustments.
        
        Args:
            peer_state: Peer state
            rtt_us: Round-trip time
            queue_delay_us: Observed queue delay
            
        Returns:
            Credit adjustments per class
        """
        self.update_rtt(peer_state, rtt_us)
        self.update_queue_delay(peer_state, queue_delay_us)
        
        adjustments = {}
        
        # Determine if congested
        is_congested = queue_delay_us > self.TARGET_DELAY_US * 2
        
        for msg_class in MessageClass:
            current = peer_state.credits.get_credit(msg_class)
            max_credit = peer_state.credits._max_credit(msg_class)
            
            if is_congested:
                # Multiplicative decrease
                new_credit = max(
                    int(current * self.DECREASE_FACTOR),
                    self.MIN_WINDOW
                )
            else:
                # Additive increase (per RTT)
                increment = max(1, int(current * (self.INCREASE_FACTOR - 1)))
                new_credit = min(current + increment, self.MAX_WINDOW, max_credit)
            
            if new_credit != current:
                adjustments[msg_class] = new_credit - current
        
        return adjustments


class MultiQueueScheduler:
    """Multi-queue scheduler with credit-based backpressure.
    
    Features:
    - Per-class priority queues (token > control > migration)
    - Credit-based flow control (receiver paces sender)
    - Adaptive pacing based on congestion signals
    - Head-of-line blocking prevention
    
    Key benefit: Token traffic protected from migration bulk traffic.
    """
    
    def __init__(
        self,
        local_lid: LogicalEndpointID,
        route_table: VersionedRouteTable,
        on_send: Optional[Callable[[QueuedMessage, VersionedRoute], bool]] = None,
        max_queue_size: int = 1000,
        credit_request_interval_s: float = 0.1,
    ):
        """Initialize multi-queue scheduler.
        
        Args:
            local_lid: Local endpoint LID
            route_table: Versioned route table
            on_send: Callback to actually send message
            max_queue_size: Max messages per queue
            credit_request_interval_s: Min interval between credit requests
        """
        self.local_lid = local_lid
        self.route_table = route_table
        self.on_send = on_send
        self.max_queue_size = max_queue_size
        self.credit_request_interval_s = credit_request_interval_s
        
        # Per-peer state
        self._peers: Dict[LogicalEndpointID, PeerQueueState] = {}
        self._peers_lock = threading.RLock()
        
        # Congestion controller
        self.congestion = CongestionController()
        
        # Scheduling state
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._schedule_interval_us = 100  # 100 microseconds
        
        # Global statistics
        self._stats = {
            "total_enqueued": 0,
            "total_sent": 0,
            "total_dropped_queue_full": 0,
            "total_dropped_no_credits": 0,
            "total_credit_waits": 0,
            "total_pacing_delays": 0,
        }
    
    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="MultiQueueScheduler"
        )
        self._scheduler_thread.start()
        logger.info("MultiQueueScheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2.0)
            self._scheduler_thread = None
        logger.info("MultiQueueScheduler stopped")
    
    def get_or_create_peer(self, peer_lid: LogicalEndpointID) -> PeerQueueState:
        """Get or create peer state."""
        with self._peers_lock:
            if peer_lid not in self._peers:
                self._peers[peer_lid] = PeerQueueState(peer_lid=peer_lid)
            return self._peers[peer_lid]
    
    def enqueue(
        self,
        peer_lid: LogicalEndpointID,
        message: MobilityMessage,
        stream_id: StreamID,
        priority: Optional[MessageClass] = None,
    ) -> bool:
        """Enqueue a message for sending.
        
        Args:
            peer_lid: Destination peer LID
            message: Message to send
            stream_id: Stream identifier
            priority: Message priority (from header if not specified)
            
        Returns:
            True if enqueued, False if queue full
        """
        if priority is None:
            priority = message.header.msg_class
        
        peer = self.get_or_create_peer(peer_lid)
        
        with self._peers_lock:
            queue = peer.queues[priority]
            
            # Check capacity
            if len(queue) >= self.max_queue_size:
                self._stats["total_dropped_queue_full"] += 1
                peer.total_dropped += 1
                logger.warning(
                    "Queue full for peer %s class %s, dropping message",
                    peer_lid, priority.name
                )
                return False
            
            # Create queued message
            queued = QueuedMessage(
                message=message,
                stream_id=stream_id,
                priority=priority,
            )
            
            # Apply pacing for bulk migration traffic
            if priority in (MessageClass.MIGRATION_BULK, MessageClass.MIGRATION_STATE):
                queued.earliest_send_time = self._compute_pacing_time(peer, message)
            
            queue.append(queued)
            peer.total_enqueued += 1
            self._stats["total_enqueued"] += 1
        
        return True
    
    def _compute_pacing_time(
        self,
        peer: PeerQueueState,
        message: MobilityMessage,
    ) -> float:
        """Compute earliest send time for paced traffic."""
        now = time.time()
        msg_size = message.header.payload_len + 64  # Header + payload
        
        if peer.pacing_rate_bps > 0:
            send_duration = msg_size / peer.pacing_rate_bps
            return max(now, peer.last_send_time + send_duration)
        
        return now
    
    def process_credit_update(
        self,
        peer_lid: LogicalEndpointID,
        credits: Dict[MessageClass, int],
    ) -> None:
        """Process credit update from peer.
        
        Args:
            peer_lid: Peer that sent credits
            credits: New credit values per class
        """
        peer = self.get_or_create_peer(peer_lid)
        
        with self._peers_lock:
            for msg_class, credit in credits.items():
                peer.credits.replenish(msg_class, credit)
            
            peer.credit_request_pending = False
        
        logger.debug("Updated credits from %s: %s", peer_lid, credits)
    
    def process_ack(
        self,
        peer_lid: LogicalEndpointID,
        rtt_us: float,
        queue_delay_us: float,
        bytes_acked: int,
    ) -> None:
        """Process ACK from peer for congestion control.
        
        Args:
            peer_lid: Peer that sent ACK
            rtt_us: Measured RTT
            queue_delay_us: Observed queue delay
            bytes_acked: Bytes acknowledged
        """
        peer = self.get_or_create_peer(peer_lid)
        
        with self._peers_lock:
            peer.bytes_in_flight = max(0, peer.bytes_in_flight - bytes_acked)
        
        # Update congestion state
        adjustments = self.congestion.on_ack_received(peer, rtt_us, queue_delay_us)
        
        # Apply credit adjustments
        for msg_class, delta in adjustments.items():
            if delta > 0:
                peer.credits.replenish(msg_class, delta)
    
    def _scheduler_loop(self) -> None:
        """Main scheduling loop."""
        while self._running:
            sent_any = False
            
            with self._peers_lock:
                for peer_lid, peer in list(self._peers.items()):
                    msg = self._try_send_one(peer)
                    if msg:
                        sent_any = True
            
            # Brief sleep if nothing to send
            if not sent_any:
                time.sleep(self._schedule_interval_us / 1_000_000)
    
    def _try_send_one(self, peer: PeerQueueState) -> Optional[QueuedMessage]:
        """Try to send one message for a peer.
        
        Implements strict priority scheduling:
        - Check queues in priority order (high to low)
        - Only send if credits available and pacing allows
        
        Returns:
            Sent message or None
        """
        route = self.route_table.get_route(peer.peer_lid)
        if not route or not route.is_usable:
            return None
        
        now = time.time()
        
        # Check queues in priority order (high to low)
        for priority in reversed(list(MessageClass)):
            queue = peer.queues[priority]
            if not queue:
                continue
            
            # Check credits
            if not peer.has_credits(priority):
                self._stats["total_credit_waits"] += 1
                peer.total_credit_waits += 1
                
                # Request more credits if needed
                self._maybe_request_credits(peer)
                continue
            
            # Peek at first message
            queued = queue[0]
            
            # Check pacing
            if not queued.is_ready_to_send:
                self._stats["total_pacing_delays"] += 1
                continue
            
            # Check bytes in flight
            msg_size = queued.message.header.payload_len + 64
            if peer.bytes_in_flight + msg_size > peer.max_bytes_in_flight:
                continue
            
            # Try to send
            if self.on_send:
                try:
                    success = self.on_send(queued, route)
                except Exception as e:
                    logger.error("Send callback error: %s", e)
                    success = False
                
                if not success:
                    # Retry later
                    queued.retry_count += 1
                    if queued.retry_count >= queued.max_retries:
                        queue.popleft()
                        peer.total_dropped += 1
                        logger.warning(
                            "Message dropped after %d retries to %s",
                            queued.max_retries, peer.peer_lid
                        )
                    return None
            
            # Success - dequeue and update state
            queue.popleft()
            peer.consume_credit(priority)
            peer.bytes_in_flight += msg_size
            peer.last_send_time = now
            peer.total_sent += 1
            self._stats["total_sent"] += 1
            
            return queued
        
        return None
    
    def _maybe_request_credits(self, peer: PeerQueueState) -> None:
        """Request credits from peer if needed."""
        now = time.time()
        
        if peer.credit_request_pending:
            return
        
        if now - peer.last_credit_request < self.credit_request_interval_s:
            return
        
        # TODO: Send credit request message
        peer.credit_request_pending = True
        peer.last_credit_request = now
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        with self._peers_lock:
            peer_stats = {}
            for lid, peer in self._peers.items():
                peer_stats[str(lid)] = {
                    "queue_depth": peer.get_queue_depth(),
                    "per_class": {
                        cls.name: len(peer.queues[cls])
                        for cls in MessageClass
                    },
                    "credits": {
                        cls.name: peer.credits.get_credit(cls)
                        for cls in MessageClass
                    },
                    "bytes_in_flight": peer.bytes_in_flight,
                    "rtt_us": peer.rtt_estimate_us,
                    "queue_delay_us": peer.queue_delay_us,
                    "sent": peer.total_sent,
                    "dropped": peer.total_dropped,
                }
            
            return {
                "global": self._stats.copy(),
                "peers": peer_stats,
            }


@dataclass
class StreamReorderState:
    """Per-stream reordering and deduplication state."""
    stream_id: StreamID
    
    # Expected next sequence number
    next_expected_seqno: int = 0
    
    # Delivered high watermark
    delivered_watermark: int = -1
    
    # Reorder buffer: seqno -> message
    reorder_buffer: Dict[int, MobilityMessage] = field(default_factory=dict)
    max_buffer_size: int = 1024
    
    # Dedup bitmap (tracks recent seqnos relative to watermark)
    dedup_bitmap: int = 0
    dedup_window: int = 64
    
    # Cutover watermark (if set, old epoch messages only up to this seqno)
    cutover_watermark: Optional[int] = None
    cutover_epoch: Optional[int] = None
    
    # Statistics
    total_received: int = 0
    total_delivered: int = 0
    duplicates_dropped: int = 0
    out_of_order_received: int = 0
    
    def is_duplicate(self, seqno: int) -> bool:
        """Check if seqno is a duplicate."""
        if seqno <= self.delivered_watermark:
            return True
        
        # Check bitmap for recently seen
        offset = seqno - self.delivered_watermark - 1
        if 0 <= offset < self.dedup_window:
            return bool(self.dedup_bitmap & (1 << offset))
        
        return False
    
    def mark_seen(self, seqno: int) -> None:
        """Mark seqno as seen."""
        offset = seqno - self.delivered_watermark - 1
        if 0 <= offset < self.dedup_window:
            self.dedup_bitmap |= (1 << offset)
    
    def advance_watermark(self, new_watermark: int) -> None:
        """Advance watermark, sliding dedup window."""
        if new_watermark <= self.delivered_watermark:
            return
        
        shift = new_watermark - self.delivered_watermark
        if shift >= self.dedup_window:
            self.dedup_bitmap = 0
        else:
            self.dedup_bitmap >>= shift
        
        self.delivered_watermark = new_watermark


class StreamMultiplexer:
    """Stream multiplexer with ordering and deduplication.
    
    Provides:
    - In-order delivery per stream
    - Deduplication across epochs
    - Reorder buffering for out-of-order messages
    - Cutover watermark handling
    """
    
    def __init__(
        self,
        local_lid: LogicalEndpointID,
        on_deliver: Optional[Callable[[StreamID, MobilityMessage], None]] = None,
        max_reorder_buffer: int = 1024,
        dedup_window: int = 64,
    ):
        """Initialize stream multiplexer.
        
        Args:
            local_lid: Local endpoint LID
            on_deliver: Callback when message ready for delivery
            max_reorder_buffer: Max messages to buffer per stream
            dedup_window: Window size for deduplication
        """
        self.local_lid = local_lid
        self.on_deliver = on_deliver
        self.max_reorder_buffer = max_reorder_buffer
        self.dedup_window = dedup_window
        
        # Per-stream state
        self._streams: Dict[str, StreamReorderState] = {}
        self._streams_lock = threading.RLock()
        
        # Next send seqno per stream (for sending side)
        self._send_seqno: Dict[str, int] = defaultdict(int)
        
        # Statistics
        self._stats = {
            "total_received": 0,
            "total_delivered": 0,
            "total_duplicates": 0,
            "total_out_of_order": 0,
        }
    
    def get_or_create_stream(self, stream_id: StreamID) -> StreamReorderState:
        """Get or create stream state."""
        key = stream_id.to_str()
        with self._streams_lock:
            if key not in self._streams:
                self._streams[key] = StreamReorderState(
                    stream_id=stream_id,
                    max_buffer_size=self.max_reorder_buffer,
                    dedup_window=self.dedup_window,
                )
            return self._streams[key]
    
    def allocate_seqno(self, stream_id: StreamID) -> int:
        """Allocate next sequence number for sending.
        
        Args:
            stream_id: Stream to allocate for
            
        Returns:
            Next sequence number
        """
        key = stream_id.to_str()
        with self._streams_lock:
            seqno = self._send_seqno[key]
            self._send_seqno[key] += 1
            return seqno
    
    def receive(
        self,
        stream_id: StreamID,
        message: MobilityMessage,
    ) -> List[MobilityMessage]:
        """Receive a message and return any ready for in-order delivery.
        
        Args:
            stream_id: Stream identifier
            message: Received message
            
        Returns:
            List of messages ready for in-order delivery
        """
        state = self.get_or_create_stream(stream_id)
        seqno = message.header.seqno
        epoch = message.header.epoch
        
        with self._streams_lock:
            state.total_received += 1
            self._stats["total_received"] += 1
            
            # Check for duplicate
            if state.is_duplicate(seqno):
                state.duplicates_dropped += 1
                self._stats["total_duplicates"] += 1
                logger.debug(
                    "Stream %s: Dropped duplicate seqno=%d",
                    stream_id, seqno
                )
                return []
            
            # Check cutover watermark
            if state.cutover_watermark is not None:
                if epoch == state.cutover_epoch and seqno > state.cutover_watermark:
                    # This is from old epoch past cutover, drop
                    logger.debug(
                        "Stream %s: Dropped post-cutover seqno=%d from epoch=%d",
                        stream_id, seqno, epoch
                    )
                    return []
            
            # Mark as seen
            state.mark_seen(seqno)
            
            # Check if in-order or needs buffering
            if seqno == state.next_expected_seqno:
                # In order - deliver immediately and drain buffer
                return self._deliver_with_buffer_drain(state, message)
            
            elif seqno > state.next_expected_seqno:
                # Out of order - buffer
                state.out_of_order_received += 1
                self._stats["total_out_of_order"] += 1
                
                if len(state.reorder_buffer) < state.max_buffer_size:
                    state.reorder_buffer[seqno] = message
                else:
                    logger.warning(
                        "Stream %s: Reorder buffer full, dropping seqno=%d",
                        stream_id, seqno
                    )
                
                return []
            
            else:
                # Old seqno (already delivered)
                state.duplicates_dropped += 1
                self._stats["total_duplicates"] += 1
                return []
    
    def _deliver_with_buffer_drain(
        self,
        state: StreamReorderState,
        first_message: MobilityMessage,
    ) -> List[MobilityMessage]:
        """Deliver message and drain consecutive buffered messages."""
        to_deliver = [first_message]
        next_seqno = first_message.header.seqno + 1
        
        # Drain consecutive from buffer
        while next_seqno in state.reorder_buffer:
            msg = state.reorder_buffer.pop(next_seqno)
            to_deliver.append(msg)
            next_seqno += 1
        
        # Update state
        state.next_expected_seqno = next_seqno
        state.advance_watermark(next_seqno - 1)
        state.total_delivered += len(to_deliver)
        self._stats["total_delivered"] += len(to_deliver)
        
        # Invoke delivery callback
        if self.on_deliver:
            for msg in to_deliver:
                try:
                    self.on_deliver(state.stream_id, msg)
                except Exception as e:
                    logger.error("Delivery callback error: %s", e)
        
        return to_deliver
    
    def set_cutover_watermark(
        self,
        stream_id: StreamID,
        watermark_seqno: int,
        old_epoch: int,
    ) -> None:
        """Set cutover watermark for stream.
        
        Messages from old epoch after this seqno will be dropped.
        
        Args:
            stream_id: Stream being migrated
            watermark_seqno: Last valid seqno from old epoch
            old_epoch: The old epoch being cut over from
        """
        state = self.get_or_create_stream(stream_id)
        
        with self._streams_lock:
            state.cutover_watermark = watermark_seqno
            state.cutover_epoch = old_epoch
        
        logger.info(
            "Set cutover watermark for %s: seqno=%d, epoch=%d",
            stream_id, watermark_seqno, old_epoch
        )
    
    def clear_cutover_watermark(self, stream_id: StreamID) -> None:
        """Clear cutover watermark after migration complete."""
        state = self.get_or_create_stream(stream_id)
        
        with self._streams_lock:
            state.cutover_watermark = None
            state.cutover_epoch = None
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        with self._streams_lock:
            stream_stats = {}
            for key, state in self._streams.items():
                stream_stats[key] = {
                    "next_expected": state.next_expected_seqno,
                    "watermark": state.delivered_watermark,
                    "buffer_size": len(state.reorder_buffer),
                    "received": state.total_received,
                    "delivered": state.total_delivered,
                    "duplicates": state.duplicates_dropped,
                    "out_of_order": state.out_of_order_received,
                    "cutover_active": state.cutover_watermark is not None,
                }
            
            return {
                "global": self._stats.copy(),
                "streams": stream_stats,
            }
