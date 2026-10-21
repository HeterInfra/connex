# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
QoS Scheduler with Credit-based Backpressure.

Implements multi-queue scheduling with:
- Traffic class prioritization (token_critical > control > migration_bulk)
- Credit-based flow control
- Pacing for migration traffic
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TrafficClass(IntEnum):
    """Traffic priority classes (higher = more priority)."""
    MIGRATION_BULK = 0    # Background migration data
    CONTROL = 1           # Control messages (heartbeat, routing)
    TOKEN_CRITICAL = 2    # Critical token data for inference


@dataclass
class Message:
    """Message in the scheduler queue."""
    traffic_class: TrafficClass
    data: Any
    peer_id: str
    timestamp: float = field(default_factory=time.time)
    seqno: int = 0
    size_bytes: int = 0
    
    # For pacing
    send_after: float = 0.0  # Don't send before this time


@dataclass
class PeerCredits:
    """Credit state for a peer."""
    peer_id: str
    
    # Credits by traffic class
    credits: dict[TrafficClass, int] = field(default_factory=lambda: {
        TrafficClass.TOKEN_CRITICAL: 100,
        TrafficClass.CONTROL: 50,
        TrafficClass.MIGRATION_BULK: 10,
    })
    
    # Credit limits
    max_credits: dict[TrafficClass, int] = field(default_factory=lambda: {
        TrafficClass.TOKEN_CRITICAL: 100,
        TrafficClass.CONTROL: 50,
        TrafficClass.MIGRATION_BULK: 10,
    })
    
    def has_credits(self, tc: TrafficClass) -> bool:
        """Check if credits available for traffic class."""
        return self.credits.get(tc, 0) > 0
        
    def consume(self, tc: TrafficClass, count: int = 1) -> bool:
        """Consume credits. Returns True if successful."""
        if self.credits.get(tc, 0) >= count:
            self.credits[tc] -= count
            return True
        return False
        
    def replenish(self, tc: TrafficClass, count: int) -> None:
        """Replenish credits."""
        max_c = self.max_credits.get(tc, 100)
        self.credits[tc] = min(self.credits.get(tc, 0) + count, max_c)


class QoSScheduler:
    """Multi-queue scheduler with credit-based backpressure.
    
    Features:
    - Per-peer, per-traffic-class queues
    - Strict priority scheduling
    - Credit-based flow control
    - Pacing for migration traffic
    """
    
    def __init__(
        self,
        on_send: Callable[[Message], bool] | None = None,
        migration_pacing_mbps: float = 1000.0,  # 1 Gbps default
        max_queue_size: int = 1000,
    ):
        """Initialize QoS scheduler.
        
        Args:
            on_send: Callback to actually send message (returns True on success)
            migration_pacing_mbps: Rate limit for migration traffic
            max_queue_size: Max messages per queue
        """
        self.on_send = on_send
        self.migration_pacing_mbps = migration_pacing_mbps
        self.max_queue_size = max_queue_size
        
        # Per-peer queues: peer_id -> {tc -> deque}
        self._queues: dict[str, dict[TrafficClass, deque]] = {}
        
        # Per-peer credits
        self._credits: dict[str, PeerCredits] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Pacing state for migration
        self._last_migration_send: dict[str, float] = {}
        
        # Statistics
        self._stats = {
            "enqueued": 0,
            "sent": 0,
            "dropped_queue_full": 0,
            "dropped_no_credits": 0,
            "blocked_pacing": 0,
        }
        
        # Background scheduler thread
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
            
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
        )
        self._scheduler_thread.start()
        logger.info("QoS scheduler started")
        
    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2.0)
            self._scheduler_thread = None
            
    def enqueue(
        self,
        peer_id: str,
        traffic_class: TrafficClass,
        data: Any,
        size_bytes: int = 0,
    ) -> bool:
        """Enqueue a message for sending.
        
        Args:
            peer_id: Target peer identifier
            traffic_class: Message priority class
            data: Message payload
            size_bytes: Message size for pacing
            
        Returns:
            True if enqueued, False if queue full
        """
        with self._lock:
            # Initialize peer state if needed
            if peer_id not in self._queues:
                self._queues[peer_id] = {tc: deque() for tc in TrafficClass}
                self._credits[peer_id] = PeerCredits(peer_id=peer_id)
                
            queue = self._queues[peer_id][traffic_class]
            
            # Check queue capacity
            if len(queue) >= self.max_queue_size:
                self._stats["dropped_queue_full"] += 1
                logger.warning(
                    "Queue full for peer=%s class=%s, dropping message",
                    peer_id, traffic_class.name
                )
                return False
                
            # Create message
            msg = Message(
                traffic_class=traffic_class,
                data=data,
                peer_id=peer_id,
                size_bytes=size_bytes,
            )
            
            # Apply pacing for migration traffic
            if traffic_class == TrafficClass.MIGRATION_BULK and size_bytes > 0:
                msg.send_after = self._calculate_send_time(peer_id, size_bytes)
                
            queue.append(msg)
            self._stats["enqueued"] += 1
            
        return True
        
    def _calculate_send_time(self, peer_id: str, size_bytes: int) -> float:
        """Calculate earliest send time for pacing."""
        last_send = self._last_migration_send.get(peer_id, 0.0)
        
        # Time to send this message at paced rate
        rate_bps = self.migration_pacing_mbps * 1024 * 1024 / 8
        send_duration = size_bytes / rate_bps
        
        return max(time.time(), last_send + send_duration)
        
    def process_credits(self, peer_id: str, credits_update: dict[TrafficClass, int]) -> None:
        """Process credit update from peer.
        
        Args:
            peer_id: Peer identifier
            credits_update: Credits to add by traffic class
        """
        with self._lock:
            if peer_id not in self._credits:
                self._credits[peer_id] = PeerCredits(peer_id=peer_id)
                
            for tc, count in credits_update.items():
                self._credits[peer_id].replenish(tc, count)
                
        logger.debug("Updated credits for peer=%s: %s", peer_id, credits_update)
        
    def _scheduler_loop(self) -> None:
        """Background scheduling loop."""
        while self._running:
            sent_any = False
            
            with self._lock:
                for peer_id in list(self._queues.keys()):
                    msg = self._try_dequeue(peer_id)
                    if msg:
                        sent_any = True
                        
            # Brief sleep if nothing to send
            if not sent_any:
                time.sleep(0.0001)  # 100 us
                
    def _try_dequeue(self, peer_id: str) -> Optional[Message]:
        """Try to dequeue and send highest priority message for peer."""
        queues = self._queues.get(peer_id, {})
        credits = self._credits.get(peer_id)
        
        if not credits:
            return None
            
        # Check queues in priority order
        for tc in reversed(TrafficClass):
            queue = queues.get(tc)
            if not queue:
                continue
                
            # Check credits
            if not credits.has_credits(tc):
                continue
                
            # Peek at first message
            msg = queue[0]
            
            # Check pacing for migration
            if tc == TrafficClass.MIGRATION_BULK:
                now = time.time()
                if msg.send_after > now:
                    self._stats["blocked_pacing"] += 1
                    continue
                    
            # Try to send
            if self.on_send:
                success = False
                try:
                    success = self.on_send(msg)
                except Exception as e:
                    logger.error("Send callback error: %s", e)
                    
                if success:
                    queue.popleft()
                    credits.consume(tc)
                    self._stats["sent"] += 1
                    
                    # Update pacing state
                    if tc == TrafficClass.MIGRATION_BULK:
                        self._last_migration_send[peer_id] = time.time()
                        
                    return msg
            else:
                # No callback, just dequeue
                queue.popleft()
                credits.consume(tc)
                self._stats["sent"] += 1
                return msg
                
        return None
        
    def get_queue_depth(self, peer_id: str, traffic_class: TrafficClass | None = None) -> int:
        """Get queue depth for peer.
        
        Args:
            peer_id: Peer identifier
            traffic_class: Specific class or None for total
            
        Returns:
            Number of queued messages
        """
        with self._lock:
            queues = self._queues.get(peer_id, {})
            if traffic_class is not None:
                return len(queues.get(traffic_class, []))
            return sum(len(q) for q in queues.values())
            
    def get_credits(self, peer_id: str) -> dict[TrafficClass, int]:
        """Get current credits for peer."""
        with self._lock:
            credits = self._credits.get(peer_id)
            if credits:
                return dict(credits.credits)
            return {}
            
    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            total_queued = sum(
                sum(len(q) for q in peer_queues.values())
                for peer_queues in self._queues.values()
            )
            
        return {
            **self._stats,
            "total_queued": total_queued,
            "peers": len(self._queues),
            "migration_pacing_mbps": self.migration_pacing_mbps,
        }
        
    def send_immediate(
        self,
        peer_id: str,
        traffic_class: TrafficClass,
        data: Any,
    ) -> bool:
        """Send a message immediately, bypassing queue.
        
        Used for urgent messages that should not wait.
        Still respects credits.
        
        Args:
            peer_id: Target peer
            traffic_class: Message priority
            data: Message payload
            
        Returns:
            True if sent successfully
        """
        with self._lock:
            # Initialize if needed
            if peer_id not in self._credits:
                self._credits[peer_id] = PeerCredits(peer_id=peer_id)
                
            credits = self._credits[peer_id]
            
            # Check credits
            if not credits.has_credits(traffic_class):
                self._stats["dropped_no_credits"] += 1
                return False
                
            msg = Message(
                traffic_class=traffic_class,
                data=data,
                peer_id=peer_id,
            )
            
            if self.on_send:
                try:
                    if self.on_send(msg):
                        credits.consume(traffic_class)
                        self._stats["sent"] += 1
                        return True
                except Exception as e:
                    logger.error("Send error: %s", e)
                    
        return False
