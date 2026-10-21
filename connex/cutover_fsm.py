# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project

"""
Cutover State Machine for Non-disruptive Endpoint Migration.

Implements the protocol for seamless endpoint mobility:
1. PREPARE: Establish new connection, warmup MR/buffers
2. CUTOVER: Switch traffic to new endpoint
3. DRAIN: Allow in-flight messages on old connection to complete
4. FINALIZE: Close old connection, update epoch
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CutoverState(Enum):
    """States in the cutover FSM."""
    IDLE = auto()           # No migration in progress
    PREPARING = auto()      # New connection being established
    PREPARED = auto()       # New connection ready, waiting for cutover
    CUTTING_OVER = auto()   # Traffic switching in progress
    DRAINING = auto()       # Old connection draining
    FINALIZING = auto()     # Cleanup in progress


class CutoverEvent(Enum):
    """Events that drive state transitions."""
    START_PREPARE = auto()     # Begin migration preparation
    PREPARE_COMPLETE = auto()  # New connection is ready
    PREPARE_FAILED = auto()    # Preparation failed
    START_CUTOVER = auto()     # Begin traffic switch
    CUTOVER_COMPLETE = auto()  # Traffic switched
    DRAIN_COMPLETE = auto()    # Old connection drained
    ABORT = auto()             # Abort migration


@dataclass
class CutoverContext:
    """Context for an ongoing cutover operation."""
    stream_id: str                      # Stream being migrated
    old_endpoint: str                   # Old endpoint address
    new_endpoint: str                   # New endpoint address
    old_epoch: int                      # Epoch before cutover
    new_epoch: int                      # Epoch after cutover
    
    # State tracking
    state: CutoverState = CutoverState.IDLE
    start_time: float = 0.0
    prepare_time: float = 0.0
    cutover_time: float = 0.0
    drain_time: float = 0.0
    finalize_time: float = 0.0
    
    # Message tracking for drain
    last_old_seqno: int = 0             # Last seqno sent on old connection
    drain_watermark: int = 0            # All messages up to this are drained
    
    # Error tracking
    error: Optional[str] = None
    retry_count: int = 0
    
    # Connection handles
    old_connection: Any = None
    new_connection: Any = None
    
    def elapsed(self) -> float:
        """Time since cutover started."""
        return time.time() - self.start_time if self.start_time else 0.0


class CutoverFSM:
    """Finite State Machine for managing endpoint cutover.
    
    Thread-safe implementation that coordinates:
    - Connection establishment (PREPARE)
    - Traffic switching (CUTOVER) 
    - In-flight message handling (DRAIN)
    - Cleanup (FINALIZE)
    """
    
    def __init__(
        self,
        stage_idx: int,
        on_state_change: Callable[[CutoverState, CutoverContext], None] | None = None,
        prepare_timeout: float = 30.0,
        drain_timeout: float = 10.0,
    ):
        """Initialize cutover FSM.
        
        Args:
            stage_idx: Local stage index
            on_state_change: Callback for state changes
            prepare_timeout: Max time for preparation phase
            drain_timeout: Max time for drain phase
        """
        self.stage_idx = stage_idx
        self.on_state_change = on_state_change
        self.prepare_timeout = prepare_timeout
        self.drain_timeout = drain_timeout
        
        # Active cutovers by stream_id
        self._cutovers: dict[str, CutoverContext] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "total_cutovers": 0,
            "successful_cutovers": 0,
            "failed_cutovers": 0,
            "aborted_cutovers": 0,
            "total_prepare_time": 0.0,
            "total_cutover_time": 0.0,
            "total_drain_time": 0.0,
        }
        
    def start_cutover(
        self,
        stream_id: str,
        old_endpoint: str,
        new_endpoint: str,
        old_epoch: int,
        new_epoch: int,
    ) -> CutoverContext:
        """Start a new cutover operation.
        
        Args:
            stream_id: Stream identifier
            old_endpoint: Current endpoint address
            new_endpoint: Target endpoint address
            old_epoch: Current epoch
            new_epoch: Target epoch
            
        Returns:
            CutoverContext for tracking the operation
        """
        with self._lock:
            if stream_id in self._cutovers:
                existing = self._cutovers[stream_id]
                if existing.state not in (CutoverState.IDLE, CutoverState.FINALIZING):
                    raise RuntimeError(
                        f"Cutover already in progress for stream {stream_id} "
                        f"(state={existing.state.name})"
                    )
                    
            ctx = CutoverContext(
                stream_id=stream_id,
                old_endpoint=old_endpoint,
                new_endpoint=new_endpoint,
                old_epoch=old_epoch,
                new_epoch=new_epoch,
                state=CutoverState.PREPARING,
                start_time=time.time(),
            )
            self._cutovers[stream_id] = ctx
            self._stats["total_cutovers"] += 1
            
        logger.info(
            "Stage %d: Starting cutover for stream=%s from %s to %s (epoch %d->%d)",
            self.stage_idx, stream_id, old_endpoint, new_endpoint, old_epoch, new_epoch
        )
        
        self._notify_state_change(ctx)
        return ctx
        
    def prepare_complete(
        self,
        stream_id: str,
        new_connection: Any,
    ) -> bool:
        """Signal that preparation is complete.
        
        Args:
            stream_id: Stream identifier
            new_connection: Handle to new connection
            
        Returns:
            True if transition successful
        """
        with self._lock:
            ctx = self._cutovers.get(stream_id)
            if not ctx:
                logger.warning("No cutover context for stream %s", stream_id)
                return False
                
            if ctx.state != CutoverState.PREPARING:
                logger.warning(
                    "Invalid state for prepare_complete: %s (expected PREPARING)",
                    ctx.state.name
                )
                return False
                
            ctx.new_connection = new_connection
            ctx.prepare_time = time.time() - ctx.start_time
            ctx.state = CutoverState.PREPARED
            
        logger.info(
            "Stage %d: Cutover prepared for stream=%s (took %.3fs)",
            self.stage_idx, stream_id, ctx.prepare_time
        )
        
        self._stats["total_prepare_time"] += ctx.prepare_time
        self._notify_state_change(ctx)
        return True
        
    def prepare_failed(self, stream_id: str, error: str) -> bool:
        """Signal that preparation failed.
        
        Args:
            stream_id: Stream identifier
            error: Error message
            
        Returns:
            True if transition successful
        """
        with self._lock:
            ctx = self._cutovers.get(stream_id)
            if not ctx:
                return False
                
            ctx.error = error
            ctx.state = CutoverState.IDLE
            ctx.retry_count += 1
            self._stats["failed_cutovers"] += 1
            
        logger.error(
            "Stage %d: Cutover prepare failed for stream=%s: %s",
            self.stage_idx, stream_id, error
        )
        
        self._notify_state_change(ctx)
        return True
        
    def start_traffic_switch(
        self,
        stream_id: str,
        last_old_seqno: int,
    ) -> bool:
        """Begin traffic switch phase.
        
        Args:
            stream_id: Stream identifier
            last_old_seqno: Last sequence number sent on old connection
            
        Returns:
            True if transition successful
        """
        with self._lock:
            ctx = self._cutovers.get(stream_id)
            if not ctx:
                return False
                
            if ctx.state != CutoverState.PREPARED:
                logger.warning(
                    "Invalid state for start_traffic_switch: %s",
                    ctx.state.name
                )
                return False
                
            ctx.last_old_seqno = last_old_seqno
            ctx.drain_watermark = last_old_seqno
            ctx.state = CutoverState.CUTTING_OVER
            
        logger.info(
            "Stage %d: Starting traffic switch for stream=%s (watermark=%d)",
            self.stage_idx, stream_id, last_old_seqno
        )
        
        self._notify_state_change(ctx)
        return True
        
    def cutover_complete(self, stream_id: str) -> bool:
        """Signal that traffic has been switched.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if transition successful
        """
        with self._lock:
            ctx = self._cutovers.get(stream_id)
            if not ctx:
                return False
                
            if ctx.state != CutoverState.CUTTING_OVER:
                return False
                
            ctx.cutover_time = time.time() - ctx.start_time - ctx.prepare_time
            ctx.state = CutoverState.DRAINING
            
        logger.info(
            "Stage %d: Traffic switched for stream=%s (cutover took %.3fs)",
            self.stage_idx, stream_id, ctx.cutover_time
        )
        
        self._stats["total_cutover_time"] += ctx.cutover_time
        self._notify_state_change(ctx)
        return True
        
    def update_drain_progress(self, stream_id: str, acked_seqno: int) -> bool:
        """Update drain progress with acknowledged sequence number.
        
        Args:
            stream_id: Stream identifier
            acked_seqno: Highest acknowledged seqno on old connection
            
        Returns:
            True if drain is complete
        """
        with self._lock:
            ctx = self._cutovers.get(stream_id)
            if not ctx or ctx.state != CutoverState.DRAINING:
                return False
                
            # Check if all in-flight messages are drained
            if acked_seqno >= ctx.last_old_seqno:
                ctx.drain_time = time.time() - ctx.start_time - ctx.prepare_time - ctx.cutover_time
                ctx.state = CutoverState.FINALIZING
                
                logger.info(
                    "Stage %d: Drain complete for stream=%s (took %.3fs)",
                    self.stage_idx, stream_id, ctx.drain_time
                )
                
                self._stats["total_drain_time"] += ctx.drain_time
                self._notify_state_change(ctx)
                return True
                
        return False
        
    def finalize(self, stream_id: str) -> Optional[CutoverContext]:
        """Finalize cutover and cleanup.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Final CutoverContext with timing info
        """
        with self._lock:
            ctx = self._cutovers.get(stream_id)
            if not ctx:
                return None
                
            if ctx.state != CutoverState.FINALIZING:
                logger.warning(
                    "Finalize called in wrong state: %s",
                    ctx.state.name
                )
                
            ctx.finalize_time = time.time() - ctx.start_time
            ctx.state = CutoverState.IDLE
            self._stats["successful_cutovers"] += 1
            
            # Remove from active cutovers
            del self._cutovers[stream_id]
            
        logger.info(
            "Stage %d: Cutover finalized for stream=%s (total %.3fs)",
            self.stage_idx, stream_id, ctx.finalize_time
        )
        
        return ctx
        
    def abort(self, stream_id: str, reason: str) -> bool:
        """Abort an in-progress cutover.
        
        Args:
            stream_id: Stream identifier
            reason: Abort reason
            
        Returns:
            True if abort successful
        """
        with self._lock:
            ctx = self._cutovers.get(stream_id)
            if not ctx:
                return False
                
            ctx.error = f"Aborted: {reason}"
            ctx.state = CutoverState.IDLE
            self._stats["aborted_cutovers"] += 1
            
            del self._cutovers[stream_id]
            
        logger.warning(
            "Stage %d: Cutover aborted for stream=%s: %s",
            self.stage_idx, stream_id, reason
        )
        
        self._notify_state_change(ctx)
        return True
        
    def get_context(self, stream_id: str) -> Optional[CutoverContext]:
        """Get cutover context for a stream."""
        with self._lock:
            return self._cutovers.get(stream_id)
            
    def is_cutting_over(self, stream_id: str) -> bool:
        """Check if a stream is in cutover."""
        with self._lock:
            ctx = self._cutovers.get(stream_id)
            return ctx is not None and ctx.state not in (
                CutoverState.IDLE,
                CutoverState.FINALIZING
            )
            
    def get_active_cutovers(self) -> list[CutoverContext]:
        """Get all active cutover contexts."""
        with self._lock:
            return [
                ctx for ctx in self._cutovers.values()
                if ctx.state != CutoverState.IDLE
            ]
            
    def get_stats(self) -> dict[str, Any]:
        """Get cutover statistics."""
        with self._lock:
            active = len([
                c for c in self._cutovers.values()
                if c.state != CutoverState.IDLE
            ])
            
        return {
            **self._stats,
            "active_cutovers": active,
            "avg_prepare_time": (
                self._stats["total_prepare_time"] / max(self._stats["successful_cutovers"], 1)
            ),
            "avg_cutover_time": (
                self._stats["total_cutover_time"] / max(self._stats["successful_cutovers"], 1)
            ),
            "avg_drain_time": (
                self._stats["total_drain_time"] / max(self._stats["successful_cutovers"], 1)
            ),
        }
        
    def _notify_state_change(self, ctx: CutoverContext) -> None:
        """Notify callback of state change."""
        if self.on_state_change:
            try:
                self.on_state_change(ctx.state, ctx)
            except Exception as e:
                logger.error("State change callback error: %s", e)
