"""
Performance Monitoring for the Crypto Narrative Pulse Tracker.

Provides:
- End-to-end latency tracking from message ingestion to alert delivery
- Throughput metrics (messages per second)
- Warning logging when latency exceeds thresholds

Requirements: 12.1, 12.2, 12.3, 12.4
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Latency threshold in seconds - log warning if exceeded
LATENCY_WARNING_THRESHOLD_SECONDS = 5.0

# Window size for throughput calculation (in seconds)
THROUGHPUT_WINDOW_SECONDS = 60.0


# =============================================================================
# LATENCY TRACKER
# =============================================================================


@dataclass
class LatencyMeasurement:
    """Single latency measurement record."""

    message_id: str
    ingestion_time: float  # Unix timestamp when message was ingested
    alert_time: Optional[float] = None  # Unix timestamp when alert was sent
    latency_ms: Optional[float] = None  # Calculated latency in milliseconds

    def complete(self, alert_time: Optional[float] = None) -> float:
        """
        Mark the measurement as complete and calculate latency.

        Args:
            alert_time: Unix timestamp when alert was sent (default: now)

        Returns:
            Latency in milliseconds
        """
        self.alert_time = alert_time or time.time()
        self.latency_ms = (self.alert_time - self.ingestion_time) * 1000
        return self.latency_ms


class LatencyTracker:
    """
    Tracks end-to-end latency from message ingestion to alert delivery.

    Measures the time between when a message is ingested into the pipeline
    and when an alert is sent to the user (via Telegram or other channels).

    Requirements: 12.1, 12.4
    """

    def __init__(
        self,
        warning_threshold_seconds: float = LATENCY_WARNING_THRESHOLD_SECONDS,
        max_pending: int = 1000,
    ):
        """
        Initialize the latency tracker.

        Args:
            warning_threshold_seconds: Log warning if latency exceeds this value
            max_pending: Maximum number of pending measurements to track
        """
        self.warning_threshold_seconds = warning_threshold_seconds
        self.max_pending = max_pending

        # Pending measurements (message_id -> LatencyMeasurement)
        self._pending: dict[str, LatencyMeasurement] = {}

        # Completed measurements (for statistics)
        self._completed: deque[LatencyMeasurement] = deque(maxlen=1000)

        # Thread lock for safe concurrent access
        self._lock = threading.Lock()

        # Statistics
        self._total_measurements = 0
        self._warnings_count = 0

        logger.info(
            f"LatencyTracker initialized with {warning_threshold_seconds}s warning threshold"
        )

    def record_ingestion(
        self,
        message_id: str,
        ingestion_time: Optional[float] = None,
    ) -> None:
        """
        Record when a message was ingested into the pipeline.

        Args:
            message_id: Unique identifier for the message
            ingestion_time: Unix timestamp (default: now)

        Requirements: 12.1
        """
        with self._lock:
            # Clean up old pending measurements if we're at capacity
            if len(self._pending) >= self.max_pending:
                # Remove oldest 10%
                to_remove = list(self._pending.keys())[: self.max_pending // 10]
                for key in to_remove:
                    del self._pending[key]

            self._pending[message_id] = LatencyMeasurement(
                message_id=message_id,
                ingestion_time=ingestion_time or time.time(),
            )

    def record_alert(
        self,
        message_id: str,
        alert_time: Optional[float] = None,
    ) -> Optional[float]:
        """
        Record when an alert was sent for a message.

        Args:
            message_id: Unique identifier for the message
            alert_time: Unix timestamp (default: now)

        Returns:
            Latency in milliseconds, or None if message_id not found

        Requirements: 12.1, 12.4
        """
        with self._lock:
            measurement = self._pending.pop(message_id, None)

            if measurement is None:
                logger.debug(f"No pending measurement for message_id: {message_id}")
                return None

            latency_ms = measurement.complete(alert_time)
            latency_seconds = latency_ms / 1000

            self._completed.append(measurement)
            self._total_measurements += 1

            # Log warning if latency exceeds threshold
            if latency_seconds > self.warning_threshold_seconds:
                self._warnings_count += 1
                logger.warning(
                    f"High latency detected: {latency_ms:.2f}ms "
                    f"(threshold: {self.warning_threshold_seconds * 1000:.0f}ms) "
                    f"for message_id: {message_id}"
                )
            else:
                logger.debug(f"Latency for {message_id}: {latency_ms:.2f}ms")

            return latency_ms

    def get_statistics(self) -> dict[str, Any]:
        """
        Get latency statistics.

        Returns:
            Dictionary with latency statistics

        Requirements: 12.1
        """
        with self._lock:
            if not self._completed:
                return {
                    "total_measurements": self._total_measurements,
                    "pending_count": len(self._pending),
                    "warnings_count": self._warnings_count,
                    "avg_latency_ms": 0.0,
                    "min_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                }

            latencies = [
                m.latency_ms for m in self._completed if m.latency_ms is not None
            ]

            if not latencies:
                return {
                    "total_measurements": self._total_measurements,
                    "pending_count": len(self._pending),
                    "warnings_count": self._warnings_count,
                    "avg_latency_ms": 0.0,
                    "min_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                }

            sorted_latencies = sorted(latencies)

            return {
                "total_measurements": self._total_measurements,
                "pending_count": len(self._pending),
                "warnings_count": self._warnings_count,
                "avg_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)]
                if len(sorted_latencies) >= 20
                else max(latencies),
                "p99_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.99)]
                if len(sorted_latencies) >= 100
                else max(latencies),
            }

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._pending.clear()
            self._completed.clear()
            self._total_measurements = 0
            self._warnings_count = 0


# =============================================================================
# THROUGHPUT TRACKER
# =============================================================================


class ThroughputTracker:
    """
    Tracks message processing throughput (messages per second).

    Uses a sliding window to calculate the current throughput rate.

    Requirements: 12.2, 12.3
    """

    def __init__(self, window_seconds: float = THROUGHPUT_WINDOW_SECONDS):
        """
        Initialize the throughput tracker.

        Args:
            window_seconds: Size of the sliding window for throughput calculation
        """
        self.window_seconds = window_seconds

        # Timestamps of processed messages (for sliding window calculation)
        self._timestamps: deque[float] = deque()

        # Thread lock for safe concurrent access
        self._lock = threading.Lock()

        # Total messages processed (all time)
        self._total_messages = 0

        # Start time for overall rate calculation
        self._start_time = time.time()

        logger.info(f"ThroughputTracker initialized with {window_seconds}s window")

    def record_message(self, timestamp: Optional[float] = None) -> None:
        """
        Record that a message was processed.

        Args:
            timestamp: Unix timestamp (default: now)

        Requirements: 12.2
        """
        now = timestamp or time.time()

        with self._lock:
            self._timestamps.append(now)
            self._total_messages += 1

            # Clean up old timestamps outside the window
            cutoff = now - self.window_seconds
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

    def get_throughput(self) -> float:
        """
        Get current throughput in messages per second.

        Returns:
            Messages per second (based on sliding window)

        Requirements: 12.2, 12.3
        """
        now = time.time()

        with self._lock:
            # Clean up old timestamps
            cutoff = now - self.window_seconds
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            count = len(self._timestamps)

            if count == 0:
                return 0.0

            # Calculate actual window duration
            if count == 1:
                return 1.0 / self.window_seconds

            window_duration = now - self._timestamps[0]
            if window_duration <= 0:
                return 0.0

            return count / window_duration

    def get_statistics(self) -> dict[str, Any]:
        """
        Get throughput statistics.

        Returns:
            Dictionary with throughput statistics

        Requirements: 12.2, 12.3
        """
        now = time.time()

        with self._lock:
            # Clean up old timestamps
            cutoff = now - self.window_seconds
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            count = len(self._timestamps)
            elapsed = now - self._start_time

            # Calculate throughput inline to avoid deadlock
            if count == 0:
                current_throughput = 0.0
            elif count == 1:
                current_throughput = 1.0 / self.window_seconds
            else:
                window_duration = now - self._timestamps[0]
                current_throughput = (
                    count / window_duration if window_duration > 0 else 0.0
                )

            return {
                "current_throughput_mps": current_throughput,
                "total_messages": self._total_messages,
                "messages_in_window": count,
                "window_seconds": self.window_seconds,
                "overall_avg_throughput_mps": self._total_messages / elapsed
                if elapsed > 0
                else 0.0,
                "uptime_seconds": elapsed,
            }

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._timestamps.clear()
            self._total_messages = 0
            self._start_time = time.time()


# =============================================================================
# PERFORMANCE MONITOR (Combined)
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Combined performance metrics snapshot."""

    # Latency metrics
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    latency_warnings_count: int = 0

    # Throughput metrics
    current_throughput_mps: float = 0.0
    total_messages: int = 0
    overall_avg_throughput_mps: float = 0.0

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API/dashboard."""
        return {
            "latency": {
                "avg_ms": self.avg_latency_ms,
                "min_ms": self.min_latency_ms,
                "max_ms": self.max_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "warnings_count": self.latency_warnings_count,
            },
            "throughput": {
                "current_mps": self.current_throughput_mps,
                "total_messages": self.total_messages,
                "overall_avg_mps": self.overall_avg_throughput_mps,
            },
            "timestamp": self.timestamp.isoformat() + "Z",
            "uptime_seconds": self.uptime_seconds,
        }


class PerformanceMonitor:
    """
    Combined performance monitoring for the pipeline.

    Tracks both latency and throughput metrics, providing a unified
    interface for performance monitoring.

    Requirements: 12.1, 12.2, 12.3, 12.4
    """

    def __init__(
        self,
        latency_warning_threshold_seconds: float = LATENCY_WARNING_THRESHOLD_SECONDS,
        throughput_window_seconds: float = THROUGHPUT_WINDOW_SECONDS,
    ):
        """
        Initialize the performance monitor.

        Args:
            latency_warning_threshold_seconds: Log warning if latency exceeds this
            throughput_window_seconds: Window size for throughput calculation
        """
        self.latency_tracker = LatencyTracker(
            warning_threshold_seconds=latency_warning_threshold_seconds
        )
        self.throughput_tracker = ThroughputTracker(
            window_seconds=throughput_window_seconds
        )

        self._start_time = time.time()

        logger.info("PerformanceMonitor initialized")

    def record_ingestion(
        self,
        message_id: str,
        ingestion_time: Optional[float] = None,
    ) -> None:
        """
        Record message ingestion for latency tracking.

        Args:
            message_id: Unique identifier for the message
            ingestion_time: Unix timestamp (default: now)

        Requirements: 12.1
        """
        self.latency_tracker.record_ingestion(message_id, ingestion_time)
        self.throughput_tracker.record_message(ingestion_time)

    def record_alert(
        self,
        message_id: str,
        alert_time: Optional[float] = None,
    ) -> Optional[float]:
        """
        Record alert delivery for latency tracking.

        Args:
            message_id: Unique identifier for the message
            alert_time: Unix timestamp (default: now)

        Returns:
            Latency in milliseconds, or None if message_id not found

        Requirements: 12.1, 12.4
        """
        return self.latency_tracker.record_alert(message_id, alert_time)

    def record_message_processed(self, timestamp: Optional[float] = None) -> None:
        """
        Record that a message was processed (for throughput only).

        Use this when you don't need latency tracking for a message.

        Args:
            timestamp: Unix timestamp (default: now)

        Requirements: 12.2
        """
        self.throughput_tracker.record_message(timestamp)

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics snapshot.

        Returns:
            PerformanceMetrics with current values

        Requirements: 12.1, 12.2, 12.3
        """
        latency_stats = self.latency_tracker.get_statistics()
        throughput_stats = self.throughput_tracker.get_statistics()

        return PerformanceMetrics(
            avg_latency_ms=latency_stats["avg_latency_ms"],
            min_latency_ms=latency_stats["min_latency_ms"],
            max_latency_ms=latency_stats["max_latency_ms"],
            p95_latency_ms=latency_stats["p95_latency_ms"],
            p99_latency_ms=latency_stats["p99_latency_ms"],
            latency_warnings_count=latency_stats["warnings_count"],
            current_throughput_mps=throughput_stats["current_throughput_mps"],
            total_messages=throughput_stats["total_messages"],
            overall_avg_throughput_mps=throughput_stats["overall_avg_throughput_mps"],
            uptime_seconds=throughput_stats["uptime_seconds"],
        )

    def get_throughput(self) -> float:
        """
        Get current throughput in messages per second.

        Returns:
            Messages per second

        Requirements: 12.2, 12.3
        """
        return self.throughput_tracker.get_throughput()

    def reset(self) -> None:
        """Reset all tracking data."""
        self.latency_tracker.reset()
        self.throughput_tracker.reset()
        self._start_time = time.time()


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_performance_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the singleton PerformanceMonitor instance.

    Returns:
        PerformanceMonitor instance (creates one if not exists)
    """
    global _performance_monitor
    with _monitor_lock:
        if _performance_monitor is None:
            _performance_monitor = PerformanceMonitor()
        return _performance_monitor


def reset_performance_monitor() -> None:
    """Reset the performance monitor (useful for testing)."""
    global _performance_monitor
    with _monitor_lock:
        if _performance_monitor is not None:
            _performance_monitor.reset()
        _performance_monitor = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def record_ingestion(message_id: str, ingestion_time: Optional[float] = None) -> None:
    """
    Record message ingestion for latency tracking.

    Convenience function using the singleton monitor.

    Args:
        message_id: Unique identifier for the message
        ingestion_time: Unix timestamp (default: now)

    Requirements: 12.1
    """
    get_performance_monitor().record_ingestion(message_id, ingestion_time)


def record_alert(
    message_id: str, alert_time: Optional[float] = None
) -> Optional[float]:
    """
    Record alert delivery for latency tracking.

    Convenience function using the singleton monitor.

    Args:
        message_id: Unique identifier for the message
        alert_time: Unix timestamp (default: now)

    Returns:
        Latency in milliseconds, or None if message_id not found

    Requirements: 12.1, 12.4
    """
    return get_performance_monitor().record_alert(message_id, alert_time)


def record_message_processed(timestamp: Optional[float] = None) -> None:
    """
    Record that a message was processed (for throughput only).

    Convenience function using the singleton monitor.

    Args:
        timestamp: Unix timestamp (default: now)

    Requirements: 12.2
    """
    get_performance_monitor().record_message_processed(timestamp)


def get_performance_metrics() -> PerformanceMetrics:
    """
    Get current performance metrics snapshot.

    Convenience function using the singleton monitor.

    Returns:
        PerformanceMetrics with current values

    Requirements: 12.1, 12.2, 12.3
    """
    return get_performance_monitor().get_metrics()


def get_current_throughput() -> float:
    """
    Get current throughput in messages per second.

    Convenience function using the singleton monitor.

    Returns:
        Messages per second

    Requirements: 12.2, 12.3
    """
    return get_performance_monitor().get_throughput()


def get_metrics_for_dashboard() -> dict[str, Any]:
    """
    Get performance metrics formatted for dashboard display.

    Returns:
        Dictionary with metrics suitable for Streamlit dashboard

    Requirements: 12.3
    """
    return get_performance_monitor().get_metrics().to_dict()
