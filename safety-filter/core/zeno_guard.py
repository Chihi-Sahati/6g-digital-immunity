# Zeno Temporal Safety Guard
# Original: Deterministic Digital Immunity for 6G Networks
# Phase 2: DSF Core — Zeno Temporal Guard (Python)
"""
zeno_guard.py — Deterministic Digital Immunity for 6G Networks
Phase 2: DSF Core — Zeno Temporal Guard (Python)

This module implements the Zeno temporal safety guard for the DSF. A Zeno
behaviour occurs when a control system issues an infinite number of
actuations in finite time — a pathological case that can arise from overly
aggressive LLM-generated intents or oscillatory feedback loops.

The ZenoGuard enforces a minimum inter-actuation time τ_min (tau_min),
ensuring that the DSF never commands the network element more rapidly
than the physical system can respond. This is critical for:

  - RAN power control: OsmoBTS has a maximum command rate (typically
    10 commands/second). Exceeding this causes command queue overflow.
  - Handover execution: Each HO requires RRC/NAS signalling (50–200 ms).
    Rapid successive HOs cause UE re-attachment failures.
  - Configuration changes: Osmocom CTRL interface has a processing latency
    that varies with load (1–50 ms per command).

Mathematical background:
    The Zeno-free condition requires:
        Δt_n = t_n - t_{n-1} >= τ_min    for all n >= 1

    where t_n is the timestamp of the n-th actuation and τ_min is the
    minimum permitted inter-actuation interval.

    When the guard detects a Zeno violation (Δt < τ_min), it enters a
    cooldown period during which all actuations are blocked. This prevents
    cascading rapid commands from overwhelming the network element.

Thread safety:
    All operations are protected by a threading.Lock, making the guard
    safe for use in multi-threaded DSF deployments where multiple
    validation requests may arrive concurrently.

Usage:
    >>> from zeno_guard import ZenoGuard
    >>> guard = ZenoGuard(tau_min=0.5)  # 500 ms minimum
    >>> result = guard.check()
    >>> if result.passed:
    ...     # Apply the intent
    ...     guard.record_actuation()
    ... else:
    ...     # Block the intent, wait for cooldown
    ...     print(f"Zeno blocked, wait {result.time_since_last}s")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("cbf.zeno_guard")
logger.setLevel(logging.DEBUG)

# ============================================================================
# Result Type
# ============================================================================


@dataclass
class ZenoResult:
    """Result of a Zeno guard check.

    Attributes:
        passed: True if the inter-actuation time Δt >= τ_min.
        time_since_last: Time elapsed since the last actuation (seconds).
            If no prior actuation exists, this is infinity.
        tau_min: The configured minimum inter-actuation time (seconds).
        guard_active: True if the guard is currently in cooldown mode
            (i.e., a recent violation triggered the cooldown).
        cooldown_remaining: Time remaining in the cooldown period (seconds).
            Zero if no cooldown is active.
    """

    passed: bool = True
    time_since_last: float = float("inf")
    tau_min: float = 0.5
    guard_active: bool = False
    cooldown_remaining: float = 0.0

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialisation."""
        return {
            "passed": self.passed,
            "time_since_last": self.time_since_last,
            "tau_min": self.tau_min,
            "guard_active": self.guard_active,
            "cooldown_remaining": self.cooldown_remaining,
        }

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "BLOCK"
        return (
            f"ZenoResult(status={status}, dt={self.time_since_last:.4f}s, "
            f"tau_min={self.tau_min:.4f}s, guard_active={self.guard_active})"
        )


# ============================================================================
# Zeno Guard Statistics
# ============================================================================


@dataclass
class ZenoStatistics:
    """Accumulated statistics for Zeno guard monitoring.

    These statistics are useful for observability dashboards and for
    detecting systemic patterns of Zeno-prone intent generation.

    Attributes:
        total_checks: Total number of guard.check() calls.
        total_passed: Number of checks that passed.
        total_violations: Number of checks that failed (Zeno detected).
        total_actuations: Total number of actuations recorded.
        total_cooldown_entries: Number of cooldown periods entered.
        min_delta_t: Minimum observed inter-actuation time (seconds).
        max_delta_t: Maximum observed inter-actuation time (seconds).
        avg_delta_t: Average inter-actuation time (seconds).
    """

    total_checks: int = 0
    total_passed: int = 0
    total_violations: int = 0
    total_actuations: int = 0
    total_cooldown_entries: int = 0
    min_delta_t: float = float("inf")
    max_delta_t: float = 0.0
    _delta_t_sum: float = 0.0
    _delta_t_count: int = 0

    @property
    def avg_delta_t(self) -> float:
        """Average inter-actuation time."""
        if self._delta_t_count == 0:
            return 0.0
        return self._delta_t_sum / self._delta_t_count

    @property
    def violation_rate(self) -> float:
        """Fraction of checks that resulted in violations."""
        if self.total_checks == 0:
            return 0.0
        return self.total_violations / self.total_checks

    def record_delta_t(self, delta_t: float) -> None:
        """Record an inter-actuation time for statistics."""
        self._delta_t_count += 1
        self._delta_t_sum += delta_t
        if delta_t < self.min_delta_t:
            self.min_delta_t = delta_t
        if delta_t > self.max_delta_t:
            self.max_delta_t = delta_t

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialisation."""
        return {
            "total_checks": self.total_checks,
            "total_passed": self.total_passed,
            "total_violations": self.total_violations,
            "total_actuations": self.total_actuations,
            "total_cooldown_entries": self.total_cooldown_entries,
            "min_delta_t": self.min_delta_t
            if self.min_delta_t != float("inf")
            else 0.0,
            "max_delta_t": self.max_delta_t,
            "avg_delta_t": self.avg_delta_t,
            "violation_rate": self.violation_rate,
        }


# ============================================================================
# Zeno Guard
# ============================================================================


@dataclass
class ZenoGuard:
    """Zeno temporal guard for the DSF actuation pipeline.

    Enforces a minimum inter-actuation time τ_min to prevent the LLM/DSF
    from issuing commands at an infinite rate. When a Zeno violation is
    detected (actuation requested before τ_min has elapsed), the guard
    enters a cooldown period during which all subsequent actuations are
    blocked.

    The guard maintains a sliding window of recent actuation timestamps
    (configurable size) for burst detection and statistical monitoring.

    Attributes:
        tau_min: Minimum inter-actuation time in seconds.
            Default: 0.5 (500 ms), matching the OsmoBTS golden baseline.
        window_size: Number of recent actuation timestamps to track.
            Used for burst detection and statistical analysis.
            Default: 5.
        cooldown_on_violation: Cooldown duration after a Zeno violation
            (seconds). During cooldown, all actuations are blocked.
            Default: 2.0 seconds.
        cooldown_multiplier: Multiplier for the cooldown after repeated
            violations within the observation window. Each consecutive
            violation increases the cooldown by this factor.
            Default: 1.5 (exponential backoff, capped at max_cooldown).
        max_cooldown: Maximum cooldown duration (seconds).
            Prevents the cooldown from growing unboundedly.
            Default: 30.0 seconds.

    Thread Safety:
        All public methods are protected by an internal threading.Lock.
        Safe for concurrent use from multiple threads.

    Example:
        >>> guard = ZenoGuard(tau_min=0.5)
        >>> result = guard.check()
        >>> if result.passed:
        ...     # Proceed with actuation
        ...     guard.record_actuation()
    """

    # Configuration parameters.
    tau_min: float = 0.5
    window_size: int = 5
    cooldown_on_violation: float = 2.0
    cooldown_multiplier: float = 1.5
    max_cooldown: float = 30.0

    # Internal state (not set via constructor).
    last_actuation_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5),
        init=False,
    )
    _last_block_time: float | None = field(default=None, init=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _stats: ZenoStatistics = field(
        default_factory=ZenoStatistics,
        init=False,
        repr=False,
    )
    _violation_count_in_window: int = field(default=0, init=False)
    _last_violation_time: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Post-initialisation: ensure window_size is applied to deque."""
        # Re-create the deque with the correct maxlen from window_size.
        object.__setattr__(self, "last_actuation_times", deque(maxlen=self.window_size))

    # ========================================================================
    # Core Methods
    # ========================================================================

    def check(self) -> ZenoResult:
        """Check whether the current time satisfies the Zeno-free condition.

        Determines if enough time has elapsed since the last actuation
        to permit a new actuation. Also checks whether the guard is
        currently in cooldown mode.

        The check is performed at the time of calling (uses time.time()).

        Returns:
            ZenoResult with:
                - passed: True if actuation is permitted
                - time_since_last: Seconds since last actuation
                - tau_min: Configured minimum inter-actuation time
                - guard_active: True if in cooldown
                - cooldown_remaining: Seconds remaining in cooldown

        Thread Safety:
            This method acquires the internal lock and is safe to call
            from multiple threads concurrently.
        """
        now = time.time()

        with self._lock:
            self._stats.total_checks += 1

            # ------------------------------------------------------------------
            # Step 1: Check cooldown status
            # ------------------------------------------------------------------
            if self._is_in_cooldown(now):
                cooldown_remaining = self._get_cooldown_remaining(now)
                result = ZenoResult(
                    passed=False,
                    time_since_last=self._time_since_last(now),
                    tau_min=self.tau_min,
                    guard_active=True,
                    cooldown_remaining=cooldown_remaining,
                )
                self._stats.total_violations += 1
                logger.debug(
                    f"Zeno check BLOCKED (cooldown): "
                    f"cooldown_remaining={cooldown_remaining:.4f}s"
                )
                return result

            # ------------------------------------------------------------------
            # Step 2: Check inter-actuation time
            # ------------------------------------------------------------------
            time_since_last = self._time_since_last(now)

            if len(self.last_actuation_times) == 0:
                # No prior actuations — always pass.
                result = ZenoResult(
                    passed=True,
                    time_since_last=float("inf"),
                    tau_min=self.tau_min,
                    guard_active=False,
                    cooldown_remaining=0.0,
                )
                self._stats.total_passed += 1
                logger.debug("Zeno check PASSED (no prior actuations)")
                return result

            if time_since_last >= self.tau_min:
                # Sufficient time has elapsed.
                result = ZenoResult(
                    passed=True,
                    time_since_last=time_since_last,
                    tau_min=self.tau_min,
                    guard_active=False,
                    cooldown_remaining=0.0,
                )
                self._stats.total_passed += 1
                logger.debug(
                    f"Zeno check PASSED: dt={time_since_last:.4f}s "
                    f">= tau_min={self.tau_min:.4f}s"
                )
                return result

            # ------------------------------------------------------------------
            # Step 3: Zeno violation detected
            # ------------------------------------------------------------------
            # Compute cooldown with exponential backoff.
            self._violation_count_in_window += 1
            cooldown_duration = self._compute_cooldown()

            # Enter cooldown.
            self._last_block_time = now
            self._stats.total_violations += 1
            self._stats.total_cooldown_entries += 1
            self._last_violation_time = now

            result = ZenoResult(
                passed=False,
                time_since_last=time_since_last,
                tau_min=self.tau_min,
                guard_active=True,
                cooldown_remaining=cooldown_duration,
            )

            logger.warning(
                f"ZENO VIOLATION: dt={time_since_last:.4f}s "
                f"< tau_min={self.tau_min:.4f}s. "
                f"Entering cooldown for {cooldown_duration:.4f}s "
                f"(violation #{self._stats.total_violations}, "
                f"consecutive_in_window={self._violation_count_in_window})"
            )

            return result

    def record_actuation(self) -> None:
        """Record that an actuation has been performed.

        This method MUST be called after a successful check() and after
        the actuation command has been sent to the network element.
        The timestamp is recorded for future Zeno checks.

        Thread Safety:
            Acquires the internal lock.
        """
        now = time.time()

        with self._lock:
            self._stats.total_actuations += 1

            # Compute and record delta_t for statistics.
            if len(self.last_actuation_times) > 0:
                last_time = self.last_actuation_times[-1]
                delta_t = now - last_time
                self._stats.record_delta_t(delta_t)

                logger.debug(
                    f"Actuation recorded: delta_t={delta_t:.4f}s, "
                    f"tau_min={self.tau_min:.4f}s, "
                    f"total_actuations={self._stats.total_actuations}"
                )
            else:
                logger.debug(
                    f"First actuation recorded at t={now:.6f}, "
                    f"total_actuations={self._stats.total_actuations}"
                )

            self.last_actuation_times.append(now)

            # Reset violation count on successful actuation.
            # (The violation count tracks consecutive violations within
            # a single "burst" — a successful actuation ends the burst.)
            self._violation_count_in_window = 0

    def get_remaining_cooldown(self) -> float:
        """Get the remaining cooldown time in seconds.

        Returns the time until the next actuation is permitted. If no
        cooldown is active, returns 0.0.

        Thread Safety:
            Acquires the internal lock.

        Returns:
            Remaining cooldown time in seconds (0.0 if no cooldown).
        """
        now = time.time()

        with self._lock:
            return self._get_cooldown_remaining(now)

    def get_statistics(self) -> ZenoStatistics:
        """Get a snapshot of the guard's accumulated statistics.

        Thread Safety:
            Acquires the internal lock.

        Returns:
            ZenoStatistics snapshot (copy).
        """
        with self._lock:
            return ZenoStatistics(
                total_checks=self._stats.total_checks,
                total_passed=self._stats.total_passed,
                total_violations=self._stats.total_violations,
                total_actuations=self._stats.total_actuations,
                total_cooldown_entries=self._stats.total_cooldown_entries,
                min_delta_t=self._stats.min_delta_t,
                max_delta_t=self._stats.max_delta_t,
                _delta_t_sum=self._stats._delta_t_sum,
                _delta_t_count=self._stats._delta_t_count,
            )

    def reset(self) -> None:
        """Reset the guard to its initial state.

        Clears all recorded actuation timestamps, violation counts,
        cooldown state, and statistics. Useful for testing or when
        reconfiguring the guard.

        Thread Safety:
            Acquires the internal lock.
        """
        with self._lock:
            self.last_actuation_times.clear()
            self._last_block_time = None
            self._stats = ZenoStatistics()
            self._violation_count_in_window = 0
            self._last_violation_time = None
            logger.info("Zeno guard reset to initial state")

    # ========================================================================
    # Private Helpers
    # ========================================================================

    def _time_since_last(self, now: float) -> float:
        """Compute the time elapsed since the last actuation.

        Args:
            now: Current wall-clock time (seconds).

        Returns:
            Time since last actuation (seconds), or infinity if no
            prior actuation exists.
        """
        if len(self.last_actuation_times) == 0:
            return float("inf")
        return now - self.last_actuation_times[-1]

    def _is_in_cooldown(self, now: float) -> bool:
        """Check whether the guard is currently in cooldown mode.

        Args:
            now: Current wall-clock time (seconds).

        Returns:
            True if cooldown is still active.
        """
        if self._last_block_time is None:
            return False

        cooldown_duration = self._compute_cooldown()
        elapsed = now - self._last_block_time
        return elapsed < cooldown_duration

    def _get_cooldown_remaining(self, now: float) -> float:
        """Get the remaining cooldown time.

        Args:
            now: Current wall-clock time (seconds).

        Returns:
            Remaining cooldown seconds (0.0 if not in cooldown).
        """
        if self._last_block_time is None:
            return 0.0

        cooldown_duration = self._compute_cooldown()
        elapsed = now - self._last_block_time
        remaining = cooldown_duration - elapsed

        return max(0.0, remaining)

    def _compute_cooldown(self) -> float:
        """Compute the cooldown duration with exponential backoff.

        The cooldown increases with consecutive violations:
            cooldown = min(base_cooldown * multiplier^consecutive, max_cooldown)

        Returns:
            Cooldown duration in seconds.
        """
        if self._violation_count_in_window <= 1:
            return self.cooldown_on_violation

        # Exponential backoff: base * multiplier^(count - 1)
        factor = self.cooldown_multiplier ** (self._violation_count_in_window - 1)
        cooldown = self.cooldown_on_violation * factor

        # Cap at maximum cooldown.
        return min(cooldown, self.max_cooldown)

    # ========================================================================
    # Context Manager (for test convenience)
    # ========================================================================

    def __enter__(self) -> "ZenoGuard":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager — no cleanup needed."""
        pass


# ============================================================================
# Convenience: create from YAML config
# ============================================================================


def create_zeno_guard_from_config(
    tau_min_ms: float = 500.0,
    window_size: int = 5,
    cooldown_on_violation_s: float = 2.0,
    cooldown_multiplier: float = 1.5,
    max_cooldown_s: float = 30.0,
) -> ZenoGuard:
    """Create a ZenoGuard from configuration parameters.

    This is a convenience factory that converts milliseconds to seconds
    and creates the guard with the specified parameters.

    Args:
        tau_min_ms: Minimum inter-actuation time in milliseconds.
        window_size: Number of recent timestamps to track.
        cooldown_on_violation_s: Cooldown duration in seconds.
        cooldown_multiplier: Exponential backoff multiplier.
        max_cooldown_s: Maximum cooldown in seconds.

    Returns:
        Configured ZenoGuard instance.

    Example:
        >>> guard = create_zeno_guard_from_config(
        ...     tau_min_ms=500,  # 500 ms from abis.yaml golden baseline
        ...     window_size=5,
        ...     cooldown_on_violation_s=2.0,
        ... )
    """
    return ZenoGuard(
        tau_min=tau_min_ms / 1000.0,
        window_size=window_size,
        cooldown_on_violation=cooldown_on_violation_s,
        cooldown_multiplier=cooldown_multiplier,
        max_cooldown=max_cooldown_s,
    )
