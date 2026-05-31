#!/usr/bin/env python3
"""
gRPC Interceptor — Deterministic Safety Filter (DSF) Service.

Exposes the ``IntentValidationService`` over gRPC so that the TelcoLLM
gateway can submit network-configuration intents for safety evaluation.
Every intent passes through three sequential gates:

    Gate 1 — CUE Schema Validation  (syntactic correctness)
    Gate 2 — CBF Gate               (mathematical safety via Control
                                     Barrier Functions)
    Gate 3 — Zeno Temporal Guard    (liveness / rate limiting)

Only when **all** gates pass is the intent marked ``ALLOW`` and forwarded
to the :mod:`network_actuator` for Osmocom actuation.

Usage
-----
    python -m safety_filter.api.grpc_interceptor          # default port 50052
    python -m safety_filter.api.grpc_interceptor --port 60051

Generated protobuf stubs must be available at
``safety_filter/api/proto/intent_pb2.py`` and ``intent_pb2_grpc.py``.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import paths — package-relative when installed, filesystem-relative for
# development / simulation environments.
# ---------------------------------------------------------------------------
# Allow running from the project root without package installation.
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import grpc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "grpcio is required for the DSF gRPC interceptor.  "
        "Install it with:  pip install grpcio grpcio-tools"
    ) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "numpy is required for CBF barrier evaluation.  "
        "Install it with:  pip install numpy"
    ) from exc

# Phase-2 CBF engine (may be a stub in simulation mode)
try:
    from safety_filter.core.cbf_engine import CBFPythonEngine
except ImportError:
    logging.getLogger(__name__).warning(
        "CBFPythonEngine not found — using stub implementation."
    )
    CBFPythonEngine = None  # type: ignore[assignment,misc]

# Zeno temporal guard (Phase-3 component)
try:
    from safety_filter.core.zeno_guard import ZenoGuard
except ImportError:
    logging.getLogger(__name__).warning(
        "ZenoGuard not found — using stub implementation."
    )
    ZenoGuard = None  # type: ignore[assignment,misc]

# Gate-1 CUE schema validator
try:
    from safety_filter.validators.cue_schema_validator import (
        CueSchemaValidator,
    )
except ImportError:
    logging.getLogger(__name__).warning(
        "CueSchemaValidator not found — using stub implementation."
    )
    CueSchemaValidator = None  # type: ignore[assignment,misc]

# Protobuf stubs (generated at build time)
try:
    from safety_filter.api.proto import (
        llm_intent_pb2 as intent_pb2,
        llm_intent_pb2_grpc as intent_pb2_grpc,
    )
except ImportError:
    # Fallback: attempt a flat import for development layout
    try:
        from api.proto import (
            llm_intent_pb2 as intent_pb2,
            llm_intent_pb2_grpc as intent_pb2_grpc,
        )  # type: ignore[no-redef]
    except ImportError:
        try:
            import llm_intent_pb2 as intent_pb2
            import llm_intent_pb2_grpc as intent_pb2_grpc
        except ImportError:
            intent_pb2 = None  # type: ignore[assignment,misc]
            intent_pb2_grpc = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("safety_filter.api.grpc_interceptor")


# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------
class Verdict(str, Enum):
    """Enumeration of possible DSF intent-verdict values."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    AMEND = "AMEND"
    DEFER = "DEFER"
    RATE_LIMIT = "RATE_LIMIT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Barrier function definitions (default set for 6G RAN parameters)
# ---------------------------------------------------------------------------
@dataclass
class BarrierConfig:
    """Configuration for a single Control Barrier Function.

    Attributes:
        name: Human-readable barrier identifier (e.g. ``"tx_power"``).
        lower_bound: Safe lower limit for the corresponding state variable.
        upper_bound: Safe upper limit for the corresponding state variable.
        state_index: Index into the state vector ``x ∈ R^n``.
        gain: Barrier gain ``k > 0`` used in ``h_dot >= -k·h(x)``.
    """

    name: str
    lower_bound: float
    upper_bound: float
    state_index: int
    gain: float = 1.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BarrierConfig(name={self.name!r}, "
            f"bounds=[{self.lower_bound}, {self.upper_bound}], "
            f"idx={self.state_index}, gain={self.gain})"
        )


@dataclass
class LyapunovConfig:
    """Configuration for the Lyapunov stability check (Gate 2 supplement).

    Attributes:
        alpha: Desired convergence rate ``alpha > 0``.
        x_ref: Reference (equilibrium) state vector ``x*``.
    """

    alpha: float = 0.1
    x_ref: List[float] = field(default_factory=list)


# Default barriers — index mapping matches the canonical state vector layout:
#   x = [tx_power, ho_hysteresis, cell_reselection_offset,
#        rxlev_min, prb_utilisation, pci, arfcn, bandwidth]
DEFAULT_BARRIERS: List[BarrierConfig] = [
    BarrierConfig(
        name="tx_power", lower_bound=0.0, upper_bound=46.0, state_index=0, gain=2.0
    ),
    BarrierConfig(
        name="ho_hysteresis", lower_bound=0.0, upper_bound=30.0, state_index=1, gain=1.0
    ),
    BarrierConfig(
        name="cell_reselection_offset",
        lower_bound=0.0,
        upper_bound=24.0,
        state_index=2,
        gain=1.5,
    ),
    BarrierConfig(
        name="rxlev_min", lower_bound=-128.0, upper_bound=-48.0, state_index=3, gain=1.0
    ),
    BarrierConfig(
        name="prb_utilisation",
        lower_bound=0.0,
        upper_bound=100.0,
        state_index=4,
        gain=3.0,
    ),
    BarrierConfig(
        name="pci", lower_bound=0.0, upper_bound=1007.0, state_index=5, gain=0.5
    ),
    BarrierConfig(
        name="arfcn", lower_bound=0.0, upper_bound=65535.0, state_index=6, gain=0.1
    ),
    BarrierConfig(
        name="bandwidth", lower_bound=1.4, upper_bound=100.0, state_index=7, gain=1.0
    ),
]

DEFAULT_LYAPUNOV: LyapunovConfig = LyapunovConfig(
    alpha=0.1,
    x_ref=[30.0, 6.0, 4.0, -80.0, 60.0, 0.0, 0.0, 20.0],
)


# ---------------------------------------------------------------------------
# Stub implementations when core modules are unavailable
# ---------------------------------------------------------------------------
class _StubCBFEngine:
    """Minimal CBF engine stub used when the real engine is not available."""

    def __init__(self, state_dim: int = 8) -> None:
        self.state_dim = state_dim

    def evaluate_barrier(self, x: "np.ndarray", lower: float, upper: float) -> float:
        """Return a heuristic barrier value.

        h(x) ∈ (-∞, 0] means safe; h(x) > 0 means unsafe.
        """
        return max(0.0, float(x) - upper) + max(0.0, lower - float(x))

    def evaluate_all_barriers(
        self,
        x: "np.ndarray",
        barriers: List[BarrierConfig],
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for b in barriers:
            if b.state_index < len(x):
                val = self.evaluate_barrier(
                    x[b.state_index], b.lower_bound, b.upper_bound
                )
                results[b.name] = val
            else:
                results[b.name] = 0.0
        return results

    def evaluate_lyapunov(
        self,
        x: "np.ndarray",
        x_ref: "np.ndarray",
        alpha: float,
    ) -> float:
        diff = x[: len(x_ref)] - x_ref[: len(x)]
        return float(0.5 * np.dot(diff, diff) - alpha * float(np.linalg.norm(diff)))


class _StubZenoGuard:
    """Minimal Zeno guard stub."""

    def __init__(self, tau_min: float = 0.5) -> None:
        self.tau_min = tau_min
        self._last_actuation: float = 0.0

    def check(self, element_id: str = "default") -> Dict[str, Any]:
        now = time.monotonic()
        elapsed = now - self._last_actuation
        active = elapsed < self.tau_min
        remaining = max(0.0, self.tau_min - elapsed)
        return {
            "active": active,
            "remaining_s": remaining,
            "element_id": element_id,
        }

    def record_actuation(self, element_id: str = "default") -> None:
        self._last_actuation = time.monotonic()

    def reset(self) -> None:
        self._last_actuation = 0.0


# ---------------------------------------------------------------------------
# Service implementation
# ---------------------------------------------------------------------------
class IntentValidationServiceImpl(
    intent_pb2_grpc.IntentValidationServiceServicer
    if intent_pb2_grpc is not None
    else object  # type: ignore[misc, assignment]
):
    """gRPC servicer implementing the three-gate Deterministic Safety Filter.

    Constructor arguments may be overridden for unit-testing; all defaults
    are tuned for the 6G Osmocom simulation environment.

    Parameters:
        state_dim: Dimensionality ``n`` of the network state vector
            ``x ∈ R^n`` used by the CBF engine.
        tau_min: Minimum inter-actuation time (seconds) enforced by the
            Zeno temporal guard.
        barriers: Explicit list of :class:`BarrierConfig` instances.  When
            ``None`` the module-level :data:`DEFAULT_BARRIERS` are used.
        lyapunov_cfg: Configuration for the Lyapunov stability check.
        cue_bin_path: Explicit CUE binary path forwarded to the schema
            validator.
        schema_dir: Explicit schema directory forwarded to the schema
            validator.
    """

    def __init__(
        self,
        state_dim: int = 8,
        tau_min: float = 0.5,
        barriers: Optional[List[BarrierConfig]] = None,
        lyapunov_cfg: Optional[LyapunovConfig] = None,
        cue_bin_path: Optional[str] = None,
        schema_dir: Optional[str] = None,
    ) -> None:
        # --- CBF Engine (Gate 2) ---
        if CBFPythonEngine is not None:
            self._cbf: _StubCBFEngine = CBFPythonEngine(state_dim=state_dim)  # type: ignore[assignment]
        else:
            self._cbf = _StubCBFEngine(state_dim=state_dim)
        logger.info("CBF engine initialised (state_dim=%d)", state_dim)

        # --- Zeno Guard (Gate 3) ---
        if ZenoGuard is not None:
            self._zeno: _StubZenoGuard = ZenoGuard(tau_min=tau_min)  # type: ignore[assignment]
        else:
            self._zeno = _StubZenoGuard(tau_min=tau_min)
        logger.info("Zeno guard initialised (tau_min=%.2fs)", tau_min)

        # --- CUE Validator (Gate 1) ---
        if CueSchemaValidator is not None:
            try:
                self._cue_validator: Optional[CueSchemaValidator] = CueSchemaValidator(
                    cue_bin_path=cue_bin_path, schema_dir=schema_dir
                )
            except Exception as exc:
                logger.warning(
                    "CUE validator init failed, proceeding without CUE: %s",
                    exc,
                )
                self._cue_validator = None
        else:
            self._cue_validator = None

        # --- Barrier / Lyapunov config ---
        self._barriers: List[BarrierConfig] = barriers or DEFAULT_BARRIERS
        self._lyapunov: LyapunovConfig = lyapunov_cfg or DEFAULT_LYAPUNOV

        logger.info(
            "IntentValidationService ready: %d barriers, Lyapunov α=%.2f",
            len(self._barriers),
            self._lyapunov.alpha,
        )

        # Internal state for GetCurrentConfig
        self._golden_config: Dict[str, Any] = {}

        # Consecutive-rejection counter for backpressure
        self._consecutive_rejections: int = 0

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _extract_intent_fields(request: Any) -> Dict[str, Any]:
        """Extract intent fields from a protobuf *ValidateIntentRequest*
        (or a plain dictionary / object with matching attributes).

        Returns a normalised dictionary suitable for downstream processing.
        """
        payload: Dict[str, Any] = {}
        # Support both protobuf objects and plain dicts
        if hasattr(request, "DESCRIPTOR"):
            # Protobuf message
            for field_desc in request.DESCRIPTOR.fields:
                value = getattr(request, field_desc.name, None)
                payload[field_desc.name] = value
        elif isinstance(request, dict):
            payload.update(request)
        else:
            for attr in dir(request):
                if not attr.startswith("_"):
                    payload[attr] = getattr(request, attr)
        return payload

    def _intent_to_state_vector(self, intent: Dict[str, Any]) -> "np.ndarray":
        """Convert a parsed intent into a fixed-dimension state vector.

        The canonical state vector layout (8-dim) is:
        ``[tx_power, ho_hysteresis, cell_reselection_offset, rxlev_min,
          prb_utilisation, pci, arfcn, bandwidth]``

        Missing fields default to 0; extra fields are ignored.
        """
        mapping = [
            ("tx_power_dbm", 0),
            ("tx_power", 0),
            ("ho_hysteresis_db", 1),
            ("ho_hysteresis", 1),
            ("cell_reselection_offset_db", 2),
            ("cell_reselection_offset", 2),
            ("rxlev_min_dbm", 3),
            ("rxlev_min", 3),
            ("prb_utilisation_pct", 4),
            ("prb_utilisation", 4),
            ("pci", 5),
            ("arfcn", 6),
            ("bandwidth_mhz", 7),
            ("bandwidth", 7),
        ]
        x = np.zeros(self._cbf.state_dim, dtype=np.float64)
        for field_name, idx in mapping:
            if field_name in intent and intent[field_name] is not None:
                x[idx] = float(intent[field_name])
                break  # first alias wins
        return x

    def _build_verdict(
        self,
        verdict: Verdict,
        intent_fields: Dict[str, Any],
        gate1: Optional[Any] = None,
        gate2: Optional[Dict[str, Any]] = None,
        gate3: Optional[Dict[str, Any]] = None,
        amended_values: Optional[Dict[str, float]] = None,
        defer_ms: Optional[int] = None,
        backpressure_hint_ms: Optional[int] = None,
    ) -> Any:
        if intent_pb2 is not None and hasattr(intent_pb2, "IntentVerdict"):
            decision = intent_pb2.VerdictDecision.ALLOW
            if verdict == Verdict.DENY:
                decision = intent_pb2.VerdictDecision.DENY
            elif verdict == Verdict.AMEND:
                decision = intent_pb2.VerdictDecision.AMEND
            elif verdict == Verdict.DEFER:
                decision = intent_pb2.VerdictDecision.DEFER
            elif verdict == Verdict.RATE_LIMIT:
                decision = intent_pb2.VerdictDecision.RATE_LIMIT

            v = intent_pb2.IntentVerdict(
                verdict_id="v-" + str(time.time_ns()),
                intent_id=intent_fields.get("intent_id", "unknown"),
                decision=decision,
                explanation=f"Verdict is {verdict.name}",
            )

            # Optionally populate CbfResult if gate2 is available
            if gate2 and "barrier_values" in gate2:
                cbf_res = intent_pb2.CbfResult()
                min_margin = float("inf")
                for b_name, b_val in gate2["barrier_values"].items():
                    chk = cbf_res.barrier_checks.add()
                    chk.barrier_id = b_name
                    chk.left_hand_side = b_val
                    chk.passed = b_val <= 0
                    margin = -b_val
                    if margin < min_margin:
                        min_margin = margin
                cbf_res.minimum_margin = (
                    min_margin if min_margin != float("inf") else 0.0
                )
                cbf_res.passed = all(chk.passed for chk in cbf_res.barrier_checks)
                v.cbf_result.CopyFrom(cbf_res)

            return v

        return {"decision": verdict.value}

    # ========================================================= Unary RPC
    def ValidateIntent(self, request: Any, context: Any) -> Any:
        """Evaluate a single intent through the three-gate DSF pipeline.

        RPC signature matches the proto definition::

            rpc ValidateIntent(ValidateIntentRequest)
                returns (IntentVerdict);

        Gate evaluation order is fixed: CUE → CBF → Zeno.  The pipeline
        short-circuits at the first failure.

        Returns:
            ``IntentVerdict`` protobuf with ``verdict`` set to one of
            ``ALLOW | DENY | AMEND | DEFER | RATE_LIMIT | INTERNAL_ERROR``.
        """
        t0 = time.monotonic()
        intent_fields = self._extract_intent_fields(request)
        intent_id = intent_fields.get("intent_id", "unknown")
        element_type = intent_fields.get("target_element_type", "bts")
        logger.info("═══ ValidateIntent request  intent_id=%s ═══", intent_id)

        try:
            # -------------------------------------------------- Gate 1: CUE
            gate1_result = None
            if self._cue_validator is not None:
                try:
                    gate1_result = self._cue_validator.validate_intent(
                        intent_json=intent_fields,
                        target_element_type=str(element_type),
                    )
                except Exception as exc:
                    logger.error("Gate 1 (CUE) exception: %s", exc)
                    gate1_result = type(gate1_result) if gate1_result else None
                    gate1_result_passed = False
                    str(exc)
                else:
                    gate1_result_passed = gate1_result.passed
            else:
                logger.warning("Gate 1 skipped — CUE validator unavailable")
                gate1_result_passed = True  # skip gate

            if not gate1_result_passed:
                self._consecutive_rejections += 1
                elapsed = (time.monotonic() - t0) * 1000
                logger.warning(
                    "Intent %s DENIED at Gate 1 (CUE) — %.1f ms", intent_id, elapsed
                )
                return self._build_verdict(
                    verdict=Verdict.DENY,
                    intent_fields=intent_fields,
                    gate1=gate1_result,
                    backpressure_hint_ms=(
                        200 * self._consecutive_rejections
                        if self._consecutive_rejections > 3
                        else None
                    ),
                )

            # -------------------------------------------------- Gate 2: CBF
            x = self._intent_to_state_vector(intent_fields)
            barrier_values = self._cbf.evaluate_all_barriers(x, self._barriers)

            x_ref = np.array(self._lyapunov.x_ref[: self._cbf.state_dim])
            lyap_value = self._cbf.evaluate_lyapunov(x, x_ref, self._lyapunov.alpha)

            gate2_result: Dict[str, Any] = {
                "barrier_values": barrier_values,
                "lyapunov_value": lyap_value,
                "state_vector": x.tolist(),
            }

            # Check barrier violations
            violations: Dict[str, float] = {}
            for name, val in barrier_values.items():
                if val > 0:
                    violations[name] = val

            if violations:
                # Attempt amendment — clamp to barrier bounds
                amended: Dict[str, float] = {}
                for b in self._barriers:
                    if b.name in violations:
                        idx = b.state_index
                        if idx < len(x):
                            original = float(x[idx])
                            clamped = float(
                                np.clip(original, b.lower_bound, b.upper_bound)
                            )
                            if clamped != original:
                                amended[b.name] = clamped
                                x[idx] = clamped

                if amended:
                    # Re-evaluate with clamped state
                    barrier_values_new = self._cbf.evaluate_all_barriers(
                        x, self._barriers
                    )
                    gate2_result["amended_barrier_values"] = barrier_values_new
                    gate2_result["amended_state_vector"] = x.tolist()

                    self._consecutive_rejections += 1
                    elapsed = (time.monotonic() - t0) * 1000
                    logger.warning(
                        "Intent %s AMENDED at Gate 2 (CBF) — violations=%s, "
                        "amended=%s — %.1f ms",
                        intent_id,
                        violations,
                        amended,
                        elapsed,
                    )
                    return self._build_verdict(
                        verdict=Verdict.AMEND,
                        intent_fields=intent_fields,
                        gate1=gate1_result,
                        gate2=gate2_result,
                        amended_values=amended,
                        backpressure_hint_ms=(
                            200 * self._consecutive_rejections
                            if self._consecutive_rejections > 3
                            else None
                        ),
                    )

                # Cannot amend — deny
                self._consecutive_rejections += 1
                elapsed = (time.monotonic() - t0) * 1000
                logger.warning(
                    "Intent %s DENIED at Gate 2 (CBF) — violations=%s — %.1f ms",
                    intent_id,
                    violations,
                    elapsed,
                )
                return self._build_verdict(
                    verdict=Verdict.DENY,
                    intent_fields=intent_fields,
                    gate1=gate1_result,
                    gate2=gate2_result,
                    backpressure_hint_ms=(
                        200 * self._consecutive_rejections
                        if self._consecutive_rejections > 3
                        else None
                    ),
                )

            # -------------------------------------------------- Gate 3: Zeno
            element_id = intent_fields.get("osmo_element_id", "default")
            zeno_status = self._zeno.check(element_id=element_id)

            if zeno_status.get("active", False):
                self._consecutive_rejections += 1
                remaining_ms = int(zeno_status.get("remaining_s", 0.0) * 1000)
                elapsed = (time.monotonic() - t0) * 1000
                logger.warning(
                    "Intent %s DEFERRED at Gate 3 (Zeno) — remaining=%d ms — %.1f ms",
                    intent_id,
                    remaining_ms,
                    elapsed,
                )
                return self._build_verdict(
                    verdict=Verdict.DEFER,
                    intent_fields=intent_fields,
                    gate1=gate1_result,
                    gate2=gate2_result,
                    gate3=zeno_status,
                    defer_ms=remaining_ms,
                    backpressure_hint_ms=(
                        200 * self._consecutive_rejections
                        if self._consecutive_rejections > 3
                        else None
                    ),
                )

            # -------------------------------------------------- ALL GATES PASS
            self._consecutive_rejections = 0
            self._zeno.record_actuation(element_id=element_id)
            self._golden_config.update(intent_fields)

            elapsed = (time.monotonic() - t0) * 1000
            logger.info(
                "Intent %s ALLOWED — all gates passed — %.1f ms",
                intent_id,
                elapsed,
            )
            return self._build_verdict(
                verdict=Verdict.ALLOW,
                intent_fields=intent_fields,
                gate1=gate1_result,
                gate2=gate2_result,
                gate3=zeno_status,
            )

        except Exception:
            logger.exception("INTERNAL ERROR processing intent %s", intent_id)
            return self._build_verdict(
                verdict=Verdict.INTERNAL_ERROR,
                intent_fields=intent_fields,
            )

    # ===================================================== Streaming RPC
    def ValidateIntentStream(self, request_iterator: Any, context: Any) -> Any:
        """Server-streaming RPC that validates a stream of intents.

        For each incoming ``ValidateIntentRequest`` the servicer runs the
        three-gate pipeline and yields an ``IntentStreamAck``.

        Backpressure mechanism: when the number of consecutive rejections
        exceeds a threshold (default 3), the ack includes a
        ``backpressure_hint_ms`` advising the client to slow down.

        Yields:
            ``IntentStreamAck`` protobuf messages.
        """
        logger.info("ValidateIntentStream session started (peer=%s)", context.peer())

        try:
            for request in request_iterator:
                t0 = time.monotonic()
                intent_fields = self._extract_intent_fields(request)
                intent_id = intent_fields.get("intent_id", "unknown")
                logger.debug("Stream intent received: %s", intent_id)

                # Run the three-gate pipeline via the unary handler.
                # We reuse the logic by extracting the verdict result.
                verdict = self.ValidateIntent(request, context)

                # Build the streaming ack
                ack_data: Dict[str, Any] = {
                    "intent_id": intent_id,
                    "timestamp_ns": time.time_ns(),
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 2),
                }

                # Extract verdict string
                if hasattr(verdict, "verdict"):
                    ack_data["verdict"] = verdict.verdict
                elif isinstance(verdict, dict):
                    ack_data["verdict"] = verdict.get("verdict", "UNKNOWN")
                else:
                    ack_data["verdict"] = "UNKNOWN"

                # Backpressure hint
                if self._consecutive_rejections > 3:
                    ack_data["backpressure_hint_ms"] = (
                        200 * self._consecutive_rejections
                    )

                # Attempt protobuf ack
                if intent_pb2 is not None and hasattr(intent_pb2, "IntentStreamAck"):
                    try:
                        ack = intent_pb2.IntentStreamAck(**ack_data)
                    except Exception:
                        ack = ack_data
                else:
                    ack = ack_data

                yield ack

        except Exception as exc:
            logger.exception("ValidateIntentStream error: %s", exc)
            raise

    # ===================================================== Config retrieval
    def GetCurrentConfig(self, request: Any, context: Any) -> Any:
        """Return the current golden configuration.

        The golden config is updated each time an intent passes all three
        gates and is marked ``ALLOW``.
        """
        logger.debug("GetCurrentConfig requested")

        config_data: Dict[str, Any] = {
            "golden_config": self._golden_config,
            "barriers": [
                {
                    "name": b.name,
                    "lower_bound": b.lower_bound,
                    "upper_bound": b.upper_bound,
                    "state_index": b.state_index,
                    "gain": b.gain,
                }
                for b in self._barriers
            ],
            "lyapunov": {
                "alpha": self._lyapunov.alpha,
                "x_ref": self._lyapunov.x_ref,
            },
            "zeno_tau_min": self._zeno.tau_min,
            "timestamp_ns": time.time_ns(),
        }

        if intent_pb2 is not None and hasattr(intent_pb2, "CurrentConfigResponse"):
            try:
                return intent_pb2.CurrentConfigResponse(**config_data)
            except Exception:
                pass

        return config_data


# ---------------------------------------------------------------------------
# gRPC server lifecycle
# ---------------------------------------------------------------------------
def serve(port: int = 50052, max_workers: int = 10) -> None:
    """Start the DSF gRPC interceptor server and block until termination.

    The server listens on ``0.0.0.0:{port}`` with a thread-pool executor
    of size *max_workers*.  ``SIGTERM`` and ``SIGINT`` trigger graceful
    shutdown (stop accepting new RPCs, finish in-flight requests).

    Parameters:
        port: TCP port to bind.
        max_workers: Maximum concurrent RPC threads.
    """
    # ---------------------------------------------------------------- logging
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )

    logger.info("Starting DSF gRPC interceptor on 0.0.0.0:%d", port)

    # ------------------------------------------------------------ server
    try:
        server = grpc.server(
            ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_receive_message_length", 4 * 1024 * 1024),
                ("grpc.max_send_message_length", 4 * 1024 * 1024),
            ],
        )
    except Exception as exc:
        logger.critical("Failed to create gRPC server: %s", exc)
        sys.exit(1)

    # Add servicer (only if protobuf stubs are available)
    if intent_pb2_grpc is not None:
        intent_pb2_grpc.add_IntentValidationServiceServicer_to_server(
            IntentValidationServiceImpl(), server
        )
    else:
        logger.warning(
            "Protobuf stubs unavailable — servicer added as no-op.  "
            "Generate stubs with:  protoc --python_out=. --grpc_python_out=. "
            "intent.proto"
        )
        # Still add a minimal servicer so the server can start
        try:
            intent_pb2_grpc.add_IntentValidationServiceServicer_to_server(
                IntentValidationServiceImpl(), server
            )
        except Exception:
            pass

    # ------------------------------------------------------------ bind
    try:
        server.add_insecure_port(f"0.0.0.0:{port}")
    except Exception as exc:
        logger.critical("Failed to bind port %d: %s", port, exc)
        sys.exit(1)

    server.start()
    logger.info("DSF gRPC server STARTED (port=%d, workers=%d)", port, max_workers)

    # -------------------------------------------------------- signal handling
    stop_event = threading.Event()

    def _handle_signal(signum: int, frame: Any) -> None:  # type: ignore[misc]
        logger.info(
            "Received signal %s — initiating graceful shutdown...",
            signal.Signals(signum).name,
        )
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # -------------------------------------------------------- wait
    try:
        # Use a simple polling loop so signals are caught promptly
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown requested via keyboard interrupt")

    # -------------------------------------------------------- graceful stop
    grace = 5.0
    logger.info("Stopping gRPC server (grace=%ss)...", grace)
    server.stop(grace=grace).wait()
    logger.info("DSF gRPC server STOPPED")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deterministic Safety Filter — gRPC Interceptor"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50052,
        help="gRPC listen port (default: 50052)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Thread-pool size (default: 10)",
    )
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.workers)
