"""
gRPC Client for Deterministic Security Framework (DSF) Communication.

Submits network configuration intents from the TelcoLLM Agent to the DSF
for safety validation.  The DSF verifies each intent against Contract-Based
Design (CBF) invariants, Lyapunov stability guarantees, Zeno-freeness, and
Composable Unified Enforcement (CUE) policies before allowing deployment.

Architecture Context:
    - TelcoLLM Agent runs in inference-net (172.29.x.0/24).
    - DSF listens on 172.29.0.20:50052 (TLS with mTLS in production).
    - The agent NEVER talks to Osmocom directly; iptables enforce this.
    - All communication with Osmocom flows through Kafka telemetry topics.

Protobuf Definition (llm_intent.proto):
    service IntentValidationService {
        rpc ValidateIntent(NetworkIntent) returns (IntentVerdict);
        rpc ValidateIntentStream(stream NetworkIntent)
            returns (stream IntentVerdict);
        rpc GetCurrentConfig(ElementRequest) returns (NetworkConfig);
    }

Dependencies:
    - grpcio
    - grpcio-tools (for generated stubs)

Usage:
    from llm_agent.api.grpc_client import DsfGrpcClient, GrpcClientConfig

    client = DsfGrpcClient()
    result = client.submit_intent(intent_payload)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Attempt protobuf imports – provide stubs for simulation environments
# ---------------------------------------------------------------------------

try:
    import grpc
    from grpc import StatusCode  # type: ignore[import-untyped]

    # Attempt to import generated protobuf modules
    try:
        # Search in common protobuf output locations
        _proto_search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "proto"),
            os.path.join(os.path.dirname(__file__), "..", "pb"),
            os.path.join(os.path.dirname(__file__), ".."),
        ]
        for _p in _proto_search_paths:
            if _p not in sys.path:
                sys.path.insert(0, os.path.abspath(_p))

        import intent_pb2  # type: ignore[import-untyped]
        import intent_pb2_grpc  # type: ignore[import-untyped]

        _PROTO_AVAILABLE = True
    except ImportError:
        _PROTO_AVAILABLE = False
        intent_pb2 = None  # type: ignore[assignment]
        intent_pb2_grpc = None  # type: ignore[assignment]

except ImportError:
    grpc = None  # type: ignore[assignment]
    _PROTO_AVAILABLE = False
    intent_pb2 = None  # type: ignore[assignment]
    intent_pb2_grpc = None  # type: ignore[assignment]

# Import intent model from the config generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from intent.config_generator import IntentPayload  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VerdictDecision(Enum):
    """DSF verdict decision types."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    AMEND = "AMEND"
    RATE_LIMIT = "RATE_LIMIT"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class GrpcClientConfig:
    """Configuration for the DSF gRPC client.

    Attributes:
        dsf_host:          DSF server hostname or IP address.
        dsf_port:          DSF gRPC port number.
        timeout_s:         Per-RPC timeout in seconds.
        max_retries:       Maximum retry attempts for transient failures.
        retry_backoff_s:   Exponential backoff base (seconds) between retries.
        use_tls:           Whether to use TLS (mTLS in production).
        ca_cert_path:      Path to the CA certificate for TLS verification.
                           ``None`` uses the system default trust store.
    """

    dsf_host: str = "172.29.0.20"
    dsf_port: int = 50052
    timeout_s: float = 10.0
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    use_tls: bool = True
    ca_cert_path: str | None = None


@dataclass
class SubmissionResult:
    """Result of an intent submission to the DSF.

    Attributes:
        intent_id:         ID of the submitted intent.
        decision:          DSF decision (ALLOW / DENY / AMEND / RATE_LIMIT).
        evaluated_at:      ISO-8601 timestamp of the DSF evaluation.
        rationale:         DSF explanation of the decision.
        cbf_passed:        Whether the intent passed CBF invariant checks.
        lyapunov_passed:   Whether the intent passed Lyapunov stability checks.
        zeno_passed:       Whether the intent passed Zeno-freeness checks.
        cue_passed:        Whether the intent passed CUE compliance checks.
        amended_intent:    For AMEND decisions, the modified intent as a dict.
        error:             Error message if the submission failed.
    """

    intent_id: str = ""
    decision: str = "ERROR"
    evaluated_at: str = ""
    rationale: str = ""
    cbf_passed: bool | None = None
    lyapunov_passed: bool | None = None
    zeno_passed: bool | None = None
    cue_passed: bool | None = None
    amended_intent: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# DSF gRPC Client
# ---------------------------------------------------------------------------


class DsfGrpcClient:
    """gRPC client for submitting intents to the Deterministic Security Framework.

    Handles channel creation (with optional mTLS), intent serialisation to
    protobuf, RPC invocation with retry logic, and verdict parsing.

    Args:
        config: Optional client configuration.  Defaults use the production
                DSF address (172.29.0.20:50052) with TLS.

    Example::

        with DsfGrpcClient() as client:
            result = client.submit_intent(intent_payload)
            if result.decision == "ALLOW":
                print("Intent approved by DSF")
    """

    def __init__(self, config: GrpcClientConfig | None = None) -> None:
        self._config = config or GrpcClientConfig()
        self._channel = None
        self._stub = None
        self._connected = False

        self._connect()

    def _connect(self) -> None:
        """Establish the gRPC channel and create the service stub."""
        if grpc is None:
            logger.warning("grpcio not installed – client operating in simulation mode")
            return

        try:
            self._channel = self._create_channel()
            if _PROTO_AVAILABLE and intent_pb2_grpc is not None:
                self._stub = intent_pb2_grpc.IntentValidationServiceStub(self._channel)
            else:
                logger.warning(
                    "Protobuf stubs not available – client operating in simulation mode"
                )
                self._stub = None

            self._connected = True
            logger.info(
                "DsfGrpcClient connected to %s:%d (TLS=%s)",
                self._config.dsf_host,
                self._config.dsf_port,
                self._config.use_tls,
            )

        except Exception as exc:
            logger.error(
                "Failed to connect to DSF at %s:%d: %s",
                self._config.dsf_host,
                self._config.dsf_port,
                exc,
                exc_info=True,
            )
            self._connected = False

    def _create_channel(self) -> Any:
        """Create a gRPC channel, either secure (mTLS) or insecure.

        Returns:
            A ``grpc.Channel`` instance.

        Raises:
            RuntimeError: If TLS credentials cannot be loaded.
        """
        target = f"{self._config.dsf_host}:{self._config.dsf_port}"

        if not self._config.use_tls:
            logger.debug("Creating insecure channel to %s", target)
            return grpc.insecure_channel(target)  # type: ignore[attr-defined]

        # TLS channel
        try:
            if self._config.ca_cert_path:
                with open(self._config.ca_cert_path, "rb") as f:
                    ca_creds = grpc.ssl_channel_credentials(f.read())
                logger.debug(
                    "Creating TLS channel with CA cert: %s",
                    self._config.ca_cert_path,
                )
            else:
                ca_creds = grpc.ssl_channel_credentials()
                logger.debug("Creating TLS channel with default credentials")

            return grpc.secure_channel(target, ca_creds)  # type: ignore[attr-defined]

        except FileNotFoundError as exc:
            raise RuntimeError(
                f"TLS CA certificate not found at {self._config.ca_cert_path}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to create TLS channel: {exc}") from exc

    def submit_intent(self, intent: IntentPayload) -> SubmissionResult:
        """Submit a single configuration intent to the DSF for validation.

        The intent is serialised to a protobuf ``NetworkIntent`` message and
        sent via the ``ValidateIntent`` RPC.  The response ``IntentVerdict``
        is parsed into a :class:`SubmissionResult`.

        Retry logic:
        - Transient failures (UNAVAILABLE, DEADLINE_EXCEEDED) are retried
          with exponential backoff up to ``max_retries``.
        - Non-retryable errors return immediately with ``ERROR`` decision.

        Args:
            intent: The configuration intent to submit.

        Returns:
            :class:`SubmissionResult` with the DSF verdict.
        """
        logger.info(
            "Submitting intent %s to DSF at %s:%d",
            intent.intent_id,
            self._config.dsf_host,
            self._config.dsf_port,
        )

        if not self._connected or self._stub is None:
            logger.warning("DSF client not connected – returning simulation result")
            return self._simulate_verdict(intent)

        last_error: str | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                proto_msg = self._intent_to_protobuf(intent)
                verdict = self._stub.ValidateIntent(
                    proto_msg,
                    timeout=self._config.timeout_s,
                )
                result = self._parse_verdict(verdict)
                logger.info(
                    "DSF verdict for intent %s: %s (CBF=%s, Lyapunov=%s, Zeno=%s, CUE=%s)",
                    intent.intent_id,
                    result.decision,
                    result.cbf_passed,
                    result.lyapunov_passed,
                    result.zeno_passed,
                    result.cue_passed,
                )
                return result

            except grpc.RpcError as exc:  # type: ignore[attr-defined]
                code = exc.code()  # type: ignore[attr-defined]
                details = exc.details()  # type: ignore[attr-defined]

                if (
                    code
                    in (  # type: ignore[attr-defined]
                        StatusCode.UNAVAILABLE,  # type: ignore[attr-defined]
                        StatusCode.DEADLINE_EXCEEDED,  # type: ignore[attr-defined]
                    )
                    and attempt < self._config.max_retries
                ):
                    backoff = self._config.retry_backoff_s * (2 ** (attempt - 1))
                    logger.warning(
                        "DSF RPC transient error (attempt %d/%d): %s – "
                        "retrying in %.1fs",
                        attempt,
                        self._config.max_retries,
                        details,
                        backoff,
                    )
                    time.sleep(backoff)
                    last_error = details
                    continue

                logger.error(
                    "DSF RPC error (attempt %d/%d): [%s] %s",
                    attempt,
                    self._config.max_retries,
                    code,  # type: ignore[attr-defined]
                    details,
                )
                return SubmissionResult(
                    intent_id=intent.intent_id,
                    decision="ERROR",
                    evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    rationale=f"gRPC error: {details}",
                    error=details,
                )

            except Exception as exc:
                logger.error(
                    "Unexpected error submitting intent %s: %s",
                    intent.intent_id,
                    exc,
                    exc_info=True,
                )
                last_error = str(exc)
                if attempt < self._config.max_retries:
                    backoff = self._config.retry_backoff_s * (2 ** (attempt - 1))
                    time.sleep(backoff)
                    continue

        return SubmissionResult(
            intent_id=intent.intent_id,
            decision="ERROR",
            evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            rationale=f"All {self._config.max_retries} attempts failed",
            error=last_error,
        )

    def submit_intent_stream(
        self,
        intents: list[IntentPayload],
    ) -> list[SubmissionResult]:
        """Submit a batch of intents via a bidirectional streaming RPC.

        Opens a streaming connection to the DSF, sends intents one by one,
        and collects verdicts from the response stream.

        Args:
            intents: List of intents to submit.

        Returns:
            List of :class:`SubmissionResult` in submission order.
        """
        if not self._connected or self._stub is None:
            logger.warning("DSF client not connected – returning simulation results")
            return [self._simulate_verdict(i) for i in intents]

        if not intents:
            return []

        results: list[SubmissionResult] = []

        try:

            def _request_iterator() -> Iterator[Any]:
                """Yield protobuf messages for the streaming RPC."""
                for intent in intents:
                    try:
                        yield self._intent_to_protobuf(intent)
                    except Exception as exc:
                        logger.error(
                            "Failed to serialise intent %s for stream: %s",
                            intent.intent_id,
                            exc,
                        )

            responses = self._stub.ValidateIntentStream(
                _request_iterator(),
                timeout=self._config.timeout_s * len(intents),
            )

            for response in responses:
                result = self._parse_verdict(response)
                results.append(result)

        except grpc.RpcError as exc:  # type: ignore[attr-defined]
            logger.error(
                "Streaming RPC error: %s",
                exc.details(),  # type: ignore[attr-defined]
            )
            # Return error results for any intents not yet processed
            processed_ids = {r.intent_id for r in results}
            for intent in intents:
                if intent.intent_id not in processed_ids:
                    results.append(
                        SubmissionResult(
                            intent_id=intent.intent_id,
                            decision="ERROR",
                            rationale=f"Stream error: {exc.details()}",  # type: ignore[attr-defined]
                            error=str(exc.details()),  # type: ignore[attr-defined]
                        )
                    )

        except Exception as exc:
            logger.error(
                "Unexpected streaming error: %s",
                exc,
                exc_info=True,
            )

        logger.info(
            "Stream submission complete: %d intents → %d results",
            len(intents),
            len(results),
        )
        return results

    def query_current_config(self, element_id: str) -> dict[str, Any]:
        """Query the DSF for the current configuration of a network element.

        Calls the ``GetCurrentConfig`` RPC with the element ID and returns
        the configuration as a dictionary.

        Args:
            element_id: The network element ID to query.

        Returns:
            Dictionary of current configuration parameters, or an empty
            dict on error / simulation mode.
        """
        logger.debug("Querying current config for element %s", element_id)

        if not self._connected or self._stub is None:
            logger.warning("DSF client not connected – returning empty config")
            return {}

        try:
            if _PROTO_AVAILABLE and intent_pb2 is not None:
                request = intent_pb2.ElementRequest(element_id=element_id)  # type: ignore[attr-defined]
                response = self._stub.GetCurrentConfig(
                    request,
                    timeout=self._config.timeout_s,
                )
                # Parse the response – attempt JSON deserialisation
                if hasattr(response, "config_json") and response.config_json:  # type: ignore[attr-defined]
                    return json.loads(response.config_json)  # type: ignore[attr-defined]
                # Fallback: iterate known fields
                config: dict[str, Any] = {}
                for desc, value in response.ListFields():  # type: ignore[attr-defined]
                    config[desc.name] = value
                return config
            else:
                return {}

        except grpc.RpcError as exc:  # type: ignore[attr-defined]
            logger.error(
                "GetCurrentConfig RPC error for %s: %s",
                element_id,
                exc.details(),  # type: ignore[attr-defined]
            )
            return {}
        except Exception as exc:
            logger.error(
                "Unexpected error querying config for %s: %s",
                element_id,
                exc,
                exc_info=True,
            )
            return {}

    # ---- Protobuf Serialisation ----------------------------------------------

    def _intent_to_protobuf(self, intent: IntentPayload) -> Any:
        """Convert an :class:`IntentPayload` to a protobuf ``NetworkIntent`` message.

        If protobuf modules are available, constructs the message using the
        generated stub types.  Otherwise, returns a plain dict as a fallback.

        Args:
            intent: The IntentPayload to serialise.

        Returns:
            Protobuf ``NetworkIntent`` message or dict fallback.
        """
        if _PROTO_AVAILABLE and intent_pb2 is not None:
            msg = intent_pb2.NetworkIntent(  # type: ignore[attr-defined]
                intent_id=intent.intent_id,
                rationale=intent.rationale,
                generated_at=intent.generated_at,
                target_element_id=intent.target_element_id,
                category=intent.category,
                confidence_score=intent.confidence_score,
            )

            # Populate radio intent sub-message
            if intent.radio_params:
                radio = intent_pb2.RadioIntent()  # type: ignore[attr-defined]
                _set_proto_field(
                    radio,
                    "tx_power_adjustment_dbm",
                    intent.radio_params.get("tx_power_adjustment_dbm", 0),
                )
                _set_proto_field(
                    radio,
                    "antenna_tilt_adjustment_deg",
                    intent.radio_params.get("antenna_tilt_adjustment_deg", 0.0),
                )
                _set_proto_field(
                    radio,
                    "cell_reselection_offset_db",
                    intent.radio_params.get("cell_reselection_offset_db", 0),
                )
                _set_proto_field(
                    radio, "a3_offset_db", intent.radio_params.get("a3_offset_db", 0)
                )
                _set_proto_field(
                    radio,
                    "ho_hysteresis_db",
                    intent.radio_params.get("ho_hysteresis_db", 0.0),
                )
                _set_proto_field(
                    radio,
                    "time_to_trigger_ms",
                    intent.radio_params.get("time_to_trigger_ms", 0),
                )
                # Store remaining params as JSON
                _set_proto_field(radio, "params_json", json.dumps(intent.radio_params))
                msg.radio_intent.CopyFrom(radio)  # type: ignore[attr-defined]

            # Populate core intent sub-message
            if intent.core_params:
                core = intent_pb2.CoreIntent()  # type: ignore[attr-defined]
                _set_proto_field(core, "qci", intent.core_params.get("qci", 0))
                _set_proto_field(core, "mbr_bps", intent.core_params.get("mbr_bps", 0))
                _set_proto_field(core, "gbr_bps", intent.core_params.get("gbr_bps", 0))
                _set_proto_field(
                    core, "arp_priority", intent.core_params.get("arp_priority", 0)
                )
                _set_proto_field(
                    core, "session_limit", intent.core_params.get("session_limit", 0)
                )
                _set_proto_field(core, "params_json", json.dumps(intent.core_params))
                msg.core_intent.CopyFrom(core)  # type: ignore[attr-defined]

            # Populate transport intent
            if intent.transport_params:
                transport = intent_pb2.TransportIntent()  # type: ignore[attr-defined]
                _set_proto_field(
                    transport, "params_json", json.dumps(intent.transport_params)
                )
                msg.transport_intent.CopyFrom(transport)  # type: ignore[attr-defined]

            # Populate security intent
            if intent.security_params:
                security = intent_pb2.SecurityIntent()  # type: ignore[attr-defined]
                _set_proto_field(
                    security, "params_json", json.dumps(intent.security_params)
                )
                msg.security_intent.CopyFrom(security)  # type: ignore[attr-defined]

            # Reasoning trace and metadata as JSON strings
            if intent.reasoning_trace:
                _set_proto_field(
                    msg, "reasoning_trace_json", json.dumps(intent.reasoning_trace)
                )
            if intent.metadata:
                _set_proto_field(msg, "metadata_json", json.dumps(intent.metadata))

            return msg
        else:
            # Fallback: return a dict that mimics the protobuf structure
            logger.warning("Protobuf not available – returning dict fallback")
            return {
                "intent_id": intent.intent_id,
                "rationale": intent.rationale,
                "generated_at": intent.generated_at,
                "target_element_id": intent.target_element_id,
                "category": intent.category,
                "confidence_score": intent.confidence_score,
                "radio_params": intent.radio_params,
                "core_params": intent.core_params,
                "transport_params": intent.transport_params,
                "security_params": intent.security_params,
                "reasoning_trace_json": json.dumps(intent.reasoning_trace),
                "metadata_json": json.dumps(intent.metadata),
            }

    # ---- Verdict Parsing -----------------------------------------------------

    def _parse_verdict(self, verdict: Any) -> SubmissionResult:
        """Parse a protobuf ``IntentVerdict`` into a :class:`SubmissionResult`.

        Extracts all safety check results (CBF, Lyapunov, Zeno, CUE) and
        the overall decision.  For AMEND decisions, attempts to parse the
        amended intent from the verdict.

        Args:
            verdict: Protobuf ``IntentVerdict`` message.

        Returns:
            Parsed :class:`SubmissionResult`.
        """
        try:
            # Extract top-level fields
            intent_id = getattr(verdict, "intent_id", "unknown")
            decision = getattr(verdict, "decision", "ERROR")
            evaluated_at = getattr(
                verdict,
                "evaluated_at",
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            rationale = getattr(verdict, "rationale", "")

            # Extract safety check results
            cbf_result = getattr(verdict, "cbf_result", None)
            lyapunov_result = getattr(verdict, "lyapunov_result", None)
            zeno_result = getattr(verdict, "zeno_result", None)
            cue_result = getattr(verdict, "cue_result", None)

            cbf_passed = getattr(cbf_result, "passed", None) if cbf_result else None
            lyapunov_passed = (
                getattr(lyapunov_result, "passed", None) if lyapunov_result else None
            )
            zeno_passed = getattr(zeno_result, "passed", None) if zeno_result else None
            cue_passed = getattr(cue_result, "passed", None) if cue_result else None

            # CBF details
            cbf_details = {}
            if cbf_result:
                cbf_details = {
                    "invariant_name": getattr(cbf_result, "invariant_name", ""),
                    "violation_details": getattr(cbf_result, "violation_details", ""),
                }

            # Lyapunov details
            lyapunov_details = {}
            if lyapunov_result:
                lyapunov_details = {
                    "lyapunov_value": getattr(lyapunov_result, "lyapunov_value", 0.0),
                    "threshold": getattr(lyapunov_result, "threshold", 0.0),
                    "stable": getattr(lyapunov_result, "stable", False),
                }

            # Zeno details
            zeno_details = {}
            if zeno_result:
                zeno_details = {
                    "min_transition_time": getattr(
                        zeno_result, "min_transition_time", 0.0
                    ),
                    "zeno_free": getattr(zeno_result, "zeno_free", False),
                }

            # CUE details
            cue_details = {}
            if cue_result:
                cue_details = {
                    "cue_policy_id": getattr(cue_result, "cue_policy_id", ""),
                    "cue_compliant": getattr(cue_result, "cue_compliant", False),
                    "violations": getattr(cue_result, "violations", []),
                }

            # Parse amended intent if present
            amended_intent: dict[str, Any] | None = None
            if decision == "AMEND":
                amended_json = getattr(verdict, "amended_intent_json", None)
                if amended_json:
                    try:
                        amended_intent = json.loads(amended_json)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to parse amended intent JSON")

            result = SubmissionResult(
                intent_id=intent_id,
                decision=decision,
                evaluated_at=evaluated_at,
                rationale=rationale,
                cbf_passed=cbf_passed,
                lyapunov_passed=lyapunov_passed,
                zeno_passed=zeno_passed,
                cue_passed=cue_passed,
                amended_intent=amended_intent,
            )

            # Log safety check details
            if cbf_details:
                logger.debug("CBF details: %s", cbf_details)
            if lyapunov_details:
                logger.debug("Lyapunov details: %s", lyapunov_details)
            if zeno_details:
                logger.debug("Zeno details: %s", zeno_details)
            if cue_details:
                logger.debug("CUE details: %s", cue_details)

            return result

        except Exception as exc:
            logger.error(
                "Error parsing verdict: %s",
                exc,
                exc_info=True,
            )
            return SubmissionResult(
                decision="ERROR",
                error=f"Verdict parsing error: {exc}",
            )

    # ---- Simulation Mode -----------------------------------------------------

    @staticmethod
    def _simulate_verdict(intent: IntentPayload) -> SubmissionResult:
        """Generate a simulated verdict for environments without a live DSF.

        In simulation mode, all intents with confidence > 0.5 are ALLOWED,
        and all others are AMENDed with minor adjustments.

        Args:
            intent: The submitted intent.

        Returns:
            Simulated :class:`SubmissionResult`.
        """
        import random

        random.seed(hash(intent.intent_id))

        if intent.confidence_score > 0.5:
            decision = "ALLOW"
            amended = None
        else:
            decision = "AMEND"
            amended = {
                "adjustment": "Reduced parameter magnitudes for safety",
                "modified_params": intent.radio_params,
            }

        return SubmissionResult(
            intent_id=intent.intent_id,
            decision=decision,
            evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            rationale=(
                f"[SIMULATED] Intent {decision} for element "
                f"{intent.target_element_id} (category={intent.category})"
            ),
            cbf_passed=True,
            lyapunov_passed=True,
            zeno_passed=True,
            cue_passed=decision != "AMEND" or random.random() > 0.5,
            amended_intent=amended,
        )

    # ---- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the gRPC channel and release all resources."""
        if self._channel is not None:
            try:
                self._channel.close()
                logger.info("DSF gRPC channel closed")
            except Exception as exc:
                logger.error("Error closing gRPC channel: %s", exc)
            finally:
                self._channel = None
                self._stub = None
                self._connected = False

    def __enter__(self) -> "DsfGrpcClient":
        """Context manager entry."""
        if not self._connected:
            self._connect()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit – closes the channel."""
        self.close()


# ---------------------------------------------------------------------------
# Protobuf field helper
# ---------------------------------------------------------------------------


def _set_proto_field(msg: Any, field_name: str, value: Any) -> None:
    """Safely set a field on a protobuf message, ignoring unknown fields.

    Args:
        msg:       Protobuf message instance.
        field_name: Name of the field to set.
        value:     Value to assign.
    """
    try:
        setattr(msg, field_name, value)
    except AttributeError:
        # Field doesn't exist in this protobuf version – silently skip
        logger.debug("Protobuf field %s not found on message %s", field_name, type(msg))


# ---------------------------------------------------------------------------
# Intent Submission Pipeline (End-to-End Orchestrator)
# ---------------------------------------------------------------------------


class IntentSubmissionPipeline:
    """End-to-end pipeline that connects RCA anomaly detection, intent
    generation, and DSF submission into a single workflow.

    This pipeline:
    1. Consumes telemetry via the RCA pipeline
    2. Detects anomalies
    3. Generates configuration intents
    4. Submits intents to the DSF via gRPC
    5. Handles DSF verdicts (ALLOW / DENY / AMEND / RATE_LIMIT)

    Args:
        rca_pipeline:      The RCA pipeline for telemetry consumption.
        config_generator:  The intent generator for config changes.
        grpc_client:       The DSF gRPC client for intent submission.
    """

    def __init__(
        self,
        rca_pipeline: Any,
        config_generator: Any,
        grpc_client: DsfGrpcClient,
    ) -> None:
        self._rca_pipeline = rca_pipeline
        self._config_generator = config_generator
        self._grpc_client = grpc_client
        self._last_submission_time: float = 0.0
        self._submission_count: int = 0
        self._allow_count: int = 0
        self._deny_count: int = 0
        self._amend_count: int = 0
        self._error_count: int = 0

        logger.info("IntentSubmissionPipeline initialised")

    def run_once(self) -> SubmissionResult | None:
        """Execute a single pipeline cycle.

        1. Run RCA on latest telemetry.
        2. If anomalies found, generate an intent.
        3. If cooldown elapsed, submit intent to DSF.
        4. Return the submission result.

        Returns:
            :class:`SubmissionResult` if an intent was submitted, ``None`` otherwise.
        """
        logger.info("Running single pipeline cycle")

        # Step 1 – RCA
        try:
            anomalies = self._rca_pipeline.run_single()
        except Exception as exc:
            logger.error("RCA pipeline error: %s", exc, exc_info=True)
            return None

        if not anomalies:
            logger.debug("No anomalies detected – skipping intent generation")
            return None

        # Step 2 – Intent generation
        try:
            intent = self._config_generator.generate_intent(anomalies)
        except Exception as exc:
            logger.error(
                "Intent generation error: %s",
                exc,
                exc_info=True,
            )
            return None

        # Step 3 – Cooldown check
        cooldown = self._config_generator._config.intent_cooldown_s
        if not self._should_submit(self._last_submission_time, cooldown):
            logger.debug(
                "Intent cooldown active (%.1fs remaining) – skipping submission",
                cooldown - (time.time() - self._last_submission_time),
            )
            return None

        # Step 4 – DSF submission
        try:
            result = self._grpc_client.submit_intent(intent)
            self._last_submission_time = time.time()
            self._submission_count += 1

            # Update counters
            if result.decision == "ALLOW":
                self._allow_count += 1
            elif result.decision == "DENY":
                self._deny_count += 1
            elif result.decision == "AMEND":
                self._amend_count += 1
            elif result.decision == "ERROR":
                self._error_count += 1

            # Handle verdict
            self._handle_verdict(result)

            return result

        except Exception as exc:
            logger.error(
                "DSF submission error: %s",
                exc,
                exc_info=True,
            )
            self._error_count += 1
            return None

    def run_continuous(self, poll_interval: float = 5.0) -> None:
        """Run the pipeline continuously in a blocking loop.

        Each cycle: run_once, sleep, log statistics.

        Args:
            poll_interval: Seconds between pipeline cycles.

        Note:
            Blocks indefinitely. Use ``KeyboardInterrupt`` or a threading
            event to stop.
        """
        logger.info(
            "IntentSubmissionPipeline starting continuous mode (poll_interval=%.1fs)",
            poll_interval,
        )
        cycle = 0

        try:
            while True:
                cycle += 1
                logger.debug("Pipeline cycle %d starting", cycle)

                try:
                    result = self.run_once()

                    if result is not None:
                        logger.info(
                            "Cycle %d result: intent=%s decision=%s "
                            "(stats: total=%d allow=%d deny=%d amend=%d error=%d)",
                            cycle,
                            result.intent_id,
                            result.decision,
                            self._submission_count,
                            self._allow_count,
                            self._deny_count,
                            self._amend_count,
                            self._error_count,
                        )

                except Exception as exc:
                    logger.error(
                        "Pipeline cycle %d unhandled error: %s",
                        cycle,
                        exc,
                        exc_info=True,
                    )

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("IntentSubmissionPipeline continuous mode stopped")
        finally:
            self._log_final_stats()

    def _handle_verdict(self, result: SubmissionResult) -> None:
        """Process a DSF verdict and take appropriate follow-up actions.

        - ALLOW: Log and accept.
        - DENY: Log denial with rationale; no resubmission.
        - AMEND: Extract amended intent; log for potential resubmission.
        - RATE_LIMIT: Apply cooldown and schedule retry.
        - ERROR: Log error for monitoring.

        Args:
            result: The DSF submission result to handle.
        """
        decision = result.decision

        if decision == "ALLOW":
            logger.info(
                "✓ Intent %s ALLOWED by DSF – CBF=%s Lyapunov=%s Zeno=%s CUE=%s",
                result.intent_id,
                result.cbf_passed,
                result.lyapunov_passed,
                result.zeno_passed,
                result.cue_passed,
            )

        elif decision == "DENY":
            logger.warning(
                "✗ Intent %s DENIED by DSF – rationale: %s",
                result.intent_id,
                result.rationale,
            )
            if result.cbf_passed is False:
                logger.warning("  → CBF invariant violation detected")
            if result.lyapunov_passed is False:
                logger.warning("  → Lyapunov stability violation detected")
            if result.zeno_passed is False:
                logger.warning("  → Zeno-freeness violation detected")
            if result.cue_passed is False:
                logger.warning("  → CUE compliance violation detected")

        elif decision == "AMEND":
            logger.info(
                "≈ Intent %s AMENDED by DSF – rationale: %s",
                result.intent_id,
                result.rationale,
            )
            if result.amended_intent:
                logger.info(
                    "  → Amended parameters: %s",
                    json.dumps(result.amended_intent, indent=2),
                )
                # Note: Resubmission of amended intents should be handled
                # by a separate approval workflow to avoid infinite loops

        elif decision == "RATE_LIMIT":
            logger.warning(
                "⏳ Intent %s RATE LIMITED by DSF – backing off",
                result.intent_id,
            )
            # Apply extended cooldown
            self._last_submission_time = time.time()

        else:
            logger.error(
                "✗ Intent %s received unexpected decision: %s (%s)",
                result.intent_id,
                decision,
                result.error or "no error details",
            )

    def _should_submit(
        self,
        last_submission_time: float,
        cooldown: float,
    ) -> bool:
        """Check whether the cooldown period has elapsed.

        Args:
            last_submission_time: Unix timestamp of the last submission.
            cooldown:             Required cooldown period in seconds.

        Returns:
            ``True`` if submission is allowed, ``False`` otherwise.
        """
        if last_submission_time == 0.0:
            return True  # Never submitted before
        elapsed = time.time() - last_submission_time
        return elapsed >= cooldown

    def _log_final_stats(self) -> None:
        """Log aggregate statistics when the pipeline stops."""
        logger.info(
            "Pipeline final statistics: total=%d allow=%d deny=%d amend=%d error=%d",
            self._submission_count,
            self._allow_count,
            self._deny_count,
            self._amend_count,
            self._error_count,
        )


# ---------------------------------------------------------------------------
# Module entry-point for direct execution
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the IntentSubmissionPipeline in standalone demo mode."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # In a real deployment, these would be injected from the application config.
    # For standalone execution, we demonstrate with simulation-mode components.

    from intent.config_generator import ConfigGenerator  # noqa: E402
    from inference.rca_chain import RCAPipeline  # noqa: E402

    logger.info("Starting DSF gRPC client in standalone mode")

    # Create components
    rca = RCAPipeline(
        kafka_config={
            "bootstrap_servers": "172.29.0.10:9092",
            "topic": "network-telemetry",
            "group_id": "intent-pipeline-group",
        },
    )

    generator = ConfigGenerator()

    grpc_config = GrpcClientConfig(
        dsf_host="172.29.0.20",
        dsf_port=50052,
        use_tls=False,  # Use insecure for local dev
    )
    client = DsfGrpcClient(config=grpc_config)

    pipeline = IntentSubmissionPipeline(
        rca_pipeline=rca,
        config_generator=generator,
        grpc_client=client,
    )

    try:
        pipeline.run_continuous(poll_interval=5.0)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        client.close()
        rca.close()


if __name__ == "__main__":
    main()
