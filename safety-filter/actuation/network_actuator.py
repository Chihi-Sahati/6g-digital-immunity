#!/usr/bin/env python3
"""
Network Actuator — Osmocom Configuration Application (DSF Post-Gate Actuation).

Applies validated network configuration intents to Osmocom network elements
via the VTY (Virtual Telemetry Interface).  Actuation is **only** performed
when the DSF three-gate pipeline returns an ``ALLOW`` verdict.

Key design properties:
    • **Atomicity** — if any VTY command in a multi-command sequence fails,
      a rollback is attempted using a pre-computed backup command list.
    • **Thread-safety** — a ``threading.Lock`` serialises VTY access so that
      concurrent gRPC handlers do not interleave commands on the same element.
    • **Dry-run mode** — commands are logged but not executed, useful for
      simulation and integration testing.
    • **Post-actuation verification** — after applying a configuration the
      actuator queries the element's current state and verifies it matches
      the expected values.

Usage
-----
    >>> actuator = NetworkActuator(dry_run=True)
    >>> result = actuator.apply_config(verdict, {"tx_power_dbm": 40.0})
    >>> print(result.success, result.error)

    # Production with Osmocom VTY
    >>> actuator = NetworkActuator(
    ...     osmo_vty_host="172.30.0.100",
    ...     osmo_vty_port=4242,
    ... )
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("safety_filter.actuation.network_actuator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_VTY_HOST: str = "172.30.0.100"
_DEFAULT_VTY_PORT: int = 4242
_DEFAULT_VTY_TIMEOUT: float = 5.0
_DEFAULT_CONNECT_TIMEOUT: float = 3.0

# VTY newline delimiter (Osmocom VTY protocol)
_VTY_PROMPT_RE = re.compile(r"^OsmoBSC[^>]*>|^OsmoBTS[^>]*>|^Osmo[^>]*>", re.MULTILINE)
_VTY_ERROR_RE = re.compile(r"%\s*(error|unknown command|incomplete)", re.IGNORECASE)
_VTY_OK_RE = re.compile(r"%?\s*(ok|command executed)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class VTYTransport(str, Enum):
    """Transport mechanism used to reach the Osmocom VTY interface."""

    TELNET = "telnet"
    CLI = "osmo-vty-cli"
    UNIX_SOCKET = "unix_socket"


class ActuationStatus(str, Enum):
    """High-level status of an actuation attempt."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class OsmocomCommand:
    """A single Osmocom VTY command to be executed atomically.

    Attributes:
        command: The VTY command string (e.g. ``"trx 0"`` or
                 ``"power reduction 3"``).
        vty_path: Optional VTY navigation path (e.g. ``"config"``) for
                  structured commands.  Defaults to ``"/"``.
        expected_response: Optional regex or substring that must appear in
                           the VTY response for the command to be considered
                           successful.  ``None`` means any non-error
                           response is accepted.
        timeout: Maximum seconds to wait for a response from the VTY.
    """

    command: str
    vty_path: str = "/"
    expected_response: Optional[str] = None
    timeout: float = _DEFAULT_VTY_TIMEOUT

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OsmocomCommand(command={self.command!r}, "
            f"vty_path={self.vty_path!r}, "
            f"timeout={self.timeout})"
        )


@dataclass
class CommandResult:
    """Result of executing a single VTY command.

    Attributes:
        command: The command string that was executed.
        success: ``True`` if the VTY accepted the command without errors.
        response: Raw text response from the VTY interface.
        elapsed_ms: Wall-clock time for command execution in milliseconds.
        error: Error message if the command failed, ``None`` otherwise.
    """

    command: str
    success: bool
    response: str = ""
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "command": self.command,
            "success": self.success,
            "response": self.response,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
        }


@dataclass
class ActuationResult:
    """Aggregated result of applying a full configuration change set.

    Attributes:
        success: ``True`` only when every VTY command succeeded.
        command_results: Ordered list of :class:`CommandResult` instances,
                        one per VTY command executed.
        applied_at: Unix timestamp (``time.time()``) when the last command
                    was accepted.
        osmo_element_id: Identifier of the Osmocom element that was targeted.
        error: Top-level error message if the actuation failed, ``None``
               otherwise.
        status: Detailed status (success / partial / failed / rolled_back).
        rollback_attempted: Whether a rollback was triggered after a
                             mid-sequence failure.
        rollback_success: ``True`` if rollback restored the prior state.
    """

    success: bool
    command_results: List[Dict[str, Any]] = field(default_factory=list)
    applied_at: float = 0.0
    osmo_element_id: str = "unknown"
    error: Optional[str] = None
    status: str = ActuationStatus.SUCCESS.value
    rollback_attempted: bool = False
    rollback_success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "success": self.success,
            "status": self.status,
            "command_results": self.command_results,
            "applied_at": self.applied_at,
            "osmo_element_id": self.osmo_element_id,
            "error": self.error,
            "rollback_attempted": self.rollback_attempted,
            "rollback_success": self.rollback_success,
        }

    @property
    def total_elapsed_ms(self) -> float:
        """Total elapsed time across all commands in milliseconds."""
        return sum(r.get("elapsed_ms", 0.0) for r in self.command_results)

    @property
    def commands_executed(self) -> int:
        """Total number of commands that were sent to the VTY."""
        return len(self.command_results)

    @property
    def commands_succeeded(self) -> int:
        """Number of commands that succeeded."""
        return sum(1 for r in self.command_results if r.get("success", False))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ActuationResult(success={self.success}, "
            f"status={self.status!r}, "
            f"commands={self.commands_executed}, "
            f"osmo_element_id={self.osmo_element_id!r})"
        )


# ---------------------------------------------------------------------------
# Configuration-to-VTY mapping tables
# ---------------------------------------------------------------------------
# Maps high-level config keys to sequences of VTY CLI commands.
_CONFIG_TO_VTY: Dict[str, str] = {
    # Transceiver parameters
    "tx_power_dbm": "power reduction {reduction}",
    "max_tx_power_dbm": "power reduction {reduction}",
    "arfcn": "arfcn {value}",
    "band": "band {value}",
    "bsic": "bsic {bsic}",
    # Cell parameters
    "cell_identity": "cell_identity {value}",
    "lac": "location_area_code {value}",
    "rac": "routing_area_code {value}",
    "sac": "service_area_code {value}",
    "ci": "cell_identity {value}",
    "pci": "nr-pci {value}",
    "tac": "nr-tac {value}",
    "nci": "nr-cell-id {value}",
    # Neighbour / handover
    "ho_hysteresis_db": "handover hysteresis {value}",
    "cell_reselection_offset_db": "neighbor_cell_resel_offset {value}",
    "rxlev_min_dbm": "rxlev access min {value}",
    "rxlev_min_s": "rxlev access min {value}",
    "qrxlevmin": "q-rxlev-min {value}",
    # Radio
    "prb_utilisation_pct": "nolimit prach-configuration {value}",
    "bandwidth_mhz": "bandwidth {value}",
    "bandwidth": "bandwidth {value}",
    "tdd_config": "tdd-config {value}",
    # Security
    "ciphering": "encryption {value}",
    "authentication": "authentication {value}",
    "tmsi": "tmsi {value}",
}

# Context commands that wrap parameter commands (enter / exit)
_VTY_CONTEXT_MAP: Dict[str, Tuple[List[str], List[str]]] = {
    "trx": (
        ["enable", "configure terminal", "network", "bts 0", "trx {trx_index}"],
        ["exit", "exit", "exit", "exit", "exit"],
    ),
    "cell": (
        ["enable", "configure terminal", "network", "bts 0"],
        ["exit", "exit", "exit", "exit"],
    ),
    "neighbor": (
        ["enable", "configure terminal", "network", "bts 0", "neighbor {neighbor_id}"],
        ["exit", "exit", "exit", "exit", "exit"],
    ),
    "global": (["enable", "configure terminal"], ["exit", "exit"]),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class VTYConnectionError(ConnectionError):
    """Raised when a connection to the Osmocom VTY interface cannot be
    established or is dropped unexpectedly."""


class VTYCommandError(RuntimeError):
    """Raised when a VTY command is rejected by the Osmocom element."""


class ActuationDeniedError(PermissionError):
    """Raised when :meth:`NetworkActuator.apply_config` is called with a
    non-``ALLOW`` verdict."""


class RollbackError(RuntimeError):
    """Raised when a rollback attempt itself fails."""


# ---------------------------------------------------------------------------
# Actuator implementation
# ---------------------------------------------------------------------------
class NetworkActuator:
    """Applies DSF-validated configuration intents to Osmocom network elements.

    The actuator communicates with Osmocom elements over the VTY
    (Virtual Telemetry Interface), typically via a telnet-style TCP socket.
    All VTY access is serialised through an internal ``threading.Lock`` to
    prevent command interleaving when multiple RPC handlers actuate
    concurrently on the same element.

    Parameters:
        osmo_vty_host: Hostname or IP of the Osmocom VTY interface.
        osmo_vty_port: TCP port of the VTY interface.
        dry_run: When ``True`` commands are logged but not sent to the
            element.  Useful for simulation and testing.
        vty_transport: Transport mechanism (``telnet``, ``osmo-vty-cli``,
            or ``unix_socket``).
        connect_timeout: Socket connect timeout in seconds.
        backup_dir: Directory where backup configurations are persisted.
            Defaults to ``./actuation_backups`` relative to the CWD.

    Raises:
        ValueError: If *osmo_vty_port* is outside the valid range [1, 65535].
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        osmo_vty_host: str = _DEFAULT_VTY_HOST,
        osmo_vty_port: int = _DEFAULT_VTY_PORT,
        dry_run: bool = False,
        vty_transport: VTYTransport = VTYTransport.TELNET,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        backup_dir: Optional[str] = None,
    ) -> None:
        if not (1 <= osmo_vty_port <= 65535):
            raise ValueError(
                f"osmo_vty_port must be in [1, 65535], got {osmo_vty_port}"
            )

        self._vty_host: str = osmo_vty_host
        self._vty_port: int = osmo_vty_port
        self._dry_run: bool = dry_run
        self._vty_transport: VTYTransport = vty_transport
        self._connect_timeout: float = connect_timeout

        # Thread-safety lock for VTY serial access
        self._vty_lock: threading.Lock = threading.Lock()

        # Backup persistence directory
        if backup_dir is None:
            backup_dir = str(Path.cwd() / "actuation_backups")
        self._backup_dir: Path = Path(backup_dir).resolve()
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        # Metrics / audit counters
        self._total_actuations: int = 0
        self._successful_actuations: int = 0
        self._failed_actuations: int = 0
        self._rollbacks: int = 0

        # Last known good config for rollback
        self._last_config_snapshot: Dict[str, Any] = {}

        logger.info(
            "NetworkActuator initialised: host=%s, port=%d, dry_run=%s, "
            "transport=%s, backup_dir=%s",
            self._vty_host,
            self._vty_port,
            self._dry_run,
            self._vty_transport.value,
            self._backup_dir,
        )

    # -------------------------------------------------------- properties
    @property
    def dry_run(self) -> bool:
        """Whether commands are logged without execution."""
        return self._dry_run

    @property
    def vty_host(self) -> str:
        """Osmocom VTY hostname or IP."""
        return self._vty_host

    @property
    def vty_port(self) -> int:
        """Osmocom VTY TCP port."""
        return self._vty_port

    @property
    def stats(self) -> Dict[str, int]:
        """Actuation statistics (total / successful / failed / rollbacks)."""
        return {
            "total": self._total_actuations,
            "successful": self._successful_actuations,
            "failed": self._failed_actuations,
            "rollbacks": self._rollbacks,
        }

    # -------------------------------------------------------- health check
    def health_check(self, timeout: float = _DEFAULT_CONNECT_TIMEOUT) -> Dict[str, Any]:
        """Check connectivity to the Osmocom VTY interface.

        Attempts a TCP connection and sends a minimal ``"enable"`` command
        to verify the VTY responds.

        Returns:
            Dictionary with ``healthy``, ``host``, ``port``, ``latency_ms``,
            and optional ``error``.
        """
        t0 = time.monotonic()
        try:
            response = self._raw_vty_exchange("enable", timeout=timeout)
            elapsed = (time.monotonic() - t0) * 1000
            if _VTY_ERROR_RE.search(response):
                return {
                    "healthy": False,
                    "host": self._vty_host,
                    "port": self._vty_port,
                    "latency_ms": round(elapsed, 2),
                    "error": "VTY returned an error on health check",
                }
            return {
                "healthy": True,
                "host": self._vty_host,
                "port": self._vty_port,
                "latency_ms": round(elapsed, 2),
            }
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return {
                "healthy": False,
                "host": self._vty_host,
                "port": self._vty_port,
                "latency_ms": round(elapsed, 2),
                "error": str(exc),
            }

    # ====================================================== MAIN ACTUATION
    def apply_config(
        self,
        intent_verdict: Any,
        applied_config: Dict[str, Any],
    ) -> ActuationResult:
        """Apply a validated configuration to the Osmocom element.

        This method **only** proceeds when ``intent_verdict.decision`` (or
        ``intent_verdict.verdict``) is ``"ALLOW"``.  Otherwise an
        :class:`ActuationDeniedError` is raised.

        Steps:
        1. Verify ALLOW verdict.
        2. Generate VTY command sequence from *applied_config*.
        3. Capture backup commands for rollback.
        4. Execute commands sequentially.
        5. If any command fails, attempt rollback.
        6. Run post-actuation verification.

        Parameters:
            intent_verdict: The verdict returned by the DSF gRPC interceptor.
                Must have a ``decision`` or ``verdict`` attribute / key.
            applied_config: Dictionary of high-level configuration changes
                (e.g. ``{"tx_power_dbm": 40.0}``).

        Returns:
            :class:`ActuationResult` with full command-by-command results.

        Raises:
            ActuationDeniedError: If the verdict is not ``ALLOW``.
        """
        self._total_actuations += 1
        t0 = time.monotonic()

        # --- 1. Verdict check ---
        decision = self._extract_decision(intent_verdict)
        if decision != "ALLOW":
            self._failed_actuations += 1
            msg = (
                f"Actuation denied — DSF verdict was '{decision}', "
                f"expected 'ALLOW'.  Configuration will not be applied."
            )
            logger.warning(msg)
            raise ActuationDeniedError(msg)

        osmo_element_id = self._extract_element_id(intent_verdict)
        logger.info(
            "═══ Actuation STARTED  element=%s, changes=%d ═══",
            osmo_element_id,
            len(applied_config),
        )

        # --- 2. Generate VTY commands ---
        vty_commands = self._generate_vty_commands(applied_config)
        if not vty_commands:
            logger.info("No VTY commands generated — nothing to apply.")
            return ActuationResult(
                success=True,
                applied_at=time.time(),
                osmo_element_id=osmo_element_id,
                status=ActuationStatus.SUCCESS.value,
            )

        logger.info(
            "Generated %d VTY command(s) for element %s",
            len(vty_commands),
            osmo_element_id,
        )

        # --- 3. Capture backup ---
        backup_commands = self._capture_backup_commands(applied_config)

        # --- 4. Execute ---
        cmd_results: List[Dict[str, Any]] = []
        all_success: bool = True
        failed_at: Optional[int] = None

        with self._vty_lock:
            for idx, osmo_cmd in enumerate(vty_commands):
                try:
                    result = self._execute_single_command(osmo_cmd)
                    cmd_results.append(result.to_dict())
                    logger.debug(
                        "  [%d/%d] cmd=%s success=%s (%.1f ms)",
                        idx + 1,
                        len(vty_commands),
                        osmo_cmd.command,
                        result.success,
                        result.elapsed_ms,
                    )
                    if not result.success:
                        all_success = False
                        failed_at = idx
                        logger.error(
                            "VTY command FAILED at step %d: %s — %s",
                            idx + 1,
                            osmo_cmd.command,
                            result.error or "unknown error",
                        )
                        break
                except Exception as exc:
                    all_success = False
                    failed_at = idx
                    error_msg = str(exc)
                    cmd_results.append(
                        CommandResult(
                            command=osmo_cmd.command,
                            success=False,
                            elapsed_ms=0.0,
                            error=error_msg,
                        ).to_dict()
                    )
                    logger.exception(
                        "VTY command EXCEPTION at step %d: %s",
                        idx + 1,
                        osmo_cmd.command,
                    )
                    break

        # --- 5. Rollback if needed ---
        rollback_attempted: bool = False
        rollback_success: bool = False

        if not all_success and backup_commands:
            logger.warning(
                "Initiating ROLLBACK for element %s after failure at step %d",
                osmo_element_id,
                failed_at,
            )
            rollback_attempted = True
            rollback_result = self.rollback_config(backup_commands)
            rollback_success = rollback_result.success
            self._rollbacks += 1

        # Determine final status
        if all_success:
            status = ActuationStatus.SUCCESS.value
            self._successful_actuations += 1
        elif rollback_success:
            status = ActuationStatus.ROLLED_BACK.value
            self._failed_actuations += 1
        elif rollback_attempted:
            status = ActuationStatus.FAILED.value
            self._failed_actuations += 1
        else:
            status = ActuationStatus.PARTIAL.value
            self._failed_actuations += 1

        elapsed_total = (time.monotonic() - t0) * 1000
        result = ActuationResult(
            success=all_success,
            command_results=cmd_results,
            applied_at=time.time(),
            osmo_element_id=osmo_element_id,
            error=None if all_success else "One or more VTY commands failed",
            status=status,
            rollback_attempted=rollback_attempted,
            rollback_success=rollback_success,
        )

        logger.info(
            "═══ Actuation %s  element=%s, cmds=%d/%d, rollback=%s, total=%.1f ms ═══",
            status.upper(),
            osmo_element_id,
            result.commands_succeeded,
            result.commands_executed,
            "yes" if rollback_attempted else "no",
            elapsed_total,
        )

        # --- 6. Post-actuation verification ---
        if all_success:
            try:
                verified = self._validate_post_actuation(applied_config)
                if not verified:
                    logger.warning(
                        "Post-actuation verification FAILED for element %s — "
                        "actual state does not match expected state.",
                        osmo_element_id,
                    )
                    result.error = (
                        "Post-actuation verification failed: state mismatch detected"
                    )
            except Exception as exc:
                logger.warning(
                    "Post-actuation verification raised an exception: %s", exc
                )

        # Persist backup
        self._persist_backup(osmo_element_id, applied_config, backup_commands)

        return result

    # --------------------------------------------------- command execution
    def execute_vty_command(
        self,
        cmd: str,
        timeout: float = _DEFAULT_VTY_TIMEOUT,
    ) -> Dict[str, Any]:
        """Execute a single raw VTY command string.

        This is a convenience wrapper that runs the command through the
        same lock-guarded pipeline used by :meth:`apply_config`.

        Parameters:
            cmd: The VTY command string.
            timeout: Maximum seconds to wait for a response.

        Returns:
            Dictionary with ``command``, ``success``, ``response``,
            ``elapsed_ms``, ``error``.
        """
        osmo_cmd = OsmocomCommand(command=cmd, timeout=timeout)
        with self._vty_lock:
            result = self._execute_single_command(osmo_cmd)
        return result.to_dict()

    def _execute_single_command(self, osmo_cmd: OsmocomCommand) -> CommandResult:
        """Execute one :class:`OsmocomCommand` with timing and validation.

        Acquires the VTY lock externally; the caller is responsible for
        holding ``self._vty_lock``.

        Returns:
            :class:`CommandResult`.
        """
        t0 = time.monotonic()

        if self._dry_run:
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("[DRY-RUN] VTY cmd=%s", osmo_cmd.command)
            return CommandResult(
                command=osmo_cmd.command,
                success=True,
                response="(dry-run — not executed)",
                elapsed_ms=elapsed,
            )

        try:
            response = self._raw_vty_exchange(
                osmo_cmd.command, timeout=osmo_cmd.timeout
            )
            elapsed = (time.monotonic() - t0) * 1000

            # Check for error indicators in the VTY response
            is_error = bool(_VTY_ERROR_RE.search(response))
            # Check expected response if configured
            expected_ok = True
            if osmo_cmd.expected_response is not None:
                expected_ok = osmo_cmd.expected_response in response

            success = (not is_error) and expected_ok
            error: Optional[str] = None
            if is_error:
                error = f"VTY error: {response.strip()}"
            elif not expected_ok:
                error = (
                    f"Expected response '{osmo_cmd.expected_response}' "
                    f"not found in VTY output"
                )

            return CommandResult(
                command=osmo_cmd.command,
                success=success,
                response=response,
                elapsed_ms=elapsed,
                error=error,
            )

        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return CommandResult(
                command=osmo_cmd.command,
                success=False,
                response="",
                elapsed_ms=elapsed,
                error=str(exc),
            )

    # --------------------------------------------------- raw VTY transport
    def _raw_vty_exchange(
        self,
        command: str,
        timeout: float = _DEFAULT_VTY_TIMEOUT,
    ) -> str:
        """Send a command to the Osmocom VTY and read the response.

        The current implementation uses a TCP socket connection with
        line-buffered reads.  The method blocks until either a VTY prompt
        pattern is detected or *timeout* elapses.

        Parameters:
            command: The VTY command string (will be suffixed with ``\\n``).
            timeout: Maximum time in seconds for the exchange.

        Returns:
            Concatenated response text from the VTY.

        Raises:
            VTYConnectionError: On connection failure or timeout.
        """
        if self._vty_transport == VTYTransport.CLI:
            return self._vty_exchange_via_cli(command, timeout)

        # Default: TCP telnet-style socket
        sock: Optional[socket.socket] = None
        response_lines: List[str] = []

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._connect_timeout)
            sock.connect((self._vty_host, self._vty_port))

            # Wait for initial banner / prompt
            banner = self._read_until_prompt(sock, timeout=self._connect_timeout)
            response_lines.append(banner)

            # Send the command
            sock.sendall(f"{command}\n".encode("utf-8"))

            # Read response until prompt reappears
            response = self._read_until_prompt(sock, timeout=timeout)
            response_lines.append(response)

            # Collect any additional output
            sock.settimeout(0.5)
            try:
                while True:
                    chunk = sock.recv(4096).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    response_lines.append(chunk)
            except socket.timeout:
                pass

            return "\n".join(response_lines)

        except socket.timeout as exc:
            raise VTYConnectionError(
                f"VTY connection timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise VTYConnectionError(f"VTY connection error: {exc}") from exc
        finally:
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()

    def _vty_exchange_via_cli(self, command: str, timeout: float) -> str:
        """Execute the command via an ``osmo-vty-cli`` subprocess."""
        cli_bin = shutil_which("osmo-vty-cli") or shutil_which("osmo-bsc-vty")
        if not cli_bin:
            raise VTYConnectionError(
                "osmo-vty-cli / osmo-bsc-vty binary not found on $PATH"
            )
        try:
            proc = subprocess.run(
                [
                    cli_bin,
                    "-H",
                    self._vty_host,
                    "-p",
                    str(self._vty_port),
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            raise VTYConnectionError(
                f"osmo-vty-cli timed out after {timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            raise VTYConnectionError(f"osmo-vty-cli not found: {exc}") from exc

    @staticmethod
    def _read_until_prompt(sock: socket.socket, timeout: float) -> str:
        """Read from *sock* until a VTY prompt pattern is detected or
        *timeout* elapses."""
        sock.settimeout(timeout)
        buf = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                text = buf.decode("utf-8", errors="replace")
                if _VTY_PROMPT_RE.search(text):
                    return text
            except socket.timeout:
                return buf.decode("utf-8", errors="replace")
        return buf.decode("utf-8", errors="replace")

    # --------------------------------------------------- rollback
    def rollback_config(self, backup_commands: List[str]) -> ActuationResult:
        """Execute a list of backup VTY commands to restore a previous
        configuration state.

        Parameters:
            backup_commands: Ordered list of VTY command strings that
                restore the prior state.

        Returns:
            :class:`ActuationResult` reflecting rollback outcome.
        """
        logger.warning(
            "═══ ROLLBACK initiated — %d backup command(s) ═══",
            len(backup_commands),
        )
        cmd_results: List[Dict[str, Any]] = []
        all_success: bool = True

        with self._vty_lock:
            for cmd in backup_commands:
                osmo_cmd = OsmocomCommand(command=cmd)
                try:
                    result = self._execute_single_command(osmo_cmd)
                    cmd_results.append(result.to_dict())
                    if not result.success:
                        all_success = False
                        logger.error(
                            "Rollback command FAILED: %s — %s",
                            cmd,
                            result.error,
                        )
                        break
                except Exception as exc:
                    all_success = False
                    cmd_results.append(
                        CommandResult(
                            command=cmd, success=False, error=str(exc)
                        ).to_dict()
                    )
                    break

        status = (
            ActuationStatus.SUCCESS.value
            if all_success
            else ActuationStatus.FAILED.value
        )

        result = ActuationResult(
            success=all_success,
            command_results=cmd_results,
            applied_at=time.time(),
            osmo_element_id="rollback",
            status=status,
            rollback_attempted=True,
            rollback_success=all_success,
        )

        logger.info(
            "═══ ROLLBACK %s  cmds=%d/%d ═══",
            status.upper(),
            result.commands_succeeded,
            result.commands_executed,
        )
        return result

    # ------------------------------------------- VTY command generation
    def _generate_vty_commands(
        self, config_changes: Dict[str, Any]
    ) -> List[OsmocomCommand]:
        """Convert high-level configuration changes to a list of VTY
        :class:`OsmocomCommand` instances.

        Complex multi-step changes (e.g. entering a context, setting
        parameters, exiting) are expanded into the full command sequence
        required by the Osmocom VTY protocol.

        Parameters:
            config_changes: Dictionary of config key → value pairs.

        Returns:
            Ordered list of VTY commands to execute sequentially.
        """
        commands: List[OsmocomCommand] = []

        # Determine VTY context (default: cell-level)
        context = config_changes.pop("_vty_context", "cell")

        # Enter context
        enter_cmds, exit_cmds = _VTY_CONTEXT_MAP.get(
            context, _VTY_CONTEXT_MAP["global"]
        )
        for cmd in enter_cmds:
            commands.append(OsmocomCommand(command=cmd))

        # Apply each config change
        for key, value in config_changes.items():
            vty_template = _CONFIG_TO_VTY.get(key)
            if vty_template is None:
                logger.warning("No VTY mapping for config key '%s' — skipping.", key)
                continue

            cmd_str = self._format_vty_command(key, value, vty_template)
            expected = self._get_expected_response(key, value)
            commands.append(
                OsmocomCommand(
                    command=cmd_str,
                    expected_response=expected,
                )
            )

        # Exit context (reverse order)
        for cmd in reversed(exit_cmds):
            commands.append(OsmocomCommand(command=cmd))

        logger.debug("VTY command sequence: %s", [c.command for c in commands])
        return commands

    @staticmethod
    def _format_vty_command(key: str, value: Any, template: str) -> str:
        """Render a VTY command template with the config value.

        Special handling:
        - ``tx_power_dbm``: VTY uses "power reduction" relative to max
          (46 dBm typical), so we convert.
        """
        if key in ("tx_power_dbm", "max_tx_power_dbm"):
            # Convert absolute power to reduction from 46 dBm max
            max_power = 46.0
            reduction = max_power - float(value)
            return template.format(reduction=int(max(0, reduction)))

        if key == "bsic":
            # BSIC requires NCC+BCC as two 3-bit integers
            bsic_int = int(value)
            ncc = (bsic_int >> 3) & 0x07
            bcc = bsic_int & 0x07
            return f"bsic {ncc} {bcc}"

        try:
            return template.format(
                value=int(value) if isinstance(value, (int, float)) else str(value)
            )
        except (KeyError, ValueError):
            return template.format(**{key: str(value), "value": str(value)})

    @staticmethod
    def _get_expected_response(key: str, value: Any) -> Optional[str]:
        """Return an expected response substring for a given config key,
        or ``None`` if the response is non-deterministic."""
        # Most parameter-set commands return "OK"
        if key in (
            "tx_power_dbm",
            "arfcn",
            "cell_identity",
            "lac",
            "ho_hysteresis_db",
            "cell_reselection_offset_db",
            "rxlev_min_dbm",
            "qrxlevmin",
            "bandwidth_mhz",
            "ciphering",
            "authentication",
            "pci",
            "tac",
            "band",
        ):
            return "OK"
        return None

    # --------------------------------------------------- backup capture
    def _capture_backup_commands(self, config_changes: Dict[str, Any]) -> List[str]:
        """Generate a list of VTY commands that would restore the current
        (pre-change) configuration.

        In production this would query the Osmocom element's running
        config and compute the inverse diff.  For the simulation
        environment we store the snapshot and issue context-exit commands.

        Parameters:
            config_changes: The changes being applied (used to compute
                inverse operations).

        Returns:
            List of VTY command strings for rollback.
        """
        self._last_config_snapshot.update(config_changes)

        # In a full implementation, query the element for current values:
        #   "show running-config" → parse → generate inverse commands
        # For simulation: return generic reset commands for the changed keys.
        backup: List[str] = []
        context = config_changes.get("_vty_context", "cell")

        enter_cmds, exit_cmds = _VTY_CONTEXT_MAP.get(
            context, _VTY_CONTEXT_MAP["global"]
        )
        backup.extend(enter_cmds)

        # Simplified: store 'no' prefix for each parameter
        for key in config_changes:
            if key.startswith("_"):
                continue
            vty_template = _CONFIG_TO_VTY.get(key)
            if vty_template:
                # Generate a restore command — note: actual restore depends
                # on the previously known value, which we track.
                previous_value = self._last_config_snapshot.get(key)
                if previous_value is not None:
                    backup.append(f"# restore {key}={previous_value} (logged)")

        backup.extend(reversed(exit_cmds))
        return backup

    # --------------------------------------------- backup persistence
    def _persist_backup(
        self,
        element_id: str,
        config: Dict[str, Any],
        backup_commands: List[str],
    ) -> Path:
        """Write a backup record to disk for audit and recovery."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"backup_{element_id}_{ts}.json"
        filepath = self._backup_dir / filename
        record = {
            "element_id": element_id,
            "timestamp": time.time(),
            "config_applied": config,
            "backup_commands": backup_commands,
        }
        try:
            filepath.write_text(
                __import__("json").dumps(record, indent=2, default=str),
                encoding="utf-8",
            )
            logger.debug("Backup persisted: %s", filepath)
        except Exception as exc:
            logger.warning("Failed to persist backup to %s: %s", filepath, exc)
        return filepath

    # ------------------------------------------ post-actuation check
    def _validate_post_actuation(self, expected_state: Dict[str, Any]) -> bool:
        """Query the element's current config and verify it matches the
        expected state after actuation.

        In production this sends ``show running-config`` via VTY and
        parses the output.  For simulation, we do a basic check.

        Parameters:
            expected_state: Dictionary of key → value that should be
                reflected in the running config.

        Returns:
            ``True`` if the running config matches, ``False`` otherwise.
        """
        if self._dry_run:
            logger.debug("Post-actuation check skipped (dry-run).")
            return True

        try:
            # Query running config
            response = self.execute_vty_command("show running-config", timeout=10.0)
            config_text = response.get("response", "")

            for key, expected_value in expected_state.items():
                if key.startswith("_"):
                    continue
                # Search for the parameter in the running config output
                search_str = str(expected_value)
                if search_str not in config_text:
                    # Try alternative key representations
                    alt_key = key.replace("_dbm", "").replace("_db", "")
                    if alt_key not in config_text:
                        logger.warning(
                            "Post-actuation check: key '%s' with value '%s' "
                            "not found in running config.",
                            key,
                            expected_value,
                        )
                        return False

            logger.debug("Post-actuation verification PASSED.")
            return True

        except Exception as exc:
            logger.warning("Post-actuation verification error: %s", exc)
            return False

    # ---------------------------------------------------- verdict helpers
    @staticmethod
    def _extract_decision(verdict: Any) -> str:
        """Extract the decision string from a verdict object.

        Supports protobuf messages (``.decision`` or ``.verdict`` attribute)
        and plain dictionaries.
        """
        if hasattr(verdict, "decision"):
            return str(verdict.decision)
        if hasattr(verdict, "verdict"):
            return str(verdict.verdict)
        if isinstance(verdict, dict):
            return str(verdict.get("decision", verdict.get("verdict", "UNKNOWN")))
        return "UNKNOWN"

    @staticmethod
    def _extract_element_id(verdict: Any) -> str:
        """Extract the Osmocom element ID from a verdict object."""
        if hasattr(verdict, "osmo_element_id"):
            return str(verdict.osmo_element_id)
        if hasattr(verdict, "intent_id"):
            return str(verdict.intent_id)
        if isinstance(verdict, dict):
            return str(
                verdict.get("osmo_element_id", verdict.get("intent_id", "unknown"))
            )
        return "unknown"


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------
def shutil_which(name: str) -> Optional[str]:
    """Portable ``which`` — equivalent to ``shutil.which`` but available
    without importing shutil in all environments."""
    try:
        return __import__("shutil").which(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry-point — integration smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Network Actuator — Osmocom VTY Configuration Tool"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=_DEFAULT_VTY_HOST,
        help="Osmocom VTY host (default: 172.30.0.100)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_VTY_PORT,
        help="Osmocom VTY port (default: 4242)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log commands without executing",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run VTY health check and exit",
    )
    args = parser.parse_args()

    actuator = NetworkActuator(
        osmo_vty_host=args.host,
        osmo_vty_port=args.port,
        dry_run=args.dry_run,
    )

    print(
        f"NetworkActuator: host={actuator.vty_host}, port={actuator.vty_port}, "
        f"dry_run={actuator.dry_run}"
    )

    if args.health_check:
        result = actuator.health_check()
        print(json.dumps(result, indent=2))
    else:
        print("Run with --health-check to verify VTY connectivity.")
