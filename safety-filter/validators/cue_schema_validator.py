#!/usr/bin/env python3
"""
CUE Schema Validator for 6G Digital Immunity — DSF Gate 1 (Syntactic).

Executes the CUE CLI to validate TelcoLLM-generated network configuration
intents against `.cue` (and adapted `.yaml`) schemas that codify Osmocom /
3GPP configuration structure.

Gate 1 is the first of three sequential gates in the Deterministic Safety
Filter (DSF).  Intent JSON that fails syntactic validation is rejected
immediately — the CBF and Zeno gates are never consulted.

Typical usage
-------------
    >>> validator = CueSchemaValidator()
    >>> result = validator.validate_intent(intent_dict, target_element_type="bts")
    >>> if result.passed:
    ...     print("Intent passed Gate 1")
    ... else:
    ...     for err in result.errors:
    ...         print(f"{err.field_path}: {err.message}")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(
    "safety_filter.validators.cue_schema_validator"
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_CUE_BIN: str = "/usr/local/bin/cue"

# Schema directory is resolved relative to this file's parent tree:
#   safety-filter/validators/cue_schema_validator.py
#   safety-filter/schemas/   (co-located schemas)
_SCHEMA_SEARCH_ROOTS: List[str] = [
    str(Path(__file__).resolve().parent.parent / "schemas"),
    str(Path(__file__).resolve().parent.parent.parent / "osmo-bsc" / "config"),
    str(
        Path(__file__).resolve().resolve().parent.parent.parent / "osmo-bts" / "config"
    ),
]

# Mapping from logical element type to schema file name
_ELEMENT_SCHEMA_MAP: Dict[str, str] = {
    "amf": "amf.cue",
    "bts": "abis.cue",
    "bsc": "bsc.cue",
    "mme": "mme.cue",
    "enb": "enb.cue",
    "gnb": "gnb.cue",
    "core": "core.cue",
    "ran": "ran.cue",
    "cell": "cell.cue",
    "nssai": "nssai.cue",
    "qos": "qos.cue",
    "slice": "slice.cue",
    "handover": "handover.cue",
    "default": "intent.cue",
}

# Regex patterns for parsing CUE error output
_CUE_ERROR_PATTERN = re.compile(r"(?P<field_path>[^\s:]+)\s*:\s*(?P<message>.+)")
_CUE_VIOLATION_TYPES: Dict[str, str] = {
    "incomplete": "missing_required_field",
    "disallowed": "disallowed_field",
    "conflicting": "conflict",
    "invalid type": "type_mismatch",
    "out of range": "range_violation",
    "not enough": "insufficient_elements",
    "too many": "excess_elements",
}

# CUE execution timeout (seconds)
_DEFAULT_CUE_TIMEOUT: float = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CueValidationError:
    """Represents a single validation error produced by CUE schema checking.

    Attributes:
        field_path: Dot-separated JSON path to the offending field (e.g.
                    ``"cell.tx_power_dbm"``).  Falls back to ``"<root>"`` when
                    CUE reports an error without a specific location.
        message: Human-readable description of the violation as reported by
                 the CUE CLI.
        violation_type: Normalised violation category string (e.g.
                        ``"missing_required_field"``, ``"type_mismatch"``).
                        Heuristically derived from the raw CUE message.
    """

    field_path: str
    message: str
    violation_type: str

    def to_dict(self) -> Dict[str, str]:
        """Serialise to a plain dictionary for JSON / protobuf transport."""
        return {
            "field_path": self.field_path,
            "message": self.message,
            "violation_type": self.violation_type,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CueValidationError(field_path={self.field_path!r}, "
            f"message={self.message!r}, "
            f"violation_type={self.violation_type!r})"
        )


@dataclass
class CueValidationResult:
    """Aggregated result of a single CUE schema validation run.

    Attributes:
        passed: ``True`` when the intent JSON satisfies the schema with zero
                errors.  Warnings do *not* affect this flag.
        errors: Ordered list of :class:`CueValidationError` instances.
        warnings: Informal messages that do not block intent progression.
        cue_version: Version string reported by ``cue version`` (or
                     ``"unknown"`` when unavailable).
        elapsed_ms: Wall-clock time for the CUE invocation in milliseconds.
    """

    passed: bool
    errors: List[CueValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cue_version: str = "unknown"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "passed": self.passed,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "cue_version": self.cue_version,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    @property
    def error_count(self) -> int:
        """Number of hard validation errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of soft warnings."""
        return len(self.warnings)

    def __repr__(self) -> str:  # pragma: no cover
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"CueValidationResult({status}, errors={self.error_count}, "
            f"warnings={self.warning_count}, cue={self.cue_version!r})"
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class CueBinaryNotFoundError(FileNotFoundError):
    """Raised when the CUE binary cannot be located on ``$PATH`` or the
    configured ``CUE_BIN_PATH``."""


class CueExecutionError(RuntimeError):
    """Raised when the CUE subprocess exits with an unexpected code or
    times out."""


class CueSchemaNotFoundError(FileNotFoundError):
    """Raised when the requested schema file does not exist in any of the
    configured search roots."""


# ---------------------------------------------------------------------------
# Validator implementation
# ---------------------------------------------------------------------------
class CueSchemaValidator:
    """Synchronous and asynchronous CUE schema validator for TelcoLLM intents.

    The validator discovers the ``cue`` binary automatically (via
    ``shutil.which`` or the ``CUE_BIN_PATH`` environment variable) and
    maintains a search path for ``.cue`` / ``.yaml`` schema files.

    Parameters:
        cue_bin_path: Explicit path to the ``cue`` binary.  When ``None``
            (default) the validator searches ``$PATH`` and falls back to
            ``/usr/local/bin/cue``.
        schema_dir: Additional schema directory added to the front of the
            search path.  Built-in search roots are always consulted as a
            fallback.

    Raises:
        CueBinaryNotFoundError: If the CUE binary cannot be found.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        cue_bin_path: Optional[str] = None,
        schema_dir: Optional[str] = None,
    ) -> None:
        # --- resolve CUE binary ---
        self._cue_bin_path: str = self._resolve_cue_binary(cue_bin_path)
        self._cue_version: str = self._probe_cue_version()
        logger.info(
            "CueSchemaValidator initialised: bin=%s, version=%s",
            self._cue_bin_path,
            self._cue_version,
        )

        # --- resolve schema directories ---
        self._schema_dirs: List[Path] = []
        if schema_dir is not None:
            p = Path(schema_dir).resolve()
            if p.is_dir():
                self._schema_dirs.append(p)
                logger.info("Added schema directory (explicit): %s", p)
            else:
                logger.warning("Explicit schema_dir %s does not exist — skipping.", p)
        for root in _SCHEMA_SEARCH_ROOTS:
            p = Path(root).resolve()
            if p.is_dir():
                self._schema_dirs.append(p)
        logger.debug("Schema search path: %s", self._schema_dirs)

    # -------------------------------------------------------- binary helpers
    @staticmethod
    def _resolve_cue_binary(cue_bin_path: Optional[str]) -> str:
        """Locate the ``cue`` executable.

        Search order:
        1. Explicit *cue_bin_path* argument.
        2. ``CUE_BIN_PATH`` environment variable.
        3. ``shutil.which("cue")``
        4. Hard-coded default ``/usr/local/bin/cue``
        """
        candidates: List[Optional[str]] = [
            cue_bin_path,
            os.environ.get("CUE_BIN_PATH"),
            shutil.which("cue"),
            _DEFAULT_CUE_BIN,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate).resolve()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        raise CueBinaryNotFoundError(
            "Cannot locate the CUE binary.  Set CUE_BIN_PATH env var or "
            "install CUE and ensure it is on $PATH."
        )

    def _probe_cue_version(self) -> str:
        """Run ``cue version`` and capture the version string."""
        try:
            proc = subprocess.run(
                [self._cue_bin_path, "version"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            # CUE prints something like "cue version v0.5.0"
            first_line = (proc.stdout or "").splitlines()[0].strip()
            return first_line
        except Exception as exc:
            logger.warning("Unable to probe CUE version: %s", exc)
            return "unknown"

    # --------------------------------------------------------- schema lookup
    def _get_schema_path(self, element_type: str) -> Path:
        """Resolve a logical element type to an on-disk schema file.

        Parameters:
            element_type: Logical element identifier (e.g. ``"amf"``,
                          ``"bts"``, ``"cell"``).

        Returns:
            Absolute :class:`~pathlib.Path` to the matching schema file.

        Raises:
            CueSchemaNotFoundError: If no schema file can be found.
        """
        schema_filename = _ELEMENT_SCHEMA_MAP.get(
            element_type.lower(), _ELEMENT_SCHEMA_MAP["default"]
        )
        for schema_dir in self._schema_dirs:
            candidate = schema_dir / schema_filename
            if candidate.is_file():
                logger.debug(
                    "Schema resolved: element_type=%s -> %s",
                    element_type,
                    candidate,
                )
                return candidate
        # Try with .yaml extension for adapted ABIS / 3GPP schemas
        yaml_candidate = schema_filename.replace(".cue", ".yaml")
        for schema_dir in self._schema_dirs:
            candidate = schema_dir / yaml_candidate
            if candidate.is_file():
                logger.debug(
                    "Schema resolved (yaml fallback): element_type=%s -> %s",
                    element_type,
                    candidate,
                )
                return candidate

        searched = [str(d) for d in self._schema_dirs]
        raise CueSchemaNotFoundError(
            f"Schema file '{schema_filename}' (or .yaml variant) not found "
            f"in search path: {searched}"
        )

    # -------------------------------------------------------- core validation
    def validate_intent(
        self,
        intent_json: Dict[str, Any],
        target_element_type: str,
        timeout: float = _DEFAULT_CUE_TIMEOUT,
    ) -> CueValidationResult:
        """Validate an intent dictionary against the CUE schema for a given
        network element type (Gate 1).

        Steps:
        1. Resolve the schema file for *target_element_type*.
        2. Serialise *intent_json* to a temporary JSON file.
        3. Execute ``cue vet --all-errors <schema> <json>``.
        4. Parse stdout / stderr for errors and warnings.

        Parameters:
            intent_json: The intent payload produced by TelcoLLM.
            target_element_type: Logical element type used to select the
                CUE schema (e.g. ``"bts"``, ``"amf"``).
            timeout: Maximum wall-clock time (seconds) for the ``cue``
                subprocess.

        Returns:
            A :class:`CueValidationResult` with ``passed=True`` if no errors.
        """
        logger.info(
            "Gate 1 — CUE validation: element_type=%s, fields=%d",
            target_element_type,
            len(intent_json),
        )
        t0 = time.monotonic()

        # 1. resolve schema
        try:
            schema_path = self._get_schema_path(target_element_type)
        except CueSchemaNotFoundError as exc:
            logger.error("Schema lookup failed: %s", exc)
            return CueValidationResult(
                passed=False,
                errors=[
                    CueValidationError(
                        field_path="<schema>",
                        message=str(exc),
                        violation_type="schema_not_found",
                    )
                ],
                warnings=[],
                cue_version=self._cue_version,
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        # 2. generic validation
        result = self.validate_json_schema(intent_json, schema_path, timeout)
        result.elapsed_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "Gate 1 result: passed=%s, errors=%d, warnings=%d (%.1f ms)",
            result.passed,
            result.error_count,
            result.warning_count,
            result.elapsed_ms,
        )
        return result

    def validate_json_schema(
        self,
        intent_json: Dict[str, Any],
        schema_path: Path,
        timeout: float = _DEFAULT_CUE_TIMEOUT,
    ) -> CueValidationResult:
        """Validate *intent_json* against an arbitrary CUE schema file.

        Parameters:
            intent_json: Intent payload dictionary.
            schema_path: Path to the ``.cue`` (or ``.yaml``) schema.
            timeout: Subprocess timeout in seconds.

        Returns:
            :class:`CueValidationResult`.
        """
        # 3. write intent to temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="intent_")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(intent_json, f, indent=2, ensure_ascii=False)

            # 4. run CUE
            stdout, stderr, returncode = self._run_cue_vet(
                schema_path, tmp_path, timeout
            )
            errors, warnings = self._parse_cue_output(stdout, stderr)

            passed = returncode == 0 and len(errors) == 0
            return CueValidationResult(
                passed=passed,
                errors=errors,
                warnings=warnings,
                cue_version=self._cue_version,
            )
        except Exception as exc:
            logger.exception("Unexpected error during CUE validation")
            return CueValidationResult(
                passed=False,
                errors=[
                    CueValidationError(
                        field_path="<runtime>",
                        message=f"Unexpected error: {exc}",
                        violation_type="internal_error",
                    )
                ],
                cue_version=self._cue_version,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ----------------------------------------------------- subprocess helpers
    def _run_cue_vet(
        self, schema_path: Path, json_path: str, timeout: float
    ) -> Tuple[str, str, int]:
        """Execute ``cue vet --all-errors`` synchronously.

        Returns:
            Tuple of (stdout, stderr, returncode).

        Raises:
            CueExecutionError: On timeout or non-zero exit that indicates
                a tool-level failure rather than a validation finding.
        """
        cmd: List[str] = [
            self._cue_bin_path,
            "vet",
            "--all-errors",
            str(schema_path),
            json_path,
        ]
        logger.debug("Executing CUE: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None,
                env=None,
            )
            logger.debug(
                "CUE exit=%d, stdout=%d bytes, stderr=%d bytes",
                proc.returncode,
                len(proc.stdout or ""),
                len(proc.stderr or ""),
            )
            return proc.stdout or "", proc.stderr or "", proc.returncode

        except subprocess.TimeoutExpired as exc:
            raise CueExecutionError(
                f"CUE execution timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise CueExecutionError(f"Failed to execute CUE binary: {exc}") from exc

    # ---------------------------------------------------- output parsing
    def _parse_cue_output(
        self, stdout: str, stderr: str
    ) -> Tuple[List[CueValidationError], List[str]]:
        """Parse CUE CLI stdout/stderr into structured error and warning
        collections.

        CUE typically emits one error per line.  Each line may contain the
        field path, violation keyword, and a human-readable message.

        Returns:
            (errors, warnings) tuple.
        """
        errors: List[CueValidationError] = []
        warnings: List[str] = []

        combined = f"{stdout}\n{stderr}"

        for line in combined.splitlines():
            line = line.strip()
            if not line:
                continue

            # Attempt structured match
            match = _CUE_ERROR_PATTERN.search(line)
            if match:
                field_path = match.group("field_path").strip()
                message = match.group("message").strip()

                # Heuristic violation-type classification
                violation_type = self._classify_violation(message)

                errors.append(
                    CueValidationError(
                        field_path=field_path,
                        message=message,
                        violation_type=violation_type,
                    )
                )
                continue

            # Unstructured line — treat as warning unless it looks like
            # an error keyword
            lower_line = line.lower()
            if any(kw in lower_line for kw in ("error", "fail", "invalid")):
                errors.append(
                    CueValidationError(
                        field_path="<unknown>",
                        message=line,
                        violation_type="unclassified_error",
                    )
                )
            else:
                warnings.append(line)

        return errors, warnings

    @staticmethod
    def _classify_violation(message: str) -> str:
        """Map a CUE error message to a normalised violation type."""
        lower = message.lower()
        for keyword, vtype in _CUE_VIOLATION_TYPES.items():
            if keyword in lower:
                return vtype
        return "generic_violation"

    # ---------------------------------------------------- async variant
    async def async_validate_intent(
        self,
        intent_json: Dict[str, Any],
        target_element_type: str,
        timeout: float = _DEFAULT_CUE_TIMEOUT,
    ) -> CueValidationResult:
        """Async counterpart of :meth:`validate_intent`.

        Uses :func:`asyncio.create_subprocess_exec` to avoid blocking the
        event loop during CUE execution.
        """
        logger.info(
            "Gate 1 (async) — CUE validation: element_type=%s, fields=%d",
            target_element_type,
            len(intent_json),
        )
        t0 = time.monotonic()

        # resolve schema
        try:
            schema_path = self._get_schema_path(target_element_type)
        except CueSchemaNotFoundError as exc:
            logger.error("Schema lookup failed: %s", exc)
            return CueValidationResult(
                passed=False,
                errors=[
                    CueValidationError(
                        field_path="<schema>",
                        message=str(exc),
                        violation_type="schema_not_found",
                    )
                ],
                warnings=[],
                cue_version=self._cue_version,
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        # write temp file (sync — small JSON, negligible overhead)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="intent_")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(intent_json, f, indent=2, ensure_ascii=False)

            cmd: List[str] = [
                self._cue_bin_path,
                "vet",
                "--all-errors",
                str(schema_path),
                tmp_path,
            ]
            logger.debug("Async CUE: %s", " ".join(cmd))

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error("Async CUE timed out after %ss", timeout)
                return CueValidationResult(
                    passed=False,
                    errors=[
                        CueValidationError(
                            field_path="<runtime>",
                            message=f"CUE timed out after {timeout}s",
                            violation_type="timeout",
                        )
                    ],
                    cue_version=self._cue_version,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                )

            stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
            stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
            returncode = proc.returncode or 0

            errors, warnings = self._parse_cue_output(stdout, stderr)
            passed = returncode == 0 and len(errors) == 0

            result = CueValidationResult(
                passed=passed,
                errors=errors,
                warnings=warnings,
                cue_version=self._cue_version,
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )
            logger.info(
                "Gate 1 (async) result: passed=%s, errors=%d, warnings=%d",
                result.passed,
                result.error_count,
                result.warning_count,
            )
            return result

        except Exception as exc:
            logger.exception("Unexpected async error during CUE validation")
            return CueValidationResult(
                passed=False,
                errors=[
                    CueValidationError(
                        field_path="<runtime>",
                        message=f"Unexpected async error: {exc}",
                        violation_type="internal_error",
                    )
                ],
                cue_version=self._cue_version,
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # --------------------------------------------------- introspection
    @property
    def cue_bin_path(self) -> str:
        """Resolved path to the CUE binary."""
        return self._cue_bin_path

    @property
    def cue_version(self) -> str:
        """Version string reported by the CUE binary."""
        return self._cue_version

    @property
    def schema_search_dirs(self) -> List[str]:
        """Ordered list of directories searched for schema files."""
        return [str(p) for p in self._schema_dirs]

    def list_available_schemas(self) -> Dict[str, List[str]]:
        """Return a mapping of element type → available schema files across
        all search directories."""
        available: Dict[str, List[str]] = {}
        for element_type, filename in _ELEMENT_SCHEMA_MAP.items():
            paths: List[str] = []
            for d in self._schema_dirs:
                for ext in ("", ".yaml"):
                    candidate = d / (
                        filename if ext == "" else filename.replace(".cue", ext)
                    )
                    if candidate.is_file():
                        paths.append(str(candidate))
            if paths:
                available[element_type] = paths
        return available


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------
def validate_intent_file(
    intent_path: str,
    target_element_type: str,
    cue_bin_path: Optional[str] = None,
    schema_dir: Optional[str] = None,
) -> CueValidationResult:
    """Utility: load intent from a JSON file and validate it.

    Parameters:
        intent_path: Filesystem path to a JSON intent file.
        target_element_type: Logical element type for schema selection.
        cue_bin_path: Optional explicit CUE binary path.
        schema_dir: Optional additional schema directory.

    Returns:
        :class:`CueValidationResult`.

    Raises:
        FileNotFoundError: If *intent_path* does not exist.
        json.JSONDecodeError: If the intent file is not valid JSON.
    """
    intent_file = Path(intent_path).resolve()
    if not intent_file.is_file():
        raise FileNotFoundError(f"Intent file not found: {intent_path}")

    intent_json = json.loads(intent_file.read_text(encoding="utf-8"))
    validator = CueSchemaValidator(cue_bin_path=cue_bin_path, schema_dir=schema_dir)
    return validator.validate_intent(intent_json, target_element_type)


# ---------------------------------------------------------------------------
# Main — smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        validator = CueSchemaValidator()
        print(f"CUE binary : {validator.cue_bin_path}")
        print(f"CUE version: {validator.cue_version}")
        print(f"Schemas    : {validator.list_available_schemas()}")
    except CueBinaryNotFoundError as exc:
        print(f"[ERROR] {exc}", file=__import__("sys").stderr)
    except Exception:
        logger.exception("Smoke-test failed")
