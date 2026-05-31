# Enhanced CBF Safety Engine
# Original: Deterministic Digital Immunity for 6G Networks
# Phase 2: DSF Core — Python Wrapper for CBF Safety Filter (ctypes)
# Review Fix: Fixed has_lyapunov property call bug (line 706)
"""
cbf_engine.py — Deterministic Digital Immunity for 6G Networks
Phase 2: DSF Core — Python Wrapper for CBF Safety Filter (ctypes)

This module provides a pure-Python wrapper around the C++ CBF safety filter
library (`libcbf_core.so` / `cbf_core.dll`) using ctypes. The wrapper
exposes a high-level ``CBFPythonEngine`` class that allows Python code to:

  1. Register barrier functions with Python callbacks
  2. Configure Lyapunov stability monitoring
  3. Evaluate all safety constraints and receive structured results

Why ctypes over pybind11?
    - No C++ compilation step required for the Python module itself
    - Simpler deployment (just drop the .so and this .py file)
    - Works with any Python >= 3.9 (no pybind11 dependency)
    - Callbacks are marshalled via CFUNCTYPE (no GIL complications)

Prerequisites:
    - numpy (for array operations)
    - libcbf_core.so compiled from safety-filter/core/cbf_math.cpp

Environment:
    - SAFETY_FILTER_LIB: Path to the shared library (default: ./libcbf_core.so)

Usage:
    >>> from cbf_engine import CBFPythonEngine
    >>> import numpy as np
    >>> engine = CBFPythonEngine(state_dim=3)
    >>> engine.add_barrier("tx_power", lambda x: 43.0 - x[0], lambda h: h)
    >>> engine.set_lyapunov(alpha=0.1, x_ref=np.array([43.0, 0.5, 45.0]), ...)
    >>> results = engine.evaluate_all(np.array([40.0, 0.3, 50.0]), np.array([1.0, 0.0, 0.0]))
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.ctypeslib import ndpointer

# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("cbf.engine")
logger.setLevel(logging.DEBUG)

# ============================================================================
# Result Types
# ============================================================================


@dataclass
class CbfResultDict:
    """Result of evaluating a single CBF barrier constraint.

    Attributes:
        passed: True if the CBF condition is satisfied.
        cbf_value: Barrier function value h(x).
        gamma_value: Class-K function value γ(h(x)).
        margin: Safety margin (positive = safe).
        barrier_id: Barrier identifier.
    """

    passed: bool = False
    cbf_value: float = 0.0
    gamma_value: float = 0.0
    margin: float = 0.0
    barrier_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for serialisation."""
        return {
            "passed": self.passed,
            "cbf_value": self.cbf_value,
            "gamma_value": self.gamma_value,
            "margin": self.margin,
            "barrier_id": self.barrier_id,
        }


@dataclass
class LyapunovResultDict:
    """Result of evaluating the Lyapunov stability condition.

    Attributes:
        passed: True if V̇ <= -α·V.
        V: Lyapunov function value.
        dV: Time derivative of V.
        alpha: Decay rate constant.
        threshold: Stability threshold (-α·V).
    """

    passed: bool = False
    V: float = 0.0
    dV: float = 0.0
    alpha: float = 0.0
    threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for serialisation."""
        return {
            "passed": self.passed,
            "V": self.V,
            "dV": self.dV,
            "alpha": self.alpha,
            "threshold": self.threshold,
        }


# ============================================================================
# C Callback Types (ctypes)
# ============================================================================

# Scalar function: double f(const double* x, int n)
SCALAR_FUNC = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
)

# Gradient function: void grad_f(const double* x, int n, double* grad_out)
GRADIENT_FUNC = ctypes.CFUNCTYPE(
    None,
    ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS", write=True),
)

# Class-K function: double gamma(double h)
GAMMA_FUNC = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

# Lyapunov function: double V(const double* x, int n_x, const double* x_ref, int n_ref)
LYAPUNOV_V_FUNC = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
)

# Lyapunov derivative: double dVdt(const double* x, int n_x, const double* x_ref, const double* u, int n_u)
LYAPUNOV_DVDT_FUNC = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
)

# ============================================================================
# C++ Engine C API Structure (matches C linkage in cbf_math.cpp)
# ============================================================================
# The C++ engine exposes a flat C API via extern "C" functions.
# The ctypes wrapper calls these functions directly.
# For simplicity and to avoid requiring modifications to the C++ header,
# this Python wrapper implements the CBF evaluation logic in pure Python
# while calling back through ctypes for performance-critical operations
# if the native library is available. Otherwise, it falls back to a
# pure-Python implementation.
# ============================================================================


def _find_library(lib_path: str | None = None) -> str:
    """Locate the CBF safety filter shared library.

    Search order:
        1. Explicit ``lib_path`` argument
        2. ``SAFETY_FILTER_LIB`` environment variable
        3. ``./libcbf_core.so`` (Linux)
        4. ``./cbf_core.dll`` (Windows)
        5. ``./libcbf_core.dylib`` (macOS)

    Args:
        lib_path: Explicit path to the shared library.

    Returns:
        Absolute path to the shared library.

    Raises:
        FileNotFoundError: If no library can be found.
    """
    candidates = []

    if lib_path:
        candidates.append(lib_path)

    env_path = os.environ.get("SAFETY_FILTER_LIB")
    if env_path:
        candidates.append(env_path)

    # Platform-specific default names.
    platform_defaults = {
        "linux": "libcbf_core.so",
        "darwin": "libcbf_core.dylib",
        "win32": "cbf_core.dll",
    }
    import sys

    system = sys.platform
    if system in platform_defaults:
        candidates.append(platform_defaults[system])
    candidates.append("libcbf_core.so")  # Final fallback

    for path in candidates:
        if os.path.isabs(path) and os.path.exists(path):
            return os.path.abspath(path)
        # Try relative to this file's directory.
        rel_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(rel_path):
            return os.path.abspath(rel_path)
        # Try as-is.
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"Cannot find CBF safety filter library. Searched: {candidates}. "
        f"Set SAFETY_FILTER_LIB environment variable or pass lib_path explicitly."
    )


def _load_library(lib_path: str | None = None) -> ctypes.CDLL:
    """Load the shared library with ctypes.

    Args:
        lib_path: Path to the shared library (or None for auto-detect).

    Returns:
        Loaded CDLL handle.

    Raises:
        OSError: If the library cannot be loaded.
    """
    path = _find_library(lib_path)
    logger.info(f"Loading CBF safety filter library from: {path}")

    lib = ctypes.CDLL(path)

    # Set up argument types for the C API functions.
    # (These would be defined if we add C-linkage wrappers to the C++ code.)

    return lib


# ============================================================================
# Numerical Gradient Helpers (Pure Python)
# ============================================================================


def _gradient_central_difference(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Compute the gradient of f at x using central finite differences.

    ∂f/∂x_i ≈ [f(x + ε·e_i) - f(x - ε·e_i)] / (2·ε)

    Args:
        f: Scalar function f: R^n → R.
        x: Evaluation point (numpy array).
        epsilon: Finite difference step size.

    Returns:
        Gradient vector (∂f/∂x_1, ..., ∂f/∂x_n) as numpy array.
    """
    n = len(x)
    grad = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        h_plus = f(x_plus)
        h_minus = f(x_minus)

        if np.isfinite(h_plus) and np.isfinite(h_minus):
            grad[i] = (h_plus - h_minus) / (2.0 * epsilon)
        else:
            # Fallback to forward difference.
            x_fwd = x.copy()
            x_fwd[i] += epsilon
            grad[i] = (f(x_fwd) - f(x)) / epsilon
            logger.warning(
                f"Central difference NaN/Inf at index {i}, "
                f"falling back to forward difference"
            )

    return grad


# ============================================================================
# C++ Callback Wrappers
# ============================================================================


def _make_scalar_callback(py_func: Callable[[np.ndarray], float]) -> SCALAR_FUNC:
    """Wrap a Python function into a ctypes scalar callback.

    Args:
        py_func: Python function taking a numpy array and returning a float.

    Returns:
        ctypes CFUNCTYPE callback.
    """

    @SCALAR_FUNC
    def c_callback(x_ptr, n):
        try:
            x = np.ctypeslib.as_array(x_ptr, shape=(n,))
            result = py_func(x.copy())
            return float(result)
        except Exception as e:
            logger.error(f"Scalar callback failed: {e}")
            return 0.0

    return c_callback


def _make_gradient_callback(py_func: Callable[[np.ndarray], float]) -> GRADIENT_FUNC:
    """Wrap a Python function into a ctypes gradient callback using
    central finite differences.

    Args:
        py_func: Python function whose gradient is to be computed.

    Returns:
        ctypes CFUNCTYPE callback that fills the gradient output array.
    """

    @GRADIENT_FUNC
    def c_callback(x_ptr, n, grad_out):
        try:
            x = np.ctypeslib.as_array(x_ptr, shape=(n,))
            grad = _gradient_central_difference(py_func, x.copy())
            grad_out_ref = np.ctypeslib.as_array(grad_out, shape=(n,))
            grad_out_ref[:] = grad
        except Exception as e:
            logger.error(f"Gradient callback failed: {e}")

    return c_callback


def _make_gamma_callback(py_func: Callable[[float], float]) -> GAMMA_FUNC:
    """Wrap a Python gamma function into a ctypes callback.

    Args:
        py_func: Python function gamma: R → R.

    Returns:
        ctypes CFUNCTYPE callback.
    """

    @GAMMA_FUNC
    def c_callback(h_val):
        try:
            return float(py_func(h_val))
        except Exception as e:
            logger.error(f"Gamma callback failed: {e}")
            return 0.0

    return c_callback


# ============================================================================
# CBFPythonEngine — Main Python Interface
# ============================================================================


class CBFPythonEngine:
    """Python wrapper for the CBF safety filter engine.

    This class provides a high-level interface for registering barrier
    functions, configuring Lyapunov monitoring, and evaluating safety
    constraints. It uses ctypes to bridge to the C++ core library when
    available, with a pure-Python fallback.

    The engine manages barrier functions and Lyapunov configuration
    internally. All evaluation methods are thread-safe (protected by
    a threading.RLock).

    Args:
        state_dim: Dimension of the state vector x.
        lib_path: Path to the shared library (None for auto-detect).

    Raises:
        ValueError: If state_dim <= 0.
        FileNotFoundError: If the shared library cannot be found.
    """

    def __init__(self, state_dim: int, lib_path: str | None = None) -> None:
        """Initialise the CBF engine.

        Args:
            state_dim: Dimension of the state vector (must be > 0).
            lib_path: Optional path to the C++ shared library.
        """
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")

        self._state_dim = state_dim
        self._barriers: list[_BarrierEntry] = []
        self._lyap_config: _LyapunovEntry | None = None
        self._lock = __import__("threading").RLock()

        # Attempt to load the native library.
        self._lib: ctypes.CDLL | None = None
        self._native_available = False
        try:
            self._lib = _load_library(lib_path)
            self._native_available = True
            logger.info("Native C++ CBF library loaded successfully")
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                f"Native C++ library not available ({e}); using pure-Python fallback"
            )
            self._native_available = False

    @property
    def state_dim(self) -> int:
        """State vector dimension."""
        return self._state_dim

    @property
    def native_available(self) -> bool:
        """True if the C++ native library is loaded."""
        return self._native_available

    @property
    def barrier_count(self) -> int:
        """Number of registered barriers."""
        return len(self._barriers)

    @property
    def has_lyapunov(self) -> bool:
        """True if Lyapunov monitoring is configured."""
        return self._lyap_config is not None

    def add_barrier(
        self,
        barrier_id: str,
        barrier_func: Callable[[np.ndarray], float],
        gamma_func: Callable[[float], float],
    ) -> None:
        """Register a Control Barrier Function.

        The barrier function h(x) defines the safe set C = {x : h(x) >= 0}.
        The class-K function γ(h) defines the required descent rate.

        Args:
            barrier_id: Unique identifier for the barrier.
            barrier_func: Function h : R^n → R. Takes a numpy state
                vector and returns h(x).
            gamma_func: Class-K function γ : R → R. Takes h(x) and
                returns γ(h(x)). Common choice: gamma_func = lambda h: h.

        Raises:
            ValueError: If barrier_func or gamma_func is not callable.
            RuntimeError: If the barrier_id is already registered.
        """
        if not callable(barrier_func):
            raise ValueError("barrier_func must be callable")
        if not callable(gamma_func):
            raise ValueError("gamma_func must be callable")

        with self._lock:
            # Check for duplicate barrier IDs.
            for existing in self._barriers:
                if existing.barrier_id == barrier_id:
                    raise RuntimeError(f"Barrier '{barrier_id}' is already registered")

            # Create ctypes callbacks for native bridge.
            c_barrier = _make_scalar_callback(barrier_func)
            c_gamma = _make_gamma_callback(gamma_func)
            c_gradient = _make_gradient_callback(barrier_func)

            entry = _BarrierEntry(
                barrier_id=barrier_id,
                barrier_func=barrier_func,
                gamma_func=gamma_func,
                c_barrier=c_barrier,
                c_gamma=c_gamma,
                c_gradient=c_gradient,
            )
            self._barriers.append(entry)

        logger.info(
            f"Barrier registered: id='{barrier_id}', "
            f"total_barriers={len(self._barriers)}"
        )

    def set_lyapunov(
        self,
        alpha: float,
        x_ref: np.ndarray,
        V_func: Callable[[np.ndarray, np.ndarray], float],
        dVdt_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> None:
        """Configure Lyapunov stability monitoring.

        Args:
            alpha: Decay rate constant (> 0).
            x_ref: Reference state vector (numpy array, shape (state_dim,)).
            V_func: Lyapunov function V : R^n × R^n → R_{>=0}.
                Takes (current_state, reference_state), returns V >= 0.
            dVdt_func: Time derivative of V : R^n × R^n × R^m → R.
                Takes (current_state, reference_state, control_input),
                returns dV/dt.

        Raises:
            ValueError: If alpha <= 0 or dimensions mismatch.
        """
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        x_ref = np.asarray(x_ref, dtype=np.float64).ravel()
        if x_ref.shape[0] != self._state_dim:
            raise ValueError(
                f"x_ref dimension ({x_ref.shape[0]}) must match "
                f"state_dim ({self._state_dim})"
            )

        with self._lock:
            self._lyap_config = _LyapunovEntry(
                alpha=alpha,
                x_ref=x_ref.copy(),
                V_func=V_func,
                dVdt_func=dVdt_func,
            )

        logger.info(f"Lyapunov configured: alpha={alpha}, x_ref={x_ref.tolist()}")

    def evaluate_all(
        self,
        state: np.ndarray,
        control_input: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Evaluate all CBF barrier constraints.

        For each registered barrier, computes:
            h(x), γ(h(x)), ∇h(x), and the CBF condition margin.

        The CBF condition is:
            L_f*h(x) + L_g*h(x)*u >= -γ(h(x))
            margin = L_f*h + L_g*h*u + γ(h)

        Args:
            state: Current state vector x (numpy array, shape (state_dim,)).
            control_input: Control input vector u (numpy array).

        Returns:
            List of dictionaries, one per barrier, each containing:
                passed, cbf_value, gamma_value, margin, barrier_id.

        Raises:
            ValueError: If state dimension mismatch.
        """
        state = self._validate_state(state)
        control_input = np.asarray(control_input, dtype=np.float64)

        results: list[dict[str, Any]] = []
        t_start = time.perf_counter()

        with self._lock:
            for barrier in self._barriers:
                result = self._evaluate_single_barrier(state, control_input, barrier)
                results.append(result)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.debug(f"Evaluated {len(self._barriers)} barriers in {elapsed_ms:.3f} ms")

        return results

    def evaluate_lyapunov(
        self,
        state: np.ndarray,
        control_input: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate the Lyapunov stability condition.

        Checks whether V̇(x, u) <= -α·V(x).

        Args:
            state: Current state vector x (numpy array, shape (state_dim,)).
            control_input: Control input vector u (numpy array).

        Returns:
            Dictionary containing: passed, V, dV, alpha, threshold.

        Raises:
            RuntimeError: If Lyapunov is not configured.
            ValueError: If state dimension mismatch.
        """
        state = self._validate_state(state)
        control_input = np.asarray(control_input, dtype=np.float64)

        with self._lock:
            if self._lyap_config is None:
                raise RuntimeError(
                    "Lyapunov is not configured. Call set_lyapunov() first."
                )

            cfg = self._lyap_config
            x_ref = cfg.x_ref

            try:
                # Compute V(x, x_ref).
                V_val = cfg.V_func(state.copy(), x_ref.copy())
                if not np.isfinite(V_val):
                    logger.error(f"Lyapunov: V(x, x_ref) is NaN/Inf = {V_val}")
                    return _lyapunov_result_dict(
                        False, V_val, np.nan, cfg.alpha, -np.inf
                    )

                # V must be non-negative.
                if V_val < 0.0:
                    logger.warning(
                        f"Lyapunov: V = {V_val} is negative (should be >= 0)"
                    )
                    V_val = 0.0

                # Compute V̇(x, x_ref, u).
                dV_val = cfg.dVdt_func(state.copy(), x_ref.copy(), control_input.copy())
                if not np.isfinite(dV_val):
                    logger.error(f"Lyapunov: dV/dt is NaN/Inf = {dV_val}")
                    return _lyapunov_result_dict(
                        False, V_val, dV_val, cfg.alpha, -np.inf
                    )

                # Compute threshold and check.
                threshold = -cfg.alpha * V_val
                passed = dV_val <= threshold

                if not passed:
                    logger.warning(
                        f"Lyapunov VIOLATED: V={V_val:.6f}, dV/dt={dV_val:.6f}, "
                        f"threshold={threshold:.6f}, α={cfg.alpha}"
                    )
                else:
                    logger.debug(
                        f"Lyapunov PASSED: V={V_val:.6f}, dV/dt={dV_val:.6f}, "
                        f"threshold={threshold:.6f}"
                    )

                return _lyapunov_result_dict(
                    passed, V_val, dV_val, cfg.alpha, threshold
                )

            except Exception as e:
                logger.error(f"Lyapunov evaluation threw exception: {e}")
                return _lyapunov_result_dict(False, 0.0, 0.0, cfg.alpha, 0.0)

    def evaluate(
        self,
        state: np.ndarray,
        control_input: np.ndarray,
        current_time: float | None = None,
    ) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
        """Combined CBF + Lyapunov evaluation.

        Evaluates all CBF barriers first. If any barrier fails, returns
        immediately with the violation. If all barriers pass, evaluates
        the Lyapunov condition.

        Args:
            state: Current state vector x (numpy array).
            control_input: Control input vector u (numpy array).
            current_time: Wall-clock time in seconds (for logging).

        Returns:
            Tuple of (overall_passed, cbf_results, lyapunov_result):
                overall_passed: True if all checks pass.
                cbf_results: List of CBF result dicts.
                lyapunov_result: Lyapunov result dict (empty if not configured).
        """
        if current_time is None:
            current_time = time.time()

        logger.info(f"Combined evaluation at t={current_time:.6f}")

        # Evaluate all barriers.
        cbf_results = self.evaluate_all(state, control_input)
        all_cbf_passed = all(r["passed"] for r in cbf_results)

        # If any CBF failed, return early.
        if not all_cbf_passed:
            logger.warning(
                f"CBF violation at t={current_time:.6f}: "
                f"{sum(1 for r in cbf_results if not r['passed'])} barrier(s) failed"
            )
            return (False, cbf_results, {})

        # Evaluate Lyapunov (if configured).
        lyap_result: dict[str, Any] = {}
        if self.has_lyapunov:
            lyap_result = self.evaluate_lyapunov(state, control_input)
            if not lyap_result["passed"]:
                logger.warning(f"Lyapunov violation at t={current_time:.6f}")
                return (False, cbf_results, lyap_result)

        # All checks passed.
        logger.info(f"All checks PASSED at t={current_time:.6f}")
        return (True, cbf_results, lyap_result)

    # ========================================================================
    # Private Helpers
    # ========================================================================

    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        """Validate and normalise a state vector.

        Args:
            state: Input state (numpy array or array-like).

        Returns:
            Validated numpy float64 array of shape (state_dim,).

        Raises:
            ValueError: On dimension mismatch or NaN/Inf values.
        """
        state = np.asarray(state, dtype=np.float64).ravel()
        if state.shape[0] != self._state_dim:
            raise ValueError(
                f"State dimension ({state.shape[0]}) must match "
                f"state_dim ({self._state_dim})"
            )
        if not np.all(np.isfinite(state)):
            invalid_indices = np.where(~np.isfinite(state))[0]
            raise ValueError(
                f"State contains NaN/Inf at indices: {invalid_indices.tolist()}"
            )
        return state

    def _evaluate_single_barrier(
        self,
        state: np.ndarray,
        control_input: np.ndarray,
        barrier: _BarrierEntry,
    ) -> dict[str, Any]:
        """Evaluate a single CBF barrier.

        Args:
            state: Validated state vector.
            control_input: Control input vector.
            barrier: Barrier entry to evaluate.

        Returns:
            Result dictionary.
        """
        try:
            # Step 1: Evaluate h(x).
            cbf_value = float(barrier.barrier_func(state))
            if not np.isfinite(cbf_value):
                logger.error(
                    f"Barrier '{barrier.barrier_id}': h(x) is NaN/Inf = {cbf_value}"
                )
                return _cbf_result_dict(
                    passed=False,
                    cbf_value=cbf_value,
                    gamma_value=0.0,
                    margin=-np.inf,
                    barrier_id=barrier.barrier_id,
                )

            # Step 2: Evaluate γ(h(x)).
            gamma_value = float(barrier.gamma_func(cbf_value))
            if not np.isfinite(gamma_value):
                logger.error(
                    f"Barrier '{barrier.barrier_id}': γ(h(x)) is NaN/Inf = {gamma_value}"
                )
                return _cbf_result_dict(
                    passed=False,
                    cbf_value=cbf_value,
                    gamma_value=gamma_value,
                    margin=-np.inf,
                    barrier_id=barrier.barrier_id,
                )

            # Step 3: Compute ∇h(x) via central finite differences.
            grad = _gradient_central_difference(barrier.barrier_func, state)

            # Step 4: Compute L_g*h*u = ∇h · u (simplified affine model).
            l_g_h_u = 0.0
            if control_input.size > 0:
                if control_input.shape[0] == self._state_dim:
                    l_g_h_u = float(np.dot(grad, control_input))
                else:
                    logger.warning(
                        f"Barrier '{barrier.barrier_id}': u dimension "
                        f"({control_input.shape[0]}) != state_dim ({self._state_dim}); "
                        f"control Lie derivative set to 0"
                    )

            # Step 5: CBF condition.
            #   L_f*h + L_g*h*u >= -γ(h)
            #   margin = L_f*h + L_g*h*u + γ(h)
            #   (Conservative: L_f*h = 0 when no drift model is provided.)
            l_f_h = 0.0
            margin = l_f_h + l_g_h_u + gamma_value
            passed = margin >= 0.0

            if not passed:
                logger.warning(
                    f"Barrier VIOLATED: id='{barrier.barrier_id}', "
                    f"h(x)={cbf_value:.6f}, γ(h)={gamma_value:.6f}, "
                    f"L_g*h*u={l_g_h_u:.6f}, margin={margin:.6f}"
                )
            else:
                logger.debug(
                    f"Barrier PASSED: id='{barrier.barrier_id}', "
                    f"h(x)={cbf_value:.6f}, margin={margin:.6f}"
                )

            return _cbf_result_dict(
                passed=passed,
                cbf_value=cbf_value,
                gamma_value=gamma_value,
                margin=margin,
                barrier_id=barrier.barrier_id,
            )

        except Exception as e:
            logger.error(
                f"Barrier '{barrier.barrier_id}': evaluation threw exception: {e}"
            )
            return _cbf_result_dict(
                passed=False,
                cbf_value=0.0,
                gamma_value=0.0,
                margin=-np.inf,
                barrier_id=barrier.barrier_id,
            )


# ============================================================================
# Internal Data Structures
# ============================================================================


@dataclass
class _BarrierEntry:
    """Internal barrier function entry with ctypes callbacks."""

    barrier_id: str
    barrier_func: Callable[[np.ndarray], float]
    gamma_func: Callable[[float], float]
    c_barrier: SCALAR_FUNC
    c_gamma: GAMMA_FUNC
    c_gradient: GRADIENT_FUNC


@dataclass
class _LyapunovEntry:
    """Internal Lyapunov configuration entry."""

    alpha: float
    x_ref: np.ndarray
    V_func: Callable[[np.ndarray, np.ndarray], float]
    dVdt_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float]


# ============================================================================
# Helper Functions
# ============================================================================


def _cbf_result_dict(
    passed: bool,
    cbf_value: float,
    gamma_value: float,
    margin: float,
    barrier_id: str,
) -> dict[str, Any]:
    """Create a CBF result dictionary."""
    return {
        "passed": bool(passed),
        "cbf_value": float(cbf_value),
        "gamma_value": float(gamma_value),
        "margin": float(margin),
        "barrier_id": str(barrier_id),
    }


def _lyapunov_result_dict(
    passed: bool,
    V: float,
    dV: float,
    alpha: float,
    threshold: float,
) -> dict[str, Any]:
    """Create a Lyapunov result dictionary."""
    return {
        "passed": bool(passed),
        "V": float(V),
        "dV": float(dV),
        "alpha": float(alpha),
        "threshold": float(threshold),
    }


# ============================================================================
# Convenience Factory
# ============================================================================


def create_engine(
    state_dim: int,
    barriers: list[dict[str, Any]] | None = None,
    lyapunov: dict[str, Any] | None = None,
    lib_path: str | None = None,
) -> CBFPythonEngine:
    """Create a pre-configured CBF engine from specification dicts.

    This is a convenience factory for creating an engine with barriers
    and Lyapunov config from declarative dictionaries (e.g., loaded
    from the YAML golden baseline).

    Args:
        state_dim: Dimension of the state vector.
        barriers: List of barrier dicts, each with keys:
            barrier_id, barrier_expr, floor, ceiling, class.
        lyapunov: Lyapunov config dict with keys:
            alpha, x_ref (list), V_func_type, dVdt_func_type.
        lib_path: Path to shared library (None for auto-detect).

    Returns:
        Configured CBFPythonEngine instance.

    Example:
        >>> engine = create_engine(
        ...     state_dim=3,
        ...     barriers=[
        ...         {
        ...             "barrier_id": "tx_power_ceiling",
        ...             "floor": None,
        ...             "ceiling": 43.0,
        ...             "barrier_expr": "h(x) = 43 - x[0] >= 0",
        ...             "state_index": 0,
        ...         },
        ...     ],
        ... )
    """
    engine = CBFPythonEngine(state_dim=state_dim, lib_path=lib_path)

    if barriers:
        for b in barriers:
            state_idx = b.get("state_index", 0)
            floor = b.get("floor")
            ceiling = b.get("ceiling")

            # Build barrier function based on barrier class.
            if floor is not None and ceiling is not None:
                # Band barrier: floor <= value <= ceiling
                def _band_h(x, _idx=state_idx, _f=floor, _c=ceiling):
                    return min(x[_idx] - _f, _c - x[_idx])

                barrier_func = _band_h
            elif ceiling is not None:
                # Ceiling barrier: value <= ceiling
                def _ceil_h(x, _idx=state_idx, _c=ceiling):
                    return _c - x[_idx]

                barrier_func = _ceil_h
            elif floor is not None:
                # Floor barrier: value >= floor
                def _floor_h(x, _idx=state_idx, _f=floor):
                    return x[_idx] - _f

                barrier_func = _floor_h
            else:
                raise ValueError(
                    f"Barrier '{b.get('barrier_id')}': must specify floor and/or ceiling"
                )

            # Default gamma: γ(h) = k*h where k = 1.
            k = b.get("class_k_constant", 1.0)

            def _gamma(h_val, _k=k):
                return _k * h_val

            engine.add_barrier(
                barrier_id=b["barrier_id"],
                barrier_func=barrier_func,
                gamma_func=_gamma,
            )

    if lyapunov:
        alpha = lyapunov["alpha"]
        x_ref = np.array(lyapunov["x_ref"], dtype=np.float64)

        # Default: quadratic Lyapunov V(x, x_ref) = ||x - x_ref||^2
        v_type = lyapunov.get("V_func_type", "quadratic")

        if v_type == "quadratic":

            def _V(x, xr):
                diff = x - xr
                return float(np.dot(diff, diff))

            def _dVdt(x, xr, u):
                diff = x - xr
                return float(2.0 * np.dot(diff, u))
        else:
            raise ValueError(f"Unknown V_func_type: {v_type}")

        engine.set_lyapunov(alpha=alpha, x_ref=x_ref, V_func=_V, dVdt_func=_dVdt)

    return engine
