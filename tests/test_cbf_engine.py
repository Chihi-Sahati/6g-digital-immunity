"""
test_cbf_engine.py — Comprehensive unit tests for the CBF safety-filter engine.

Tests the pure-Python fallback path of CBFPythonEngine, including:
  - Engine initialisation (valid / invalid state_dim)
  - Barrier registration and duplicate detection
  - Ceiling, floor, and band barrier semantics
  - CBF condition pass / fail
  - Class-K gamma function variations
  - NaN / Inf state rejection
  - Dimension mismatch handling
  - Lyapunov configuration and stability checks
  - Combined CBF + Lyapunov evaluation
  - Central-difference gradient accuracy
  - Thread-safety under concurrent evaluation
  - The create_engine() factory function
"""

from __future__ import annotations

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root and safety-filter/core are on sys.path so that we
# can import cbf_engine without any package installation step.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SAFETY_FILTER_CORE = os.path.join(_PROJECT_ROOT, "safety-filter", "core")

for _p in (_PROJECT_ROOT, _SAFETY_FILTER_CORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Force pure-Python fallback by ensuring no native library is found.
# We test the Python path exclusively.
os.environ.pop("SAFETY_FILTER_LIB", None)

# ---------------------------------------------------------------------------
# Monkey-patch ndpointer to tolerate the unsupported ``write=True`` keyword
# argument that cbf_engine.py uses.  The ``write`` flag is never consulted
# in the pure-Python fallback path that we test here.
# ---------------------------------------------------------------------------
_orig_ndpointer = np.ctypeslib.ndpointer


def _patched_ndpointer(*args, **kwargs):
    kwargs.pop("write", None)
    return _orig_ndpointer(*args, **kwargs)


np.ctypeslib.ndpointer = _patched_ndpointer

from cbf_engine import (  # noqa: E402
    CBFPythonEngine,
    create_engine,
    _gradient_central_difference,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine_3d() -> CBFPythonEngine:
    """Create a 3-dimensional CBF engine with no barriers."""
    return CBFPythonEngine(state_dim=3)


@pytest.fixture
def engine_1d() -> CBFPythonEngine:
    """Create a 1-dimensional CBF engine with no barriers."""
    return CBFPythonEngine(state_dim=1)


@pytest.fixture
def ceiling_engine(engine_3d: CBFPythonEngine) -> CBFPythonEngine:
    """3-D engine with a tx-power ceiling barrier: h(x) = 43 - x[0]."""
    engine_3d.add_barrier(
        barrier_id="tx_power_ceiling",
        barrier_func=lambda x: 43.0 - x[0],
        gamma_func=lambda h: h,
    )
    return engine_3d


# ===========================================================================
# 1. Engine Initialisation
# ===========================================================================


class TestEngineInitialization:
    """Tests for CBFPythonEngine.__init__."""

    def test_valid_state_dim(self):
        """Engine should accept any positive integer state_dim."""
        for dim in (1, 2, 3, 10, 100):
            eng = CBFPythonEngine(state_dim=dim)
            assert eng.state_dim == dim

    def test_invalid_state_dim_zero(self):
        """state_dim=0 should raise ValueError."""
        with pytest.raises(ValueError, match="state_dim must be positive"):
            CBFPythonEngine(state_dim=0)

    def test_invalid_state_dim_negative(self):
        """Negative state_dim should raise ValueError."""
        with pytest.raises(ValueError, match="state_dim must be positive"):
            CBFPythonEngine(state_dim=-1)

    def test_default_properties(self):
        """New engine should have no barriers and no Lyapunov config."""
        eng = CBFPythonEngine(state_dim=3)
        assert eng.barrier_count == 0
        assert eng.has_lyapunov is False

    def test_pure_python_fallback(self):
        """Without a native library, the engine should report native_available=False."""
        eng = CBFPythonEngine(state_dim=3)
        # In CI / test environments the .so is almost never present.
        assert isinstance(eng.native_available, bool)


# ===========================================================================
# 2. Barrier Registration
# ===========================================================================


class TestBarrierRegistration:
    """Tests for add_barrier()."""

    def test_add_single_barrier(self, engine_3d: CBFPythonEngine):
        """Adding one barrier should increment barrier_count to 1."""
        engine_3d.add_barrier(
            barrier_id="b1",
            barrier_func=lambda x: 1.0 - x[0],
            gamma_func=lambda h: h,
        )
        assert engine_3d.barrier_count == 1

    def test_add_multiple_barriers(self, engine_3d: CBFPythonEngine):
        """Multiple barriers with unique IDs should all be stored."""
        for i in range(5):
            engine_3d.add_barrier(
                barrier_id=f"b{i}",
                barrier_func=lambda x, idx=i: 10.0 - x[idx % 3],
                gamma_func=lambda h: h,
            )
        assert engine_3d.barrier_count == 5

    def test_duplicate_barrier_id_raises(self, engine_3d: CBFPythonEngine):
        """Registering the same barrier_id twice should raise RuntimeError."""
        engine_3d.add_barrier(
            barrier_id="dup",
            barrier_func=lambda x: 5.0 - x[0],
            gamma_func=lambda h: h,
        )
        with pytest.raises(RuntimeError, match="already registered"):
            engine_3d.add_barrier(
                barrier_id="dup",
                barrier_func=lambda x: 3.0 - x[1],
                gamma_func=lambda h: h,
            )

    def test_non_callable_barrier_func_raises(self, engine_3d: CBFPythonEngine):
        """Passing a non-callable barrier_func should raise ValueError."""
        with pytest.raises(ValueError, match="barrier_func must be callable"):
            engine_3d.add_barrier(
                barrier_id="bad_bf",
                barrier_func="not a function",
                gamma_func=lambda h: h,
            )

    def test_non_callable_gamma_func_raises(self, engine_3d: CBFPythonEngine):
        """Passing a non-callable gamma_func should raise ValueError."""
        with pytest.raises(ValueError, match="gamma_func must be callable"):
            engine_3d.add_barrier(
                barrier_id="bad_gf",
                barrier_func=lambda x: 1.0,
                gamma_func=42,
            )


# ===========================================================================
# 3. Ceiling Barrier — h(x) = ceiling - x[0]
# ===========================================================================


class TestBarrierCeiling:
    """Tests for ceiling-type barrier: h(x) = C - x[0] >= 0 ⟹ x[0] <= C."""

    CEILING = 43.0

    @pytest.fixture
    def engine(self):
        eng = CBFPythonEngine(state_dim=2)
        eng.add_barrier(
            barrier_id="ceiling",
            barrier_func=lambda x: self.CEILING - x[0],
            gamma_func=lambda h: h,
        )
        return eng

    def test_safe_state_below_ceiling(self, engine: CBFPythonEngine):
        """State well below ceiling → h(x) > 0 → CBF should pass."""
        state = np.array([40.0, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        assert len(results) == 1
        assert results[0]["passed"] is True
        assert results[0]["cbf_value"] == pytest.approx(self.CEILING - 40.0)

    def test_boundary_state_at_ceiling(self, engine: CBFPythonEngine):
        """State exactly at ceiling → h(x) = 0 → margin depends on gamma."""
        state = np.array([self.CEILING, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        assert results[0]["cbf_value"] == pytest.approx(0.0)

    def test_unsafe_state_above_ceiling(self, engine: CBFPythonEngine):
        """State above ceiling → h(x) < 0, γ(h) < 0, margin likely negative."""
        state = np.array([50.0, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        assert results[0]["cbf_value"] < 0
        # With gamma(h) = h (negative) and L_g*h*u = 0, margin = γ(h) < 0
        assert results[0]["passed"] is False


# ===========================================================================
# 4. Floor Barrier — h(x) = x[0] - floor
# ===========================================================================


class TestBarrierFloor:
    """Tests for floor-type barrier: h(x) = x[0] - F >= 0 ⟹ x[0] >= F."""

    FLOOR = 5.0

    @pytest.fixture
    def engine(self):
        eng = CBFPythonEngine(state_dim=2)
        eng.add_barrier(
            barrier_id="floor",
            barrier_func=lambda x: x[0] - self.FLOOR,
            gamma_func=lambda h: h,
        )
        return eng

    def test_safe_state_above_floor(self, engine: CBFPythonEngine):
        """State above floor → h(x) > 0."""
        state = np.array([10.0, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        assert results[0]["passed"] is True
        assert results[0]["cbf_value"] == pytest.approx(5.0)

    def test_boundary_at_floor(self, engine: CBFPythonEngine):
        """State at floor → h(x) = 0."""
        state = np.array([self.FLOOR, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        assert results[0]["cbf_value"] == pytest.approx(0.0)

    def test_unsafe_state_below_floor(self, engine: CBFPythonEngine):
        """State below floor → h(x) < 0 → CBF fails."""
        state = np.array([2.0, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        assert results[0]["cbf_value"] < 0
        assert results[0]["passed"] is False


# ===========================================================================
# 5. Band Barrier — floor <= x[0] <= ceiling
# ===========================================================================


class TestBarrierBand:
    """Tests for band-type barrier: h(x) = min(x[0] - F, C - x[0])."""

    FLOOR = 5.0
    CEILING = 43.0

    @pytest.fixture
    def engine(self):
        eng = CBFPythonEngine(state_dim=2)

        def band_h(x):
            return min(x[0] - self.FLOOR, self.CEILING - x[0])

        eng.add_barrier(
            barrier_id="band",
            barrier_func=band_h,
            gamma_func=lambda h: h,
        )
        return eng

    def test_safe_in_band_center(self, engine: CBFPythonEngine):
        """State in the centre of the band → positive margin."""
        state = np.array([24.0, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        # h = min(24-5, 43-24) = min(19, 19) = 19
        assert results[0]["cbf_value"] == pytest.approx(19.0)
        assert results[0]["passed"] is True

    def test_safe_near_floor(self, engine: CBFPythonEngine):
        """State near the floor boundary → small positive h."""
        state = np.array([6.0, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        # h = min(6-5, 43-6) = min(1, 37) = 1
        assert results[0]["cbf_value"] == pytest.approx(1.0)

    def test_unsafe_below_floor(self, engine: CBFPythonEngine):
        """State below the floor → negative h → fail."""
        state = np.array([3.0, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        assert results[0]["cbf_value"] < 0
        assert results[0]["passed"] is False

    def test_unsafe_above_ceiling(self, engine: CBFPythonEngine):
        """State above the ceiling → negative h → fail."""
        state = np.array([50.0, 0.0])
        control = np.array([0.0, 0.0])
        results = engine.evaluate_all(state, control)
        assert results[0]["cbf_value"] < 0
        assert results[0]["passed"] is False


# ===========================================================================
# 6. CBF Condition Pass
# ===========================================================================


class TestCBFConditionPass:
    """Verify margin >= 0 when the control is safe."""

    def test_zero_control_safe_state(self, ceiling_engine: CBFPythonEngine):
        """With u=0 and state in safe set, margin = γ(h) >= 0."""
        state = np.array([40.0, 0.0, 0.0])
        control = np.array([0.0, 0.0, 0.0])
        results = ceiling_engine.evaluate_all(state, control)
        r = results[0]
        assert r["passed"] is True
        assert r["margin"] >= 0

    def test_control_away_from_boundary(self, ceiling_engine: CBFPythonEngine):
        """Negative u[0] pushes x[0] down (away from ceiling) → safe."""
        state = np.array([40.0, 0.0, 0.0])
        control = np.array([-1.0, 0.0, 0.0])
        results = ceiling_engine.evaluate_all(state, control)
        r = results[0]
        # h = 43-40 = 3, gamma = 3, grad_h[0] = -1
        # margin = 0 + (-1)*(-1) + 3 = 1 + 3 = 4 > 0
        assert r["passed"] is True
        assert r["margin"] > 0


# ===========================================================================
# 7. CBF Condition Fail
# ===========================================================================


class TestCBFConditionFail:
    """Verify margin < 0 when the control is unsafe."""

    def test_control_toward_boundary_exceeds_margin(
        self, ceiling_engine: CBFPythonEngine
    ):
        """Large positive u[0] pushes toward ceiling, may exceed margin."""
        state = np.array([42.9, 0.0, 0.0])
        control = np.array([100.0, 0.0, 0.0])
        results = ceiling_engine.evaluate_all(state, control)
        r = results[0]
        # h = 43-42.9 = 0.1, gamma = 0.1, grad_h[0] = -1
        # margin = 0 + (-1)*100 + 0.1 = -99.9
        assert r["passed"] is False
        assert r["margin"] < 0


# ===========================================================================
# 8. Gamma Function Variations
# ===========================================================================


class TestGammaFunction:
    """Test different class-K gamma functions."""

    def test_linear_gamma(self, engine_1d: CBFPythonEngine):
        """γ(h) = h (identity)."""
        engine_1d.add_barrier(
            barrier_id="linear",
            barrier_func=lambda x: 5.0 - x[0],
            gamma_func=lambda h: h,
        )
        state = np.array([3.0])
        results = engine_1d.evaluate_all(state, np.array([0.0]))
        assert results[0]["gamma_value"] == pytest.approx(2.0)

    def test_quadratic_gamma(self, engine_1d: CBFPythonEngine):
        """γ(h) = h^2."""
        engine_1d.add_barrier(
            barrier_id="quad",
            barrier_func=lambda x: 5.0 - x[0],
            gamma_func=lambda h: h**2,
        )
        state = np.array([3.0])
        results = engine_1d.evaluate_all(state, np.array([0.0]))
        assert results[0]["gamma_value"] == pytest.approx(4.0)

    def test_constant_gamma(self, engine_1d: CBFPythonEngine):
        """γ(h) = constant (always adds fixed margin)."""
        k = 0.5
        engine_1d.add_barrier(
            barrier_id="const",
            barrier_func=lambda x: 5.0 - x[0],
            gamma_func=lambda h: k,
        )
        state = np.array([3.0])
        results = engine_1d.evaluate_all(state, np.array([0.0]))
        assert results[0]["gamma_value"] == pytest.approx(k)

    def test_scaled_gamma(self, engine_1d: CBFPythonEngine):
        """γ(h) = 2*h."""
        engine_1d.add_barrier(
            barrier_id="scaled",
            barrier_func=lambda x: 5.0 - x[0],
            gamma_func=lambda h: 2.0 * h,
        )
        state = np.array([3.0])
        results = engine_1d.evaluate_all(state, np.array([0.0]))
        assert results[0]["gamma_value"] == pytest.approx(4.0)


# ===========================================================================
# 9. NaN / Inf State Rejection
# ===========================================================================


class TestNanStateRejection:
    """Tests that NaN and Inf in the state vector raise ValueError."""

    def test_nan_state(self, ceiling_engine: CBFPythonEngine):
        with pytest.raises(ValueError, match="NaN/Inf"):
            ceiling_engine.evaluate_all(
                np.array([np.nan, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            )

    def test_inf_state(self, ceiling_engine: CBFPythonEngine):
        with pytest.raises(ValueError, match="NaN/Inf"):
            ceiling_engine.evaluate_all(
                np.array([np.inf, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            )

    def test_neg_inf_state(self, ceiling_engine: CBFPythonEngine):
        with pytest.raises(ValueError, match="NaN/Inf"):
            ceiling_engine.evaluate_all(
                np.array([-np.inf, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            )

    def test_nan_in_second_position(self, ceiling_engine: CBFPythonEngine):
        """NaN anywhere in the state should be rejected."""
        with pytest.raises(ValueError, match="NaN/Inf"):
            ceiling_engine.evaluate_all(
                np.array([1.0, np.nan, 3.0]),
                np.array([0.0, 0.0, 0.0]),
            )


# ===========================================================================
# 10. Dimension Mismatch
# ===========================================================================


class TestDimensionMismatch:
    """Tests that wrong state dimensions are caught."""

    def test_state_too_short(self, ceiling_engine: CBFPythonEngine):
        with pytest.raises(ValueError, match="State dimension"):
            ceiling_engine.evaluate_all(
                np.array([1.0, 2.0]),
                np.array([0.0, 0.0]),
            )

    def test_state_too_long(self, ceiling_engine: CBFPythonEngine):
        with pytest.raises(ValueError, match="State dimension"):
            ceiling_engine.evaluate_all(
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0.0, 0.0, 0.0, 0.0]),
            )


# ===========================================================================
# 11. Lyapunov Configuration
# ===========================================================================


class TestLyapunovConfiguration:
    """Tests for set_lyapunov()."""

    def test_valid_configuration(self, engine_3d: CBFPythonEngine):
        """Valid Lyapunov config should be accepted."""
        x_ref = np.array([40.0, 0.3, 0.0])
        engine_3d.set_lyapunov(
            alpha=0.1,
            x_ref=x_ref,
            V_func=lambda x, xr: float(np.sum((x - xr) ** 2)),
            dVdt_func=lambda x, xr, u: float(2.0 * np.dot(x - xr, u)),
        )
        assert engine_3d.has_lyapunov is True

    def test_invalid_alpha_zero(self, engine_3d: CBFPythonEngine):
        """alpha=0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            engine_3d.set_lyapunov(
                alpha=0.0,
                x_ref=np.array([1.0, 2.0, 3.0]),
                V_func=lambda x, xr: 0.0,
                dVdt_func=lambda x, xr, u: 0.0,
            )

    def test_invalid_alpha_negative(self, engine_3d: CBFPythonEngine):
        """Negative alpha should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            engine_3d.set_lyapunov(
                alpha=-0.1,
                x_ref=np.array([1.0, 2.0, 3.0]),
                V_func=lambda x, xr: 0.0,
                dVdt_func=lambda x, xr, u: 0.0,
            )

    def test_x_ref_dimension_mismatch(self, engine_3d: CBFPythonEngine):
        """x_ref of wrong dimension should raise ValueError."""
        with pytest.raises(ValueError, match="x_ref dimension"):
            engine_3d.set_lyapunov(
                alpha=0.1,
                x_ref=np.array([1.0, 2.0]),
                V_func=lambda x, xr: 0.0,
                dVdt_func=lambda x, xr, u: 0.0,
            )


# ===========================================================================
# 12. Lyapunov Stability Pass
# ===========================================================================


class TestLyapunovStabilityPass:
    """Tests where V_dot <= -alpha * V (stability satisfied)."""

    @pytest.fixture
    def stable_engine(self):
        """Engine with Lyapunov config where the state is at the reference."""
        eng = CBFPythonEngine(state_dim=2)
        x_ref = np.array([10.0, 20.0])
        eng.set_lyapunov(
            alpha=0.5,
            x_ref=x_ref,
            V_func=lambda x, xr: float(np.sum((x - xr) ** 2)),
            dVdt_func=lambda x, xr, u: float(2.0 * np.dot(x - xr, u)),
        )
        return eng

    def test_at_reference_zero_dV(self, stable_engine: CBFPythonEngine):
        """At x=x_ref, V=0, any dV <= 0 → passed."""
        result = stable_engine.evaluate_lyapunov(
            np.array([10.0, 20.0]),
            np.array([0.0, 0.0]),
        )
        assert result["passed"] is True
        assert result["V"] == pytest.approx(0.0)
        assert result["threshold"] == pytest.approx(0.0)

    def test_converging_state(self, stable_engine: CBFPythonEngine):
        """State close to ref with control toward ref → stable."""
        # x = [11, 20], x_ref = [10, 20], V = 1, threshold = -0.5
        # u = [-1, 0], dV = 2*(1)*(-1) = -2, -2 <= -0.5 → passed
        result = stable_engine.evaluate_lyapunov(
            np.array([11.0, 20.0]),
            np.array([-1.0, 0.0]),
        )
        assert result["V"] == pytest.approx(1.0)
        assert result["dV"] == pytest.approx(-2.0)
        assert result["threshold"] == pytest.approx(-0.5)
        assert result["passed"] is True


# ===========================================================================
# 13. Lyapunov Stability Fail
# ===========================================================================


class TestLyapunovStabilityFail:
    """Tests where V_dot > -alpha * V (stability violated)."""

    @pytest.fixture
    def engine(self):
        eng = CBFPythonEngine(state_dim=2)
        x_ref = np.array([10.0, 20.0])
        eng.set_lyapunov(
            alpha=0.5,
            x_ref=x_ref,
            V_func=lambda x, xr: float(np.sum((x - xr) ** 2)),
            dVdt_func=lambda x, xr, u: float(2.0 * np.dot(x - xr, u)),
        )
        return eng

    def test_diverging_control(self, engine: CBFPythonEngine):
        """Control away from ref → dV > 0 > -alpha*V → fail."""
        # x = [10, 20], V = 0 → threshold = 0 → dV=0 passes trivially
        # Use non-zero V: x = [12, 20], V = 4, threshold = -2
        # u = [10, 0], dV = 2*2*10 = 40, 40 > -2 → fail
        result = engine.evaluate_lyapunov(
            np.array([12.0, 20.0]),
            np.array([10.0, 0.0]),
        )
        assert result["V"] == pytest.approx(4.0)
        assert result["dV"] == pytest.approx(40.0)
        assert result["passed"] is False

    def test_small_positive_dV(self, engine: CBFPythonEngine):
        """Even a small positive dV fails when V > 0."""
        # x = [11, 20], V = 1, threshold = -0.5
        # u = [0.1, 0], dV = 2*1*0.1 = 0.2, 0.2 > -0.5 → fail
        result = engine.evaluate_lyapunov(
            np.array([11.0, 20.0]),
            np.array([0.1, 0.0]),
        )
        assert result["dV"] == pytest.approx(0.2)
        assert result["passed"] is False


# ===========================================================================
# 14. Combined CBF + Lyapunov Evaluation
# ===========================================================================


class TestCombinedEvaluation:
    """Tests for the evaluate() method (CBF + Lyapunov together)."""

    @pytest.fixture
    def full_engine(self):
        eng = CBFPythonEngine(state_dim=2)
        eng.add_barrier(
            barrier_id="ceiling",
            barrier_func=lambda x: 50.0 - x[0],
            gamma_func=lambda h: h,
        )
        eng.set_lyapunov(
            alpha=0.5,
            x_ref=np.array([25.0, 10.0]),
            V_func=lambda x, xr: float(np.sum((x - xr) ** 2)),
            dVdt_func=lambda x, xr, u: float(2.0 * np.dot(x - xr, u)),
        )
        return eng

    def test_all_pass(self, full_engine: CBFPythonEngine):
        """Both CBF and Lyapunov pass → overall True."""
        state = np.array([30.0, 10.0])
        # u[0] = -2: CBF margin = 0 + (-1)*(-2) + 20 = 22 > 0 → pass
        # Lyapunov: V=25, dV=2*5*(-2)=-20, threshold=-12.5, -20<=-12.5 → pass
        control = np.array([-2.0, 0.0])
        passed, cbf_results, lyap_result = full_engine.evaluate(state, control)
        assert passed is True
        assert all(r["passed"] for r in cbf_results)
        assert lyap_result["passed"] is True

    def test_cbf_fail_short_circuits(self, full_engine: CBFPythonEngine):
        """If CBF fails, Lyapunov result should be empty dict."""
        state = np.array([60.0, 10.0])  # h = 50-60 = -10 → fail
        control = np.array([0.0, 0.0])
        passed, cbf_results, lyap_result = full_engine.evaluate(state, control)
        assert passed is False
        assert any(not r["passed"] for r in cbf_results)
        assert lyap_result == {}

    def test_lyapunov_fail(self, full_engine: CBFPythonEngine):
        """CBF passes but Lyapunov fails → overall False."""
        # CBF: h = 50 - 30 = 20 > 0, gamma = 20
        #   grad_h[0] = -1, L_g*h*u = (-1)*10 = -10, margin = -10 + 20 = 10 > 0 → pass
        # Lyapunov: x=[30,10], x_ref=[25,10], V=25, threshold=-12.5
        #   u=[10, 0], dV = 2*5*10 = 100, 100 > -12.5 → fail
        state = np.array([30.0, 10.0])
        control = np.array([10.0, 0.0])
        passed, cbf_results, lyap_result = full_engine.evaluate(state, control)
        assert passed is False
        assert all(r["passed"] for r in cbf_results)
        assert lyap_result["passed"] is False

    def test_no_lyapunov_configured(self):
        """With no Lyapunov, only CBF is evaluated."""
        eng = CBFPythonEngine(state_dim=2)
        eng.add_barrier(
            barrier_id="ceil",
            barrier_func=lambda x: 50.0 - x[0],
            gamma_func=lambda h: h,
        )
        state = np.array([30.0, 5.0])
        passed, cbf_results, lyap_result = eng.evaluate(state, np.array([0.0, 0.0]))
        assert passed is True
        assert lyap_result == {}


# ===========================================================================
# 15. Gradient Central Difference Accuracy
# ===========================================================================


class TestGradientCentralDifference:
    """Verify the _gradient_central_difference helper."""

    def test_linear_function(self):
        """f(x) = 2*x[0] + 3*x[1] → grad = [2, 3]."""

        def f(x):
            return 2.0 * x[0] + 3.0 * x[1]

        x = np.array([5.0, 7.0])
        grad = _gradient_central_difference(f, x)
        assert grad[0] == pytest.approx(2.0, abs=1e-6)
        assert grad[1] == pytest.approx(3.0, abs=1e-6)

    def test_quadratic_function(self):
        """f(x) = x[0]^2 + x[1]^2 → grad = [2*x[0], 2*x[1]]."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        x = np.array([3.0, -2.0])
        grad = _gradient_central_difference(f, x)
        assert grad[0] == pytest.approx(6.0, abs=1e-5)
        assert grad[1] == pytest.approx(-4.0, abs=1e-5)

    def test_sinusoidal_function(self):
        """f(x) = sin(x[0]) → grad = [cos(x[0])]."""

        def f(x):
            return np.sin(x[0])

        x = np.array([1.0])
        grad = _gradient_central_difference(f, x)
        assert grad[0] == pytest.approx(np.cos(1.0), abs=1e-6)

    def test_zero_gradient(self):
        """f(x) = constant → grad = [0, 0]."""

        def f(x):
            return 42.0

        x = np.array([1.0, 2.0])
        grad = _gradient_central_difference(f, x)
        assert grad[0] == pytest.approx(0.0, abs=1e-8)
        assert grad[1] == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# 16. Thread Safety — Concurrent Evaluations
# ===========================================================================


class TestThreadSafety:
    """Ensure the engine is safe for concurrent access."""

    def test_concurrent_barrier_add_and_eval(self, engine_3d: CBFPythonEngine):
        """Multiple threads should be able to evaluate without data races."""
        engine_3d.add_barrier(
            barrier_id="ceil",
            barrier_func=lambda x: 100.0 - x[0],
            gamma_func=lambda h: h,
        )

        errors = []
        results = []

        def eval_worker(idx):
            try:
                state = np.array([float(idx), 0.0, 0.0])
                control = np.array([0.0, 0.0, 0.0])
                r = engine_3d.evaluate_all(state, control)
                results.append(r)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(eval_worker, i) for i in range(50)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 50

    def test_concurrent_lyapunov_eval(self):
        """Concurrent Lyapunov evaluations should not corrupt state."""
        eng = CBFPythonEngine(state_dim=2)
        eng.set_lyapunov(
            alpha=0.1,
            x_ref=np.array([0.0, 0.0]),
            V_func=lambda x, xr: float(np.sum((x - xr) ** 2)),
            dVdt_func=lambda x, xr, u: float(2.0 * np.dot(x - xr, u)),
        )

        errors = []
        results = []

        def lyap_worker(idx):
            try:
                state = np.array([float(idx), float(idx)])
                control = np.array([-1.0, -1.0])
                r = eng.evaluate_lyapunov(state, control)
                results.append(r)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(lyap_worker, i) for i in range(30)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 30
        # All results should be valid dicts
        for r in results:
            assert "passed" in r
            assert "V" in r


# ===========================================================================
# 17. create_engine Factory Function
# ===========================================================================


class TestCreateEngineFactory:
    """Tests for the create_engine() convenience factory."""

    def test_factory_with_ceiling_barrier(self):
        """Factory should create an engine with a ceiling barrier."""
        eng = create_engine(
            state_dim=3,
            barriers=[
                {
                    "barrier_id": "tx_power",
                    "state_index": 0,
                    "ceiling": 43.0,
                },
            ],
        )
        assert eng.state_dim == 3
        assert eng.barrier_count == 1

        # Test at safe state
        results = eng.evaluate_all(
            np.array([40.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        )
        assert results[0]["passed"] is True

        # Test at unsafe state
        results = eng.evaluate_all(
            np.array([50.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        )
        assert results[0]["passed"] is False

    def test_factory_with_floor_barrier(self):
        """Factory should create an engine with a floor barrier."""
        eng = create_engine(
            state_dim=2,
            barriers=[
                {
                    "barrier_id": "power_floor",
                    "state_index": 0,
                    "floor": 5.0,
                },
            ],
        )
        assert eng.barrier_count == 1

        results = eng.evaluate_all(np.array([10.0, 0.0]), np.array([0.0, 0.0]))
        assert results[0]["passed"] is True

        results = eng.evaluate_all(np.array([2.0, 0.0]), np.array([0.0, 0.0]))
        assert results[0]["passed"] is False

    def test_factory_with_band_barrier(self):
        """Factory should create an engine with a band barrier."""
        eng = create_engine(
            state_dim=1,
            barriers=[
                {
                    "barrier_id": "band",
                    "state_index": 0,
                    "floor": 5.0,
                    "ceiling": 43.0,
                },
            ],
        )
        assert eng.barrier_count == 1

        # In band → pass
        results = eng.evaluate_all(np.array([20.0]), np.array([0.0]))
        assert results[0]["passed"] is True

        # Below floor → fail
        results = eng.evaluate_all(np.array([3.0]), np.array([0.0]))
        assert results[0]["passed"] is False

        # Above ceiling → fail
        results = eng.evaluate_all(np.array([50.0]), np.array([0.0]))
        assert results[0]["passed"] is False

    def test_factory_with_lyapunov(self):
        """Factory should configure Lyapunov from dict."""
        eng = create_engine(
            state_dim=3,
            lyapunov={
                "alpha": 0.2,
                "x_ref": [10.0, 20.0, 30.0],
                "V_func_type": "quadratic",
            },
        )
        assert eng.has_lyapunov is True
        result = eng.evaluate_lyapunov(
            np.array([10.0, 20.0, 30.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert result["passed"] is True

    def test_factory_with_barriers_and_lyapunov(self):
        """Full factory with barriers + Lyapunov."""
        eng = create_engine(
            state_dim=3,
            barriers=[
                {"barrier_id": "ceil", "state_index": 0, "ceiling": 43.0},
            ],
            lyapunov={
                "alpha": 0.1,
                "x_ref": [20.0, 0.0, 0.0],
                "V_func_type": "quadratic",
            },
        )
        assert eng.barrier_count == 1
        assert eng.has_lyapunov is True

    def test_factory_no_barriers_no_lyapunov(self):
        """Factory with no args should create a bare engine."""
        eng = create_engine(state_dim=5)
        assert eng.state_dim == 5
        assert eng.barrier_count == 0
        assert eng.has_lyapunov is False
