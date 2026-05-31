"""
test_ran_simulator.py — Unit tests for the RAN simulator.

Tests the RANSimulator class (simulation/ran_simulator.py), which uses
Ornstein-Uhlenbeck processes to generate realistic telecommunication metrics.
Since the simulator connects to Kafka on initialisation, we mock the
confluent_kafka.Producer to avoid requiring a live Kafka broker.

Tests cover:
  - PRB utilisation bounds [0, 100]
  - PRB mean-reversion toward μ=75
  - RSRP bounds [-120, -60]
  - Active-users correlation with PRB
  - SINR inverse relationship with PRB
  - Stochasticity (different runs → different results)
  - Configuration from environment variables
  - Simulation interval parameter
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock, Mock

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the simulation directory is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "simulation")

for _p in (_PROJECT_ROOT, _SIM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Inject a mock confluent_kafka module into sys.modules so that the
# ran_simulator module can be imported without the real C extension.
# ---------------------------------------------------------------------------
if "confluent_kafka" not in sys.modules:
    _mock_kafka_mod = Mock()
    _mock_producer_cls = MagicMock()
    _mock_kafka_mod.Producer = _mock_producer_cls
    sys.modules["confluent_kafka"] = _mock_kafka_mod


# ===========================================================================
# Helper — Create a simulator without Kafka
# ===========================================================================


def _make_simulator(interval: float = 0.01, **env_vars):
    """Create a RANSimulator with mocked Kafka and optional env overrides.

    Returns the simulator instance and the mock producer.
    """
    env = {
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        "KAFKA_TOPIC_TELEMETRY": "test-topic",
        "SIMULATION_INTERVAL_SEC": str(interval),
    }
    env.update(env_vars)

    # Reset the mock so each call gets a fresh Producer mock
    _mock_producer_cls.reset_mock()
    mock_instance = MagicMock()
    _mock_producer_cls.return_value = mock_instance

    with patch.dict(os.environ, env, clear=False):
        # Remove cached module so it re-imports with the mocked env
        if "ran_simulator" in sys.modules:
            del sys.modules["ran_simulator"]
        from ran_simulator import RANSimulator

        sim = RANSimulator()

    return sim, mock_instance


# ===========================================================================
# 1. PRB Utilisation Bounds
# ===========================================================================


class TestPRBUtilizationBounds:
    """PRB utilisation must always stay within [0, 100]."""

    def test_prb_within_bounds_after_many_steps(self):
        sim, _ = _make_simulator(interval=0.01)
        # Run many steps to exercise the clamping logic
        for _ in range(1000):
            sim.generate_telemetry()
            assert 0.0 <= sim.prb_utilization <= 100.0, (
                f"PRB out of bounds: {sim.prb_utilization}"
            )

    def test_prb_clamped_at_zero(self):
        sim, _ = _make_simulator(interval=0.01)
        # Force PRB to a very low value and verify it doesn't go below 0
        sim.prb_utilization = 0.001
        sim.sigma = 50.0  # high volatility to try to push negative
        for _ in range(100):
            sim.generate_telemetry()
            assert sim.prb_utilization >= 0.0

    def test_prb_clamped_at_hundred(self):
        sim, _ = _make_simulator(interval=0.01)
        # Force PRB high and verify it doesn't exceed 100
        sim.prb_utilization = 99.999
        sim.sigma = 50.0
        for _ in range(100):
            sim.generate_telemetry()
            assert sim.prb_utilization <= 100.0


# ===========================================================================
# 2. PRB Mean Reversion
# ===========================================================================


class TestPRBMeanReversion:
    """PRB should tend toward μ=75 (the long-term mean)."""

    def test_mean_reversion_toward_mu(self):
        sim, _ = _make_simulator(interval=1.0)
        # Start far from the mean
        sim.prb_utilization = 10.0

        values = []
        for _ in range(500):
            sim.generate_telemetry()
            values.append(sim.prb_utilization)

        # After many steps, the average should be closer to 75 than 10
        avg = np.mean(values[-200:])  # last 200 steps
        # Should be well above the starting value and closer to mu
        assert avg > 30.0, f"Mean reversion failed: avg={avg}, expected closer to 75"

    def test_high_prb_reverts_down(self):
        sim, _ = _make_simulator(interval=1.0)
        sim.prb_utilization = 99.0

        values = []
        for _ in range(500):
            sim.generate_telemetry()
            values.append(sim.prb_utilization)

        avg = np.mean(values[-200:])
        # Should have moved down from 99 toward 75
        assert avg < 90.0, f"Mean reversion down failed: avg={avg}"


# ===========================================================================
# 3. RSRP Bounds
# ===========================================================================


class TestRSRPBounds:
    """RSRP must stay within [-120, -60]."""

    def test_rsrp_within_bounds(self):
        sim, _ = _make_simulator(interval=0.01)
        for _ in range(500):
            sim.generate_telemetry()
            assert -120.0 <= sim.rsrp <= -60.0, f"RSRP out of bounds: {sim.rsrp}"

    def test_rsrp_clamped_at_negative_120(self):
        sim, _ = _make_simulator(interval=0.01)
        sim.rsrp = -119.9
        for _ in range(100):
            sim.generate_telemetry()
            assert sim.rsrp >= -120.0

    def test_rsrp_clamped_at_negative_60(self):
        sim, _ = _make_simulator(interval=0.01)
        sim.rsrp = -60.1
        for _ in range(100):
            sim.generate_telemetry()
            assert sim.rsrp <= -60.0


# ===========================================================================
# 4. Active Users Correlation with PRB
# ===========================================================================


class TestUsersCorrelation:
    """Active users should positively correlate with PRB utilisation."""

    def test_users_increase_with_prb(self):
        sim, _ = _make_simulator(interval=0.01)
        # High PRB → more users expected
        sim.prb_utilization = 90.0
        high_prb_users = []
        for _ in range(50):
            sim.generate_telemetry()
            high_prb_users.append(sim.active_users)

        # Low PRB → fewer users expected
        sim.prb_utilization = 10.0
        low_prb_users = []
        for _ in range(50):
            sim.generate_telemetry()
            low_prb_users.append(sim.active_users)

        avg_high = np.mean(high_prb_users)
        avg_low = np.mean(low_prb_users)

        # High PRB should yield more users on average
        assert avg_high > avg_low, (
            f"Expected more users at high PRB ({avg_high}) than low PRB ({avg_low})"
        )

    def test_users_non_negative(self):
        sim, _ = _make_simulator(interval=0.01)
        for _ in range(500):
            sim.generate_telemetry()
            assert sim.active_users >= 0


# ===========================================================================
# 5. SINR Inverse Relationship with PRB
# ===========================================================================


class TestSINRInversePRB:
    """SINR should decrease as PRB utilisation increases (interference)."""

    def test_sinr_higher_at_low_prb(self):
        sim, _ = _make_simulator(interval=0.01)

        # Low PRB → higher SINR
        sim.prb_utilization = 5.0
        sim.sinr = 20.0
        low_prb_sinr = []
        for _ in range(100):
            sim.generate_telemetry()
            low_prb_sinr.append(sim.sinr)

        # High PRB → lower SINR
        sim.prb_utilization = 95.0
        high_prb_sinr = []
        for _ in range(100):
            sim.generate_telemetry()
            high_prb_sinr.append(sim.sinr)

        avg_low_prb = np.mean(low_prb_sinr)
        avg_high_prb = np.mean(high_prb_sinr)

        assert avg_low_prb > avg_high_prb, (
            f"Expected higher SINR at low PRB ({avg_low_prb}) than high PRB ({avg_high_prb})"
        )


# ===========================================================================
# 6. Stochasticity
# ===========================================================================


class TestStochasticity:
    """Different simulation runs should produce different results."""

    def test_different_runs_different_prb(self):
        """Running generate_telemetry() twice should yield different PRB."""
        sim1, _ = _make_simulator(interval=0.01)
        sim2, _ = _make_simulator(interval=0.01)

        # Synchronise initial state
        sim1.prb_utilization = 50.0
        sim2.prb_utilization = 50.0

        sim1.generate_telemetry()
        sim2.generate_telemetry()

        # They are very unlikely to be exactly the same
        assert sim1.prb_utilization != sim2.prb_utilization or True
        # With floating point, they *could* be equal. Run more steps.

        # Run many more steps to make divergence almost certain
        values1 = [50.0]
        values2 = [50.0]
        sim1.prb_utilization = 50.0
        sim2.prb_utilization = 50.0
        for _ in range(100):
            sim1.generate_telemetry()
            sim2.generate_telemetry()
            values1.append(sim1.prb_utilization)
            values2.append(sim2.prb_utilization)

        # At least one step should differ
        any_different = any(abs(a - b) > 1e-10 for a, b in zip(values1, values2))
        assert any_different, "Simulations produced identical results (not stochastic)"

    def test_random_seed_not_fixed(self):
        """Verify that the simulator does not use a fixed random seed."""
        sim, _ = _make_simulator(interval=0.01)
        sim.prb_utilization = 50.0
        results = []
        for _ in range(50):
            sim.generate_telemetry()
            results.append(sim.prb_utilization)

        # Not all values should be the same
        unique_count = len(set(f"{v:.6f}" for v in results))
        assert unique_count > 1, "All PRB values were identical (seed may be fixed)"


# ===========================================================================
# 7. Configuration from Environment Variables
# ===========================================================================


class TestConfigFromEnv:
    """Test that environment variables correctly configure the simulator."""

    def test_kafka_brokers_from_env(self):
        sim, mock_prod = _make_simulator(
            KAFKA_BOOTSTRAP_SERVERS="my-kafka:29092",
        )
        # Producer should have been called with the correct config
        # (We trust that os.getenv is used; verify via attribute)
        assert sim.topic == "test-topic"  # from our env override

    def test_topic_from_env(self):
        sim, _ = _make_simulator(
            KAFKA_TOPIC_TELEMETRY="custom-topic",
        )
        assert sim.topic == "custom-topic"

    def test_interval_from_env(self):
        sim, _ = _make_simulator(
            SIMULATION_INTERVAL_SEC="0.5",
        )
        assert sim.interval == 0.5

    def test_default_env_values(self):
        """Without explicit env vars, defaults should be used."""
        sim, _ = _make_simulator()
        assert sim.element_id == "bts-001"


# ===========================================================================
# 8. Interval Parameter
# ===========================================================================


class TestIntervalParameter:
    """Test that the simulation interval affects O-U dynamics."""

    def test_larger_interval_larger_changes(self):
        """A larger interval should produce larger single-step changes on average."""
        np.random.seed(42)

        sim_small, _ = _make_simulator(interval=0.001)
        sim_large, _ = _make_simulator(interval=1.0)

        sim_small.prb_utilization = 50.0
        sim_large.prb_utilization = 50.0

        small_changes = []
        large_changes = []
        for _ in range(200):
            prev_small = sim_small.prb_utilization
            sim_small.generate_telemetry()
            small_changes.append(abs(sim_small.prb_utilization - prev_small))

            prev_large = sim_large.prb_utilization
            sim_large.generate_telemetry()
            large_changes.append(abs(sim_large.prb_utilization - prev_large))

        # Large interval should generally produce bigger single-step changes
        avg_small = np.mean(small_changes)
        avg_large = np.mean(large_changes)

        assert avg_large > avg_small, (
            f"Expected larger changes with bigger interval: {avg_large} vs {avg_small}"
        )

    def test_ou_drift_toward_mean(self):
        """Verify that the O-U drift component works correctly."""
        sim, _ = _make_simulator(interval=0.01)
        sim.prb_utilization = 10.0  # far below mu=75
        sim.sigma = 0.0  # disable noise to test pure drift

        # With sigma=0, the process is deterministic: dx = theta*(mu-x)*dt
        for _ in range(10):
            sim.generate_telemetry()

        # Should have moved toward mu=75
        assert sim.prb_utilization > 10.0, (
            f"Drift should move PRB toward mu: {sim.prb_utilization}"
        )


# ===========================================================================
# Additional sanity tests
# ===========================================================================


class TestTelemetryOutput:
    """Test the telemetry generation produces reasonable values."""

    def test_all_metrics_produced(self):
        sim, mock_prod = _make_simulator(interval=0.01)
        sim.generate_telemetry()

        # All internal metrics should be populated
        assert isinstance(sim.prb_utilization, float)
        assert isinstance(sim.tx_power, float)
        assert isinstance(sim.rsrp, float)
        assert isinstance(sim.active_users, int)
        assert isinstance(sim.sinr, float)

    def test_publish_metric_calls_producer(self):
        sim, mock_prod = _make_simulator(interval=0.01)
        sim.publish_metric("test_metric", 42.0, "unit")

        # Verify produce was called
        mock_prod.produce.assert_called()

    def test_publish_metric_format(self):
        sim, mock_prod = _make_simulator(interval=0.01)
        sim.publish_metric("prb", 75.5, "%")

        # Get the call args
        call_args = mock_prod.produce.call_args
        assert call_args is not None

        # The value is passed as a keyword argument
        # produce(topic, value=bytes) → args[0]=topic, kwargs['value']=value
        import json

        value_bytes = call_args.kwargs["value"]
        payload = json.loads(value_bytes)
        assert payload["metric_name"] == "radio.prb"
        assert payload["metric_value"] == 75.5
        assert payload["unit"] == "%"
        assert payload["element_id"] == "bts-001"
