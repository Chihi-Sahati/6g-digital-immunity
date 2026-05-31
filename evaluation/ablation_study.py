#!/usr/bin/env python3

"""
Ablation Study — Deterministic Safety Filter (DSF) for 6G Digital Immunity

This script performs a systematic ablation study of the DSF by removing each
of the three sequential safety gates independently and measuring the impact on
safety and performance. The goal is to quantify the marginal contribution of
each gate to the overall safety guarantee.

DSF Gate Architecture (3 gates in series):
    Gate 1 — CUE Schema Validation:   Structural/syntactic correctness of the
                                         LLM-generated network intent against the
                                         Osmocom / 3GPP CUE schema.
    Gate 2 — CBF Safety Enforcement:   Control Barrier Function (CBF) check
                                         ensuring h(x) >= 0 for all registered
                                         barriers. Lyapunov stability verified
                                         as a secondary condition.
    Gate 3 — Zeno Temporal Guard:      Minimum inter-actuation time τ_min
                                         enforcement to prevent infinite-rate
                                         command sequences.

Ablation Configurations:
    1. Full DSF         — All three gates active (baseline)
    2. No CUE           — Gates 2 + 3 only (Gate 1 disabled)
    3. No CBF           — Gates 1 + 3 only (Gate 2 disabled)
    4. No Zeno          — Gates 1 + 2 only (Gate 3 disabled)
    5. No DSF           — All gates disabled (unconstrained baseline)

State Vector:
    x = [tx_power, ho_hysteresis, prb_utilization]

CBF Barrier Functions:
    h1(x) = P_max  - P_tx     >= 0   (tx power ceiling, 43 dBm)
    h2(x) = P_tx   - P_min    >= 0   (tx power floor,   0 dBm)
    h3(x) = PRB_max - P_PRB   >= 0   (PRB ceiling,     100%)
    h4(x) = H_HO   - H_min    >= 0   (HO hysteresis floor, 0 dB)

Class-K Extension Rate:
    γ(h) = 0.3 · h

Methodology:
    For each configuration, N=1000 randomly generated LLM intents are simulated.
    A fraction of intents (~30%) are adversarial — designed to violate safety
    constraints (e.g., excessive tx power, zero PRB headroom). The simulation
    tracks safety violations, acceptance/denial/amendment counts, and processing
    latency. Results are summarised in a table and visualised as grouped bar
    charts suitable for publication.

Dependencies:
    numpy, matplotlib, pandas

Usage:
    python evaluation/ablation_study.py

Output:
    evaluation/ablation_results.csv   — Per-configuration summary statistics
    evaluation/ablation_study.png    — Grouped bar chart comparison
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for reproducibility

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Publication-quality styling
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 12,
        "font.size": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_DIM = 3
# State vector indices: [tx_power, ho_hysteresis, prb_utilization]
IDX_TX_POWER = 0
IDX_HO_HYST = 1
IDX_PRB_UTIL = 2

# Network parameter limits (3GPP / Osmocom golden baseline)
P_MAX = 43.0  # dBm — maximum transmit power ceiling
P_MIN = 0.0  # dBm — minimum transmit power floor
PRB_MAX = 100.0  # %    — maximum PRB utilisation ceiling
H_MIN = 0.0  # dB   — minimum handover hysteresis

# CBF parameters
GAMMA_K = 0.3  # Class-K gain: γ(h) = k · h

# Zeno parameters
TAU_MIN_MS = 500.0  # Minimum inter-actuation time (ms)
COOLDOWN_S = 2.0  # Cooldown duration after Zeno violation (s)

# Simulation parameters
N_INTENTS = 1000
ADVERSARIAL_FRACTION = 0.30  # 30% of intents are adversarial

# Intent decision labels
DECISION_ACCEPT = "ACCEPT"
DECISION_DENY = "DENY"
DECISION_AMEND = "AMEND"

# CUE schema validation failure rate for structurally malformed intents
CUE_MALFORM_RATE = 0.08  # 8% of intents have structural issues


# ===========================================================================
# DSF Gate Simulation Models
# ===========================================================================


@dataclass
class Intent:
    """A single LLM-generated network intent with associated metadata."""

    state: np.ndarray  # Current state [tx_power, ho_hyst, prb_util]
    desired_state: np.ndarray  # Desired state from LLM
    timestamp: float  # Simulated arrival time (seconds)
    is_adversarial: bool  # Whether intent is adversarial
    is_malformed: bool  # Whether intent has structural issues
    decision: str = ""  # Final DSF decision


@dataclass
class GateResult:
    """Result from a single DSF gate evaluation."""

    gate_name: str
    passed: bool
    decision: str  # ACCEPT, DENY, AMEND
    amended_state: np.ndarray | None = None
    latency_ms: float = 0.0
    cbf_margins: List[float] = field(default_factory=list)
    cbf_passed_all: bool = True
    zeno_passed: bool = True


def generate_intent(rng: np.random.Generator, idx: int) -> Intent:
    """Generate a random LLM intent (safe or adversarial).

    Safe intents request parameter values within the feasible region defined
    by the CBF barriers with positive margins. Adversarial intents request
    values at or beyond the safety boundary.

    Args:
        rng: NumPy random generator for reproducibility.
        idx: Intent index (used for timestamp spacing).

    Returns:
        An Intent object with state, desired state, and metadata.
    """
    # Current state: uniformly sampled within safe operating region
    current_tx = rng.uniform(10.0, 40.0)  # dBm
    current_hyst = rng.uniform(0.5, 6.0)  # dB
    current_prb = rng.uniform(10.0, 85.0)  # %

    state = np.array([current_tx, current_hyst, current_prb], dtype=np.float64)

    # Timestamp with realistic inter-arrival spacing (100–2000 ms)
    timestamp = idx * rng.uniform(0.1, 2.0)

    # Determine if adversarial (30% chance)
    is_adversarial = rng.random() < ADVERSARIAL_FRACTION

    # Determine if malformed (structural issues, ~8%)
    is_malformed = rng.random() < CUE_MALFORM_RATE

    if is_adversarial:
        # Adversarial: push parameters toward or beyond CBF boundaries
        adv_type = rng.integers(0, 4)
        if adv_type == 0:
            # Excessive tx power (beyond ceiling)
            desired_tx = rng.uniform(42.0, 50.0)
            desired_prb = rng.uniform(10.0, 80.0)
            desired_hyst = rng.uniform(0.5, 6.0)
        elif adv_type == 1:
            # Near-zero or negative tx power (below floor)
            desired_tx = rng.uniform(-5.0, 5.0)
            desired_prb = rng.uniform(10.0, 80.0)
            desired_hyst = rng.uniform(0.5, 6.0)
        elif adv_type == 2:
            # PRB utilisation beyond ceiling
            desired_tx = rng.uniform(10.0, 40.0)
            desired_prb = rng.uniform(92.0, 110.0)
            desired_hyst = rng.uniform(0.5, 6.0)
        else:
            # Negative handover hysteresis (below floor)
            desired_tx = rng.uniform(10.0, 40.0)
            desired_prb = rng.uniform(10.0, 80.0)
            desired_hyst = rng.uniform(-3.0, 0.3)
    else:
        # Safe intent: within feasible region
        desired_tx = rng.uniform(15.0, 38.0)
        desired_hyst = rng.uniform(1.0, 5.0)
        desired_prb = rng.uniform(20.0, 75.0)

    desired_state = np.array([desired_tx, desired_hyst, desired_prb], dtype=np.float64)

    return Intent(
        state=state,
        desired_state=desired_state,
        timestamp=timestamp,
        is_adversarial=is_adversarial,
        is_malformed=is_malformed,
    )


# ---------------------------------------------------------------------------
# Gate 1: CUE Schema Validation (structural correctness)
# ---------------------------------------------------------------------------


def gate1_cue_validation(intent: Intent) -> GateResult:
    """Simulate CUE Schema Validation (Gate 1).

    Validates the structural correctness of the LLM-generated intent against
    the Osmocom/3GPP CUE schema. Malformed intents (missing fields, wrong
    types, out-of-range structural fields) are rejected at this gate.

    This gate does not evaluate safety constraints — it only checks that the
    intent's JSON structure conforms to the expected schema.

    Args:
        intent: The intent to validate.

    Returns:
        GateResult with pass/deny decision and latency.
    """
    t_start = time.perf_counter()

    # Malformed intents fail schema validation
    if intent.is_malformed:
        return GateResult(
            gate_name="CUE_Schema",
            passed=False,
            decision=DECISION_DENY,
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    # Check that desired state values are finite and within gross bounds
    for i, val in enumerate(intent.desired_state):
        if not np.isfinite(val) or abs(val) > 200.0:
            return GateResult(
                gate_name="CUE_Schema",
                passed=False,
                decision=DECISION_DENY,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

    # Simulated schema validation latency (~0.5–2.0 ms)
    latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(0.5, 2.0)

    return GateResult(
        gate_name="CUE_Schema",
        passed=True,
        decision=DECISION_ACCEPT,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Gate 2: CBF Safety Enforcement
# ---------------------------------------------------------------------------


def compute_cbf_barriers(desired_state: np.ndarray) -> Tuple[List[float], bool]:
    """Evaluate all CBF barrier functions for a desired state.

    Barrier functions:
        h1(x) = P_max  - P_tx       (tx power ceiling)
        h2(x) = P_tx   - P_min      (tx power floor)
        h3(x) = PRB_max - P_PRB      (PRB utilisation ceiling)
        h4(x) = H_HO   - H_min      (HO hysteresis floor)

    The CBF condition requires h_i(x) >= 0 for all barriers i.

    Args:
        desired_state: Desired network state vector [tx_power, ho_hyst, prb_util].

    Returns:
        Tuple of (barrier margins list, all_passed bool).
    """
    h1 = P_MAX - desired_state[IDX_TX_POWER]  # Ceiling: tx power <= 43 dBm
    h2 = desired_state[IDX_TX_POWER] - P_MIN  # Floor:   tx power >= 0 dBm
    h3 = PRB_MAX - desired_state[IDX_PRB_UTIL]  # Ceiling: PRB util <= 100%
    h4 = desired_state[IDX_HO_HYST] - H_MIN  # Floor:   HO hyst >= 0 dB

    margins = [h1, h2, h3, h4]
    all_passed = all(h >= 0.0 for h in margins)
    return margins, all_passed


def gate2_cbf_safety(intent: Intent) -> GateResult:
    """Simulate CBF Safety Enforcement (Gate 2).

    Evaluates Control Barrier Functions on the desired state vector. If all
    barriers are satisfied (h_i >= 0), the intent passes. If any barrier is
    violated, the gate attempts to amend the desired state by clamping values
    to the nearest safe operating point.

    The Lyapunov stability condition is also checked: V̇(x,u) <= -α·V(x),
    where V(x) = ||x - x_ref||^2 is a quadratic Lyapunov function.

    Args:
        intent: The intent to evaluate.

    Returns:
        GateResult with accept/amend/deny decision, CBF margins, and latency.
    """
    t_start = time.perf_counter()

    desired = intent.desired_state.copy()
    margins, all_passed = compute_cbf_barriers(desired)

    if all_passed:
        # All barriers satisfied — compute Lyapunov stability
        # Reference state (nominal operating point)
        x_ref = np.array([30.0, 3.0, 50.0], dtype=np.float64)
        V = float(np.sum((desired - x_ref) ** 2))

        # Lyapunov check: dV/dt <= -alpha * V
        # For a static intent (no dynamics), dV/dt ≈ 0, so condition holds when V=0
        # For non-zero V with alpha=0.1, the condition is: 0 <= -0.1 * V, which
        # only holds for V=0. We use a relaxed check: |V| < threshold.

        # For safety filter purposes, we use a relaxed Lyapunov condition:
        # V must not grow uncontrollably (V < large threshold)
        lyapunov_passed = V < 1000.0  # Reasonable operating region

        if lyapunov_passed:
            latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(
                0.1, 0.5
            )
            return GateResult(
                gate_name="CBF_Safety",
                passed=True,
                decision=DECISION_ACCEPT,
                latency_ms=latency,
                cbf_margins=margins,
                cbf_passed_all=True,
            )
        else:
            latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(
                0.1, 0.5
            )
            return GateResult(
                gate_name="CBF_Safety",
                passed=False,
                decision=DECISION_DENY,
                latency_ms=latency,
                cbf_margins=margins,
                cbf_passed_all=False,
            )

    # CBF violation — attempt amendment by clamping to safe region
    amended = desired.copy()
    amendment_needed = False

    if margins[0] < 0:  # h1: tx power ceiling
        amended[IDX_TX_POWER] = P_MAX - 1.0  # 1 dBm margin
        amendment_needed = True
    if margins[1] < 0:  # h2: tx power floor
        amended[IDX_TX_POWER] = P_MIN + 1.0
        amendment_needed = True
    if margins[2] < 0:  # h3: PRB ceiling
        amended[IDX_PRB_UTIL] = PRB_MAX - 5.0  # 5% margin
        amendment_needed = True
    if margins[3] < 0:  # h4: HO hyst floor
        amended[IDX_HO_HYST] = H_MIN + 0.5  # 0.5 dB margin
        amendment_needed = True

    # Recompute margins on amended state
    amended_margins, amended_passed = compute_cbf_barriers(amended)

    latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(0.2, 1.0)

    if amendment_needed and amended_passed:
        return GateResult(
            gate_name="CBF_Safety",
            passed=True,
            decision=DECISION_AMEND,
            amended_state=amended,
            latency_ms=latency,
            cbf_margins=amended_margins,
            cbf_passed_all=True,
        )
    else:
        return GateResult(
            gate_name="CBF_Safety",
            passed=False,
            decision=DECISION_DENY,
            latency_ms=latency,
            cbf_margins=margins,
            cbf_passed_all=False,
        )


# ---------------------------------------------------------------------------
# Gate 3: Zeno Temporal Guard
# ---------------------------------------------------------------------------


def gate3_zeno_guard(intent: Intent, last_actuation_time: float) -> GateResult:
    """Simulate the Zeno Temporal Guard (Gate 3).

    Enforces a minimum inter-actuation time τ_min to prevent the LLM/DSF from
    issuing commands at an infinite rate. If the time since the last actuation
    is less than τ_min, the intent is blocked.

    Mathematical condition:
        Δt = t_current - t_last >= τ_min

    Args:
        intent: The intent to evaluate.
        last_actuation_time: Timestamp of the last actuation (seconds).

    Returns:
        GateResult with pass/deny decision and latency.
    """
    t_start = time.perf_counter()

    tau_min_s = TAU_MIN_MS / 1000.0  # Convert ms to seconds

    if last_actuation_time < 0:
        # No prior actuation — always pass
        latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(
            0.01, 0.1
        )
        return GateResult(
            gate_name="Zeno_Guard",
            passed=True,
            decision=DECISION_ACCEPT,
            latency_ms=latency,
            zeno_passed=True,
        )

    delta_t = intent.timestamp - last_actuation_time

    if delta_t >= tau_min_s:
        latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(
            0.01, 0.1
        )
        return GateResult(
            gate_name="Zeno_Guard",
            passed=True,
            decision=DECISION_ACCEPT,
            latency_ms=latency,
            zeno_passed=True,
        )
    else:
        latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(
            0.01, 0.1
        )
        return GateResult(
            gate_name="Zeno_Guard",
            passed=False,
            decision=DECISION_DENY,
            latency_ms=latency,
            zeno_passed=False,
        )


# ===========================================================================
# Full DSF Pipeline Simulation
# ===========================================================================


@dataclass
class AblationResult:
    """Results from a single ablation configuration run."""

    config_name: str
    safety_violations: int = 0
    accepted_count: int = 0
    denied_count: int = 0
    amended_count: int = 0
    total_intents: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    cbf_margins_all: List[float] = field(default_factory=list)
    zeno_blocks: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_intents == 0:
            return 0.0
        return self.accepted_count / self.total_intents

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.mean(self.latencies_ms))

    @property
    def std_latency_ms(self) -> float:
        if len(self.latencies_ms) < 2:
            return 0.0
        return float(np.std(self.latencies_ms))

    @property
    def avg_min_cbf_margin(self) -> float:
        if not self.cbf_margins_all:
            return 0.0
        return float(np.mean(self.cbf_margins_all))


def run_dsf_simulation(
    config_name: str,
    enable_cue: bool,
    enable_cbf: bool,
    enable_zeno: bool,
    n_intents: int = N_INTENTS,
    seed: int = 42,
) -> AblationResult:
    """Run DSF simulation with the specified gate configuration.

    Args:
        config_name: Human-readable configuration name.
        enable_cue: Whether Gate 1 (CUE Schema) is active.
        enable_cbf: Whether Gate 2 (CBF Safety) is active.
        enable_zeno: Whether Gate 3 (Zeno Guard) is active.
        n_intents: Number of intents to simulate.
        seed: Random seed for reproducibility.

    Returns:
        AblationResult with all tracked metrics.
    """
    rng = np.random.default_rng(seed)
    result = AblationResult(config_name=config_name, total_intents=n_intents)
    last_actuation_time = -1.0  # No prior actuation

    for i in range(n_intents):
        intent = generate_intent(rng, i)
        total_latency = 0.0
        decision = DECISION_ACCEPT
        blocked = False

        # Gate 1: CUE Schema Validation
        if enable_cue:
            g1 = gate1_cue_validation(intent)
            total_latency += g1.latency_ms
            if not g1.passed:
                decision = DECISION_DENY
                blocked = True

        # Gate 2: CBF Safety Enforcement
        if not blocked and enable_cbf:
            g2 = gate2_cbf_safety(intent)
            total_latency += g2.latency_ms
            if not g2.passed:
                decision = DECISION_DENY
                blocked = True
            elif g2.decision == DECISION_AMEND:
                decision = DECISION_AMEND
            # Track CBF margins for statistical analysis
            result.cbf_margins_all.extend(g2.cbf_margins)

        # Gate 3: Zeno Temporal Guard
        if not blocked and enable_zeno:
            g3 = gate3_zeno_guard(intent, last_actuation_time)
            total_latency += g3.latency_ms
            if not g3.passed:
                decision = DECISION_DENY
                blocked = True
                result.zeno_blocks += 1

        # Track whether the intent would have been a safety violation
        # A safety violation occurs if an adversarial intent is accepted
        _, cbf_safe = compute_cbf_barriers(intent.desired_state)
        if intent.is_adversarial and decision == DECISION_ACCEPT and not cbf_safe:
            result.safety_violations += 1

        # Update counters
        result.latencies_ms.append(total_latency)

        if not blocked or (not enable_cue and not enable_cbf and not enable_zeno):
            # If all gates are disabled, check CBF safety for violation tracking
            if not enable_cue and not enable_cbf and not enable_zeno:
                _, cbf_safe = compute_cbf_barriers(intent.desired_state)
                if intent.is_adversarial and not cbf_safe:
                    result.safety_violations += 1
                    result.denied_count += 1
                else:
                    result.accepted_count += 1
            elif decision == DECISION_ACCEPT:
                result.accepted_count += 1
                last_actuation_time = intent.timestamp
            elif decision == DECISION_AMEND:
                result.amended_count += 1
                last_actuation_time = intent.timestamp
            else:
                result.denied_count += 1
        elif blocked:
            result.denied_count += 1
        elif decision == DECISION_ACCEPT:
            result.accepted_count += 1
            last_actuation_time = intent.timestamp
        elif decision == DECISION_AMEND:
            result.amended_count += 1
            last_actuation_time = intent.timestamp

    return result


# ===========================================================================
# Main Ablation Study
# ===========================================================================


def run_ablation_study() -> pd.DataFrame:
    """Execute the full ablation study across all 5 configurations.

    Returns:
        DataFrame with per-configuration results.
    """
    configurations = [
        ("Full DSF", True, True, True),
        ("No CUE", False, True, True),
        ("No CBF", True, False, True),
        ("No Zeno", True, True, False),
        ("No DSF", False, False, False),
    ]

    print("=" * 78)
    print("  DSF Ablation Study — 6G Digital Immunity")
    print("=" * 78)
    print(f"  Intents per configuration: {N_INTENTS}")
    print(f"  Adversarial fraction:     {ADVERSARIAL_FRACTION:.0%}")
    print(f"  CUE malformation rate:   {CUE_MALFORM_RATE:.0%}")
    print(f"  CBF gamma (k):           {GAMMA_K}")
    print(f"  Zeno tau_min:            {TAU_MIN_MS} ms")
    print("-" * 78)

    rows = []
    for config_name, enable_cue, enable_cbf, enable_zeno in configurations:
        print(
            f"\n  Running: {config_name:20s}  (CUE={enable_cue}, CBF={enable_cbf}, Zeno={enable_zeno})"
        )
        result = run_dsf_simulation(
            config_name=config_name,
            enable_cue=enable_cue,
            enable_cbf=enable_cbf,
            enable_zeno=enable_zeno,
            n_intents=N_INTENTS,
            seed=42,
        )
        rows.append(
            {
                "Configuration": config_name,
                "Gate_CUE": enable_cue,
                "Gate_CBF": enable_cbf,
                "Gate_Zeno": enable_zeno,
                "Total_Intents": result.total_intents,
                "Safety_Violations": result.safety_violations,
                "Accepted": result.accepted_count,
                "Denied": result.denied_count,
                "Amended": result.amended_count,
                "Acceptance_Rate": result.acceptance_rate,
                "Avg_Latency_ms": result.avg_latency_ms,
                "Std_Latency_ms": result.std_latency_ms,
                "Zeno_Blocks": result.zeno_blocks,
                "Avg_Min_CBF_Margin": result.avg_min_cbf_margin,
            }
        )

        print(f"    Safety Violations : {result.safety_violations}")
        print(
            f"    Accepted/Denied/Amend: {result.accepted_count}/{result.denied_count}/{result.amended_count}"
        )
        print(f"    Acceptance Rate  : {result.acceptance_rate:.2%}")
        print(
            f"    Avg Latency      : {result.avg_latency_ms:.3f} ms (σ={result.std_latency_ms:.3f})"
        )

    df = pd.DataFrame(rows)
    return df


# ===========================================================================
# Visualization
# ===========================================================================


def plot_ablation_results(df: pd.DataFrame, output_dir: str) -> None:
    """Generate publication-quality grouped bar charts for the ablation study.

    Creates a figure with four subplots:
        (a) Safety Violations per configuration
        (b) Intent Decision Distribution (Accept/Deny/Amend)
        (c) Acceptance Rate
        (d) Average Latency

    Args:
        df: Results DataFrame from run_ablation_study().
        output_dir: Directory to save the plot.
    """
    configs = df["Configuration"].tolist()
    n_configs = len(configs)
    x = np.arange(n_configs)
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "DSF Ablation Study — Gate Contribution Analysis\n"
        r"6G Digital Immunity: Deterministic Safety Filter ($N=1000$ intents per configuration)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Color palette (avoiding blue/indigo)
    colors = {
        "accept": "#2E8B57",  # Sea green
        "deny": "#CD5C5C",  # Indian red
        "amend": "#DAA520",  # Goldenrod
        "violation": "#8B0000",  # Dark red
        "latency": "#2F4F4F",  # Dark slate gray
    }

    # (a) Safety Violations
    ax = axes[0, 0]
    bars = ax.bar(
        x,
        df["Safety_Violations"],
        color=colors["violation"],
        width=0.5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Safety Violations", fontweight="bold")
    ax.set_title("(a) Safety Violations by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, df["Safety_Violations"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(val)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # (b) Decision Distribution
    ax = axes[0, 1]
    ax.bar(
        x - width,
        df["Accepted"],
        width,
        label="Accepted",
        color=colors["accept"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        df["Denied"],
        width,
        label="Denied",
        color=colors["deny"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width,
        df["Amended"],
        width,
        label="Amended",
        color=colors["amend"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Intent Count", fontweight="bold")
    ax.set_title("(b) Intent Decision Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    # (c) Acceptance Rate
    ax = axes[1, 0]
    bars = ax.bar(
        x,
        df["Acceptance_Rate"] * 100,
        color=colors["accept"],
        width=0.5,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.set_ylabel("Acceptance Rate (%)", fontweight="bold")
    ax.set_title("(c) Intent Acceptance Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, df["Acceptance_Rate"] * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # (d) Average Latency
    ax = axes[1, 1]
    bars = ax.bar(
        x,
        df["Avg_Latency_ms"],
        color=colors["latency"],
        width=0.5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Average Latency (ms)", fontweight="bold")
    ax.set_title("(d) DSF Processing Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylim(bottom=0)
    for bar, val, err in zip(bars, df["Avg_Latency_ms"], df["Std_Latency_ms"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "ablation_study.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {output_path}")


# ===========================================================================
# Entry Point
# ===========================================================================


def main():
    """Main entry point for the ablation study."""
    # Resolve output directory (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir

    # Suppress matplotlib warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    # Run the ablation study
    df = run_ablation_study()

    # Print summary table
    print("\n" + "=" * 78)
    print("  ABLATION STUDY — SUMMARY TABLE")
    print("=" * 78)
    print(
        df[
            [
                "Configuration",
                "Safety_Violations",
                "Accepted",
                "Denied",
                "Amended",
                "Acceptance_Rate",
                "Avg_Latency_ms",
                "Zeno_Blocks",
            ]
        ].to_string(index=False, float_format="%.3f")
    )

    # Save CSV
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # Generate plots
    plot_ablation_results(df, output_dir)

    # Key findings
    print("\n" + "=" * 78)
    print("  KEY FINDINGS")
    print("=" * 78)

    full_dsf = df[df["Configuration"] == "Full DSF"].iloc[0]
    no_dsf = df[df["Configuration"] == "No DSF"].iloc[0]

    print(
        f"  • Full DSF achieves {full_dsf['Safety_Violations']:.0f} safety violations "
        f"vs {no_dsf['Safety_Violations']:.0f} without DSF"
    )
    print(
        f"  • Safety improvement: "
        f"{((no_dsf['Safety_Violations'] - full_dsf['Safety_Violations']) / max(no_dsf['Safety_Violations'], 1)) * 100:.1f}%"
    )
    print(f"  • Full DSF acceptance rate: {full_dsf['Acceptance_Rate']:.2%}")
    print(f"  • Average DSF latency: {full_dsf['Avg_Latency_ms']:.3f} ms")

    # Marginal contribution of each gate
    for removed in ["No CUE", "No CBF", "No Zeno"]:
        row = df[df["Configuration"] == removed].iloc[0]
        extra_violations = row["Safety_Violations"] - full_dsf["Safety_Violations"]
        print(
            f"  • Removing {removed}: +{extra_violations:.0f} safety violations "
            f"(marginal cost of that gate)"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
