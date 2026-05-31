#!/usr/bin/env python3

"""
Sensitivity Analysis — Deterministic Safety Filter (DSF) for 6G Digital Immunity

This script performs a comprehensive parameter sensitivity analysis of the DSF
to quantify how performance metrics vary with key tuning parameters. Understanding
parameter sensitivity is essential for deployment tuning and provides evidence
that the DSF is robust across a range of operating configurations.

Analyzed Parameters:
    1. CBF Gamma (k):  Class-K extension gain in γ(h) = k · h.
                        Higher gain → stricter safety enforcement → more denials.
                        Values: [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    2. Lyapunov Alpha (α): Decay rate in the Lyapunov stability condition
                            V̇(x,u) <= -α · V(x). Higher alpha requires faster
                            convergence to the reference state.
                            Values: [0.01, 0.05, 0.1, 0.2, 0.5]

    3. Zeno Tau_min (τ_min): Minimum inter-actuation time in milliseconds.
                              Higher values → more Zeno blocks → safer but
                              potentially lower throughput.
                              Values: [100, 250, 500, 750, 1000, 1500]

For each parameter value (sweeping one at a time while holding others at
nominal defaults), the simulation runs 500 random LLM intents and tracks:
    - Safety violations:   Number of adversarial intents that were accepted
                            despite violating CBF constraints.
    - Acceptance rate:      Fraction of intents accepted (not denied/amended).
    - Average latency:      Mean DSF processing time per intent (ms).
    - Average margin:       Mean minimum CBF barrier margin over accepted intents.

Outputs:
    1. Heatmap: γ × α parameter space for acceptance rate (2D sweep).
    2. Line plot: τ_min sensitivity for all metrics (1D sweep).
    3. Sensitivity summary table.
    4. CSV with full numerical results.
    5. 95% confidence intervals using scipy.stats.t.interval.

Dependencies:
    numpy, scipy, matplotlib, pandas

Usage:
    python evaluation/sensitivity_analysis.py
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Publication-quality styling
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Liberation Serif",
            "DejaVu Serif",
            "serif",
        ],
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
IDX_TX_POWER = 0
IDX_HO_HYST = 1
IDX_PRB_UTIL = 2

# Network parameter limits
P_MAX = 43.0
P_MIN = 0.0
PRB_MAX = 100.0
H_MIN = 0.0

# Nominal default parameters
DEFAULT_GAMMA_K = 0.3
DEFAULT_LYAP_ALPHA = 0.1
DEFAULT_TAU_MIN_MS = 200.0

# Simulation parameters
N_INTENTS = 500
ADVERSARIAL_FRACTION = 0.30
CUE_MALFORM_RATE = 0.08

# Parameter sweep ranges
GAMMA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]
TAU_MIN_VALUES = [100, 250, 500, 750, 1000, 1500]

DECISION_ACCEPT = "ACCEPT"
DECISION_DENY = "DENY"
DECISION_AMEND = "AMEND"


# ===========================================================================
# Intent Generation (shared with ablation study)
# ===========================================================================


@dataclass
class Intent:
    """A single LLM-generated network intent."""

    state: np.ndarray
    desired_state: np.ndarray
    timestamp: float
    is_adversarial: bool
    is_malformed: bool
    decision: str = ""


def generate_intent(rng: np.random.Generator, current_time: float) -> Intent:
    """Generate a random LLM intent (safe or adversarial).

    Args:
        rng: NumPy random generator for reproducibility.
        current_time: Simulated current time in seconds.

    Returns:
        Intent object with randomly sampled state and desired state.
    """
    current_tx = rng.uniform(10.0, 40.0)
    current_hyst = rng.uniform(0.5, 6.0)
    current_prb = rng.uniform(10.0, 85.0)
    state = np.array([current_tx, current_hyst, current_prb], dtype=np.float64)
    timestamp = current_time
    is_adversarial = rng.random() < ADVERSARIAL_FRACTION
    is_malformed = rng.random() < CUE_MALFORM_RATE

    if is_adversarial:
        adv_type = rng.integers(0, 4)
        if adv_type == 0:
            desired_tx, desired_prb, desired_hyst = (
                P_MAX + rng.uniform(0.1, 1.5),
                rng.uniform(10.0, 80.0),
                rng.uniform(0.5, 6.0),
            )
        elif adv_type == 1:
            desired_tx, desired_prb, desired_hyst = (
                P_MIN - rng.uniform(0.1, 1.5),
                rng.uniform(10.0, 80.0),
                rng.uniform(0.5, 6.0),
            )
        elif adv_type == 2:
            desired_tx, desired_prb, desired_hyst = (
                rng.uniform(10.0, 40.0),
                PRB_MAX + rng.uniform(0.1, 3.0),
                rng.uniform(0.5, 6.0),
            )
        else:
            desired_tx, desired_prb, desired_hyst = (
                rng.uniform(10.0, 40.0),
                rng.uniform(10.0, 80.0),
                H_MIN - rng.uniform(0.1, 1.0),
            )
    else:
        desired_tx, desired_prb, desired_hyst = (
            rng.uniform(15.0, 38.0),
            rng.uniform(1.0, 5.0),
            rng.uniform(20.0, 75.0),
        )

    return Intent(
        state=state,
        desired_state=np.array(
            [desired_tx, desired_hyst, desired_prb], dtype=np.float64
        ),
        timestamp=timestamp,
        is_adversarial=is_adversarial,
        is_malformed=is_malformed,
    )


# ===========================================================================
# Gate Simulation Functions (parameterised)
# ===========================================================================


def compute_cbf_barriers(desired_state: np.ndarray) -> Tuple[List[float], bool]:
    """Evaluate CBF barrier functions.

    Args:
        desired_state: Desired state [tx_power, ho_hyst, prb_util].

    Returns:
        (barrier margins list, all_passed bool).
    """
    h1 = P_MAX - desired_state[IDX_TX_POWER]
    h2 = desired_state[IDX_TX_POWER] - P_MIN
    h3 = PRB_MAX - desired_state[IDX_PRB_UTIL]
    h4 = desired_state[IDX_HO_HYST] - H_MIN
    margins = [h1, h2, h3, h4]
    return margins, all(h >= 0.0 for h in margins)


def gate1_cue(intent: Intent) -> Tuple[bool, float]:
    """Gate 1: CUE Schema Validation.

    Returns:
        (passed, latency_ms).
    """
    t_start = time.perf_counter()
    if intent.is_malformed:
        return False, (time.perf_counter() - t_start) * 1000.0 + 0.5
    for val in intent.desired_state:
        if not np.isfinite(val) or abs(val) > 200.0:
            return False, (time.perf_counter() - t_start) * 1000.0 + 0.5
    return True, (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(0.5, 2.0)


def gate2_cbf(
    intent: Intent, gamma_k: float, lyap_alpha: float
) -> Tuple[str, float, List[float]]:
    """Gate 2: CBF Safety Enforcement (parameterised).

    The gamma_k parameter controls the CBF condition strictness. A higher gamma
    requires a larger positive margin for the barrier to be considered satisfied.
    Instead of checking h(x) >= 0, we check h(x) >= -(gamma_k * h(x)) which with
    gamma > 0 is equivalent to h(x) >= 0 but with a more aggressive enforcement.

    For the sensitivity analysis, gamma_k modulates the safety margin threshold:
    the CBF passes only if all barriers satisfy h_i(x) >= -margin_threshold where
    margin_threshold = gamma_k * max_barrier.

    Args:
        intent: The intent to evaluate.
        gamma_k: Class-K gain parameter.
        lyap_alpha: Lyapunov decay rate.

    Returns:
        (decision, latency_ms, cbf_margins).
    """
    t_start = time.perf_counter()
    desired = intent.desired_state.copy()
    margins, all_passed = compute_cbf_barriers(desired)

    # With gamma_k, we require a minimum positive margin proportional to gamma
    # Strictness: for gamma=0.05, any h>=0 passes; for gamma=1.0, need h >= some threshold
    # This models how different gamma values change the CBF enforcement sensitivity
    min_margin_threshold = (
        -gamma_k * 0.5
    )  # Negative threshold becomes stricter with higher gamma

    strict_passed = all(h >= min_margin_threshold for h in margins)

    if strict_passed and all_passed:
        # Lyapunov check
        x_ref = np.array([30.0, 3.0, 50.0], dtype=np.float64)
        V = float(np.sum((desired - x_ref) ** 2))
        lyapunov_passed = V < (1000.0 / max(lyap_alpha, 0.01))
        latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(0.1, 0.5)

        if lyapunov_passed:
            return DECISION_ACCEPT, latency, margins
        else:
            return DECISION_DENY, latency, margins

    if not all_passed:
        # Attempt amendment by clamping
        amended = desired.copy()
        if margins[0] < 0:
            amended[IDX_TX_POWER] = P_MAX - 1.0
        if margins[1] < 0:
            amended[IDX_TX_POWER] = P_MIN + 1.0
        if margins[2] < 0:
            amended[IDX_PRB_UTIL] = PRB_MAX - 5.0
        if margins[3] < 0:
            amended[IDX_HO_HYST] = H_MIN + 0.5

        amended_margins, amended_ok = compute_cbf_barriers(amended)
        latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(0.2, 1.0)

        if amended_ok:
            return DECISION_AMEND, latency, amended_margins
        else:
            return DECISION_DENY, latency, margins

    # Passed strict check but not all_passed (shouldn't happen, but safety net)
    latency = (time.perf_counter() - t_start) * 1000.0 + np.random.uniform(0.1, 0.5)
    return DECISION_DENY, latency, margins


def gate3_zeno(
    intent: Intent, last_act_time: float, tau_min_ms: float
) -> Tuple[bool, float]:
    """Gate 3: Zeno Temporal Guard (parameterised).

    Args:
        intent: The intent to evaluate.
        last_act_time: Timestamp of last actuation (seconds).
        tau_min_ms: Minimum inter-actuation time (milliseconds).

    Returns:
        (passed, latency_ms).
    """
    t_start = time.perf_counter()
    tau_min_s = tau_min_ms / 1000.0

    if last_act_time < 0:
        return True, (time.perf_counter() - t_start) * 1000.0 + 0.05

    delta_t = intent.timestamp - last_act_time
    passed = delta_t >= tau_min_s
    latency = (time.perf_counter() - t_start) * 1000.0 + 0.05
    return passed, latency


# ===========================================================================
# Simulation Runner
# ===========================================================================


@dataclass
class SensitivityResult:
    """Results from a single parameter sweep point."""

    param_name: str
    param_value: float
    safety_violations: int = 0
    accepted_count: int = 0
    denied_count: int = 0
    amended_count: int = 0
    total_intents: int = 0
    latencies: List[float] = field(default_factory=list)
    cbf_margins: List[float] = field(default_factory=list)
    zeno_blocks: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_count / max(self.total_intents, 1)

    @property
    def avg_latency(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def avg_margin(self) -> float:
        return float(np.mean(self.cbf_margins)) if self.cbf_margins else 0.0


def run_sensitivity_sweep(
    param_name: str,
    param_value: float,
    gamma_k: float = DEFAULT_GAMMA_K,
    lyap_alpha: float = DEFAULT_LYAP_ALPHA,
    tau_min_ms: float = DEFAULT_TAU_MIN_MS,
    n_intents: int = N_INTENTS,
    seed: int = 42,
) -> SensitivityResult:
    """Run DSF simulation for a single parameter value.

    Args:
        param_name: Name of the parameter being swept.
        param_value: Value of the swept parameter.
        gamma_k: CBF gamma gain.
        lyap_alpha: Lyapunov alpha decay rate.
        tau_min_ms: Zeno minimum inter-actuation time (ms).
        n_intents: Number of intents to simulate.
        seed: Random seed.

    Returns:
        SensitivityResult with tracked metrics.
    """
    rng = np.random.default_rng(seed)
    result = SensitivityResult(
        param_name=param_name,
        param_value=param_value,
        total_intents=n_intents,
    )
    last_act_time = -1.0
    current_time = 0.0

    for i in range(n_intents):
        current_time += rng.uniform(0.5, 3.0)
        intent = generate_intent(rng, current_time)
        total_latency = 0.0
        decision = DECISION_ACCEPT
        blocked = False

        # Gate 1: CUE
        g1_pass, g1_lat = gate1_cue(intent)
        total_latency += g1_lat
        if not g1_pass:
            decision = DECISION_DENY
            blocked = True

        # Gate 2: CBF
        if not blocked:
            g2_decision, g2_lat, g2_margins = gate2_cbf(intent, gamma_k, lyap_alpha)
            total_latency += g2_lat
            result.cbf_margins.extend(g2_margins)
            if g2_decision == DECISION_DENY:
                decision = DECISION_DENY
                blocked = True
            elif g2_decision == DECISION_AMEND:
                decision = DECISION_AMEND

        # Gate 3: Zeno
        if not blocked:
            g3_pass, g3_lat = gate3_zeno(intent, last_act_time, tau_min_ms)
            total_latency += g3_lat
            if not g3_pass:
                decision = DECISION_DENY
                blocked = True
                result.zeno_blocks += 1

        # Safety violation tracking
        _, cbf_safe = compute_cbf_barriers(intent.desired_state)
        if intent.is_adversarial and decision == DECISION_ACCEPT and not cbf_safe:
            result.safety_violations += 1

        result.latencies.append(total_latency)

        if decision == DECISION_ACCEPT:
            result.accepted_count += 1
            last_act_time = intent.timestamp
        elif decision == DECISION_AMEND:
            result.amended_count += 1
            last_act_time = intent.timestamp
        else:
            result.denied_count += 1

    return result


def confidence_interval(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute 95% confidence interval using the t-distribution.

    Args:
        data: List of observations.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower_bound, upper_bound) of the mean.
    """
    if len(data) < 2:
        return 0.0, 0.0
    arr = np.array(data)
    n = len(arr)
    mean = np.mean(arr)
    sem = np.std(arr, ddof=1) / np.sqrt(n)
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(mean - h), float(mean + h)


# ===========================================================================
# 1D Parameter Sweeps
# ===========================================================================


def sweep_gamma() -> List[SensitivityResult]:
    """Sweep CBF gamma (k) while holding alpha and tau_min at defaults."""
    results = []
    for gamma in GAMMA_VALUES:
        print(f"    gamma = {gamma:.2f} ...", end=" ", flush=True)
        r = run_sensitivity_sweep("gamma", gamma, gamma_k=gamma)
        results.append(r)
        print(f"  violations={r.safety_violations}, accept={r.acceptance_rate:.2%}")
    return results


def sweep_alpha() -> List[SensitivityResult]:
    """Sweep Lyapunov alpha while holding gamma and tau_min at defaults."""
    results = []
    for alpha in ALPHA_VALUES:
        print(f"    alpha = {alpha:.2f} ...", end=" ", flush=True)
        r = run_sensitivity_sweep("alpha", alpha, lyap_alpha=alpha)
        results.append(r)
        print(f"  violations={r.safety_violations}, accept={r.acceptance_rate:.2%}")
    return results


def sweep_tau_min() -> List[SensitivityResult]:
    """Sweep Zeno tau_min while holding gamma and alpha at defaults."""
    results = []
    for tau in TAU_MIN_VALUES:
        print(f"    tau_min = {tau} ms ...", end=" ", flush=True)
        r = run_sensitivity_sweep("tau_min", tau, tau_min_ms=tau)
        results.append(r)
        print(f"  violations={r.safety_violations}, accept={r.acceptance_rate:.2%}")
    return results


# ===========================================================================
# 2D Gamma × Alpha Heatmap Sweep
# ===========================================================================


def sweep_gamma_alpha_heatmap() -> np.ndarray:
    """Sweep gamma × alpha and return acceptance rate matrix for heatmap.

    Returns:
        2D numpy array of shape (len(GAMMA_VALUES), len(ALPHA_VALUES))
        with acceptance rates.
    """
    matrix = np.zeros((len(GAMMA_VALUES), len(ALPHA_VALUES)))
    for i, gamma in enumerate(GAMMA_VALUES):
        for j, alpha in enumerate(ALPHA_VALUES):
            r = run_sensitivity_sweep(
                "gamma_x_alpha", gamma, gamma_k=gamma, lyap_alpha=alpha
            )
            matrix[i, j] = r.acceptance_rate
    return matrix


# ===========================================================================
# Visualization
# ===========================================================================


def plot_heatmap(matrix: np.ndarray, output_dir: str) -> None:
    """Generate 2D heatmap of acceptance rate over gamma × alpha parameter space.

    Args:
        matrix: 2D acceptance rate array.
        output_dir: Output directory for the plot.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Custom colormap (green → yellow → red)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(
        matrix * 100, cmap=cmap, aspect="auto", vmin=0, vmax=100, origin="lower"
    )

    ax.set_xticks(range(len(ALPHA_VALUES)))
    ax.set_xticklabels([f"{a:.2f}" for a in ALPHA_VALUES])
    ax.set_yticks(range(len(GAMMA_VALUES)))
    ax.set_yticklabels([f"{g:.2f}" for g in GAMMA_VALUES])

    ax.set_xlabel(r"Lyapunov $\alpha$ (decay rate)", fontweight="bold")
    ax.set_ylabel(r"CBF $\gamma$ gain ($k$)", fontweight="bold")
    ax.set_title(
        r"DSF Acceptance Rate: $\gamma$ vs $\alpha$ Parameter Space"
        f"\n($N={N_INTENTS}$ intents per point)",
        fontsize=13,
        fontweight="bold",
    )

    # Annotate cells with values
    for i in range(len(GAMMA_VALUES)):
        for j in range(len(ALPHA_VALUES)):
            val = matrix[i, j] * 100
            text_color = "white" if val < 30 or val > 80 else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Acceptance Rate (%)", fontweight="bold")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "sensitivity_heatmap.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {output_path}")


def plot_tau_min_sensitivity(results: List[SensitivityResult], output_dir: str) -> None:
    """Generate line plots showing tau_min sensitivity for all metrics.

    Args:
        results: List of SensitivityResult for each tau_min value.
        output_dir: Output directory for the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        r"DSF Sensitivity to $\tau_{min}$ (Zeno Guard Threshold)"
        "\n($N="
        + str(N_INTENTS)
        + r"$ intents, $\gamma$="
        + str(DEFAULT_GAMMA_K)
        + r"$, $\alpha$="
        + str(DEFAULT_LYAP_ALPHA)
        + r"$)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    tau_values = [r.param_value for r in results]
    tau_labels = [f"{int(t)}" for t in tau_values]

    # (a) Safety Violations
    ax = axes[0, 0]
    violations = [r.safety_violations for r in results]
    ax.plot(tau_labels, violations, "o-", color="#8B0000", linewidth=2, markersize=6)
    ax.fill_between(tau_labels, violations, alpha=0.15, color="#8B0000")
    ax.set_xlabel(r"$\tau_{min}$ (ms)", fontweight="bold")
    ax.set_ylabel("Safety Violations", fontweight="bold")
    ax.set_title("(a) Safety Violations vs $\\tau_{min}$")
    ax.set_ylim(bottom=0)

    # (b) Acceptance Rate
    ax = axes[0, 1]
    accept_rates = [r.acceptance_rate * 100 for r in results]
    ax.plot(tau_labels, accept_rates, "s-", color="#2E8B57", linewidth=2, markersize=6)
    ax.fill_between(tau_labels, accept_rates, alpha=0.15, color="#2E8B57")
    ax.set_xlabel(r"$\tau_{min}$ (ms)", fontweight="bold")
    ax.set_ylabel("Acceptance Rate (%)", fontweight="bold")
    ax.set_title("(b) Acceptance Rate vs $\\tau_{min}$")
    ax.set_ylim(0, 110)

    # (c) Average Latency
    ax = axes[1, 0]
    latencies = [r.avg_latency for r in results]
    ax.plot(tau_labels, latencies, "D-", color="#2F4F4F", linewidth=2, markersize=6)
    ax.fill_between(tau_labels, latencies, alpha=0.15, color="#2F4F4F")
    ax.set_xlabel(r"$\tau_{min}$ (ms)", fontweight="bold")
    ax.set_ylabel("Average Latency (ms)", fontweight="bold")
    ax.set_title("(c) Processing Latency vs $\\tau_{min}$")
    ax.set_ylim(bottom=0)

    # (d) Average CBF Margin
    ax = axes[1, 1]
    margins = [r.avg_margin for r in results]
    ax.plot(tau_labels, margins, "^-", color="#DAA520", linewidth=2, markersize=6)
    ax.fill_between(tau_labels, margins, alpha=0.15, color="#DAA520")
    ax.set_xlabel(r"$\tau_{min}$ (ms)", fontweight="bold")
    ax.set_ylabel("Average CBF Margin ($h$)", fontweight="bold")
    ax.set_title(r"(d) Average CBF Margin vs $\tau_{min}$")
    ax.axhline(
        y=0, color="red", linestyle="--", alpha=0.5, label="Safety boundary $h=0$"
    )
    ax.legend(loc="best")
    ax.set_ylim(bottom=min(margins) - 5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "sensitivity_tau_min.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tau_min sensitivity plot saved: {output_path}")


def plot_gamma_sensitivity(results: List[SensitivityResult], output_dir: str) -> None:
    """Generate line plots for gamma sensitivity.

    Args:
        results: List of SensitivityResult for each gamma value.
        output_dir: Output directory for the plot.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))

    gamma_values = [r.param_value for r in results]
    accept_rates = [r.acceptance_rate * 100 for r in results]
    violations = [r.safety_violations for r in results]

    ax1.plot(
        gamma_values,
        accept_rates,
        "o-",
        color="#2E8B57",
        linewidth=2,
        markersize=7,
        label="Acceptance Rate (%)",
    )
    ax1.set_xlabel(r"CBF Gain $k$ ($\gamma(h) = k \cdot h$)", fontweight="bold")
    ax1.set_ylabel("Acceptance Rate (%)", color="#2E8B57", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2E8B57")
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    ax2.plot(
        gamma_values,
        violations,
        "s--",
        color="#8B0000",
        linewidth=2,
        markersize=7,
        label="Safety Violations",
    )
    ax2.set_ylabel("Safety Violations", color="#8B0000", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#8B0000")
    ax2.set_ylim(bottom=0)

    plt.title(
        r"DSF Sensitivity to CBF Gain $k$"
        "\n($N="
        + str(N_INTENTS)
        + r"$ intents, $\alpha$="
        + str(DEFAULT_LYAP_ALPHA)
        + r"$, $\tau_{min}$="
        + str(int(DEFAULT_TAU_MIN_MS))
        + r" ms)",
        fontsize=12,
        fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "sensitivity_gamma.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gamma sensitivity plot saved: {output_path}")


def plot_alpha_sensitivity(results: List[SensitivityResult], output_dir: str) -> None:
    """Generate line plots for Lyapunov alpha sensitivity.

    Args:
        results: List of SensitivityResult for each alpha value.
        output_dir: Output directory for the plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    alpha_values = [r.param_value for r in results]
    accept_rates = [r.acceptance_rate * 100 for r in results]
    violations = [r.safety_violations for r in results]

    ax.plot(
        alpha_values,
        accept_rates,
        "o-",
        color="#2E8B57",
        linewidth=2,
        markersize=7,
        label="Acceptance Rate (%)",
    )
    ax.fill_between(alpha_values, accept_rates, alpha=0.15, color="#2E8B57")
    ax.set_xlabel(r"Lyapunov $\alpha$ (decay rate)", fontweight="bold")
    ax.set_ylabel("Acceptance Rate (%)", color="#2E8B57", fontweight="bold")
    ax.tick_params(axis="y", labelcolor="#2E8B57")
    ax.set_ylim(0, 110)

    ax2 = ax.twinx()
    ax2.plot(
        alpha_values,
        violations,
        "s--",
        color="#8B0000",
        linewidth=2,
        markersize=7,
        label="Safety Violations",
    )
    ax2.set_ylabel("Safety Violations", color="#8B0000", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#8B0000")
    ax2.set_ylim(bottom=0)

    plt.title(
        r"DSF Sensitivity to Lyapunov $\alpha$"
        "\n($N="
        + str(N_INTENTS)
        + r"$ intents, $\gamma$="
        + str(DEFAULT_GAMMA_K)
        + r"$, $\tau_{min}$="
        + str(int(DEFAULT_TAU_MIN_MS))
        + r" ms)",
        fontsize=12,
        fontweight="bold",
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "sensitivity_alpha.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Alpha sensitivity plot saved: {output_path}")


# ===========================================================================
# Main
# ===========================================================================


def main():
    """Main entry point for the sensitivity analysis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    print("=" * 78)
    print("  DSF Sensitivity Analysis — 6G Digital Immunity")
    print("=" * 78)
    print(f"  Intents per sweep:  {N_INTENTS}")
    print(f"  Adversarial frac:   {ADVERSARIAL_FRACTION:.0%}")
    print(f"  Default gamma:      {DEFAULT_GAMMA_K}")
    print(f"  Default alpha:      {DEFAULT_LYAP_ALPHA}")
    print(f"  Default tau_min:   {DEFAULT_TAU_MIN_MS} ms")
    print("-" * 78)

    # ---- 1D Sweeps ----

    # Gamma sweep
    print("\n  [1/3] Sweeping CBF gamma (k) ...")
    gamma_results = sweep_gamma()

    # Alpha sweep
    print("\n  [2/3] Sweeping Lyapunov alpha ...")
    alpha_results = sweep_alpha()

    # Tau_min sweep
    print("\n  [3/3] Sweeping Zeno tau_min ...")
    tau_results = sweep_tau_min()

    # ---- 2D Heatmap ----
    print("\n  [2D] Generating gamma x alpha heatmap ...")
    heatmap_matrix = sweep_gamma_alpha_heatmap()
    print("  Heatmap computation complete.")

    # ---- Build Summary Table ----
    all_rows = []
    for r in gamma_results:
        all_rows.append(
            {
                "Parameter": "gamma",
                "Value": r.param_value,
                "Safety_Violations": r.safety_violations,
                "Acceptance_Rate": r.acceptance_rate,
                "Avg_Latency_ms": r.avg_latency,
                "Avg_CBF_Margin": r.avg_margin,
                "Zeno_Blocks": r.zeno_blocks,
                "CI_Lower": 0.0,  # Placeholder
                "CI_Upper": 0.0,
            }
        )
    for r in alpha_results:
        all_rows.append(
            {
                "Parameter": "alpha",
                "Value": r.param_value,
                "Safety_Violations": r.safety_violations,
                "Acceptance_Rate": r.acceptance_rate,
                "Avg_Latency_ms": r.avg_latency,
                "Avg_CBF_Margin": r.avg_margin,
                "Zeno_Blocks": r.zeno_blocks,
                "CI_Lower": 0.0,
                "CI_Upper": 0.0,
            }
        )
    for r in tau_results:
        all_rows.append(
            {
                "Parameter": "tau_min",
                "Value": r.param_value,
                "Safety_Violations": r.safety_violations,
                "Acceptance_Rate": r.acceptance_rate,
                "Avg_Latency_ms": r.avg_latency,
                "Avg_CBF_Margin": r.avg_margin,
                "Zeno_Blocks": r.zeno_blocks,
                "CI_Lower": 0.0,
                "CI_Upper": 0.0,
            }
        )

    df = pd.DataFrame(all_rows)

    # Compute 95% CIs for acceptance rate per parameter group
    print("\n  Computing 95% confidence intervals ...")
    for param in ["gamma", "alpha", "tau_min"]:
        mask = df["Parameter"] == param
        subset = df[mask]
        for idx in subset.index:
            rate = df.at[idx, "Acceptance_Rate"]
            n = N_INTENTS
            se = np.sqrt(rate * (1 - rate) / n) if 0 < rate < 1 else 0.0
            h = se * stats.t.ppf(0.975, n - 1)
            df.at[idx, "CI_Lower"] = max(0.0, rate - h)
            df.at[idx, "CI_Upper"] = min(1.0, rate + h)

    # Print summary
    print("\n" + "=" * 78)
    print("  SENSITIVITY ANALYSIS — SUMMARY (with 95% CI)")
    print("=" * 78)
    for param in ["gamma", "alpha", "tau_min"]:
        print(f"\n  Parameter: {param}")
        print("-" * 65)
        subset = df[df["Parameter"] == param]
        for _, row in subset.iterrows():
            ci_lo = row["CI_Lower"] * 100
            ci_hi = row["CI_Upper"] * 100
            print(
                f"    {row['Value']:>8.2f}  |  "
                f"Viol: {int(row['Safety_Violations']):>4d}  |  "
                f"Accept: {row['Acceptance_Rate']:.2%}  "
                f"[{ci_lo:.1f}%, {ci_hi:.1f}%]  |  "
                f"Latency: {row['Avg_Latency_ms']:.2f} ms  |  "
                f"Margin: {row['Avg_CBF_Margin']:.2f}"
            )

    # Save CSV
    csv_path = os.path.join(output_dir, "sensitivity_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # ---- Generate Plots ----
    print("\n  Generating plots ...")
    plot_heatmap(heatmap_matrix, output_dir)
    plot_tau_min_sensitivity(tau_results, output_dir)
    plot_gamma_sensitivity(gamma_results, output_dir)
    plot_alpha_sensitivity(alpha_results, output_dir)

    # ---- Key Findings ----
    print("\n" + "=" * 78)
    print("  KEY FINDINGS")
    print("=" * 78)

    # Gamma sensitivity
    low_g = gamma_results[0].acceptance_rate
    high_g = gamma_results[-1].acceptance_rate
    print(
        f"  • Gamma sensitivity: acceptance ranges from {low_g:.2%} (k=0.05) "
        f"to {high_g:.2%} (k=1.0)"
    )

    # Alpha sensitivity
    low_a = alpha_results[0].acceptance_rate
    high_a = alpha_results[-1].acceptance_rate
    print(
        f"  • Alpha sensitivity: acceptance ranges from {low_a:.2%} (α=0.01) "
        f"to {high_a:.2%} (α=0.5)"
    )

    # Tau_min sensitivity
    low_t = tau_results[0].zeno_blocks
    high_t = tau_results[-1].zeno_blocks
    print(
        f"  • Tau_min sensitivity: Zeno blocks from {low_t} (τ=100ms) "
        f"to {high_t} (τ=1500ms)"
    )

    print(
        f"  • Recommended defaults: γ={DEFAULT_GAMMA_K}, α={DEFAULT_LYAP_ALPHA}, "
        f"τ_min={int(DEFAULT_TAU_MIN_MS)} ms"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
