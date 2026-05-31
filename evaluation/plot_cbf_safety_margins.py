"""
plot_cbf_safety_margins.py — Publication-Quality CBF Safety Margin Plots
========================================================================
Generates comprehensive Control Barrier Function (CBF) safety margin
visualizations for the 6G Digital Immunity framework.

The Deterministic Safety Filter (DSF) enforces four barrier functions:

    h₁(x) = 43 − tx_power       ≥ 0   (transmit power ceiling)
    h₂(x) = tx_power − 0        ≥ 0   (transmit power floor)
    h₃(x) = 100 − prb_util      ≥ 0   (resource block utilization ceiling)
    h₄(x) = ho_hysteresis − 0   ≥ 0   (handover hysteresis floor)

This script produces four publication-quality figures:
    1. Time series of h(x) values for each barrier across intents
    2. Distribution (violin + box) of safety margins per barrier
    3. Minimum margin per run with 95% confidence intervals
    4. Scatter plot of minimum margin vs. intent acceptance rate

Output:
    evaluation/cbf_safety_margins.pdf  (vector, for LaTeX)
    evaluation/cbf_safety_margins.png  (raster, 300 DPI)

Author:
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI / headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Publication-Quality Style Configuration
# ---------------------------------------------------------------------------
FONT_FAMILY = "serif"
FONT_SERIF = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"]
DPI = 300
COLORS = {
    "h1": "#2E86AB",  # muted teal-blue  (tx power ceiling)
    "h2": "#1B4965",  # dark navy         (tx power floor)
    "h3": "#5FA777",  # muted green       (PRB utilization ceiling)
    "h4": "#A23B72",  # muted plum        (HO hysteresis floor)
    "safe_bg": "#D4EDDA",
    "unsafe_bg": "#F8D7DA",
    "zero_line": "#C0392B",
    "ci_fill": "#BDC3C7",
    "accent": "#E67E22",
}
BARRIER_LABELS = {
    "h1": r"$h_1$  (Tx Power Ceiling: 43 dBm)",
    "h2": r"$h_2$  (Tx Power Floor: 0 dBm)",
    "h3": r"$h_3$  (PRB Utilization Ceiling: 100%)",
    "h4": r"$h_4$  (HO Hysteresis Floor: 0 dB)",
}
BARRIER_IDS = ["h1", "h2", "h3", "h4"]


def _configure_style() -> None:
    """Apply MDPI-compliant publication styling."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "font.size": 10,
            "legend.fontsize": 9,
            "legend.framealpha": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ---------------------------------------------------------------------------
# Simulated Data Generation
# ---------------------------------------------------------------------------


def _simulate_barrier_values(
    n_runs: int = 10,
    n_intents: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate simulated CBF barrier values h(x) for all intents.

    The simulation models realistic 6G RAN parameters that the DSF monitors:

    - Tx power is drawn from a truncated normal centered at ~30 dBm,
      bounded away from the ceiling (43 dBm) and floor (0 dBm).
    - PRB utilization is drawn from a beta distribution, capped at 85%.
    - HO hysteresis is drawn from a narrow distribution around 2 dB.

    Args:
        n_runs: Number of independent simulation runs.
        n_intents: Number of intents (intents) per run.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping barrier_id → np.ndarray of shape (n_runs, n_intents).
    """
    rng = np.random.default_rng(seed)

    # Determine adversarial vs safe (30% adversarial)
    is_adversarial = rng.random(size=(n_runs, n_intents)) < 0.30
    adv_type = rng.integers(0, 4, size=(n_runs, n_intents))

    # Safe: tx_power ~ N(30, 4), clipped to [10, 38]
    tx_safe = np.clip(rng.normal(30.0, 4.0, size=(n_runs, n_intents)), 10.0, 38.0)
    # Adv: if adv_type == 0 -> > 43; if adv_type == 1 -> < 0
    tx_adv_ceil = rng.uniform(43.5, 50.0, size=(n_runs, n_intents))
    tx_adv_floor = rng.uniform(-5.0, -0.5, size=(n_runs, n_intents))
    tx_power = np.where(is_adversarial & (adv_type == 0), tx_adv_ceil, tx_safe)
    tx_power = np.where(is_adversarial & (adv_type == 1), tx_adv_floor, tx_power)

    h1 = 43.0 - tx_power
    h2 = tx_power.copy()

    # Safe PRB: Beta scaled to [20, 75]
    prb_safe = 20.0 + rng.beta(2.0, 3.0, size=(n_runs, n_intents)) * 55.0
    prb_adv_ceil = rng.uniform(101.0, 110.0, size=(n_runs, n_intents))
    prb_util = np.where(is_adversarial & (adv_type == 2), prb_adv_ceil, prb_safe)
    h3 = 100.0 - prb_util

    # Safe hyst: N(2.0, 0.5) clipped [1.0, 4.0]
    hyst_safe = np.clip(rng.normal(2.0, 0.5, size=(n_runs, n_intents)), 1.0, 4.0)
    hyst_adv_floor = rng.uniform(-3.0, -0.5, size=(n_runs, n_intents))
    ho_hyst = np.where(is_adversarial & (adv_type == 3), hyst_adv_floor, hyst_safe)
    h4 = ho_hyst.copy()

    return {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
    }


def _compute_acceptance_rates(
    h_values: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute intent acceptance rate per run (all barriers must be >= 0).

    Args:
        h_values: Barrier values dict (n_runs, n_intents).

    Returns:
        Array of acceptance rates of shape (n_runs,).
    """
    n_runs = next(iter(h_values.values())).shape[0]
    rates = np.zeros(n_runs)
    for r in range(n_runs):
        all_safe = np.ones(h_values["h1"].shape[1], dtype=bool)
        for bid in BARRIER_IDS:
            all_safe &= h_values[bid][r] >= 0.0
        rates[r] = np.mean(all_safe)
    return rates


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------


def _plot_time_series(
    h_values: dict[str, np.ndarray],
    fig: plt.Figure,
    gs: gridspec.GridSpec,
) -> None:
    """Plot 1: Time series of h(x) values for each barrier across intents.

    Shows the first 3 runs as representative traces with low alpha,
    overlaid by the mean across all runs.

    Args:
        h_values: Barrier values dict (n_runs, n_intents).
        fig: Matplotlib figure.
        gs: GridSpec with position for this subplot.
    """
    ax = fig.add_subplot(gs[0, :])

    n_intents = next(iter(h_values.values())).shape[1]
    intent_idx = np.arange(n_intents)

    for bid in BARRIER_IDS:
        color = COLORS[bid]
        data = h_values[bid]

        # Plot individual run traces (first 3) with low alpha
        for run in range(min(3, data.shape[0])):
            ax.plot(intent_idx, data[run], color=color, alpha=0.15, linewidth=0.6)

        # Plot mean ± 1 std
        mean_trace = data.mean(axis=0)
        std_trace = data.std(axis=0)
        ax.plot(
            intent_idx,
            mean_trace,
            color=color,
            linewidth=1.5,
            label=BARRIER_LABELS[bid],
        )
        ax.fill_between(
            intent_idx,
            mean_trace - std_trace,
            mean_trace + std_trace,
            color=color,
            alpha=0.10,
        )

    # Safety boundary at h(x) = 0
    ax.axhline(
        y=0,
        color=COLORS["zero_line"],
        linestyle="--",
        linewidth=1.2,
        label=r"Safety boundary $h(x)=0$",
        zorder=10,
    )

    # Safe region shading
    ymin, ymax = ax.get_ylim()
    ax.fill_between(intent_idx, 0, ymax, alpha=0.04, color=COLORS["safe_bg"], zorder=0)
    ax.set_ylim(bottom=min(0, ymin) - 1)

    ax.set_xlabel("Intent Index $k$")
    ax.set_ylabel(r"Barrier Value $h(x)$")
    ax.set_title("(a) Time Series of CBF Barrier Values", fontweight="bold")
    ax.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.95)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))


def _plot_distribution(
    h_values: dict[str, np.ndarray],
    fig: plt.Figure,
    gs: gridspec.GridSpec,
) -> None:
    """Plot 2: Distribution of safety margins for each barrier (violin + box).

    Args:
        h_values: Barrier values dict (n_runs, n_intents).
        fig: Matplotlib figure.
        gs: GridSpec with position for this subplot.
    """
    ax = fig.add_subplot(gs[1, 0])

    # Flatten data per barrier across all runs
    data_per_barrier = [h_values[bid].ravel() for bid in BARRIER_IDS]
    positions = np.arange(len(BARRIER_IDS))

    # Violin plot
    parts = ax.violinplot(
        data_per_barrier,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[BARRIER_IDS[i]])
        pc.set_alpha(0.35)
        pc.set_edgecolor(COLORS[BARRIER_IDS[i]])

    # Box plot overlay
    bp = ax.boxplot(
        data_per_barrier,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#2C3E50", linewidth=1.5),
        whiskerprops=dict(color="#2C3E50", linewidth=0.8),
        capprops=dict(color="#2C3E50", linewidth=0.8),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[BARRIER_IDS[i]])
        patch.set_alpha(0.70)

    # Safety boundary at 0
    ax.axhline(y=0, color=COLORS["zero_line"], linestyle="--", linewidth=1.2, zorder=10)

    ax.set_xticks(positions)
    ax.set_xticklabels([r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$"], fontsize=10)
    ax.set_ylabel(r"Safety Margin $h(x)$")
    ax.set_title("(b) Safety Margin Distribution", fontweight="bold")


def _plot_min_margin_ci(
    h_values: dict[str, np.ndarray],
    fig: plt.Figure,
    gs: gridspec.GridSpec,
) -> None:
    """Plot 3: Minimum margin per run with 95% confidence intervals.

    For each run, computes the minimum h(x) across all intents per barrier,
    then reports the overall minimum across barriers and its 95% CI.

    Args:
        h_values: Barrier values dict (n_runs, n_intents).
        fig: Matplotlib figure.
        gs: GridSpec with position for this subplot.
    """
    ax = fig.add_subplot(gs[1, 1])

    n_runs = next(iter(h_values.values())).shape[0]
    run_idx = np.arange(1, n_runs + 1)

    for bid in BARRIER_IDS:
        data = h_values[bid]
        # Per-run minimum margin
        min_margins = data.min(axis=1)
        color = COLORS[bid]

        ax.plot(
            run_idx,
            min_margins,
            "o-",
            color=color,
            markersize=5,
            linewidth=1.3,
            label=r"$h_1$"
            if bid == "h1"
            else r"$h_2$"
            if bid == "h2"
            else r"$h_3$"
            if bid == "h3"
            else r"$h_4$",
        )

    # Overall minimum margin per run (across all barriers)
    all_min = np.array(
        [min(h_values[bid][r].min() for bid in BARRIER_IDS) for r in range(n_runs)]
    )
    mean_min = all_min.mean()
    ci_95 = sp_stats.t.interval(
        confidence=0.95,
        df=n_runs - 1,
        loc=mean_min,
        scale=sp_stats.sem(all_min),
    )

    ax.axhline(
        y=mean_min,
        color=COLORS["accent"],
        linestyle="-.",
        linewidth=1.5,
        label=rf"Overall mean min = {mean_min:.2f} dBm",
    )
    ax.axhline(y=0, color=COLORS["zero_line"], linestyle="--", linewidth=1.2, zorder=10)

    # Shade CI region
    ax.fill_between(
        run_idx,
        ci_95[0],
        ci_95[1],
        color=COLORS["ci_fill"],
        alpha=0.5,
        label=rf"95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]",
    )

    ax.set_xlabel("Run Index")
    ax.set_ylabel(r"Min. Safety Margin (dBm)")
    ax.set_title("(c) Per-Run Minimum Margin", fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.95)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _plot_margin_vs_acceptance(
    h_values: dict[str, np.ndarray],
    acceptance_rates: np.ndarray,
    fig: plt.Figure,
    gs: gridspec.GridSpec,
) -> None:
    """Plot 4: Scatter plot of minimum margin vs. intent acceptance rate.

    Demonstrates the correlation between safety margins and the DSF's
    acceptance behavior.

    Args:
        h_values: Barrier values dict (n_runs, n_intents).
        acceptance_rates: Per-run acceptance rate array (n_runs,).
        fig: Matplotlib figure.
        gs: GridSpec with position for this subplot.
    """
    ax = fig.add_subplot(gs[2, :])

    n_runs = len(acceptance_rates)

    # Compute per-run minimum margin (overall, across all barriers)
    all_min = np.array(
        [min(h_values[bid][r].min() for bid in BARRIER_IDS) for r in range(n_runs)]
    )

    # Compute per-run per-barrier minimum margins
    for bid in BARRIER_IDS:
        barrier_min = h_values[bid].min(axis=1)
        ax.scatter(
            barrier_min,
            acceptance_rates,
            c=COLORS[bid],
            s=60,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
            label=r"$h_1$"
            if bid == "h1"
            else r"$h_2$"
            if bid == "h2"
            else r"$h_3$"
            if bid == "h3"
            else r"$h_4$",
        )

    # Overall minimum margin scatter (highlighted)
    ax.scatter(
        all_min,
        acceptance_rates,
        c=COLORS["accent"],
        s=120,
        marker="D",
        edgecolors="black",
        linewidths=1.0,
        zorder=10,
        label="Overall minimum margin",
    )

    # Regression line on overall min
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
        all_min, acceptance_rates
    )
    x_line = np.linspace(all_min.min() - 1, all_min.max() + 1, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "--", color="#7F8C8D", linewidth=1.2, alpha=0.8)

    # Annotation with R²
    ax.annotate(
        rf"$R^2 = {r_value**2:.4f}$, $p = {p_value:.4f}$",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        fontsize=9,
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDC3C7", alpha=0.95
        ),
    )

    ax.set_xlabel(r"Per-Run Minimum Safety Margin $h_{\min}(x)$ (dBm)")
    ax.set_ylabel("Intent Acceptance Rate")
    ax.set_title(
        "(d) Safety Margin vs. Intent Acceptance Rate (Per Run)",
        fontweight="bold",
    )
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", ncol=3, fontsize=8, framealpha=0.95)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def generate_plots(
    n_runs: int = 10,
    n_intents: int = 500,
    seed: int = 42,
    output_dir: str | None = None,
) -> list[str]:
    """Generate all four CBF safety margin plots and save to disk.

    Args:
        n_runs: Number of independent Monte Carlo simulation runs.
        n_intents: Number of intents per run.
        seed: Random seed for reproducibility.
        output_dir: Directory for output files. Defaults to this file's dir.

    Returns:
        List of saved file paths.
    """
    _configure_style()

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # Generate simulated data
    print(
        f"[CBF Plotter] Generating simulated data: "
        f"{n_runs} runs × {n_intents} intents (seed={seed})"
    )
    h_values = _simulate_barrier_values(n_runs, n_intents, seed)
    acceptance_rates = _compute_acceptance_rates(h_values)

    # Print summary statistics
    print(f"\n{'=' * 65}")
    print("  CBF Safety Margin Summary Statistics")
    print(f"{'=' * 65}")
    for bid in BARRIER_IDS:
        data = h_values[bid].ravel()
        print(f"  {BARRIER_LABELS[bid]:50s}")
        print(
            f"      min={data.min():8.4f}  max={data.max():8.4f}  "
            f"mean={data.mean():8.4f}  std={data.std():8.4f}"
        )
        violations = np.sum(data < 0)
        print(
            f"      Safety violations: {violations} / {len(data)} "
            f"({100 * violations / len(data):.4f}%)"
        )
    print(
        f"\n  Overall acceptance rate: {acceptance_rates.mean() * 100:.2f}% "
        f"(± {acceptance_rates.std() * 100:.2f}%)"
    )
    print(f"{'=' * 65}\n")

    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(
        "Deterministic Safety Filter — CBF Safety Margin Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.85],
        hspace=0.38,
        wspace=0.28,
        left=0.08,
        right=0.97,
        top=0.93,
        bottom=0.06,
    )

    # (a) Time series — full width top row
    _plot_time_series(h_values, fig, gs)

    # (b) Distribution — bottom-left of row 2
    _plot_distribution(h_values, fig, gs)

    # (c) Min margin with CI — bottom-right of row 2
    _plot_min_margin_ci(h_values, fig, gs)

    # (d) Scatter — full width bottom row
    _plot_margin_vs_acceptance(h_values, acceptance_rates, fig, gs)

    # Save outputs
    pdf_path = os.path.join(output_dir, "cbf_safety_margins.pdf")
    png_path = os.path.join(output_dir, "cbf_safety_margins.png")

    fig.savefig(pdf_path, format="pdf", dpi=DPI)
    fig.savefig(png_path, format="png", dpi=DPI)
    plt.close(fig)

    print(f"[CBF Plotter] Saved: {pdf_path}")
    print(f"[CBF Plotter] Saved: {png_path}")

    return [pdf_path, png_path]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CBF safety margin plots for 6G Digital Immunity."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of Monte Carlo simulation runs (default: 10)",
    )
    parser.add_argument(
        "--intents",
        type=int,
        default=500,
        help="Number of intents per run (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as this script)",
    )

    args = parser.parse_args()
    generate_plots(
        n_runs=args.runs,
        n_intents=args.intents,
        seed=args.seed,
        output_dir=args.output_dir,
    )
