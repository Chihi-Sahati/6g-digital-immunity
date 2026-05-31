import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Q1 Journal Styling Best Practices
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    }
)


def plot_cbf_safety_margins(audit_logs):
    """Plot the CBF margins over time to show how the filter maintains safety."""
    data = []
    for log in audit_logs:
        # Assuming the audit log contains 'cbf_margin' and 'timestamp'
        if "cbf_result" in log:
            margin = log["cbf_result"].get("margin", 0)
            decision = log.get("decision", "UNKNOWN")
            ts = pd.to_datetime(log.get("evaluated_at", datetime.now().isoformat()))
            data.append(
                {"Timestamp": ts, "CBF Margin (h)": margin, "Verdict": decision}
            )

    if not data:
        print("No CBF margin data found for plotting.")
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 4))

    # Plot safe region
    plt.axhline(y=0, color="r", linestyle="--", label="Safety Boundary ($h(x) = 0$)")
    plt.fill_between(
        df["Timestamp"],
        0,
        df["CBF Margin (h)"].max() + 10,
        alpha=0.1,
        color="green",
        label="Safe Set ($\mathcal{C}$)",
    )
    plt.fill_between(
        df["Timestamp"],
        df["CBF Margin (h)"].min() - 10,
        0,
        alpha=0.1,
        color="red",
        label="Unsafe Set",
    )

    # Plot margins
    sns.scatterplot(
        data=df,
        x="Timestamp",
        y="CBF Margin (h)",
        hue="Verdict",
        palette={"ALLOW": "blue", "AMEND": "orange", "DENY": "red"},
        s=50,
    )
    sns.lineplot(
        data=df,
        x="Timestamp",
        y="CBF Margin (h)",
        color="black",
        alpha=0.5,
        linewidth=1,
    )

    plt.title("Deterministic Safety Filter: Control Barrier Function (CBF) Margins")
    plt.ylabel("CBF Margin $h(x)$")
    plt.xlabel("Evaluation Time")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("cbf_safety_margins.pdf", bbox_inches="tight")
    plt.savefig("cbf_safety_margins.png", bbox_inches="tight")
    plt.close()
    print("Saved plot to cbf_safety_margins.pdf and .png")


def plot_telemetry_prb(telemetry):
    """Plot PRB Utilization and Active Users from the RAN simulator."""
    data = []
    for point in telemetry:
        if point.get("metric_name") in [
            "radio.prb_utilisation_pct",
            "radio.active_ue_count",
        ]:
            data.append(
                {
                    "Timestamp": pd.to_datetime(point["timestamp"]),
                    "Metric": point["metric_name"],
                    "Value": point["metric_value"],
                }
            )

    if not data:
        print("No telemetry data found for plotting.")
        return

    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Filter data
    df_prb = df[df["Metric"] == "radio.prb_utilisation_pct"]
    df_ue = df[df["Metric"] == "radio.active_ue_count"]

    color = "tab:blue"
    ax1.set_xlabel("Time")
    ax1.set_ylabel("PRB Utilization (%)", color=color)
    ax1.plot(
        df_prb["Timestamp"], df_prb["Value"], color=color, linewidth=2, label="PRB Util"
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Active Users", color=color)
    ax2.plot(
        df_ue["Timestamp"],
        df_ue["Value"],
        color=color,
        linestyle="--",
        linewidth=2,
        label="Active Users",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("6G RAN Telemetry: Traffic Dynamics")
    fig.tight_layout()
    plt.savefig("ran_telemetry.pdf", bbox_inches="tight")
    plt.savefig("ran_telemetry.png", bbox_inches="tight")
    plt.close()
    print("Saved plot to ran_telemetry.pdf and .png")


if __name__ == "__main__":
    try:
        with open("metrics_log.json", "r") as f:
            metrics = json.load(f)

        plot_telemetry_prb(metrics.get("telemetry", []))
        plot_cbf_safety_margins(metrics.get("audit_logs", []))

    except FileNotFoundError:
        print("metrics_log.json not found. Please run collect_metrics.py first.")
