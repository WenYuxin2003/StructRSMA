"""Plot the released Unified SCA Mechanism Ablation summary.

This script consumes only the machine-readable final summary tables under
revision_2026/results/unified_sca. It does not train or evaluate a model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "revision_2026" / "results" / "unified_sca"
OUTPUT = ROOT / "revision_2026" / "figures_r2"


def main():
    fair = pd.read_csv(RESULTS / "fair_control_ablation.csv")
    feature = pd.read_csv(RESULTS / "feature_ablation.csv")
    views = pd.read_csv(RESULTS / "view_randomization.csv")
    cold = pd.read_csv(RESULTS / "cold_results.csv")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), constrained_layout=True)

    # Panel A: fair-control ablation.
    y = np.arange(len(fair))
    axes[0].errorbar(
        fair["rmse_mean"],
        y,
        xerr=fair["rmse_sd"],
        fmt="o",
        color="#2a9d8f",
        capsize=3,
    )
    axes[0].set_yticks(y, fair["variant"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (mean +/- SD)")
    axes[0].set_title("a  Fair-control ablation", loc="left", fontweight="bold")

    # Panel B: feature removal and representation randomization.
    feature_plot = feature[feature["setting"] != "Full D+M+R+A"].copy()
    view_plot = views[views["attribution_control"] != "Full SCA"].copy()
    labels = feature_plot["setting"].tolist() + view_plot["attribution_control"].tolist()
    deltas = feature_plot["delta_rmse_vs_full"].tolist() + (view_plot["rmse"] - 0.918).tolist()
    colors = ["#457b9d"] * len(feature_plot) + ["#e9a24a"] * len(view_plot)
    y = np.arange(len(labels))
    axes[1].barh(y, deltas, color=colors)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("RMSE increase vs Full SCA")
    axes[1].set_title("b  Mechanism perturbations", loc="left", fontweight="bold")

    # Panel C: cold-setting calibration.
    order = ["cold-RNA", "cold-scaffold", "double-cold"]
    x = np.arange(len(order))
    width = 0.34
    no_sca = cold[cold["model"] == "Contact500 no SCA"].set_index("split").loc[order]
    full = cold[cold["model"] == "Full SCA"].set_index("split").loc[order]
    axes[2].bar(
        x - width / 2,
        no_sca["rmse_mean"],
        width,
        yerr=no_sca["rmse_sd"],
        label="Contact500, no SCA",
        color="#457b9d",
        capsize=3,
    )
    axes[2].bar(
        x + width / 2,
        full["rmse_mean"],
        width,
        yerr=full["rmse_sd"],
        label="Full SCA",
        color="#2a9d8f",
        capsize=3,
    )
    axes[2].set_xticks(x, order, rotation=20, ha="right")
    axes[2].set_ylabel("RMSE (mean +/- SD)")
    axes[2].set_title("c  Cold generalization", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Mechanistic and fairness analysis of the Structural Contact Adapter", fontweight="bold")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "fig_r2_3_unified_sca_mechanism"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

