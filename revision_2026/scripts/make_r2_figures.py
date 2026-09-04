import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_2026"
OUT = REV / "figures_r2"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

COLORS = {
    "baseline": "#484878",
    "contact": "#5B8DB8",
    "struct": "#2A9D8F",
    "control": "#B8B8B8",
    "negative": "#C75B5B",
    "accent": "#E3A857",
    "light": "#DCE8F2",
    "dark": "#303030",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication-ready evaluation figures.")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=("curation", "contact", "similarity", "ablation", "cold", "attribution"),
    )
    return parser.parse_args()


def save_figure(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel(ax, label):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)


def style_axis(ax):
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.tick_params(length=3)


def figure_curation():
    manifest = pd.read_csv(REV / "audit/contact500_v1/contact500_manifest.csv")
    decisions = pd.read_csv(REV / "audit/contact246_conservative_qc/contact_qc_decisions.csv")
    qc_ids = set(decisions.loc[decisions["keep_conservative_qc"], "pdb_id"])
    manifest["set"] = np.where(manifest["pdb_id"].isin(qc_ids), "Conservative QC (n=246)", "Full contact set (n=484)")

    fig = plt.figure(figsize=(7.2, 4.7))
    grid = fig.add_gridspec(2, 4, height_ratios=(0.8, 1.6), hspace=0.5, wspace=0.45)
    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    nodes = [
        (0.06, "500", "queried PDB IDs"),
        (0.32, "486", "buildable metadata\nrecords"),
        (0.58, "484", "complexes with >=1\ncontact at 4.0 A"),
        (0.86, "246", "conservative QC\nsensitivity set"),
    ]
    for index, (x, number, label) in enumerate(nodes):
        color = COLORS["struct"] if index == len(nodes) - 1 else COLORS["contact"]
        ax.text(
            x,
            0.56,
            number,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor="none"),
        )
        ax.text(x, 0.13, label, ha="center", va="center", fontsize=7)
        if index < len(nodes) - 1:
            ax.annotate("", xy=(nodes[index + 1][0] - 0.075, 0.56), xytext=(x + 0.075, 0.56), arrowprops=dict(arrowstyle="->", lw=1.2, color=COLORS["dark"]))
    ax.text(0.19, 0.83, "14 metadata exclusions", ha="center", color=COLORS["negative"], fontsize=6.5)
    ax.text(0.45, 0.83, "2 zero-contact exclusions", ha="center", color=COLORS["negative"], fontsize=6.5)
    ax.text(0.72, 0.83, "238 conservative QC exclusions", ha="center", color=COLORS["negative"], fontsize=6.5)
    panel(ax, "a")

    variables = [
        ("resolution_angstrom", "Resolution (A)", np.linspace(0.5, 8, 18)),
        ("rna_length", "RNA length (nt)", np.geomspace(4, 420, 18)),
        ("ligand_heavy_atoms", "Ligand heavy atoms", np.linspace(5, 115, 18)),
        ("contact_density", "Contact density", np.linspace(0, 0.18, 19)),
    ]
    for idx, (column, xlabel, bins) in enumerate(variables):
        ax = fig.add_subplot(grid[1, idx])
        full = manifest[column].dropna().to_numpy(float)
        qc = manifest.loc[manifest["pdb_id"].isin(qc_ids), column].dropna().to_numpy(float)
        ax.hist(full, bins=bins, color=COLORS["light"], edgecolor=COLORS["contact"], linewidth=0.5, label="Contact484")
        ax.hist(qc, bins=bins, histtype="step", color=COLORS["struct"], linewidth=1.3, label="QC246")
        if column == "rna_length":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Complexes" if idx == 0 else "")
        style_axis(ax)
        panel(ax, chr(ord("b") + idx))
    fig.axes[-1].legend(loc="upper right", fontsize=6)
    fig.suptitle("PDB-derived contact-data curation and structural distributions", y=1.01, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_r2_2_contact_curation")


def figure_contact():
    aggregate = pd.read_csv(REV / "statistics/contact/contact_grouped_aggregate.csv").iloc[0]
    cutoff = pd.read_csv(REV / "statistics/contact/contact_cutoff_sensitivity.csv")
    strata = pd.read_csv(REV / "statistics/contact_stratified/contact_stratified_across_splits.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw={"hspace": 0.48, "wspace": 0.38})

    ax = axes[0, 0]
    metrics = ["AUPRC", "AUROC", "MCC", "F1", "P@5", "P@10", "P@Nc"]
    stems = ["macro_auprc", "macro_auroc", "macro_mcc", "macro_f1", "macro_p_at_5", "macro_p_at_10", "macro_p_at_nc"]
    values = [aggregate[f"{stem}_mean"] for stem in stems]
    error_low = [aggregate[f"{stem}_mean"] - aggregate[f"{stem}_ci95_low"] for stem in stems]
    error_high = [aggregate[f"{stem}_ci95_high"] - aggregate[f"{stem}_mean"] for stem in stems]
    x = np.arange(len(metrics))
    ax.bar(x, values, color=[COLORS["contact"] if m in {"AUPRC", "AUROC"} else COLORS["light"] for m in metrics], edgecolor=COLORS["dark"], linewidth=0.5, zorder=2)
    ax.errorbar(x, values, yerr=np.array([error_low, error_high]), fmt="none", ecolor=COLORS["dark"], capsize=2, lw=0.8, zorder=3)
    ax.set_xticks(x, metrics, rotation=35, ha="right")
    ax.set_ylabel("Macro-average across complexes")
    ax.set_ylim(-0.03, 0.68)
    ax.text(0.98, 0.96, "mean and 95% CI; 3 grouped splits", transform=ax.transAxes, ha="right", va="top", fontsize=6)
    style_axis(ax)
    panel(ax, "a")

    ax = axes[0, 1]
    ax.plot(cutoff["cutoff_angstrom"], cutoff["macro_auprc"], marker="o", color=COLORS["struct"], label="Macro-AUPRC")
    ax.plot(cutoff["cutoff_angstrom"], cutoff["macro_density"], marker="s", color=COLORS["control"], label="Positive density")
    ax.set_xlabel("Contact cutoff (A)")
    ax.set_ylabel("Value")
    ax.set_xticks(cutoff["cutoff_angstrom"])
    ax.legend(fontsize=6)
    style_axis(ax)
    panel(ax, "b")

    ax = axes[1, 0]
    subset = strata[(strata["metric"] == "auprc") & strata["stratifier"].isin(["rna_length_quartile", "ligand_atoms_quartile", "density_quartile"])]
    offsets = {"rna_length_quartile": -0.18, "ligand_atoms_quartile": 0.0, "density_quartile": 0.18}
    colors = {"rna_length_quartile": COLORS["baseline"], "ligand_atoms_quartile": COLORS["contact"], "density_quartile": COLORS["struct"]}
    labels = {"rna_length_quartile": "RNA length", "ligand_atoms_quartile": "Ligand size", "density_quartile": "Contact density"}
    for key in offsets:
        part = subset[subset["stratifier"] == key].set_index("stratum").reindex(["Q1", "Q2", "Q3", "Q4"])
        positions = np.arange(4) + offsets[key]
        ax.plot(positions, part["mean"], marker="o", lw=1.1, color=colors[key], label=labels[key])
        ax.errorbar(
            positions,
            part["mean"],
            yerr=np.vstack((part["mean"] - part["ci95_low"], part["ci95_high"] - part["mean"])),
            fmt="none",
            color=colors[key],
            capsize=1.5,
            lw=0.6,
        )
    ax.set_xticks(np.arange(4), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Quartile (low to high)")
    ax.set_ylabel("Macro-AUPRC")
    ax.legend(fontsize=6, ncol=3, loc="upper center")
    style_axis(ax)
    panel(ax, "c")

    ax = axes[1, 1]
    enrichment = cutoff["macro_auprc"] / cutoff["macro_density"]
    ax.plot(cutoff["cutoff_angstrom"], enrichment, marker="o", color=COLORS["accent"], lw=1.4)
    ax.axhline(1, color=COLORS["control"], ls="--", lw=0.8)
    ax.set_xlabel("Contact cutoff (A)")
    ax.set_ylabel("AUPRC / positive density")
    ax.set_xticks(cutoff["cutoff_angstrom"])
    style_axis(ax)
    panel(ax, "d")
    fig.suptitle("Complex-level contact prediction is enriched but threshold performance remains weak", y=0.995, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_r2_5_contact_evaluation")


def model_color(name):
    if name == "DeepRSMA":
        return COLORS["baseline"]
    if "StructRSMA" in name:
        return COLORS["struct"]
    if "Shuffled" in name or "zero contact" in name or "without contact" in name:
        return COLORS["control"]
    if "descriptor" in name.lower() or "ridge" in name.lower():
        return COLORS["accent"]
    return COLORS["contact"]


def figure_ablation():
    aggregate = pd.read_csv(REV / "statistics/r2_complete/r2_aggregate_metrics.csv")
    bootstrap = pd.read_csv(REV / "statistics/r2_complete/r2_paired_bootstrap.csv")
    data = aggregate[aggregate["setting"] == "independent"].sort_values("rmse_mean")
    short_names = {
        "DeepRSMA": "DeepRSMA",
        "Contact500 pretraining without SCA": "Contact500 pretrain",
        "Global-deoverlap Contact270 pretraining without SCA": "Contact270 pretrain",
        "SCA without contact pretraining": "Random-head SCA",
        "StructRSMA: Contact500 pretraining plus SCA": "Contact500 + SCA",
        "StructRSMA: Contact270 pretraining plus SCA": "Contact270 + SCA",
        "Shuffled-contact pretraining plus SCA": "Shuffled contact + SCA",
        "Parameter-matched residual adapter with zero contact statistics": "Matched adapter, zero contact",
        "Sequence-SMILES ridge": "Sequence-SMILES ridge",
        "Ligand-only descriptors": "Ligand descriptors",
        "RNA+ligand descriptors": "RNA + ligand descriptors",
    }
    labels = [short_names.get(name, name) for name in data["model"]]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 4.4),
        sharey=True,
        gridspec_kw={"width_ratios": (1.1, 0.9, 1.2), "wspace": 0.16},
    )
    y = np.arange(len(data))
    for ax, metric, direction in ((axes[0], "rmse", "Lower is better"), (axes[1], "pcc", "Higher is better")):
        values = data[f"{metric}_mean"].to_numpy()
        low = values - data[f"{metric}_ci95_low"].to_numpy()
        high = data[f"{metric}_ci95_high"].to_numpy() - values
        for index, (value, lo, hi, name) in enumerate(zip(values, low, high, data["model"])):
            ax.errorbar(value, index, xerr=np.array([[lo], [hi]]), fmt="o", color=model_color(name), ecolor=model_color(name), capsize=2, ms=4)
        ax.set_yticks(y)
        if metric == "rmse":
            ax.set_yticklabels(labels)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(metric.upper())
        ax.set_title(direction, fontsize=7)
        ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    panel(axes[0], "a")
    panel(axes[1], "b")

    ax = axes[2]
    delta = bootstrap[(bootstrap["setting"] == "independent") & (bootstrap["metric"] == "rmse")].set_index("model")
    for index, name in enumerate(data["model"]):
        if name not in delta.index:
            continue
        row = delta.loc[name]
        ax.errorbar(row["delta"], index, xerr=np.array([[row["delta"] - row["ci95_low"]], [row["ci95_high"] - row["delta"]]]), fmt="o", color=model_color(name), capsize=2, ms=4)
    ax.axvline(0, color=COLORS["dark"], lw=0.8)
    ax.set_yticks(y)
    ax.tick_params(labelleft=False)
    ax.set_xlabel("Paired Delta RMSE vs DeepRSMA")
    ax.text(0.02, 0.02, "Left favors comparator", transform=ax.transAxes, fontsize=5.5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    panel(ax, "c")
    axes[0].invert_yaxis()
    fig.suptitle("Controlled independent-test ablations under scaffold-grouped validation", y=0.995, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_r2_3_affinity_ablation")


def figure_cold():
    aggregate = pd.read_csv(REV / "statistics/r2_complete/r2_aggregate_metrics.csv")
    bootstrap = pd.read_csv(REV / "statistics/r2_complete/r2_paired_bootstrap.csv")
    settings = ["cold_rna", "cold_scaffold", "double_cold"]
    preferred = [
        "DeepRSMA",
        "Contact500 pretraining without SCA",
        "StructRSMA: Contact500 pretraining plus SCA",
        "Sequence-SMILES ridge",
        "Ligand-only descriptors",
    ]
    data = aggregate[aggregate["setting"].isin(settings) & aggregate["model"].isin(preferred)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.42})
    x = np.arange(3)
    offsets = np.linspace(-0.25, 0.25, len(preferred))
    for model, offset in zip(preferred, offsets):
        part = data[data["model"] == model].set_index("setting").reindex(settings)
        axes[0].errorbar(x + offset, part["rmse_mean"], yerr=part["rmse_sd"], fmt="o", color=model_color(model), capsize=2, ms=3.5, label=model.replace(" pretraining without SCA", " pretrain"))
        axes[1].errorbar(x + offset, part["pcc_mean"], yerr=part["pcc_sd"], fmt="o", color=model_color(model), capsize=2, ms=3.5)
    for ax, ylabel in ((axes[0], "RMSE"), (axes[1], "PCC")):
        ax.set_xticks(x, ["Cold RNA", "Cold scaffold", "Double cold"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        style_axis(ax)
    axes[0].legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(1.05, -0.32), ncol=3)
    panel(axes[0], "a")
    panel(axes[1], "b")

    ax = axes[2]
    delta = bootstrap[(bootstrap["setting"].isin(settings)) & (bootstrap["metric"] == "rmse") & (bootstrap["model"].isin(["Contact500 pretraining without SCA", "StructRSMA: Contact500 pretraining plus SCA"]))]
    positions = []
    labels = []
    pos = 0
    for setting in settings:
        for model in ("Contact500 pretraining without SCA", "StructRSMA: Contact500 pretraining plus SCA"):
            row = delta[(delta["setting"] == setting) & (delta["model"] == model)]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(row["delta"], pos, xerr=np.array([[row["delta"] - row["ci95_low"]], [row["ci95_high"] - row["delta"]]]), fmt="o", color=model_color(model), capsize=2, ms=4)
            positions.append(pos)
            labels.append(f"{setting.replace('_', ' ')} | {'SCA' if 'StructRSMA' in model else 'pretrain'}")
            pos += 1
    ax.axvline(0, color=COLORS["dark"], lw=0.8)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Paired Delta RMSE vs DeepRSMA")
    ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    panel(ax, "c")
    fig.suptitle("Generalization across three leakage-resistant cold settings", y=1.01, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_r2_4_cold_generalization")


def figure_similarity():
    coverage = pd.read_csv(REV / "statistics/similarity_performance/rna_ligand_coverage_counts.csv")
    paired = pd.read_csv(REV / "statistics/similarity_performance/paired_error_improvement_with_similarity.csv")
    bins = pd.read_csv(REV / "statistics/similarity_performance/similarity_bin_performance.csv")
    distributions = pd.read_csv(REV / "statistics/similarity_performance/contact500_retained_excluded_features.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw={"hspace": 0.5, "wspace": 0.42})

    ax = axes[0, 0]
    part = coverage[coverage["source"] == "R-SIM_all"]
    pivot = part.pivot(index="rna_bin", columns="ligand_bin", values="n").reindex(index=["<0.5", "0.5-0.8", ">=0.8"], columns=["<0.4", "0.4-0.8", ">=0.8"]).fillna(0)
    image = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", color="white" if pivot.iloc[i, j] > pivot.to_numpy().max() * 0.55 else COLORS["dark"], fontsize=7)
    ax.set_xticks(range(3), pivot.columns)
    ax.set_yticks(range(3), pivot.index)
    ax.set_xlabel("Maximum ligand Tanimoto to Contact500")
    ax.set_ylabel("Maximum RNA identity to Contact500")
    panel(ax, "a")

    ax = axes[0, 1]
    retained = distributions[distributions["partition"] == "retained_global_deoverlap"]["contact_density"].dropna()
    excluded = distributions[distributions["partition"] == "excluded_by_global_deoverlap"]["contact_density"].dropna()
    box = ax.boxplot([retained, excluded], labels=[f"Retained\n(n={len(retained)})", f"Excluded\n(n={len(excluded)})"], patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], (COLORS["contact"], COLORS["accent"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Contact density")
    style_axis(ax)
    panel(ax, "b")

    ax = axes[1, 0]
    primary_model = "Contact500 pretraining without SCA"
    sca_model = "StructRSMA: Contact500 pretraining plus SCA"
    part = paired[(paired["setting"] == "independent") & (paired["model"] == primary_model)]
    averaged = part.groupby("entry_id").agg(ligand_similarity=("max_ligand_tanimoto", "first"), improvement=("absolute_error_improvement", "mean")).reset_index()
    ax.scatter(averaged["ligand_similarity"], averaged["improvement"], s=16, alpha=0.7, color=COLORS["struct"], edgecolor="white", linewidth=0.3)
    if len(averaged) >= 2:
        coefficient = np.polyfit(averaged["ligand_similarity"], averaged["improvement"], 1)
        xline = np.linspace(averaged["ligand_similarity"].min(), averaged["ligand_similarity"].max(), 100)
        ax.plot(xline, np.polyval(coefficient, xline), color=COLORS["dark"], lw=1)
    ax.axhline(0, color=COLORS["control"], lw=0.8)
    ax.set_xlabel("Maximum ligand Tanimoto to Contact500")
    ax.set_ylabel("Absolute-error improvement")
    panel(ax, "c")

    ax = axes[1, 1]
    part = bins[(bins["setting"] == "independent") & (bins["stratifier"] == "ligand_similarity_bin") & bins["model"].isin([primary_model, sca_model])]
    categories = ["<0.4", "0.4-0.8", ">=0.8"]
    for method, marker in ((primary_model, "o"), (sca_model, "s")):
        row = part[part["model"] == method].set_index("bin").reindex(categories)
        ax.plot(range(3), row["delta_rmse"], marker=marker, color=model_color(method), label="SCA" if "StructRSMA" in method else "Pretrain only")
    ax.axhline(0, color=COLORS["dark"], lw=0.8)
    ax.set_xticks(range(3), categories)
    ax.set_xlabel("Ligand-similarity bin")
    ax.set_ylabel("Delta RMSE vs DeepRSMA")
    ax.legend(fontsize=6)
    style_axis(ax)
    panel(ax, "d")
    fig.suptitle("Structural transfer varies with pretraining-set coverage", y=0.995, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_r2_1_similarity_coverage")


def figure_attribution():
    summary = pd.read_csv(REV / "statistics/r2_attribution/permutation_summary.csv")
    order = [
        "rna_representations", "ligand_representations", "all_contact_statistics",
        "density", "maxprob", "rnafocus", "atomfocus",
        "rna_sequence_view", "rna_graph_view", "molecule_sequence_view", "molecule_graph_view",
    ]
    summary = summary.set_index("perturbation").reindex(order).dropna(how="all").reset_index()
    labels = [value.replace("_", " ") for value in summary["perturbation"]]
    y = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"wspace": 0.58})
    for ax, metric, xlabel in (
        (axes[0], "delta_pcc_drop", "PCC drop after permutation"),
        (axes[1], "delta_rmse_increase", "RMSE increase after permutation"),
    ):
        center = summary[f"{metric}_median"].to_numpy()
        low = center - summary[f"{metric}_ci95_low"].to_numpy()
        high = summary[f"{metric}_ci95_high"].to_numpy() - center
        colors = [COLORS["struct"] if "contact" in name or name in {"density", "maxprob", "rnafocus", "atomfocus"} else COLORS["contact"] for name in summary["perturbation"]]
        for index, (value, lo, hi, color) in enumerate(zip(center, low, high, colors)):
            ax.errorbar(value, index, xerr=np.array([[lo], [hi]]), fmt="o", color=color, capsize=2, ms=4)
        ax.axvline(0, color=COLORS["dark"], lw=0.8)
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.suptitle("Representation-level permutation attribution of locked StructRSMA", y=1.01, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_r2_6_permutation_attribution")


def main():
    args = parse_args()
    functions = {
        "curation": figure_curation,
        "contact": figure_contact,
        "similarity": figure_similarity,
        "ablation": figure_ablation,
        "cold": figure_cold,
        "attribution": figure_attribution,
    }
    for name in args.figures:
        functions[name]()
        print(f"generated {name}")


if __name__ == "__main__":
    main()
