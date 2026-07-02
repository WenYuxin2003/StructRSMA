import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from summarize_independent_log import parse_log


METHOD_LOGS = {
    "Original": [
        "runs/independent_full_gpu.log",
        "runs/independent_full_gpu_seed2.log",
        "runs/independent_full_gpu_seed3.log",
    ],
    "Contact100": [
        "runs/independent_contact_rna_only_100_seed1_skipaff.log",
        "runs/independent_contact_rna_only_100_seed2_skipaff.log",
        "runs/independent_contact_rna_only_100_seed3_skipaff_full.log",
    ],
    "Contact500": [
        "runs/independent_contact_rna_only_500_seed1_skipaff.log",
        "runs/independent_contact_rna_only_500_seed2_skipaff.log",
        "runs/independent_contact_rna_only_500_seed3_skipaff.log",
    ],
}

METHOD_COLORS = {
    "Original": "#6B7280",
    "Contact100": "#3B82F6",
    "Contact500": "#EF4444",
    "ShuffledContact500": "#9CA3AF",
}

DISPLAY_LABELS = {
    "Original": "Reproduced\nDeepRSMA",
    "Contact100": "Contact100\npretraining only",
    "Contact500": "Contact500\npretraining only",
    "ShuffledContact500": "Shuffled\nContact500",
}

SHUFFLE_CONTROL_LOGS = {
    "Original": METHOD_LOGS["Original"],
    "ShuffledContact500": [
        "runs/independent_contact_rna_only_500_shuffle_seed1_skipaff.log",
        "runs/independent_contact_rna_only_500_shuffle_seed2_skipaff.log",
        "runs/independent_contact_rna_only_500_shuffle_seed3_skipaff.log",
    ],
    "Contact500": METHOD_LOGS["Contact500"],
}

PRETRAIN_PATTERN = re.compile(
    r"epoch:\s+(\d+).*?"
    r"train_loss:\s+([-+0-9.eE]+).*?"
    r"train_topk_precision:\s+([-+0-9.eE]+).*?"
    r"train_density:\s+([-+0-9.eE]+).*?"
    r"val_loss:\s+([-+0-9.eE]+).*?"
    r"val_topk_precision:\s+([-+0-9.eE]+).*?"
    r"val_density:\s+([-+0-9.eE]+)"
)


def ensure_out_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, out_dir, name):
    out_dir = Path(out_dir)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def best_row(rows, selection):
    if selection == "best_pcc":
        return max(rows, key=lambda row: row["pcc"])
    if selection == "best_scc":
        return max(rows, key=lambda row: row["scc"])
    if selection == "best_rmse":
        return min(rows, key=lambda row: row["rmse"])
    raise ValueError(selection)


def summarize_method(logs, selection):
    rows = [best_row(parse_log(log), selection) for log in logs]
    metrics = {}
    for metric in ("pcc", "scc", "rmse"):
        values = [row[metric] for row in rows]
        metrics[metric] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return metrics


def performance_figure(out_dir, selection):
    methods = list(METHOD_LOGS)
    metric_labels = [("pcc", "PCC", "higher is better"), ("scc", "SCC", "higher is better"), ("rmse", "RMSE", "lower is better")]
    summaries = {method: summarize_method(logs, selection) for method, logs in METHOD_LOGS.items()}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (metric, label, subtitle) in zip(axes, metric_labels):
        xs = np.arange(len(methods))
        means = [summaries[method][metric]["mean"] for method in methods]
        errs = [summaries[method][metric]["std"] for method in methods]
        colors = [METHOD_COLORS[method] for method in methods]
        ax.bar(xs, means, yerr=errs, capsize=4, color=colors, edgecolor="#111827", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([DISPLAY_LABELS[method] for method in methods], rotation=20, ha="right")
        ax.set_title(f"{label}\n{subtitle}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for x, value in zip(xs, means):
            ax.text(x, value + (0.015 if metric != "rmse" else 0.025), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Local reproduction protocol ({selection.replace('_', '-')}, mean +/- std over 3 seeds)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, f"fig_performance_{selection}")


def scaling_figure(out_dir):
    selections = "best_pcc"
    points = [
        ("Original", 0),
        ("Contact100", 94),
        ("Contact500", 484),
    ]
    summaries = {method: summarize_method(METHOD_LOGS[method], selections) for method, _ in points}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, metric, label in zip(axes, ("pcc", "scc", "rmse"), ("PCC", "SCC", "RMSE")):
        xs = [count for _, count in points]
        ys = [summaries[method][metric]["mean"] for method, _ in points]
        errs = [summaries[method][metric]["std"] for method, _ in points]
        ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2.0, capsize=4, color="#111827")
        ax.scatter(xs, ys, s=58, c=[METHOD_COLORS[method] for method, _ in points], edgecolors="#111827", zorder=3)
        ax.set_xlabel("PDB contact pretraining samples")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        for x, y, (method, _) in zip(xs, ys, points):
            ax.text(x, y, f" {DISPLAY_LABELS[method].replace(chr(10), ' ')}\n {y:.3f}", fontsize=8, va="bottom")
    fig.suptitle("Contact-supervised pretraining under the local protocol", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_contact_data_scaling")


def downstream_shuffle_control_figure(out_dir):
    selection = "best_rmse"
    methods = list(SHUFFLE_CONTROL_LOGS)
    summaries = {method: summarize_method(logs, selection) for method, logs in SHUFFLE_CONTROL_LOGS.items()}
    metric_labels = [
        ("pcc", "PCC", "higher is better"),
        ("scc", "SCC", "higher is better"),
        ("rmse", "RMSE", "lower is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (metric, label, subtitle) in zip(axes, metric_labels):
        xs = np.arange(len(methods))
        means = [summaries[method][metric]["mean"] for method in methods]
        errs = [summaries[method][metric]["std"] for method in methods]
        colors = [METHOD_COLORS[method] for method in methods]
        ax.bar(xs, means, yerr=errs, capsize=4, color=colors, edgecolor="#111827", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Reproduced\nDeepRSMA", "Shuffled\nContact500", "True\nContact500"], rotation=0)
        ax.set_title(f"{label}\n{subtitle}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for x, value in zip(xs, means):
            ax.text(x, value + (0.015 if metric != "rmse" else 0.025), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Downstream shuffled-contact control (best-RMSE view)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_downstream_shuffle_control")


def load_sca_summary():
    rows = {}
    with open("docs/tables/structrsma_sca_summary.csv", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[(row["method"], row["selection"])] = {
                "pcc": float(row["pcc_mean"]),
                "pcc_std": float(row["pcc_sd"]),
                "scc": float(row["scc_mean"]),
                "scc_std": float(row["scc_sd"]),
                "rmse": float(row["rmse_mean"]),
                "rmse_std": float(row["rmse_sd"]),
            }
    return rows


def sca_adapter_figure(out_dir):
    rows = load_sca_summary()
    methods = [
        ("Contact500\npretraining only", rows[("Contact500SkipAff", "initial_best_pcc_checkpoint")], "#EF4444"),
        ("StructRSMA\nbest-PCC", rows[("StructRSMA", "best_pcc")], "#7C3AED"),
        ("StructRSMA\nbest-RMSE", rows[("StructRSMA", "best_rmse")], "#8B5CF6"),
    ]
    metric_labels = [
        ("pcc", "PCC", "higher is better"),
        ("scc", "SCC", "higher is better"),
        ("rmse", "RMSE", "lower is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (metric, label, subtitle) in zip(axes, metric_labels):
        xs = np.arange(len(methods))
        means = [row[metric] for _, row, _ in methods]
        errs = [row[f"{metric}_std"] for _, row, _ in methods]
        colors = [color for _, _, color in methods]
        ax.bar(xs, means, yerr=errs, capsize=4, color=colors, edgecolor="#111827", linewidth=0.8, alpha=0.88)
        ax.set_xticks(xs)
        ax.set_xticklabels([name for name, _, _ in methods])
        ax.set_title(f"{label}\n{subtitle}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for x, value in zip(xs, means):
            ax.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Metric value")
    fig.suptitle("Structural Contact Adapter as residual calibration under the local protocol", fontsize=12)
    fig.text(0.5, 0.01, "Mean +/- SD over three seeds", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_figure(fig, out_dir, "fig_sca_adapter_results")


def load_contact_dataset_stats(data_dir):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.pt"))
    ligand_names = set()
    rna_lens = []
    atom_counts = []
    positives = []
    pairs = []
    for path in files:
        item = torch.load(path, weights_only=False)
        contact = item["contact_map"].float()
        meta = item.get("meta", {})
        ligand_names.add(meta.get("ligand_resname", ""))
        rna_lens.append(contact.size(0))
        atom_counts.append(contact.size(1))
        positives.append(float(contact.sum().item()))
        pairs.append(float(contact.numel()))
    return {
        "samples": len(files),
        "ligands": len([name for name in ligand_names if name]),
        "mean_rna_len": float(np.mean(rna_lens)),
        "mean_atoms": float(np.mean(atom_counts)),
        "positive_contacts": float(np.sum(positives)),
        "total_pairs": float(np.sum(pairs)),
        "density": float(np.sum(positives) / max(np.sum(pairs), 1.0)),
    }


def dataset_summary_figure(out_dir):
    stats = {
        "Contact100": load_contact_dataset_stats("dataset/pdb_contact_rna_only_100"),
        "Contact500": load_contact_dataset_stats("dataset/pdb_contact_rna_only_500"),
    }
    panels = [
        ("samples", "Contact samples", False),
        ("ligands", "Ligand types", False),
        ("positive_contacts", "Positive contacts", True),
        ("total_pairs", "Nucleotide-atom pairs", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for ax, (key, title, log_scale) in zip(axes.flatten(), panels):
        methods = list(stats)
        values = [stats[method][key] for method in methods]
        ax.bar(methods, values, color=[METHOD_COLORS[m] for m in methods], edgecolor="#111827", linewidth=0.8)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values):
            label = f"{int(value):,}" if value >= 10 else f"{value:.3f}"
            ax.text(i, value, label, ha="center", va="bottom", fontsize=8)
    fig.suptitle("PDB-derived contact pretraining data", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_contact_dataset_summary")


def parse_pretrain_log(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = PRETRAIN_PATTERN.search(line)
        if not match:
            continue
        rows.append(
            {
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "train_topk": float(match.group(3)),
                "train_density": float(match.group(4)),
                "val_loss": float(match.group(5)),
                "val_topk": float(match.group(6)),
                "val_density": float(match.group(7)),
            }
        )
    return rows


def contact_pretrain_curve(out_dir):
    rows = parse_pretrain_log("runs/contact_pretrain_rna_only_500.log")
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [row["train_topk"] for row in rows], marker="o", label="Train top-k precision", color="#3B82F6")
    ax.plot(epochs, [row["val_topk"] for row in rows], marker="o", label="Val top-k precision", color="#EF4444")
    if rows:
        ax.axhline(rows[-1]["val_density"], color="#6B7280", linestyle="--", label="Val contact density")
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Top-k precision")
    ax.set_title("Contact prediction improves during structural pretraining")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_contact_pretrain_curve")


def load_contact_metrics(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["metrics"]


def contact_shuffle_control_figure(out_dir):
    real = load_contact_metrics("docs/figures/contact_checkpoint_metrics_500.json")
    shuffled = load_contact_metrics("docs/figures/shuffle_control/contact_checkpoint_metrics_500.json")

    metrics = [
        ("topk_precision_mean", "Mean top-k\nprecision"),
        ("auprc", "AUPRC"),
        ("auroc", "AUROC"),
    ]
    labels = ["True contact\npretraining", "Shuffled contact\npretraining"]
    colors = ["#EF4444", "#9CA3AF"]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(metrics))
    width = 0.34
    real_values = [real[key] for key, _ in metrics]
    shuffled_values = [shuffled[key] for key, _ in metrics]
    ax.bar(x - width / 2, real_values, width, label=labels[0], color=colors[0], edgecolor="#111827", linewidth=0.8)
    ax.bar(x + width / 2, shuffled_values, width, label=labels[1], color=colors[1], edgecolor="#111827", linewidth=0.8)
    ax.axhline(real["density"], color="#6B7280", linestyle="--", linewidth=1.0, label=f"Contact density={real['density']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score on true contact validation split")
    ax.set_title("Shuffled labels remove true contact enrichment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for offset, values in ((-width / 2, real_values), (width / 2, shuffled_values)):
        for xi, value in zip(x, values):
            ax.text(xi + offset, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_contact_shuffle_control")


def method_overview_figure(out_dir):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")

    boxes = [
        (0.02, 0.55, 0.18, 0.28, "PDB RNA-ligand\ncomplexes"),
        (0.25, 0.55, 0.18, 0.28, "Distance labels\n< 4 A contacts"),
        (0.48, 0.55, 0.18, 0.28, "Contact-supervised\nDeepRSMA pretraining"),
        (0.72, 0.55, 0.24, 0.28, "Contact-pretrained\nbackbone"),
        (0.25, 0.10, 0.18, 0.28, "R-SIM affinity data\nRNA + SMILES + pKd"),
        (0.48, 0.10, 0.18, 0.28, "pKd fine-tuning\nMSE loss"),
        (0.72, 0.10, 0.24, 0.28, "Affinity prediction\nPCC / SCC / RMSE"),
    ]

    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.1,
            edgecolor="#111827",
            facecolor="#F9FAFB",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.20, 0.69), (0.25, 0.69)),
        ((0.43, 0.69), (0.48, 0.69)),
        ((0.66, 0.69), (0.72, 0.69)),
        ((0.84, 0.55), (0.84, 0.38)),
        ((0.43, 0.24), (0.48, 0.24)),
        ((0.66, 0.24), (0.72, 0.24)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.2, color="#111827"))

    ax.text(0.02, 0.94, "A. Structure-derived contact pretraining", fontsize=11, weight="bold")
    ax.text(0.25, 0.43, "B. Transfer to affinity prediction", fontsize=11, weight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_method_overview")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready DeepRSMA contact-pretraining figures.")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    performance_figure(out_dir, "best_pcc")
    performance_figure(out_dir, "best_rmse")
    scaling_figure(out_dir)
    downstream_shuffle_control_figure(out_dir)
    sca_adapter_figure(out_dir)
    dataset_summary_figure(out_dir)
    contact_pretrain_curve(out_dir)
    contact_shuffle_control_figure(out_dir)
    method_overview_figure(out_dir)
    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
