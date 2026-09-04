import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = (
    "macro_auprc",
    "macro_auroc",
    "macro_mcc",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "macro_p_at_5",
    "macro_r_at_5",
    "macro_p_at_10",
    "macro_r_at_10",
    "macro_p_at_nc",
    "macro_density",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize strict complex-level contact experiments.")
    parser.add_argument("--revision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_summary(path):
    return json.loads(path.read_text(encoding="utf-8"))


def t_ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    mean = float(values.mean())
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean - half_width, mean + half_width


def summary_row(summary, cutoff, split):
    selected = summary["test_metrics_selected_threshold"]
    fixed = summary["test_metrics_fixed_0.5"]
    row = {
        "cutoff_angstrom": cutoff,
        "split": split,
        "selected_epoch": summary["selected_epoch"],
        "validation_selected_threshold": summary["validation_selected_probability_threshold"],
        "test_complexes": pd.read_csv(Path(summary["output_dir"]) / "test_complex_metrics.csv").shape[0]
        if "output_dir" in summary
        else None,
    }
    row.update(selected)
    row["fixed_0.5_mcc"] = fixed["macro_mcc"]
    row["fixed_0.5_f1"] = fixed["macro_f1"]
    density = selected["macro_density"]
    row["auprc_over_density"] = selected["macro_auprc"] / density if density else None
    row["p_at_5_over_density"] = selected["macro_p_at_5"] / density if density else None
    row["p_at_10_over_density"] = selected["macro_p_at_10"] / density if density else None
    row["p_at_nc_over_density"] = selected["macro_p_at_nc"] / density if density else None
    return row


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows = []
    grouped_root = args.revision_dir / "results/contact_grouped/cutoff40"
    for split_dir in sorted(grouped_root.glob("split*")):
        summary_path = split_dir / "summary.json"
        if summary_path.exists():
            summary = load_summary(summary_path)
            row = summary_row(summary, 4.0, split_dir.name)
            row["test_complexes"] = pd.read_csv(split_dir / "test_complex_metrics.csv").shape[0]
            grouped_rows.append(row)
    grouped = pd.DataFrame(grouped_rows)
    grouped.to_csv(args.output_dir / "contact_grouped_runs.csv", index=False)

    aggregate = {"cutoff_angstrom": 4.0, "splits": len(grouped)}
    for metric in METRICS + (
        "auprc_over_density",
        "p_at_5_over_density",
        "p_at_10_over_density",
        "p_at_nc_over_density",
        "fixed_0.5_mcc",
        "fixed_0.5_f1",
    ):
        aggregate[f"{metric}_mean"] = float(grouped[metric].mean())
        aggregate[f"{metric}_sd"] = float(grouped[metric].std(ddof=1)) if len(grouped) > 1 else None
        ci_low, ci_high = t_ci95(grouped[metric])
        aggregate[f"{metric}_ci95_low"] = ci_low
        aggregate[f"{metric}_ci95_high"] = ci_high
    pd.DataFrame([aggregate]).to_csv(args.output_dir / "contact_grouped_aggregate.csv", index=False)

    sensitivity_rows = []
    cutoff_locations = {
        3.5: args.revision_dir / "results/contact_cutoff_sensitivity/cutoff35/split2026",
        4.0: args.revision_dir / "results/contact_grouped/cutoff40/split2026",
        4.5: args.revision_dir / "results/contact_cutoff_sensitivity/cutoff45/split2026",
        5.0: args.revision_dir / "results/contact_cutoff_sensitivity/cutoff50/split2026",
    }
    for cutoff, run_dir in cutoff_locations.items():
        summary = load_summary(run_dir / "summary.json")
        row = summary_row(summary, cutoff, "split2026")
        row["test_complexes"] = pd.read_csv(run_dir / "test_complex_metrics.csv").shape[0]
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values("cutoff_angstrom")
    sensitivity.to_csv(args.output_dir / "contact_cutoff_sensitivity.csv", index=False)

    print(grouped.to_string(index=False))
    print(pd.DataFrame([aggregate]).to_string(index=False))
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
