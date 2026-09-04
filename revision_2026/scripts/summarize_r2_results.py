import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr, t
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_2026"
OUT = REV / "statistics/r2_complete"


MODEL_NAMES = {
    "sequence_smiles": "Sequence-SMILES ridge",
    "ligand_descriptor": "Ligand-only descriptors",
    "rna_ligand_descriptor": "RNA+ligand descriptors",
}


def normalize_independent_id(value):
    value = str(value)
    sample = re.match(r"sample_(\d+)", value)
    if sample:
        return f"independent_{int(sample.group(1)):03d}"
    independent = re.match(r"independent_(\d+)", value)
    if independent:
        return f"independent_{int(independent.group(1)):03d}"
    return value


def safe_corr(function, labels, predictions):
    if len(labels) < 2 or np.std(labels) == 0 or np.std(predictions) == 0:
        return np.nan
    value = function(labels, predictions)[0]
    return float(value) if np.isfinite(value) else np.nan


def metric_dict(labels, predictions):
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    return {
        "pcc": safe_corr(pearsonr, labels, predictions),
        "scc": safe_corr(spearmanr, labels, predictions),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def add_run(runs, path, setting, split, model=None):
    summary_path = path / "summary.json"
    predictions_path = path / "test_predictions.csv"
    if not summary_path.exists() or not predictions_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    label = model or summary.get("model_label", "unknown")
    label = MODEL_NAMES.get(label, label)
    prediction = pd.read_csv(predictions_path, dtype={"entry_id": str})
    if setting == "independent":
        prediction["entry_id"] = prediction["entry_id"].map(normalize_independent_id)
    prediction["setting"] = setting
    prediction["split"] = split
    prediction["model"] = label
    prediction["run_seed"] = int(summary.get("seed", 0))
    prediction["source_dir"] = str(path.resolve())
    runs.append((summary, prediction))


def discover_runs():
    runs = []
    ablation = REV / "results/r2_independent_ablation"
    for variant in sorted(ablation.glob("*")):
        for seed_dir in sorted(variant.glob("seed*")):
            add_run(runs, seed_dir, "independent", seed_dir.name)

    baseline = REV / "results/r2_classical_baselines"
    for path in sorted(baseline.glob("independent_*")):
        match = re.match(r"independent_(.+)_seed(\d+)$", path.name)
        if match:
            add_run(runs, path, "independent", f"seed{match.group(2)}")

    cold_root = REV / "results/cold"
    for setting in ("cold_rna", "cold_scaffold", "double_cold"):
        for split_dir in sorted((cold_root / setting).glob("split*")):
            add_run(runs, split_dir / "deeprsma", setting, split_dir.name, "DeepRSMA")
            add_run(
                runs,
                split_dir / "contact500",
                setting,
                split_dir.name,
                "Contact500 pretraining without SCA",
            )

    sca_root = REV / "results/r2_cold_structrsma"
    for setting in ("cold_rna", "cold_scaffold", "double_cold"):
        for split_dir in sorted((sca_root / setting).glob("split*")):
            add_run(runs, split_dir, setting, split_dir.name)

    for path in sorted(item for item in baseline.iterdir() if item.is_dir()):
        match = re.match(r"(cold_rna|cold_scaffold|double_cold)_(.+)_split(\d+)$", path.name)
        if match:
            add_run(runs, path, match.group(1), f"split{match.group(3)}")
    return runs


def aggregate(runs):
    rows = []
    predictions = []
    for summary, prediction in runs:
        metrics = summary["test_metrics"]
        row = {
            "setting": prediction["setting"].iat[0],
            "split": prediction["split"].iat[0],
            "model": prediction["model"].iat[0],
            "run_seed": prediction["run_seed"].iat[0],
            "n": metrics["n"],
            "selected_epoch": summary.get("selected_epoch"),
            "gradient_bearing_parameter_count": summary.get("gradient_bearing_parameter_count"),
            **{metric: metrics[metric] for metric in ("pcc", "scc", "rmse", "mae")},
        }
        rows.append(row)
        predictions.append(prediction)
    runs_frame = pd.DataFrame(rows)
    aggregate_rows = []
    for (setting, model), group in runs_frame.groupby(["setting", "model"]):
        result = {"setting": setting, "model": model, "runs": len(group)}
        for metric in ("pcc", "scc", "rmse", "mae"):
            values = group[metric].to_numpy(float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            margin = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else np.nan
            result.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_sd": sd,
                    f"{metric}_ci95_low": mean - margin if np.isfinite(margin) else np.nan,
                    f"{metric}_ci95_high": mean + margin if np.isfinite(margin) else np.nan,
                }
            )
        aggregate_rows.append(result)
    return runs_frame, pd.DataFrame(aggregate_rows), pd.concat(predictions, ignore_index=True)


def paired_bootstrap(predictions, baseline="DeepRSMA", iterations=2000, seed=2026):
    rng = np.random.default_rng(seed)
    result_rows = []
    for setting in sorted(predictions["setting"].unique()):
        subset = predictions[predictions["setting"] == setting]
        for model in sorted(subset["model"].unique()):
            if model == baseline:
                continue
            units = []
            # Independent-test variants share a split and are paired by training seed.
            # Cold experiments are paired by the released split membership; their
            # internal training seeds need not equal the split-generation seed.
            unit_columns = ["run_seed"] if setting == "independent" else ["split"]
            for _, part in subset.groupby(unit_columns):
                base = part[part["model"] == baseline]
                other = part[part["model"] == model]
                if base.empty or other.empty:
                    continue
                merged = base[["entry_id", "label", "prediction"]].merge(
                    other[["entry_id", "label", "prediction"]],
                    on="entry_id",
                    suffixes=("_base", "_other"),
                )
                if len(merged) >= 2:
                    units.append(merged)
            if not units:
                continue
            point = {metric: [] for metric in ("pcc", "scc", "rmse", "mae")}
            for unit in units:
                base_metrics = metric_dict(unit["label_base"], unit["prediction_base"])
                other_metrics = metric_dict(unit["label_other"], unit["prediction_other"])
                for metric in point:
                    point[metric].append(other_metrics[metric] - base_metrics[metric])
            samples = {metric: [] for metric in point}
            for _ in range(iterations):
                selected_units = rng.integers(0, len(units), size=len(units))
                values = {metric: [] for metric in point}
                for unit_index in selected_units:
                    unit = units[int(unit_index)]
                    sampled = rng.integers(0, len(unit), size=len(unit))
                    base_metrics = metric_dict(
                        unit["label_base"].to_numpy()[sampled],
                        unit["prediction_base"].to_numpy()[sampled],
                    )
                    other_metrics = metric_dict(
                        unit["label_other"].to_numpy()[sampled],
                        unit["prediction_other"].to_numpy()[sampled],
                    )
                    for metric in values:
                        delta = other_metrics[metric] - base_metrics[metric]
                        if np.isfinite(delta):
                            values[metric].append(delta)
                for metric in samples:
                    if values[metric]:
                        samples[metric].append(float(np.mean(values[metric])))
            for metric in point:
                boot = np.asarray(samples[metric], dtype=float)
                result_rows.append(
                    {
                        "setting": setting,
                        "model": model,
                        "baseline": baseline,
                        "metric": metric,
                        "paired_units": len(units),
                        "delta": float(np.nanmean(point[metric])),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "empirical_two_sided_p": float(
                            min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
                        ),
                    }
                )
    return pd.DataFrame(result_rows)


def per_target(predictions):
    rsim = pd.read_csv(ROOT / "data/RSM_data/All_sf_dataset_v1.csv", sep="\t")
    rsim["entry_id"] = rsim["Entry_ID"].map(lambda value: str(int(value)))
    lookup = rsim[["entry_id", "Target_RNA_ID", "Target_RNA_name"]]
    cold = predictions[predictions["setting"] != "independent"].merge(
        lookup, on="entry_id", how="left"
    )
    rows = []
    fisher_rows = []
    for keys, group in cold.groupby(["setting", "split", "model"]):
        eligible = []
        for target, target_frame in group.groupby("Target_RNA_ID"):
            values = metric_dict(target_frame["label"], target_frame["prediction"])
            rows.append(
                {
                    "setting": keys[0],
                    "split": keys[1],
                    "model": keys[2],
                    "target": target,
                    "target_name": target_frame["Target_RNA_name"].iat[0],
                    "n": len(target_frame),
                    **values,
                }
            )
            if len(target_frame) >= 5 and np.isfinite(values["pcc"]) and abs(values["pcc"]) < 1:
                eligible.append((values["pcc"], len(target_frame)))
        if eligible:
            weights = np.asarray([n - 3 for _, n in eligible], dtype=float)
            combined = float(
                np.tanh(np.average([np.arctanh(value) for value, _ in eligible], weights=weights))
            )
        else:
            combined = np.nan
        fisher_rows.append(
            {
                "setting": keys[0],
                "split": keys[1],
                "model": keys[2],
                "eligible_targets_n_ge_5": len(eligible),
                "fisher_z_aggregated_pcc": combined,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(fisher_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize affinity evaluation experiments.")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    run_metrics, aggregate_metrics, predictions = aggregate(runs)
    bootstrap = paired_bootstrap(
        predictions,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    target, fisher = per_target(predictions)
    run_metrics.to_csv(OUT / "r2_run_metrics.csv", index=False)
    aggregate_metrics.to_csv(OUT / "r2_aggregate_metrics.csv", index=False)
    predictions.to_csv(OUT / "r2_all_test_predictions.csv", index=False)
    bootstrap.to_csv(OUT / "r2_paired_bootstrap.csv", index=False)
    target.to_csv(OUT / "r2_per_target_metrics.csv", index=False)
    fisher.to_csv(OUT / "r2_fisher_z_pcc.csv", index=False)
    (OUT / "r2_statistical_config.json").write_text(
        json.dumps(
            {
                "training_or_split_repeats": 3,
                "bootstrap_type": "paired hierarchical resampling of matched runs and test examples",
                "bootstrap_iterations": args.bootstrap_iterations,
                "bootstrap_seed": args.bootstrap_seed,
                "aggregate_ci": "two-sided t interval across three repeated runs/splits",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
