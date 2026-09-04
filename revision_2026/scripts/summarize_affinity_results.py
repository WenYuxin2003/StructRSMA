import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


MODEL_DIRS = {
    "DeepRSMA": "deeprsma",
    "Contact500": "contact500",
    "Contact270GlobalDeoverlap": "global_deoverlap",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize strict affinity runs and paired bootstrap differences.")
    parser.add_argument("--revision-dir", type=Path, required=True)
    parser.add_argument("--rsim-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def safe_corr(function, labels, predictions):
    if len(labels) < 2 or np.std(labels) == 0 or np.std(predictions) == 0:
        return None
    value = function(labels, predictions)[0]
    return float(value) if np.isfinite(value) else None


def metrics(frame):
    labels = frame["label"].to_numpy(float)
    predictions = frame["prediction"].to_numpy(float)
    return {
        "n": len(frame),
        "pcc": safe_corr(pearsonr, labels, predictions),
        "scc": safe_corr(spearmanr, labels, predictions),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def load_run(path, setting, split, model):
    summary_path = path / "summary.json"
    prediction_path = path / "test_predictions.csv"
    if not summary_path.exists() or not prediction_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path, dtype={"entry_id": str})
    predictions["setting"] = setting
    predictions["split"] = str(split)
    predictions["model"] = model
    predictions["run_seed"] = summary["seed"]
    return summary, predictions


def discover_runs(revision_dir):
    runs = []
    independent_root = revision_dir / "results/independent"
    for model, directory in MODEL_DIRS.items():
        for seed_dir in sorted((independent_root / directory).glob("seed*")):
            loaded = load_run(seed_dir, "independent", seed_dir.name, model)
            if loaded:
                runs.append(loaded)

    cold_root = revision_dir / "results/cold"
    for setting_dir in sorted(cold_root.glob("*")):
        if not setting_dir.is_dir():
            continue
        for split_dir in sorted(setting_dir.glob("split*")):
            for model, directory in MODEL_DIRS.items():
                loaded = load_run(split_dir / directory, setting_dir.name, split_dir.name, model)
                if loaded:
                    runs.append(loaded)
    return runs


def aggregate_run_metrics(runs):
    rows = []
    for summary, _ in runs:
        if "global_deoverlap" in summary["run_name"]:
            model = "Contact270GlobalDeoverlap"
        elif "contact500" in summary["run_name"]:
            model = "Contact500"
        else:
            model = "DeepRSMA"
        row = {
            "setting": "independent" if "independent" in summary["run_name"] else summary["run_name"].split("_split")[0],
            "split_seed": summary["split_seed"],
            "run_seed": summary["seed"],
            "model": model,
            "selected_epoch": summary["selected_epoch"],
            **summary["test_metrics"],
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    aggregates = []
    for (setting, model), group in frame.groupby(["setting", "model"]):
        row = {"setting": setting, "model": model, "runs": len(group)}
        for metric in ("pcc", "scc", "rmse", "mae"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else None
        aggregates.append(row)
    return frame, pd.DataFrame(aggregates)


def array_corr(x, y):
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denominator == 0:
        return None
    return float(np.sum(x_centered * y_centered) / denominator)


def array_metrics(labels, predictions):
    pcc = array_corr(labels, predictions)
    scc = array_corr(rankdata(labels), rankdata(predictions))
    residual = predictions - labels
    return {
        "pcc": pcc,
        "scc": scc,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def paired_metric_delta_arrays(labels, base_predictions, contact_predictions, sampled_indices):
    sampled_labels = labels[sampled_indices]
    base_metrics = array_metrics(sampled_labels, base_predictions[sampled_indices])
    contact_metrics = array_metrics(sampled_labels, contact_predictions[sampled_indices])
    return {
        metric: contact_metrics[metric] - base_metrics[metric]
        for metric in ("pcc", "scc", "rmse", "mae")
        if contact_metrics[metric] is not None and base_metrics[metric] is not None
    }


def paired_bootstrap(predictions, iterations, seed):
    rng = np.random.default_rng(seed)
    results = {}
    comparisons = [
        ("Contact500", "DeepRSMA"),
        ("Contact270GlobalDeoverlap", "DeepRSMA"),
        ("Contact270GlobalDeoverlap", "Contact500"),
    ]
    for comparator, baseline in comparisons:
        for setting in sorted(predictions["setting"].unique()):
            setting_frame = predictions[predictions["setting"] == setting]
            split_values = sorted(setting_frame["split"].unique())
            paired = []
            for split in split_values:
                split_frame = setting_frame[setting_frame["split"] == split]
                base = split_frame[split_frame["model"] == baseline].reset_index(drop=True)
                contact = split_frame[split_frame["model"] == comparator].reset_index(drop=True)
                if len(base) == 0 or len(contact) == 0:
                    continue
                merged = base[["entry_id", "label", "prediction"]].merge(
                    contact[["entry_id", "label", "prediction"]],
                    on="entry_id",
                    suffixes=("_base", "_contact"),
                )
                if len(merged) == 0:
                    continue
                paired.append(
                    (
                        merged["label_base"].to_numpy(float),
                        merged["prediction_base"].to_numpy(float),
                        merged["prediction_contact"].to_numpy(float),
                    )
                )
            if not paired:
                continue

            samples = {metric: [] for metric in ("pcc", "scc", "rmse", "mae")}
            for _ in range(iterations):
                per_split = {metric: [] for metric in samples}
                for labels, base_predictions, contact_predictions in paired:
                    indices = rng.integers(0, len(labels), size=len(labels))
                    delta = paired_metric_delta_arrays(
                        labels, base_predictions, contact_predictions, indices
                    )
                    for metric, value in delta.items():
                        per_split[metric].append(value)
                for metric, values in per_split.items():
                    if values:
                        samples[metric].append(float(np.mean(values)))

            point = {metric: [] for metric in samples}
            for labels, base_predictions, contact_predictions in paired:
                delta = paired_metric_delta_arrays(
                    labels,
                    base_predictions,
                    contact_predictions,
                    np.arange(len(labels)),
                )
                for metric, value in delta.items():
                    point[metric].append(value)
            results[f"{setting}:{comparator}_vs_{baseline}"] = {
                "definition": f"{comparator} minus {baseline}; negative RMSE/MAE favors {comparator}",
                "paired_splits": len(paired),
                "metrics": {
                    metric: {
                        "delta": float(np.mean(point[metric])) if point[metric] else None,
                        "ci95_low": float(np.quantile(samples[metric], 0.025)) if samples[metric] else None,
                        "ci95_high": float(np.quantile(samples[metric], 0.975)) if samples[metric] else None,
                    }
                    for metric in samples
                },
            }
    return results


def per_target_analysis(predictions, rsim):
    lookup = rsim.copy()
    lookup["entry_id"] = lookup["Entry_ID"].map(lambda value: str(int(value)) if float(value).is_integer() else str(value))
    lookup = lookup[["entry_id", "Target_RNA_ID"]]
    cold = predictions[predictions["setting"] != "independent"].merge(lookup, on="entry_id", how="left")
    rows = []
    fisher_rows = []
    for (setting, split, model), group in cold.groupby(["setting", "split", "model"]):
        target_pcc = []
        for target, target_frame in group.groupby("Target_RNA_ID"):
            result = metrics(target_frame)
            rows.append(
                {
                    "setting": setting,
                    "split": split,
                    "model": model,
                    "target": target,
                    **result,
                }
            )
            if result["n"] >= 5 and result["pcc"] is not None and abs(result["pcc"]) < 1:
                target_pcc.append((result["pcc"], result["n"]))
        if target_pcc:
            weights = np.asarray([n - 3 for _, n in target_pcc], dtype=float)
            z_values = np.asarray([np.arctanh(value) for value, _ in target_pcc])
            fisher_pcc = float(np.tanh(np.average(z_values, weights=weights)))
        else:
            fisher_pcc = None
        fisher_rows.append(
            {
                "setting": setting,
                "split": split,
                "model": model,
                "eligible_targets_n_ge_5": len(target_pcc),
                "fisher_z_aggregated_pcc": fisher_pcc,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(fisher_rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.revision_dir)
    run_metrics, aggregates = aggregate_run_metrics(runs)
    predictions = pd.concat([prediction for _, prediction in runs], ignore_index=True)
    bootstrap = paired_bootstrap(predictions, args.bootstrap, args.seed)
    rsim = pd.read_csv(args.rsim_csv, sep="\t")
    per_target, fisher = per_target_analysis(predictions, rsim)

    run_metrics.to_csv(args.output_dir / "affinity_run_metrics.csv", index=False)
    aggregates.to_csv(args.output_dir / "affinity_aggregate.csv", index=False)
    predictions.to_csv(args.output_dir / "all_test_predictions.csv", index=False)
    per_target.to_csv(args.output_dir / "per_target_metrics.csv", index=False)
    fisher.to_csv(args.output_dir / "fisher_z_pcc.csv", index=False)
    (args.output_dir / "paired_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(aggregates.to_string(index=False))
    print(json.dumps(bootstrap, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
