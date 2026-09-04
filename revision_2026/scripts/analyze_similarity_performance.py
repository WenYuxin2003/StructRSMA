import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_2026"
PREDICTIONS = REV / "statistics/r2_complete/r2_all_test_predictions.csv"
OVERLAP = REV / "audit/overlap_v1/downstream_to_contact500_overlap.csv"
CONTACT_OVERLAP = REV / "audit/overlap_v1/contact500_to_downstream_overlap.csv"
CONTACT_MANIFEST = REV / "audit/contact500_v1/contact500_manifest.csv"
OUT = REV / "statistics/similarity_performance"


COMPARATORS = (
    "Contact500 pretraining without SCA",
    "Global-deoverlap Contact270 pretraining without SCA",
    "StructRSMA: Contact500 pretraining plus SCA",
    "StructRSMA: Contact270 pretraining plus SCA",
)


def normalize_overlap_id(value):
    value = str(value)
    match = re.match(r"independent_(\d+)", value)
    if match:
        return f"independent_{int(match.group(1)):03d}"
    return value


def metrics(frame, suffix):
    label = frame["label"].to_numpy(float)
    prediction = frame[f"prediction_{suffix}"].to_numpy(float)
    return {
        f"pcc_{suffix}": float(pearsonr(label, prediction)[0]) if len(frame) >= 3 else np.nan,
        f"scc_{suffix}": float(spearmanr(label, prediction)[0]) if len(frame) >= 3 else np.nan,
        f"rmse_{suffix}": float(np.sqrt(mean_squared_error(label, prediction))),
        f"mae_{suffix}": float(mean_absolute_error(label, prediction)),
    }


def cliffs_delta(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not len(left) or not len(right):
        return np.nan
    greater = sum(np.sum(value > right) for value in left)
    lower = sum(np.sum(value < right) for value in left)
    return float((greater - lower) / (len(left) * len(right)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS, dtype={"entry_id": str})
    overlap = pd.read_csv(OVERLAP, dtype={"downstream_id": str})
    overlap["entry_id"] = overlap["downstream_id"].map(normalize_overlap_id)
    overlap_lookup = overlap.set_index(["source", "entry_id"])

    paired_rows = []
    for (setting, split, run_seed), unit in predictions.groupby(["setting", "split", "run_seed"]):
        base = unit[unit["model"] == "DeepRSMA"]
        if base.empty:
            continue
        for comparator in COMPARATORS:
            other = unit[unit["model"] == comparator]
            if other.empty:
                continue
            merged = base[["entry_id", "label", "prediction"]].merge(
                other[["entry_id", "label", "prediction"]],
                on="entry_id",
                suffixes=("_base", "_model"),
            )
            merged["label"] = merged["label_base"]
            merged["abs_error_base"] = np.abs(merged["prediction_base"] - merged["label"])
            merged["abs_error_model"] = np.abs(merged["prediction_model"] - merged["label"])
            merged["absolute_error_improvement"] = (
                merged["abs_error_base"] - merged["abs_error_model"]
            )
            source = "independent_test" if setting == "independent" else "R-SIM_all"
            merged["max_rna_global_identity"] = [
                overlap_lookup.loc[(source, entry_id), "max_rna_global_identity"]
                for entry_id in merged["entry_id"]
            ]
            merged["max_ligand_tanimoto"] = [
                overlap_lookup.loc[(source, entry_id), "max_ligand_tanimoto"]
                for entry_id in merged["entry_id"]
            ]
            merged["pair_min_similarity"] = np.minimum(
                merged["max_rna_global_identity"], merged["max_ligand_tanimoto"]
            )
            merged["setting"] = setting
            merged["split"] = split
            merged["run_seed"] = run_seed
            merged["model"] = comparator
            paired_rows.append(merged)
    paired = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    paired.to_csv(OUT / "paired_error_improvement_with_similarity.csv", index=False)

    if len(paired):
        paired["rna_similarity_bin"] = pd.cut(
            paired["max_rna_global_identity"],
            (-np.inf, 0.5, 0.8, np.inf),
            labels=("<0.5", "0.5-0.8", ">=0.8"),
            right=False,
        )
        paired["ligand_similarity_bin"] = pd.cut(
            paired["max_ligand_tanimoto"],
            (-np.inf, 0.4, 0.8, np.inf),
            labels=("<0.4", "0.4-0.8", ">=0.8"),
            right=False,
        )
        bin_rows = []
        for stratifier in ("rna_similarity_bin", "ligand_similarity_bin"):
            for keys, group in paired.groupby(
                ["setting", "model", stratifier], observed=True
            ):
                if len(group) < 2:
                    continue
                row = {
                    "setting": keys[0],
                    "model": keys[1],
                    "stratifier": stratifier,
                    "bin": keys[2],
                    "n_pair_seed_observations": len(group),
                    "n_unique_pairs": group["entry_id"].nunique(),
                    "absolute_error_improvement_mean": float(
                        group["absolute_error_improvement"].mean()
                    ),
                }
                row.update(metrics(group, "base"))
                row.update(metrics(group, "model"))
                row["delta_pcc"] = row["pcc_model"] - row["pcc_base"]
                row["delta_scc"] = row["scc_model"] - row["scc_base"]
                row["delta_rmse"] = row["rmse_model"] - row["rmse_base"]
                row["delta_mae"] = row["mae_model"] - row["mae_base"]
                bin_rows.append(row)
        pd.DataFrame(bin_rows).to_csv(OUT / "similarity_bin_performance.csv", index=False)

        corr_rows = []
        for (setting, model), group in paired.groupby(["setting", "model"]):
            for similarity in (
                "max_rna_global_identity",
                "max_ligand_tanimoto",
                "pair_min_similarity",
            ):
                corr_rows.append(
                    {
                        "setting": setting,
                        "model": model,
                        "similarity": similarity,
                        "n": len(group),
                        "spearman_rho": float(
                            spearmanr(group[similarity], group["absolute_error_improvement"])[0]
                        ),
                        "spearman_p": float(
                            spearmanr(group[similarity], group["absolute_error_improvement"])[1]
                        ),
                    }
                )
        pd.DataFrame(corr_rows).to_csv(OUT / "similarity_improvement_correlations.csv", index=False)

    coverage = overlap.copy()
    coverage["rna_bin"] = pd.cut(
        coverage["max_rna_global_identity"],
        (-np.inf, 0.5, 0.8, np.inf),
        labels=("<0.5", "0.5-0.8", ">=0.8"),
        right=False,
    )
    coverage["ligand_bin"] = pd.cut(
        coverage["max_ligand_tanimoto"],
        (-np.inf, 0.4, 0.8, np.inf),
        labels=("<0.4", "0.4-0.8", ">=0.8"),
        right=False,
    )
    coverage_table = (
        coverage.groupby(["source", "rna_bin", "ligand_bin"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    coverage_table.to_csv(OUT / "rna_ligand_coverage_counts.csv", index=False)

    contact_overlap = pd.read_csv(CONTACT_OVERLAP)
    manifest = pd.read_csv(CONTACT_MANIFEST)
    contact = manifest.merge(
        contact_overlap[["pdb_id", "exclude_global_or_rule"]], on="pdb_id", how="left"
    )
    contact["partition"] = np.where(
        contact["exclude_global_or_rule"].astype(str).str.lower() == "true",
        "excluded_by_global_deoverlap",
        "retained_global_deoverlap",
    )
    distribution_rows = []
    retained = contact[contact["partition"] == "retained_global_deoverlap"]
    excluded = contact[contact["partition"] == "excluded_by_global_deoverlap"]
    for variable in ("rna_length", "ligand_heavy_atoms", "contact_density"):
        left = retained[variable].dropna().to_numpy(float)
        right = excluded[variable].dropna().to_numpy(float)
        distribution_rows.append(
            {
                "variable": variable,
                "retained_n": len(left),
                "excluded_n": len(right),
                "retained_median": float(np.median(left)),
                "excluded_median": float(np.median(right)),
                "mannwhitney_p": float(mannwhitneyu(left, right, alternative="two-sided").pvalue),
                "ks_statistic": float(ks_2samp(left, right).statistic),
                "ks_p": float(ks_2samp(left, right).pvalue),
                "cliffs_delta_retained_vs_excluded": cliffs_delta(left, right),
            }
        )
    pd.DataFrame(distribution_rows).to_csv(
        OUT / "retained_excluded_distribution_tests.csv", index=False
    )
    contact.to_csv(OUT / "contact500_retained_excluded_features.csv", index=False)
    print(json.dumps({"paired_rows": len(paired), "coverage_rows": len(coverage)}, indent=2))


if __name__ == "__main__":
    main()
