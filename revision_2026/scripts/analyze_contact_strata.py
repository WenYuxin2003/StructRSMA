import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_2026/statistics/contact_stratified"
SPLIT_DIR = ROOT / "revision_2026/splits_contact_cutoff_common_v1"
RESULT_DIR = ROOT / "revision_2026/results/contact_grouped/cutoff40"
MANIFEST = ROOT / "revision_2026/audit/contact500_v1/contact500_manifest.csv"
RAW_METADATA = ROOT / "data/pdb_contacts/metadata_rna_only_500.csv"


def pdb_from_file(file_name):
    return Path(file_name).stem.split("_", 1)[1].lower()


def fingerprint(smiles):
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(molecule)


def max_similarity(query, references):
    if query is None or not references:
        return np.nan
    return float(max(DataStructs.TanimotoSimilarity(query, ref) for ref in references))


def ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(2026)
    boot = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize_group(frame, group_column, metrics):
    rows = []
    for (split, group), subset in frame.groupby(["split", group_column], observed=True):
        for metric in metrics:
            low, high = ci95(subset[metric])
            rows.append(
                {
                    "split": split,
                    "stratifier": group_column,
                    "stratum": str(group),
                    "metric": metric,
                    "n_complexes": len(subset),
                    "mean": float(subset[metric].mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    manifest["pdb_id"] = manifest["pdb_id"].str.lower()
    raw_metadata = pd.read_csv(RAW_METADATA).rename(columns={"smiles": "canonical_smiles"})
    raw_metadata["pdb_id"] = raw_metadata["pdb_id"].str.lower()
    missing = raw_metadata.loc[
        ~raw_metadata["pdb_id"].isin(manifest["pdb_id"]), ["pdb_id", "canonical_smiles"]
    ].copy()
    if len(missing):
        manifest = pd.concat(
            [manifest, missing.reindex(columns=manifest.columns)], ignore_index=True
        )
    manifest_lookup = manifest.set_index("pdb_id")
    assignments = pd.read_csv(SPLIT_DIR / "contact_group_assignments.csv")
    assignments["pdb_id"] = assignments["pdb_id"].str.lower()
    assignment_lookup = assignments.set_index("pdb_id")
    fingerprint_lookup = {
        pdb_id: fingerprint(row["canonical_smiles"])
        for pdb_id, row in manifest_lookup.iterrows()
    }

    all_rows = []
    for seed in (2026, 2027, 2028):
        split_name = f"split{seed}"
        split = json.loads(
            (SPLIT_DIR / f"contact_grouped_seed{seed}.json").read_text(encoding="utf-8")
        )
        train_pdbs = [pdb_from_file(name) for name in split["train_files"]]
        train_fingerprints = [fingerprint_lookup[pdb_id] for pdb_id in train_pdbs]
        train_fingerprints = [value for value in train_fingerprints if value is not None]
        metrics = pd.read_csv(RESULT_DIR / split_name / "test_complex_metrics.csv")
        metrics["pdb_id"] = metrics["pdb_id"].str.lower()
        metrics["split"] = split_name
        metrics["rna_group"] = metrics["pdb_id"].map(assignment_lookup["rna_group"])
        metrics["max_ligand_similarity_to_train"] = metrics["pdb_id"].map(
            lambda pdb_id: max_similarity(fingerprint_lookup.get(pdb_id), train_fingerprints)
        )
        all_rows.append(metrics)
    combined = pd.concat(all_rows, ignore_index=True)

    for column in ("rna_length", "ligand_atoms", "density"):
        combined[f"{column}_quartile"] = pd.qcut(
            combined[column], 4, labels=("Q1", "Q2", "Q3", "Q4"), duplicates="drop"
        )
    combined["ligand_similarity_bin"] = pd.cut(
        combined["max_ligand_similarity_to_train"],
        bins=(-np.inf, 0.4, 0.8, np.inf),
        labels=("<0.4", "0.4-0.8", ">=0.8"),
        right=False,
    )
    combined.to_csv(OUT / "contact_test_complexes_with_strata.csv", index=False)

    metric_columns = ("auprc", "auroc", "mcc", "f1", "p_at_5", "p_at_10", "p_at_nc")
    rows = []
    for stratifier in (
        "rna_length_quartile",
        "ligand_atoms_quartile",
        "density_quartile",
        "ligand_similarity_bin",
    ):
        rows.extend(summarize_group(combined, stratifier, metric_columns))
    stratified = pd.DataFrame(rows)
    stratified.to_csv(OUT / "contact_stratified_by_split.csv", index=False)

    split_level = (
        stratified.groupby(["stratifier", "stratum", "metric"], observed=True)
        .agg(splits=("split", "nunique"), mean=("mean", "mean"), sd=("mean", "std"))
        .reset_index()
    )
    half_width = stats.t.ppf(0.975, split_level["splits"] - 1) * split_level["sd"] / np.sqrt(
        split_level["splits"]
    )
    split_level["ci95_low"] = split_level["mean"] - half_width
    split_level["ci95_high"] = split_level["mean"] + half_width
    split_level.to_csv(OUT / "contact_stratified_across_splits.csv", index=False)

    cluster_rows = []
    for group, subset in combined.groupby("rna_group"):
        if len(subset) < 3:
            continue
        cluster_rows.append(
            {
                "rna_sequence_cluster": group,
                "n_test_occurrences": len(subset),
                "n_splits": subset["split"].nunique(),
                **{f"{metric}_mean": float(subset[metric].mean()) for metric in metric_columns},
            }
        )
    pd.DataFrame(cluster_rows).sort_values("n_test_occurrences", ascending=False).to_csv(
        OUT / "contact_rna_cluster_proxy_metrics.csv", index=False
    )
    print(split_level.to_string(index=False))


if __name__ == "__main__":
    main()
