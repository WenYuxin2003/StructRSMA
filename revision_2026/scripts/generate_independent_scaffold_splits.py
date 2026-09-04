import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Create fixed scaffold-grouped validation splits.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/RSM_data/Viral_RNA_independent_dataset_v1.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def scaffold_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return f"INVALID::{smiles}"
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    text = Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold.GetNumAtoms() else ""
    if text:
        return text
    # Acyclic compounds are grouped by canonical molecular identity.
    return "ACYCLIC::" + Chem.MolToSmiles(molecule, isomericSmiles=False)


def choose_validation_groups(frame, fraction, seed):
    groups = []
    for scaffold, subset in frame.groupby("scaffold", sort=True):
        groups.append(
            {
                "scaffold": scaffold,
                "indices": subset.index.tolist(),
                "n": len(subset),
                "mean": float(subset["pKd"].mean()),
            }
        )

    target_n = int(round(len(frame) * fraction))
    global_mean = float(frame["pKd"].mean())
    best = None
    # Search deterministic randomized group orderings and retain the split with
    # the closest sample count and affinity mean to the full training pool.
    for attempt in range(5000):
        rng = np.random.default_rng(seed * 10000 + attempt)
        order = rng.permutation(len(groups))
        selected = []
        count = 0
        for group_index in order:
            group = groups[int(group_index)]
            if count < target_n or not selected:
                selected.append(group)
                count += group["n"]
        indices = sorted(index for group in selected for index in group["indices"])
        val_mean = float(frame.loc[indices, "pKd"].mean())
        score = abs(count - target_n) / max(target_n, 1) + abs(val_mean - global_mean)
        candidate = (score, indices)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best[1]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input, sep="\t")
    frame["scaffold"] = frame["SMILES"].map(scaffold_from_smiles)

    summary = []
    for seed in args.seeds:
        val_indices = choose_validation_groups(frame, args.validation_fraction, seed)
        val_set = set(val_indices)
        train_indices = [index for index in frame.index if index not in val_set]
        train_scaffolds = set(frame.loc[train_indices, "scaffold"])
        val_scaffolds = set(frame.loc[val_indices, "scaffold"])
        if train_scaffolds & val_scaffolds:
            raise RuntimeError("Scaffold leakage detected while generating validation split")

        payload = {
            "protocol": "independent_scaffold_validation",
            "seed": seed,
            "validation_fraction_requested": args.validation_fraction,
            "train_entry_ids": frame.loc[train_indices, "Entry_ID"].astype(str).tolist(),
            "validation_entry_ids": frame.loc[val_indices, "Entry_ID"].astype(str).tolist(),
            "test_source": "fixed 48-pair Viral RNA independent test set",
            "scaffold_definition": "Bemis-Murcko; acyclic compounds grouped by canonical SMILES",
        }
        (args.output_dir / f"split_seed{seed}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        assignment = frame[["Entry_ID", "SMILES", "scaffold", "pKd"]].copy()
        assignment["partition"] = ["validation" if i in val_set else "train" for i in frame.index]
        assignment.to_csv(args.output_dir / f"split_seed{seed}.csv", index=False)
        summary.append(
            {
                "seed": seed,
                "n_train": len(train_indices),
                "n_validation": len(val_indices),
                "train_scaffolds": len(train_scaffolds),
                "validation_scaffolds": len(val_scaffolds),
                "scaffold_overlap": len(train_scaffolds & val_scaffolds),
                "train_pkd_mean": float(frame.loc[train_indices, "pKd"].mean()),
                "validation_pkd_mean": float(frame.loc[val_indices, "pKd"].mean()),
                "train_pkd_sd": float(frame.loc[train_indices, "pKd"].std(ddof=1)),
                "validation_pkd_sd": float(frame.loc[val_indices, "pKd"].std(ddof=1)),
            }
        )
    pd.DataFrame(summary).to_csv(args.output_dir / "split_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
