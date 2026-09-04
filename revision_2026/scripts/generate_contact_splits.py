import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate grouped contact train/validation/test splits.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--rna-identity-threshold", type=float, default=0.8)
    parser.add_argument("--test-group-fraction", type=float, default=0.35)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def canonical_scaffold(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "INVALID_EXACT::" + str(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return scaffold
    return "ACYCLIC_EXACT::" + Chem.MolToSmiles(mol, canonical=True)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[right] = left


def cluster_sequences(sequences, threshold):
    unique = sorted(set(sequences), key=lambda value: (-len(value), value))
    union_find = UnionFind(len(unique))
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 0.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0
    for left in range(len(unique)):
        for right in range(left + 1, len(unique)):
            distance = -float(aligner.score(unique[left], unique[right]))
            identity = 1.0 - distance / max(len(unique[left]), len(unique[right]), 1)
            if identity >= threshold:
                union_find.union(left, right)
    roots = {}
    mapping = {}
    for index, sequence in enumerate(unique):
        root = union_find.find(index)
        if root not in roots:
            roots[root] = f"RNA_CLUSTER_{len(roots):04d}"
        mapping[sequence] = roots[root]
    return mapping, len(roots)


def select_groups(frame, column, fraction, seed):
    sizes = frame.groupby(column).size().to_dict()
    groups = list(sizes)
    random.Random(seed).shuffle(groups)
    target = round(len(frame) * fraction)
    selected = set()
    count = 0
    for group in groups:
        if selected and count >= target:
            break
        selected.add(group)
        count += sizes[group]
    return selected


def split_double_group(frame, seed, test_group_fraction, validation_fraction):
    test_rna = select_groups(frame, "rna_group", test_group_fraction, seed)
    test_scaffold = select_groups(frame, "scaffold_group", test_group_fraction, seed + 10000)
    test_mask = frame["rna_group"].isin(test_rna) & frame["scaffold_group"].isin(test_scaffold)
    candidate = frame[~frame["rna_group"].isin(test_rna) & ~frame["scaffold_group"].isin(test_scaffold)]

    validation_group_fraction = math.sqrt(validation_fraction)
    val_rna = select_groups(candidate, "rna_group", validation_group_fraction, seed + 20000)
    val_scaffold = select_groups(
        candidate, "scaffold_group", validation_group_fraction, seed + 30000
    )
    validation_mask = candidate["rna_group"].isin(val_rna) & candidate["scaffold_group"].isin(
        val_scaffold
    )
    train_mask = ~candidate["rna_group"].isin(val_rna) & ~candidate["scaffold_group"].isin(
        val_scaffold
    )
    train = candidate[train_mask]
    validation = candidate[validation_mask]
    test = frame[test_mask]
    used = set(train.index) | set(validation.index) | set(test.index)
    discarded = frame[~frame.index.isin(used)]
    return train, validation, test, discarded


def overlap_check(train, validation, test):
    result = {}
    for column in ("rna_group", "scaffold_group"):
        result[column] = {
            "train_validation": len(set(train[column]) & set(validation[column])),
            "train_test": len(set(train[column]) & set(test[column])),
            "validation_test": len(set(validation[column]) & set(test[column])),
        }
        if any(result[column].values()):
            raise RuntimeError(f"Grouped contact split leakage for {column}: {result[column]}")
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_sets = [{path.name for path in directory.glob("*.pt")} for directory in args.data_dirs]
    common_files = sorted(set.intersection(*file_sets))
    if not common_files:
        raise RuntimeError("No contact sample files are shared by all cutoff datasets")

    rows = []
    reference_dir = args.data_dirs[0]
    for file_name in common_files:
        item = load_torch(reference_dir / file_name)
        meta = item.get("meta", {})
        rows.append(
            {
                "file_name": file_name,
                "pdb_id": str(meta.get("pdb_id", "")),
                "sequence": str(meta.get("sequence", "")).upper().replace("T", "U"),
                "scaffold_group": canonical_scaffold(meta.get("smiles", "")),
            }
        )
    frame = pd.DataFrame(rows)
    rna_mapping, cluster_count = cluster_sequences(
        frame["sequence"].tolist(), args.rna_identity_threshold
    )
    frame["rna_group"] = frame["sequence"].map(rna_mapping)

    summary = {
        "data_dirs": [str(path.resolve()) for path in args.data_dirs],
        "common_samples": len(frame),
        "rna_clusters": cluster_count,
        "scaffold_groups": int(frame["scaffold_group"].nunique()),
        "rna_identity_threshold": args.rna_identity_threshold,
        "splits": [],
    }
    for seed in args.seeds:
        train, validation, test, discarded = split_double_group(
            frame, seed, args.test_group_fraction, args.validation_fraction
        )
        if min(len(train), len(validation), len(test)) < 2:
            raise RuntimeError(f"Contact split seed {seed} is too small")
        checks = overlap_check(train, validation, test)
        payload = {
            "split_seed": seed,
            "grouping": {
                "rna": f"single-linkage global identity >= {args.rna_identity_threshold}",
                "ligand": "Bemis-Murcko scaffold with exact fallback",
                "construction": "test/validation require held-out RNA and scaffold; mixed edges discarded",
            },
            "counts": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
                "discarded": len(discarded),
            },
            "overlap_checks": checks,
            "train_files": train["file_name"].tolist(),
            "validation_files": validation["file_name"].tolist(),
            "test_files": test["file_name"].tolist(),
            "discarded_files": discarded["file_name"].tolist(),
        }
        file_name = f"contact_grouped_seed{seed}.json"
        (args.output_dir / file_name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary["splits"].append({"file": file_name, **payload["counts"]})

    frame.to_csv(args.output_dir / "contact_group_assignments.csv", index=False)
    (args.output_dir / "contact_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
