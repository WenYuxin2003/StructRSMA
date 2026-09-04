import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[2]
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate leakage-resistant R-SIM clustered splits.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/RSM_data/All_sf_dataset_v1.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--double-group-fraction", type=float, default=0.35)
    parser.add_argument("--rna-identity-threshold", type=float, default=0.8)
    return parser.parse_args()


def normalized_id(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def scaffold_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID_EXACT::" + smiles
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return scaffold
    return "ACYCLIC_EXACT::" + Chem.MolToSmiles(mol, canonical=True)


def normalize_rna(sequence):
    return str(sequence).upper().replace("T", "U")


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_rna_sequences(sequences, threshold):
    unique_sequences = sorted(set(sequences), key=lambda value: (-len(value), value))
    union_find = UnionFind(len(unique_sequences))
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 0.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0

    comparisons = 0
    linked_pairs = 0
    for left in range(len(unique_sequences)):
        for right in range(left + 1, len(unique_sequences)):
            comparisons += 1
            max_length = max(len(unique_sequences[left]), len(unique_sequences[right]))
            if max_length == 0:
                identity = 1.0
            else:
                edit_distance = -float(aligner.score(unique_sequences[left], unique_sequences[right]))
                identity = 1.0 - edit_distance / max_length
            if identity >= threshold:
                union_find.union(left, right)
                linked_pairs += 1

    roots = {}
    sequence_to_cluster = {}
    for index, sequence in enumerate(unique_sequences):
        root = union_find.find(index)
        if root not in roots:
            roots[root] = f"RNA_CLUSTER_{len(roots):04d}"
        sequence_to_cluster[sequence] = roots[root]
    cluster_sizes = pd.Series(list(sequence_to_cluster.values())).value_counts()
    stats = {
        "threshold": threshold,
        "identity_definition": "1 - global_edit_distance / max(sequence lengths)",
        "clustering": "single-linkage connected components",
        "unique_sequences": len(unique_sequences),
        "clusters": len(roots),
        "comparisons": comparisons,
        "pairs_at_or_above_threshold": linked_pairs,
        "largest_cluster_unique_sequences": int(cluster_sizes.max()),
    }
    return sequence_to_cluster, stats


def grouped_partition(frame, group_column, test_fraction, validation_fraction, seed):
    group_sizes = frame.groupby(group_column).size().to_dict()
    groups = list(group_sizes)
    random.Random(seed).shuffle(groups)

    total = len(frame)
    target_test = round(total * test_fraction)
    test_groups = set()
    count = 0
    for group in groups:
        if count >= target_test and test_groups:
            break
        test_groups.add(group)
        count += group_sizes[group]

    remaining = [group for group in groups if group not in test_groups]
    target_validation = round(total * validation_fraction)
    validation_groups = set()
    count = 0
    for group in remaining:
        if count >= target_validation and validation_groups:
            break
        validation_groups.add(group)
        count += group_sizes[group]

    test_mask = frame[group_column].isin(test_groups)
    validation_mask = frame[group_column].isin(validation_groups)
    train_mask = ~(test_mask | validation_mask)
    return frame[train_mask], frame[validation_mask], frame[test_mask]


def select_groups_by_pair_count(frame, column, fraction, seed):
    sizes = frame.groupby(column).size().to_dict()
    groups = list(sizes)
    random.Random(seed).shuffle(groups)
    target = round(len(frame) * fraction)
    selected = set()
    count = 0
    for group in groups:
        if count >= target and selected:
            break
        selected.add(group)
        count += sizes[group]
    return selected


def double_cold_partition(frame, validation_fraction, group_fraction, seed):
    held_rna = select_groups_by_pair_count(frame, "rna_group", group_fraction, seed)
    held_scaffold = select_groups_by_pair_count(frame, "scaffold_group", group_fraction, seed + 10000)
    test_mask = frame["rna_group"].isin(held_rna) & frame["scaffold_group"].isin(held_scaffold)
    candidate_train = frame[~frame["rna_group"].isin(held_rna) & ~frame["scaffold_group"].isin(held_scaffold)]
    test = frame[test_mask]

    validation_group_fraction = math.sqrt(validation_fraction)
    validation_rna = select_groups_by_pair_count(
        candidate_train, "rna_group", validation_group_fraction, seed + 20000
    )
    validation_scaffold = select_groups_by_pair_count(
        candidate_train, "scaffold_group", validation_group_fraction, seed + 30000
    )
    validation_mask = candidate_train["rna_group"].isin(validation_rna) & candidate_train[
        "scaffold_group"
    ].isin(validation_scaffold)
    validation = candidate_train[validation_mask]
    train = candidate_train[~candidate_train["rna_group"].isin(validation_rna) & ~candidate_train[
        "scaffold_group"
    ].isin(validation_scaffold)]
    used = set(train.index) | set(validation.index) | set(test.index)
    discarded = frame[~frame.index.isin(used)]
    return train, validation, test, discarded


def verify_no_overlap(train, validation, test, columns):
    checks = {}
    for column in columns:
        train_values = set(train[column])
        validation_values = set(validation[column])
        test_values = set(test[column])
        checks[column] = {
            "train_validation": len(train_values & validation_values),
            "train_test": len(train_values & test_values),
            "validation_test": len(validation_values & test_values),
        }
        if any(checks[column].values()):
            raise RuntimeError(f"Leakage detected for {column}: {checks[column]}")
    return checks


def write_split(output_path, setting, seed, train, validation, test, discarded, checks, metadata):
    payload = {
        "setting": setting,
        "split_seed": seed,
        "grouping": metadata,
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "discarded": len(discarded),
        },
        "overlap_checks": checks,
        "train_entry_ids": train["Entry_ID"].map(normalized_id).tolist(),
        "validation_entry_ids": validation["Entry_ID"].map(normalized_id).tolist(),
        "test_entry_ids": test["Entry_ID"].map(normalized_id).tolist(),
        "discarded_entry_ids": discarded["Entry_ID"].map(normalized_id).tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input, sep="\t")
    frame = frame.copy()
    frame["normalized_rna_sequence"] = frame["Target_RNA_sequence"].map(normalize_rna)
    sequence_to_cluster, rna_cluster_stats = cluster_rna_sequences(
        frame["normalized_rna_sequence"].tolist(), args.rna_identity_threshold
    )
    frame["rna_group"] = frame["normalized_rna_sequence"].map(sequence_to_cluster)
    frame["scaffold_group"] = frame["SMILES"].map(scaffold_key)

    summary = {
        "source": str(args.input.resolve()),
        "rows": len(frame),
        "unique_exact_rna_sequences": int(frame["normalized_rna_sequence"].nunique()),
        "rna_cluster_stats": rna_cluster_stats,
        "unique_bemis_murcko_scaffolds_with_exact_fallback": int(frame["scaffold_group"].nunique()),
        "splits": [],
    }

    for seed in args.seeds:
        for setting, group_column in (("cold_rna", "rna_group"), ("cold_scaffold", "scaffold_group")):
            train, validation, test = grouped_partition(
                frame, group_column, args.test_fraction, args.validation_fraction, seed
            )
            discarded = frame.iloc[0:0]
            checks = verify_no_overlap(train, validation, test, [group_column])
            payload = write_split(
                args.output_dir / f"{setting}_seed{seed}.json",
                setting,
                seed,
                train,
                validation,
                test,
                discarded,
                checks,
                {
                    "rna": f"single-linkage global sequence identity clusters at {args.rna_identity_threshold:.2f}"
                    if setting == "cold_rna"
                    else "not constrained",
                    "ligand": "Bemis-Murcko scaffold; invalid/acyclic molecules use exact canonical fallback"
                    if setting == "cold_scaffold"
                    else "not constrained",
                },
            )
            summary["splits"].append({"file": f"{setting}_seed{seed}.json", **payload["counts"]})

        train, validation, test, discarded = double_cold_partition(
            frame, args.validation_fraction, args.double_group_fraction, seed
        )
        if min(len(train), len(validation), len(test)) < 2:
            raise RuntimeError(f"Double-cold seed {seed} produced an unusable split")
        checks = verify_no_overlap(train, validation, test, ["rna_group", "scaffold_group"])
        payload = write_split(
            args.output_dir / f"double_cold_seed{seed}.json",
            "double_cold",
            seed,
            train,
            validation,
            test,
            discarded,
            checks,
            {
                "rna": f"single-linkage global sequence identity clusters at {args.rna_identity_threshold:.2f}",
                "ligand": "Bemis-Murcko scaffold; invalid/acyclic molecules use exact canonical fallback",
                "construction": "test requires both held-out RNA and scaffold; mixed pairs are discarded",
                "held_out_group_fraction": args.double_group_fraction,
            },
        )
        summary["splits"].append({"file": f"double_cold_seed{seed}.json", **payload["counts"]})

    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    frame[["Entry_ID", "normalized_rna_sequence", "rna_group", "scaffold_group"]].to_csv(
        args.output_dir / "group_assignments.csv", index=False
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
