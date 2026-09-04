import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
INDEPENDENT_RNA = "GGCAGAUCUGAGCCUGGGAGCUCUCUGCC"


def parse_args():
    parser = argparse.ArgumentParser(description="Strict non-neural baselines for R-SIM.")
    parser.add_argument(
        "--model",
        choices=("sequence_smiles", "ligand_descriptor", "rna_ligand_descriptor"),
        required=True,
    )
    parser.add_argument("--protocol", choices=("independent", "clustered"), required=True)
    parser.add_argument("--independent-split-file", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=2026)
    return parser.parse_args()


def normalized_id(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def load_partitions(args):
    if args.protocol == "independent":
        if args.independent_split_file is None:
            raise ValueError("--independent-split-file is required")
        source = pd.read_csv(
            ROOT / "data/RSM_data/Viral_RNA_independent_dataset_v1.csv", sep="\t"
        )
        split = json.loads(args.independent_split_file.read_text(encoding="utf-8"))
        source["id"] = source["Entry_ID"].map(normalized_id)
        lookup = source.set_index("id", drop=False)
        train = lookup.loc[[normalized_id(v) for v in split["train_entry_ids"]]].copy()
        validation = lookup.loc[[normalized_id(v) for v in split["validation_entry_ids"]]].copy()

        test_raw = pd.read_csv(ROOT / "data/independent_data.csv")
        test = pd.DataFrame(
            {
                "id": [
                    f"independent_{i:03d}_{str(name).replace(' ', '_')}"
                    for i, name in enumerate(test_raw["Name"])
                ],
                "SMILES": test_raw["SMILES"],
                "Target_RNA_sequence": INDEPENDENT_RNA,
                "pKd": -np.log10(test_raw["KD"].astype(float)),
            }
        )
        return train, validation, test

    if args.split_file is None:
        raise ValueError("--split-file is required")
    source = pd.read_csv(ROOT / "data/RSM_data/All_sf_dataset_v1.csv", sep="\t")
    source["id"] = source["Entry_ID"].map(normalized_id)
    lookup = source.set_index("id", drop=False)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    train = lookup.loc[[normalized_id(v) for v in split["train_entry_ids"]]].copy()
    validation = lookup.loc[[normalized_id(v) for v in split["validation_entry_ids"]]].copy()
    test = lookup.loc[[normalized_id(v) for v in split["test_entry_ids"]]].copy()
    return train, validation, test


def safe_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    return {
        "n": int(len(labels)),
        "pcc": float(pearsonr(labels, predictions)[0]) if len(labels) > 1 else None,
        "scc": float(spearmanr(labels, predictions)[0]) if len(labels) > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def rna_descriptors(sequence):
    sequence = str(sequence).upper().replace("T", "U")
    bases = "AUGC"
    result = [len(sequence), (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)]
    for k in (1, 2, 3):
        denominator = max(len(sequence) - k + 1, 1)
        words = [""]
        for _ in range(k):
            words = [prefix + base for prefix in words for base in bases]
        result.extend(sequence.count(word) / denominator for word in words)
    return np.asarray(result, dtype=np.float32)


def ligand_descriptors(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return np.zeros(2048 + 12, dtype=np.float32)
    fingerprint = AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(molecule)
    bits = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    scalar = np.asarray(
        [
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule),
            Descriptors.NumRotatableBonds(molecule),
            Descriptors.RingCount(molecule),
            Descriptors.FractionCSP3(molecule),
            Descriptors.HeavyAtomCount(molecule),
            Descriptors.MolMR(molecule),
            Descriptors.NHOHCount(molecule),
            Descriptors.NOCount(molecule),
        ],
        dtype=np.float32,
    )
    return np.concatenate((bits, scalar))


def fit_sequence_smiles(train, validation, test):
    rna_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=False)
    smiles_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False)
    train_rna = rna_vectorizer.fit_transform(train["Target_RNA_sequence"].astype(str))
    train_smiles = smiles_vectorizer.fit_transform(train["SMILES"].astype(str))
    train_x = sparse.hstack((train_rna, train_smiles), format="csr")
    val_x = sparse.hstack(
        (
            rna_vectorizer.transform(validation["Target_RNA_sequence"].astype(str)),
            smiles_vectorizer.transform(validation["SMILES"].astype(str)),
        ),
        format="csr",
    )
    test_x = sparse.hstack(
        (
            rna_vectorizer.transform(test["Target_RNA_sequence"].astype(str)),
            smiles_vectorizer.transform(test["SMILES"].astype(str)),
        ),
        format="csr",
    )
    candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(train_x, train["pKd"].to_numpy(float))
        prediction = model.predict(val_x)
        candidates.append((safe_metrics(validation["pKd"], prediction)["rmse"], alpha, model))
    _, alpha, model = min(candidates, key=lambda item: item[0])
    return model.predict(test_x), {"alpha": alpha, "feature_count": int(train_x.shape[1])}


def descriptor_matrix(frame, include_rna):
    ligand = np.stack([ligand_descriptors(value) for value in frame["SMILES"]])
    if not include_rna:
        return ligand
    rna = np.stack([rna_descriptors(value) for value in frame["Target_RNA_sequence"]])
    return np.concatenate((rna, ligand), axis=1)


def fit_descriptor_model(train, validation, test, seed, include_rna):
    train_x = descriptor_matrix(train, include_rna)
    validation_x = descriptor_matrix(validation, include_rna)
    test_x = descriptor_matrix(test, include_rna)
    candidates = []
    for min_leaf in (1, 2, 4, 8):
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=min_leaf,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_x, train["pKd"].to_numpy(float))
        prediction = model.predict(validation_x)
        candidates.append((safe_metrics(validation["pKd"], prediction)["rmse"], min_leaf, model))
    _, min_leaf, model = min(candidates, key=lambda item: item[0])
    return model.predict(test_x), {"min_samples_leaf": min_leaf, "feature_count": int(train_x.shape[1])}


def main():
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test = load_partitions(args)

    if args.model == "sequence_smiles":
        predictions, selected = fit_sequence_smiles(train, validation, test)
    else:
        predictions, selected = fit_descriptor_model(
            train,
            validation,
            test,
            args.seed,
            include_rna=args.model == "rna_ligand_descriptor",
        )
    labels = test["pKd"].to_numpy(float)
    metrics = safe_metrics(labels, predictions)
    prediction_frame = pd.DataFrame(
        {
            "entry_id": test["id"].map(normalized_id).tolist(),
            "label": labels,
            "prediction": predictions,
            "error": predictions - labels,
        }
    )
    prediction_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)
    config = {
        **vars(args),
        "independent_split_file": str(args.independent_split_file) if args.independent_split_file else None,
        "split_file": str(args.split_file) if args.split_file else None,
        "output_dir": str(args.output_dir),
        "selection_metric": "validation_rmse",
        "selected_hyperparameters": selected,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "python": sys.version,
        "platform": platform.platform(),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    summary = {
        "status": "COMPLETED",
        "model_label": args.model,
        "protocol": args.protocol,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "selection_metric": "validation_rmse",
        "test_metrics": metrics,
        "test_evaluations_total": 1,
        "test_evaluations_during_training": 0,
        "elapsed_seconds": time.time() - started,
        "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
