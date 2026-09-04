import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import PDBContactPairDataset, contact_collate  # noqa: E402
from model import DeepRSMAContact  # noqa: E402


class FileSubset(Dataset):
    def __init__(self, dataset, file_names):
        lookup = {path.name: index for index, path in enumerate(dataset.files)}
        missing = [name for name in file_names if name not in lookup]
        if missing:
            raise KeyError(f"Contact split files absent from dataset: {missing[:5]}")
        self.dataset = dataset
        self.indices = [lookup[name] for name in file_names]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def parse_args():
    parser = argparse.ArgumentParser(description="Grouped contact training with validation-only selection.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("strict_contact")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def focal_loss(logits, target, mask, alpha, gamma):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    alpha_t = torch.where(target > 0.5, alpha, 1.0 - alpha)
    loss = alpha_t * torch.pow(1.0 - pt, gamma) * bce
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def ranking_metric(target, probability, k, mode):
    if len(target) == 0:
        return None
    k = min(int(k), len(target))
    if k <= 0:
        return None
    indices = np.argpartition(probability, -k)[-k:]
    hits = float(target[indices].sum())
    if mode == "precision":
        return hits / k
    positives = float(target.sum())
    return hits / positives if positives > 0 else None


def threshold_metrics(target, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "mcc": float(matthews_corrcoef(target, prediction)) if len(np.unique(target)) > 1 else None,
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
    }


def complex_metrics(target, probability, threshold):
    positives = int(target.sum())
    metrics = {
        "pairs": len(target),
        "positives": positives,
        "density": positives / max(len(target), 1),
        "auprc": float(average_precision_score(target, probability))
        if positives > 0 and positives < len(target)
        else None,
        "auroc": float(roc_auc_score(target, probability))
        if positives > 0 and positives < len(target)
        else None,
        "p_at_5": ranking_metric(target, probability, 5, "precision"),
        "r_at_5": ranking_metric(target, probability, 5, "recall"),
        "p_at_10": ranking_metric(target, probability, 10, "precision"),
        "r_at_10": ranking_metric(target, probability, 10, "recall"),
        "p_at_nc": ranking_metric(target, probability, positives, "precision")
        if positives > 0
        else None,
    }
    metrics.update(threshold_metrics(target, probability, threshold))
    return metrics


def safe_mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def macro_summary(rows):
    metrics = [
        "auprc",
        "auroc",
        "mcc",
        "f1",
        "precision",
        "recall",
        "p_at_5",
        "r_at_5",
        "p_at_10",
        "r_at_10",
        "p_at_nc",
        "density",
    ]
    return {f"macro_{metric}": safe_mean([row[metric] for row in rows]) for metric in metrics}


def predict_complexes(model, loader, device, threshold=0.5):
    model.eval()
    rows = []
    with torch.no_grad():
        for rna_batch, molecule_batch, target, mask, metas in loader:
            output = model(
                rna_batch.to(device), molecule_batch.to(device), device=device, return_contact=True
            )
            logits = output["contact_logits"][:, : target.size(1), : target.size(2)]
            probabilities = torch.sigmoid(logits).detach().cpu()
            for index, meta in enumerate(metas):
                valid = mask[index] > 0.5
                valid_rows = valid.any(dim=1)
                valid_columns = valid.any(dim=0)
                target_values = target[index][valid].numpy().astype(int)
                probability_values = probabilities[index][valid].numpy()
                row = {
                    "pdb_id": str(meta.get("pdb_id", "")),
                    "rna_length": int(valid_rows.sum().item()),
                    "ligand_atoms": int(valid_columns.sum().item()),
                }
                row.update(complex_metrics(target_values, probability_values, threshold))
                row["target"] = target_values
                row["probability"] = probability_values
                rows.append(row)
    model.train()
    return rows


def train_epoch(model, loader, optimizer, device, alpha, gamma):
    model.train()
    total_loss = 0.0
    batches = 0
    for rna_batch, molecule_batch, target, mask, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        target = target.to(device)
        mask = mask.to(device)
        output = model(
            rna_batch.to(device), molecule_batch.to(device), device=device, return_contact=True
        )
        logits = output["contact_logits"][:, : target.size(1), : target.size(2)]
        loss = focal_loss(logits, target, mask, alpha, gamma)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def select_threshold(rows):
    best_threshold = 0.5
    best_mcc = -float("inf")
    for threshold in np.linspace(0.05, 0.95, 19):
        mcc_values = []
        for row in rows:
            metrics = threshold_metrics(row["target"], row["probability"], float(threshold))
            if metrics["mcc"] is not None:
                mcc_values.append(metrics["mcc"])
        macro_mcc = safe_mean(mcc_values)
        if macro_mcc is not None and macro_mcc > best_mcc:
            best_mcc = macro_mcc
            best_threshold = float(threshold)
    return best_threshold, best_mcc


def recompute_threshold(rows, threshold):
    updated = []
    for row in rows:
        new_row = dict(row)
        new_row.update(threshold_metrics(row["target"], row["probability"], threshold))
        updated.append(new_row)
    return updated


def bootstrap_ci(rows, iterations, seed):
    metric_names = [
        "auprc",
        "auroc",
        "mcc",
        "f1",
        "precision",
        "recall",
        "p_at_5",
        "r_at_5",
        "p_at_10",
        "r_at_10",
        "p_at_nc",
    ]
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in metric_names}
    for _ in range(iterations):
        indices = rng.integers(0, len(rows), size=len(rows))
        for metric in metric_names:
            value = safe_mean([rows[index][metric] for index in indices])
            if value is not None:
                samples[metric].append(value)
    result = {}
    for metric in metric_names:
        values = samples[metric]
        result[f"macro_{metric}"] = {
            "mean": safe_mean([row[metric] for row in rows]),
            "ci95_low": float(np.quantile(values, 0.025)) if values else None,
            "ci95_high": float(np.quantile(values, 0.975)) if values else None,
        }
    return result


def serializable_rows(rows):
    return [
        {key: value for key, value in row.items() if key not in {"target", "probability"}}
        for row in rows
    ]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    logger = configure_logging(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.time()

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    full_dataset = PDBContactPairDataset(args.data_dir)
    train_dataset = FileSubset(full_dataset, split["train_files"])
    validation_dataset = FileSubset(full_dataset, split["validation_files"])
    test_dataset = FileSubset(full_dataset, split["test_files"])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=contact_collate
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=contact_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=contact_collate
    )
    logger.info("STRICT_CONTACT_START run=%s device=%s", args.run_name, device)
    logger.info("Test-set policy: zero evaluations during training; exactly one after selection")

    config = vars(args).copy()
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in config.items()}
    config["split_sizes"] = {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset),
    }
    config["selection_metric"] = "validation_macro_auprc"
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    model = DeepRSMAContact(contact_mode="none").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value = -float("inf")
    best_epoch = -1
    no_improvement = 0
    history = []
    checkpoint_path = args.output_dir / "best_validation_macro_auprc.pt"

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.alpha, args.gamma)
        validation_rows = predict_complexes(model, validation_loader, device, threshold=0.5)
        validation_summary = macro_summary(validation_rows)
        value = validation_summary["macro_auprc"]
        history.append(
            {
                "epoch": epoch,
                "train_focal_loss": train_loss,
                **validation_summary,
            }
        )
        logger.info(
            "epoch=%d train_focal=%.6f val_macro_auprc=%s val_macro_auroc=%s val_p_at_nc=%s",
            epoch,
            train_loss,
            value,
            validation_summary["macro_auroc"],
            validation_summary["macro_p_at_nc"],
        )
        if value is not None and value > best_value + 1e-8:
            best_value = value
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                {
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": epoch,
                    "selection_metric": "validation_macro_auprc",
                    "selection_value": value,
                    "run_name": args.run_name,
                },
                checkpoint_path,
            )
            logger.info("CHECKPOINT_SELECTED epoch=%d validation_macro_auprc=%.6f", epoch, value)
        else:
            no_improvement += 1
        if args.patience > 0 and no_improvement >= args.patience:
            logger.info("EARLY_STOP epoch=%d patience=%d", epoch, args.patience)
            break

    if best_epoch < 0:
        raise RuntimeError("No contact checkpoint selected")
    write_csv(args.output_dir / "history.csv", history)
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    validation_rows = predict_complexes(model, validation_loader, device, threshold=0.5)
    threshold, validation_macro_mcc = select_threshold(validation_rows)
    logger.info("VALIDATION_THRESHOLD_SELECTED threshold=%.2f macro_mcc=%.6f", threshold, validation_macro_mcc)

    logger.info("FINAL_TEST_BEGIN selected_epoch=%d", best_epoch)
    test_rows = predict_complexes(model, test_loader, device, threshold=threshold)
    fixed_rows = recompute_threshold(test_rows, 0.5)
    test_summary = macro_summary(test_rows)
    fixed_summary = macro_summary(fixed_rows)
    confidence_intervals = bootstrap_ci(test_rows, args.bootstrap, args.seed)
    logger.info("FINAL_TEST_END metrics=%s", json.dumps(test_summary))

    write_csv(args.output_dir / "test_complex_metrics.csv", serializable_rows(test_rows))
    summary = {
        "status": "COMPLETED",
        "run_name": args.run_name,
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "selection_metric": "validation_macro_auprc",
        "best_validation_macro_auprc": best_value,
        "validation_selected_probability_threshold": threshold,
        "validation_macro_mcc_at_selected_threshold": validation_macro_mcc,
        "test_metrics_selected_threshold": test_summary,
        "test_metrics_fixed_0.5": fixed_summary,
        "complex_bootstrap_ci95": confidence_intervals,
        "test_evaluations_total": 1,
        "test_evaluations_during_training": 0,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("STRICT_CONTACT_COMPLETE summary=%s", json.dumps(summary))


if __name__ == "__main__":
    main()
