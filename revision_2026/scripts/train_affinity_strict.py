import argparse
import csv
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    Molecule_dataset,
    Molecule_dataset_independent,
    RNA_dataset,
    RNA_dataset_independent,
)
from model import DeepRSMAContact  # noqa: E402


AFFINITY_PREFIXES = (
    "line1.",
    "line2.",
    "line3.",
    "guided_line1.",
    "guided_line2.",
    "guided_line3.",
    "residual_line1.",
    "residual_line2.",
    "residual_line3.",
    "pair_energy_head.",
    "pair_energy_delta.",
    "cmif_fusion.",
    "cmif_line1.",
    "cmif_line2.",
    "cmif_line3.",
    "cmif_delta_line1.",
    "cmif_delta_line2.",
    "cmif_delta_line3.",
    "rna1.",
    "rna2.",
    "mole1.",
    "mole2.",
)


class DualDataset(Dataset):
    def __init__(self, rna_dataset, molecule_dataset):
        if len(rna_dataset) != len(molecule_dataset):
            raise ValueError("RNA and molecule datasets have different lengths")
        self.rna_dataset = rna_dataset
        self.molecule_dataset = molecule_dataset

    def __getitem__(self, index):
        return self.rna_dataset[index], self.molecule_dataset[index]

    def __len__(self):
        return len(self.rna_dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leakage-resistant affinity training with validation-only checkpoint selection."
    )
    parser.add_argument("--protocol", choices=("independent", "clustered"), required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--independent-split-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--contact-checkpoint", type=Path)
    parser.add_argument("--contact-mode", default="none")
    parser.add_argument(
        "--contact-stat-mode",
        choices=("predicted", "zero"),
        default="predicted",
        help="Use predicted contact statistics or a zero-valued parameter-matched control.",
    )
    parser.add_argument("--freeze-contact-head", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--shuffle-train", action="store_true")
    parser.add_argument("--max-train", type=int, default=0, help="Smoke-test only.")
    parser.add_argument("--max-val", type=int, default=0, help="Smoke-test only.")
    parser.add_argument("--max-test", type=int, default=0, help="Smoke-test only.")
    return parser.parse_args()


def configure_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("strict_affinity")
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


def normalized_id(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def ids_to_indices(frame, entry_ids):
    lookup = {normalized_id(value): i for i, value in enumerate(frame["Entry_ID"].tolist())}
    missing = [normalized_id(value) for value in entry_ids if normalized_id(value) not in lookup]
    if missing:
        raise KeyError(f"Split contains {len(missing)} Entry_ID values absent from R-SIM: {missing[:5]}")
    return [lookup[normalized_id(value)] for value in entry_ids]


def random_train_val_indices(size, val_ratio, seed):
    if size < 2:
        raise ValueError("At least two training samples are required")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    val_size = max(1, min(size - 1, int(round(size * val_ratio))))
    return indices[val_size:], indices[:val_size]


def load_datasets(args):
    if args.protocol == "independent":
        train_frame = pd.read_csv(ROOT / "data/RSM_data/Viral_RNA_independent_dataset_v1.csv", sep="\t")
        test_frame = pd.read_csv(ROOT / "data/independent_data.csv")
        train_pool = DualDataset(
            RNA_dataset("Viral_RNA_independent"),
            Molecule_dataset("Viral_RNA_independent"),
        )
        test_dataset = DualDataset(RNA_dataset_independent(), Molecule_dataset_independent())
        if args.independent_split_file is not None:
            independent_split = json.loads(
                args.independent_split_file.resolve().read_text(encoding="utf-8")
            )
            train_indices = ids_to_indices(train_frame, independent_split["train_entry_ids"])
            val_indices = ids_to_indices(train_frame, independent_split["validation_entry_ids"])
        else:
            train_indices, val_indices = random_train_val_indices(
                len(train_pool), args.val_ratio, args.split_seed
            )
        test_indices = list(range(len(test_dataset)))
        independent_ids = [
            f"independent_{index:03d}_{str(name).replace(' ', '_')}"
            for index, name in enumerate(test_frame["Name"].tolist())
        ]
        split_records = {
            "train": train_frame.iloc[train_indices]["Entry_ID"].map(normalized_id).tolist(),
            "validation": train_frame.iloc[val_indices]["Entry_ID"].map(normalized_id).tolist(),
            "test": independent_ids,
        }
        datasets = {
            "train": Subset(train_pool, train_indices),
            "validation": Subset(train_pool, val_indices),
            "test": Subset(test_dataset, test_indices),
        }
        return datasets, split_records

    if args.split_file is None:
        raise ValueError("--split-file is required for clustered protocol")
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / "data/RSM_data/All_sf_dataset_v1.csv", sep="\t")
    full_dataset = DualDataset(RNA_dataset("All_sf"), Molecule_dataset("All_sf"))
    train_indices = ids_to_indices(frame, split["train_entry_ids"])
    val_indices = ids_to_indices(frame, split["validation_entry_ids"])
    test_indices = ids_to_indices(frame, split["test_entry_ids"])
    datasets = {
        "train": Subset(full_dataset, train_indices),
        "validation": Subset(full_dataset, val_indices),
        "test": Subset(full_dataset, test_indices),
    }
    split_records = {
        "train": [normalized_id(v) for v in split["train_entry_ids"]],
        "validation": [normalized_id(v) for v in split["validation_entry_ids"]],
        "test": [normalized_id(v) for v in split["test_entry_ids"]],
    }
    return datasets, split_records


def trim_subset(dataset, maximum):
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, list(range(maximum)))


def load_contact_checkpoint(model, checkpoint_path, logger):
    if checkpoint_path is None:
        logger.info("Initialization: random DeepRSMA backbone (no contact checkpoint)")
        return {"loaded": False, "missing_keys": 0, "unexpected_keys": 0, "skipped_keys": 0}
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    filtered = {key: value for key, value in state.items() if not key.startswith(AFFINITY_PREFIXES)}
    skipped = len(state) - len(filtered)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(
        "Loaded contact checkpoint %s | skipped affinity=%d missing=%d unexpected=%d",
        checkpoint_path,
        skipped,
        len(missing),
        len(unexpected),
    )
    return {
        "loaded": True,
        "path": str(checkpoint_path),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_keys": skipped,
    }


def safe_correlation(function, labels, predictions):
    try:
        value = function(labels, predictions)[0]
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def evaluate(model, loader, device, include_predictions=False):
    model.eval()
    labels = []
    predictions = []
    entry_ids = []
    with torch.no_grad():
        for rna_batch, molecule_batch in loader:
            label = rna_batch.y.detach().cpu().float().numpy().reshape(-1)
            output = model(rna_batch.to(device), molecule_batch.to(device), device=device)["affinity"]
            prediction = output.detach().cpu().numpy().reshape(-1)
            labels.extend(label.tolist())
            predictions.extend(prediction.tolist())
            if hasattr(rna_batch, "e_id") and rna_batch.e_id is not None:
                raw_ids = rna_batch.e_id.detach().cpu().numpy().reshape(-1).tolist()
                entry_ids.extend([normalized_id(value) for value in raw_ids])
            else:
                start = len(entry_ids)
                entry_ids.extend([f"sample_{start + index:05d}" for index in range(len(label))])
    metrics = {
        "n": len(labels),
        "pcc": safe_correlation(pearsonr, labels, predictions),
        "scc": safe_correlation(spearmanr, labels, predictions),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "mae": float(mean_absolute_error(labels, predictions)),
    }
    model.train()
    if include_predictions:
        return metrics, [
            {"entry_id": entry_id, "label": label, "prediction": prediction, "error": prediction - label}
            for entry_id, label, prediction in zip(entry_ids, labels, predictions)
        ]
    return metrics


def train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    gradient_parameter_names = set()
    for rna_batch, molecule_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        rna_batch = rna_batch.to(device)
        molecule_batch = molecule_batch.to(device)
        prediction = model(rna_batch, molecule_batch, device=device)["affinity"].squeeze(1)
        loss = loss_function(prediction, rna_batch.y.float())
        loss.backward()
        gradient_parameter_names.update(
            name for name, parameter in model.named_parameters() if parameter.grad is not None
        )
        optimizer.step()
        batch_size = int(rna_batch.y.numel())
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    gradient_parameter_count = int(
        sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in gradient_parameter_names
        )
    )
    return total_loss / max(total_samples, 1), gradient_parameter_count


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def environment_snapshot(device):
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def parameter_snapshot(model):
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "contact_head_total": int(sum(parameter.numel() for parameter in model.contact_head.parameters())),
        "contact_head_trainable": int(
            sum(parameter.numel() for parameter in model.contact_head.parameters() if parameter.requires_grad)
        ),
    }


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    logger = configure_logging(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.time()

    logger.info("STRICT_PROTOCOL_START run=%s model=%s", args.run_name, args.model_label)
    logger.info("Test-set policy: zero evaluations during training; exactly one after checkpoint selection")
    logger.info("Device: %s", environment_snapshot(device))

    datasets, split_records = load_datasets(args)
    datasets = {
        "train": trim_subset(datasets["train"], args.max_train),
        "validation": trim_subset(datasets["validation"], args.max_val),
        "test": trim_subset(datasets["test"], args.max_test),
    }
    if min(len(value) for value in datasets.values()) < 2:
        raise ValueError("Every strict split must contain at least two samples")

    split_payload = {
        "protocol": args.protocol,
        "split_seed": args.split_seed,
        "source_split_file": str(args.split_file.resolve()) if args.split_file else None,
        "train_entry_ids": split_records["train"],
        "validation_entry_ids": split_records["validation"],
        "test_entry_ids": split_records["test"],
    }
    (args.output_dir / "split.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        drop_last=False,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = DeepRSMAContact(
        contact_mode=args.contact_mode,
        contact_stat_mode=args.contact_stat_mode,
    ).to(device)
    checkpoint_info = load_contact_checkpoint(model, args.contact_checkpoint, logger)
    if args.freeze_contact_head:
        for parameter in model.contact_head.parameters():
            parameter.requires_grad = False
    parameter_info = parameter_snapshot(model)
    logger.info(
        "Network state: contact_mode=%s contact_stat_mode=%s contact_head_frozen=%s parameters=%s",
        args.contact_mode,
        args.contact_stat_mode,
        args.freeze_contact_head,
        json.dumps(parameter_info),
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = nn.MSELoss()

    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update(
        {
            "selection_metric": "validation_rmse",
            "test_evaluations_during_training": 0,
            "split_sizes": {key: len(value) for key, value in datasets.items()},
            "environment": environment_snapshot(device),
            "checkpoint_initialization": checkpoint_info,
            "parameter_counts": parameter_info,
            "shared_encoders_finetuned": True,
            "contact_head_frozen": bool(args.freeze_contact_head),
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Config: %s", json.dumps(config, ensure_ascii=False))

    history = []
    best_rmse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "best_validation_rmse.pt"
    gradient_parameter_count = 0

    for epoch in range(args.epochs):
        train_loss, gradient_parameter_count = train_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        validation_metrics = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_mse": train_loss, **{f"val_{k}": v for k, v in validation_metrics.items()}}
        history.append(row)
        logger.info(
            "epoch=%d train_mse=%.6f val_rmse=%.6f val_mae=%.6f val_pcc=%s val_scc=%s",
            epoch,
            train_loss,
            validation_metrics["rmse"],
            validation_metrics["mae"],
            validation_metrics["pcc"],
            validation_metrics["scc"],
        )

        if validation_metrics["rmse"] < best_rmse - args.min_delta:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "selection_metric": "validation_rmse",
                    "run_name": args.run_name,
                    "model_label": args.model_label,
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                },
                checkpoint_path,
            )
            logger.info("CHECKPOINT_SELECTED epoch=%d validation_rmse=%.6f", epoch, best_rmse)
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            logger.info("EARLY_STOP epoch=%d patience=%d", epoch, args.patience)
            break

    if best_epoch < 0:
        raise RuntimeError("No validation checkpoint was selected")

    write_csv(args.output_dir / "history.csv", history, list(history[0].keys()))
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True)

    logger.info("FINAL_TEST_BEGIN selected_epoch=%d", best_epoch)
    test_metrics, test_predictions = evaluate(model, test_loader, device, include_predictions=True)
    logger.info("FINAL_TEST_END metrics=%s", json.dumps(test_metrics))
    write_csv(
        args.output_dir / "test_predictions.csv",
        test_predictions,
        ["entry_id", "label", "prediction", "error"],
    )

    summary = {
        "status": "COMPLETED",
        "run_name": args.run_name,
        "model_label": args.model_label,
        "protocol": args.protocol,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "selected_epoch": best_epoch,
        "selection_metric": "validation_rmse",
        "best_validation_metrics": selected["validation_metrics"],
        "test_metrics": test_metrics,
        "test_evaluations_total": 1,
        "test_evaluations_during_training": 0,
        "gradient_bearing_parameter_count": gradient_parameter_count,
        "elapsed_seconds": time.time() - started,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("STRICT_PROTOCOL_COMPLETE summary=%s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
