import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data import PDBContactPairDataset, contact_collate
from model import DeepRSMAContact


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ShuffledContactDataset(torch.utils.data.Dataset):
    """Break nucleotide-atom correspondence while preserving per-sample density."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rna, molecule, contact, meta = self.dataset[index]
        generator = torch.Generator().manual_seed(self.seed + int(index))
        flat = contact.reshape(-1)
        shuffled = flat[torch.randperm(flat.numel(), generator=generator)].reshape_as(contact)
        meta = dict(meta)
        meta["label_mode"] = "shuffle"
        return rna, molecule, shuffled, meta


def masked_bce_with_logits(logits, target, mask, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_focal_loss(logits, target, mask, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    alpha_t = torch.where(target > 0.5, alpha, 1.0 - alpha)
    loss = alpha_t * torch.pow(1.0 - pt, gamma) * bce
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def contact_metrics(logits, target, mask):
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        valid_prob = prob[mask > 0.5]
        valid_pred = pred[mask > 0.5]
        valid_target = target[mask > 0.5]
        tp = ((valid_pred == 1) & (valid_target == 1)).sum().item()
        fp = ((valid_pred == 1) & (valid_target == 0)).sum().item()
        fn = ((valid_pred == 0) & (valid_target == 1)).sum().item()
        total = valid_target.numel()
        positives = (valid_target == 1).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        density = positives / max(total, 1)
        if positives > 0:
            topk = min(int(positives), valid_prob.numel())
            top_indices = torch.topk(valid_prob, k=topk).indices
            topk_precision = valid_target[top_indices].sum().item() / max(topk, 1)
            pos_prob = valid_prob[valid_target == 1].mean().item()
        else:
            topk_precision = 0.0
            pos_prob = 0.0
        neg_prob = valid_prob[valid_target == 0].mean().item() if (valid_target == 0).any() else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "density": density,
            "topk_precision": topk_precision,
            "pos_prob": pos_prob,
            "neg_prob": neg_prob,
        }


def compute_loss(logits, target, mask, args):
    if args.loss == "focal":
        return masked_focal_loss(logits, target, mask, alpha=args.alpha, gamma=args.gamma)
    return masked_bce_with_logits(logits, target, mask)


def run_epoch(model, loader, device, args, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    metric_values = {
        "precision": [],
        "recall": [],
        "density": [],
        "topk_precision": [],
        "pos_prob": [],
        "neg_prob": [],
    }

    for rna_batch, molecule_batch, target, mask, _ in loader:
        rna_batch = rna_batch.to(device)
        molecule_batch = molecule_batch.to(device)
        target = target.to(device)
        mask = mask.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            output = model(rna_batch, molecule_batch, device=device, return_contact=True)
            logits = output["contact_logits"][:, : target.size(1), : target.size(2)]
            loss = compute_loss(logits, target, mask, args)
            if is_train:
                loss.backward()
                optimizer.step()

        metrics = contact_metrics(logits.detach(), target, mask)
        for key in metric_values:
            metric_values[key].append(metrics[key])
        total_loss += loss.item()
        total_batches += 1

    reduced = {key: float(np.mean(values)) if values else 0.0 for key, values in metric_values.items()}
    reduced["loss"] = total_loss / max(total_batches, 1)
    return reduced


def format_metrics(prefix, metrics):
    return (
        f"{prefix}_loss: {metrics['loss']:.6f} "
        f"{prefix}_precision: {metrics['precision']:.4f} "
        f"{prefix}_recall: {metrics['recall']:.4f} "
        f"{prefix}_topk_precision: {metrics['topk_precision']:.4f} "
        f"{prefix}_density: {metrics['density']:.4f} "
        f"{prefix}_pos_prob: {metrics['pos_prob']:.4f} "
        f"{prefix}_neg_prob: {metrics['neg_prob']:.4f}"
    )


def metric_is_better(metric_name, value, best_value):
    if metric_name == "loss":
        return value < best_value
    return value > best_value


def main():
    parser = argparse.ArgumentParser(description="Pretrain DeepRSMA with nucleotide-atom contact supervision.")
    parser.add_argument("--data-dir", default="dataset/pdb_contact", help="Directory of prebuilt .pt contact samples.")
    parser.add_argument("--save-path", default="save/contact_pretrain.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=["loss", "topk_precision"], default="loss")
    parser.add_argument("--loss", choices=["focal", "bce"], default="focal")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument(
        "--label-mode",
        choices=["true", "shuffle"],
        default="true",
        help="Use true contact labels or a deterministic within-sample shuffled-label control.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    full_dataset = PDBContactPairDataset(args.data_dir)
    if args.label_mode == "shuffle":
        full_dataset = ShuffledContactDataset(full_dataset, args.seed)
    if args.val_ratio > 0 and len(full_dataset) > 1:
        val_size = max(1, int(round(len(full_dataset) * args.val_ratio)))
        train_size = len(full_dataset) - val_size
        if train_size < 1:
            raise SystemExit("Validation split leaves no training samples.")
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=contact_collate,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=contact_collate,
        )

    device = torch.device(args.device)
    model = DeepRSMAContact().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-7)

    best_metric = float("inf") if args.selection_metric == "loss" else -float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, args, optimizer=optimizer)
        monitor_metrics = train_metrics
        message = f"epoch: {epoch} {format_metrics('train', train_metrics)}"
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, device, args, optimizer=None)
            monitor_metrics = val_metrics
            message += f" {format_metrics('val', val_metrics)}"
        print(message)

        metric_value = monitor_metrics[args.selection_metric]
        if metric_is_better(args.selection_metric, metric_value, best_metric):
            best_metric = metric_value
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_metric": args.selection_metric,
                    "selection_value": metric_value,
                    "train_metrics": train_metrics,
                    "val_metrics": monitor_metrics if val_loader is not None else None,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"Best contact checkpoint saved: {args.save_path}")


if __name__ == "__main__":
    main()
