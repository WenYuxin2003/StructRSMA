import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import PDBContactPairDataset, contact_collate
from model import DeepRSMAContact


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_value": checkpoint.get("selection_value"),
    }


def split_dataset(dataset, val_ratio, seed):
    val_size = max(1, int(round(len(dataset) * val_ratio)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Validation split leaves no training samples")
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def sample_topk_precision(prob, target, mask):
    values = []
    batch_size = prob.size(0)
    for i in range(batch_size):
        valid_prob = prob[i][mask[i] > 0.5]
        valid_target = target[i][mask[i] > 0.5]
        positives = int((valid_target > 0.5).sum().item())
        if positives <= 0:
            continue
        k = min(positives, valid_prob.numel())
        indices = torch.topk(valid_prob, k=k).indices
        values.append(float(valid_target[indices].sum().item() / k))
    return values


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    topk_values = []
    threshold_tp = 0
    threshold_fp = 0
    threshold_fn = 0
    total_pairs = 0
    total_positive = 0

    with torch.no_grad():
        for rna_batch, molecule_batch, target, mask, _ in loader:
            rna_batch = rna_batch.to(device)
            molecule_batch = molecule_batch.to(device)
            target = target.to(device)
            mask = mask.to(device)

            output = model(rna_batch, molecule_batch, device=device, return_contact=True)
            logits = output["contact_logits"][:, : target.size(1), : target.size(2)]
            prob = torch.sigmoid(logits)

            valid_prob = prob[mask > 0.5].detach().cpu()
            valid_target = target[mask > 0.5].detach().cpu()
            all_probs.append(valid_prob)
            all_targets.append(valid_target)
            topk_values.extend(sample_topk_precision(prob.detach(), target, mask))

            pred = (prob >= 0.5).float()
            threshold_tp += int(((pred == 1) & (target == 1) & (mask > 0.5)).sum().item())
            threshold_fp += int(((pred == 1) & (target == 0) & (mask > 0.5)).sum().item())
            threshold_fn += int(((pred == 0) & (target == 1) & (mask > 0.5)).sum().item())
            total_pairs += int((mask > 0.5).sum().item())
            total_positive += int(((target == 1) & (mask > 0.5)).sum().item())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    metrics = {
        "pairs": total_pairs,
        "positives": total_positive,
        "density": total_positive / max(total_pairs, 1),
        "topk_precision_mean": float(np.mean(topk_values)),
        "topk_precision_std": float(np.std(topk_values)),
        "threshold_precision": threshold_tp / max(threshold_tp + threshold_fp, 1),
        "threshold_recall": threshold_tp / max(threshold_tp + threshold_fn, 1),
        "auprc": float(average_precision_score(targets, probs)),
        "auroc": float(roc_auc_score(targets, probs)),
        "positive_probability_mean": float(probs[targets > 0.5].mean()),
        "negative_probability_mean": float(probs[targets <= 0.5].mean()),
    }
    return metrics


def contact_layout_stats(contact):
    row_contacts = contact.sum(dim=1)
    col_contacts = contact.sum(dim=0)
    return {
        "contact_rows": int((row_contacts > 0).sum().item()),
        "contact_cols": int((col_contacts > 0).sum().item()),
    }


def choose_example(
    dataset,
    subset,
    min_rna_len,
    max_rna_len,
    min_atoms,
    max_atoms,
    min_contacts,
    min_contact_rows,
    min_contact_cols,
):
    best = None
    best_score = -1
    for index in subset.indices:
        rna, molecule, contact, meta = dataset[index]
        positives = int(contact.sum().item())
        rna_len, atom_count = contact.shape
        layout = contact_layout_stats(contact)
        if (
            positives < min_contacts
            or rna_len < min_rna_len
            or rna_len > max_rna_len
            or atom_count < min_atoms
            or atom_count > max_atoms
            or layout["contact_rows"] < min_contact_rows
            or layout["contact_cols"] < min_contact_cols
        ):
            continue
        score = (
            positives
            + 2.0 * layout["contact_rows"]
            + 0.5 * layout["contact_cols"]
            - 0.01 * (rna_len * atom_count)
        )
        if score > best_score:
            best = (index, rna, molecule, contact, meta)
            best_score = score
    if best is not None:
        return best

    for index in subset.indices:
        rna, molecule, contact, meta = dataset[index]
        positives = int(contact.sum().item())
        if positives > best_score:
            best = (index, rna, molecule, contact, meta)
            best_score = positives
    return best


def choose_predicted_example(
    model,
    dataset,
    subset,
    device,
    min_rna_len,
    max_rna_len,
    min_atoms,
    max_atoms,
    min_contacts,
    min_contact_rows,
    min_contact_cols,
    example_index=None,
):
    if example_index is not None:
        if example_index < 0 or example_index >= len(dataset):
            raise IndexError(f"example_index {example_index} is outside dataset size {len(dataset)}")
        sample = dataset[example_index]
        target, prob, predicted_meta = predict_single(model, sample, device)
        return example_index, target, prob, predicted_meta

    best = None
    best_score = -1.0
    for index in subset.indices:
        rna, molecule, contact, meta = dataset[index]
        positives = int(contact.sum().item())
        rna_len, atom_count = contact.shape
        layout = contact_layout_stats(contact)
        if (
            positives < min_contacts
            or rna_len < min_rna_len
            or rna_len > max_rna_len
            or atom_count < min_atoms
            or atom_count > max_atoms
            or layout["contact_rows"] < min_contact_rows
            or layout["contact_cols"] < min_contact_cols
        ):
            continue
        target, prob, predicted_meta = predict_single(model, (rna, molecule, contact, meta), device)
        k = max(1, int(target.sum()))
        flat_prob = prob.reshape(-1)
        flat_target = target.reshape(-1)
        top_indices = np.argsort(flat_prob)[-k:]
        topk_precision = float(flat_target[top_indices].sum() / k)
        density = float(flat_target.mean())
        score = (
            topk_precision
            - density
            + 0.001 * positives
            + 0.035 * np.log1p(layout["contact_rows"])
            + 0.015 * np.log1p(layout["contact_cols"])
        )
        if score > best_score:
            best_score = score
            best = (index, target, prob, predicted_meta)
    if best is not None:
        return best

    fallback = choose_example(
        dataset,
        subset,
        min_rna_len,
        max_rna_len,
        min_atoms,
        max_atoms,
        min_contacts,
        min_contact_rows,
        min_contact_cols,
    )
    if fallback is None:
        return None
    index, rna, molecule, contact, meta = fallback
    target, prob, predicted_meta = predict_single(model, (rna, molecule, contact, meta), device)
    return index, target, prob, predicted_meta


def predict_single(model, sample, device):
    rna, molecule, contact, meta = sample
    loader = DataLoader([(rna, molecule, contact, meta)], batch_size=1, collate_fn=contact_collate)
    model.eval()
    with torch.no_grad():
        for rna_batch, molecule_batch, target, mask, metas in loader:
            rna_batch = rna_batch.to(device)
            molecule_batch = molecule_batch.to(device)
            output = model(rna_batch, molecule_batch, device=device, return_contact=True)
            logits = output["contact_logits"][:, : target.size(1), : target.size(2)]
            prob = torch.sigmoid(logits)[0].detach().cpu().numpy()
            return target[0].numpy(), prob, metas[0]
    raise RuntimeError("No sample predicted")


def plot_contact_example(target, prob, meta, out_dir):
    positives = int(target.sum())
    k = max(1, positives)
    flat_prob = prob.reshape(-1)
    flat_target = target.reshape(-1)
    top_indices = np.argsort(flat_prob)[-k:]
    topk_hits = int(flat_target[top_indices].sum())
    topk_precision = float(topk_hits / k)
    topk_mask = np.zeros_like(flat_target, dtype=bool)
    topk_mask[top_indices] = True
    topk_mask = topk_mask.reshape(target.shape)
    hit_y, hit_x = np.where(topk_mask & (target > 0.5))
    fp_y, fp_x = np.where(topk_mask & (target <= 0.5))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    im0 = axes[0].imshow(target, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_title(
        "Ground-truth contacts\n"
        f"{positives} positives, {int((target.sum(axis=1) > 0).sum())} nt, "
        f"{int((target.sum(axis=0) > 0).sum())} atoms"
    )
    axes[0].set_xlabel("Ligand atom")
    axes[0].set_ylabel("RNA nucleotide")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(prob, aspect="auto", cmap="magma", interpolation="nearest", vmin=0, vmax=1)
    if fp_x.size:
        axes[1].scatter(fp_x, fp_y, marker="x", s=20, linewidths=0.9, c="#38BDF8", label="Top-k false")
    if hit_x.size:
        axes[1].scatter(
            hit_x,
            hit_y,
            marker="o",
            s=22,
            facecolors="none",
            edgecolors="#34D399",
            linewidths=1.0,
            label="Top-k true",
        )
    axes[1].set_title(f"Predicted contact probability\nTop-k precision={topk_precision:.3f} ({topk_hits}/{k})")
    axes[1].set_xlabel("Ligand atom")
    axes[1].set_ylabel("RNA nucleotide")
    if hit_x.size or fp_x.size:
        axes[1].legend(loc="upper right", fontsize=7, frameon=True)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    title = (
        f"PDB {meta.get('pdb_id', 'unknown').upper()} "
        f"ligand {meta.get('ligand_resname', 'unknown')} "
        f"({target.shape[0]} nt x {target.shape[1]} atoms)"
    )
    fig.suptitle(title, fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(Path(out_dir) / f"fig_contact_map_example.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "pdb_id": meta.get("pdb_id", ""),
        "ligand_resname": meta.get("ligand_resname", ""),
        "rna_len": int(target.shape[0]),
        "atom_count": int(target.shape[1]),
        "positive_contacts": positives,
        "contact_rows": int((target.sum(axis=1) > 0).sum()),
        "contact_cols": int((target.sum(axis=0) > 0).sum()),
        "topk_hits": topk_hits,
        "example_topk_precision": topk_precision,
    }


def write_markdown(metrics, checkpoint_info, example_info, out_path):
    lines = [
        "# Contact checkpoint validation metrics",
        "",
        f"- checkpoint epoch: {checkpoint_info.get('epoch')}",
        f"- selection metric: {checkpoint_info.get('selection_metric')}",
        f"- selection value: {checkpoint_info.get('selection_value')}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "pairs",
        "positives",
        "density",
        "topk_precision_mean",
        "topk_precision_std",
        "auprc",
        "auroc",
        "threshold_precision",
        "threshold_recall",
        "positive_probability_mean",
        "negative_probability_mean",
    ):
        value = metrics[key]
        if isinstance(value, int):
            text = f"{value}"
        else:
            text = f"{value:.6f}"
        lines.append(f"| {key} | {text} |")

    lines.extend(
        [
            "",
            "## Contact-map example",
            "",
            "| Field | Value |",
            "|---|---:|",
        ]
    )
    for key, value in example_info.items():
        if isinstance(value, float):
            value = f"{value:.6f}"
        lines.append(f"| {key} | {value} |")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a contact-pretrained DeepRSMA checkpoint.")
    parser.add_argument("--data-dir", default="dataset/pdb_contact_rna_only_500")
    parser.add_argument("--checkpoint", default="save/contact_pretrain_rna_only_500.pth")
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--metrics-out", default="docs/contact_checkpoint_metrics_500.md")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--example-min-rna-len", type=int, default=18)
    parser.add_argument("--example-max-rna-len", type=int, default=90)
    parser.add_argument("--example-min-atoms", type=int, default=18)
    parser.add_argument("--example-max-atoms", type=int, default=70)
    parser.add_argument("--example-min-contacts", type=int, default=15)
    parser.add_argument("--example-min-contact-rows", type=int, default=5)
    parser.add_argument("--example-min-contact-cols", type=int, default=8)
    parser.add_argument("--example-index", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PDBContactPairDataset(args.data_dir)
    _, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=contact_collate,
    )

    device = torch.device(args.device)
    model = DeepRSMAContact().to(device)
    checkpoint_info = load_checkpoint(model, args.checkpoint, device)
    metrics = evaluate(model, loader, device)

    chosen = choose_predicted_example(
        model,
        dataset,
        val_dataset,
        device,
        args.example_min_rna_len,
        args.example_max_rna_len,
        args.example_min_atoms,
        args.example_max_atoms,
        args.example_min_contacts,
        args.example_min_contact_rows,
        args.example_min_contact_cols,
        args.example_index,
    )
    if chosen is None:
        raise RuntimeError("Could not choose a contact-map example")
    index, target, prob, meta = chosen
    example_info = plot_contact_example(target, prob, meta, out_dir)
    example_info["dataset_index"] = int(index)

    metrics_json = {
        "checkpoint": checkpoint_info,
        "metrics": metrics,
        "example": example_info,
    }
    (out_dir / "contact_checkpoint_metrics_500.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    write_markdown(metrics, checkpoint_info, example_info, args.metrics_out)
    print(json.dumps(metrics_json, indent=2))


if __name__ == "__main__":
    main()
