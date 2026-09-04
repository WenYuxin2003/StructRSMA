import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import Molecule_dataset_independent, RNA_dataset_independent  # noqa: E402
from model import DeepRSMAContact  # noqa: E402


class DualDataset(Dataset):
    def __init__(self, rna, molecule):
        self.rna = rna
        self.molecule = molecule

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, index):
        return self.rna[index], self.molecule[index]


def parse_args():
    parser = argparse.ArgumentParser(description="Representation-level permutation attribution.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def metrics(labels, predictions):
    return {
        "pcc": float(pearsonr(labels, predictions)[0]),
        "scc": float(spearmanr(labels, predictions)[0]),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def predict_from_representations(model, rna_cross, molecule_cross, views, contact_stats):
    base = model.affinity_from_cross_features(rna_cross, molecule_cross)
    refined, _ = model.cmif_fusion(views, contact_stats)
    delta = torch.cat((refined.flatten(start_dim=1), contact_stats), dim=1)
    delta = model.cmif_delta_line1(delta)
    delta = model.dropout(model.relu(delta))
    delta = model.cmif_delta_line2(delta)
    delta = model.dropout(model.relu(delta))
    delta = model.cmif_delta_line3(delta)
    return (base + delta).squeeze(1)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DeepRSMAContact(contact_mode="cmif_residual").to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = DualDataset(RNA_dataset_independent(), Molecule_dataset_independent())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    arrays = {key: [] for key in ("rna_cross", "molecule_cross", "views", "contact_stats", "labels")}
    direct_predictions = []
    with torch.no_grad():
        for rna_batch, molecule_batch in loader:
            labels = rna_batch.y.detach().cpu().float()
            rna_batch = rna_batch.to(device)
            molecule_batch = molecule_batch.to(device)
            encoded = model.encode(rna_batch, molecule_batch, device)
            rna_cross, molecule_cross = model.cross_features_from_encoded(encoded)
            views = model.multiview_features_from_encoded(encoded)
            stats = model.contact_prior_stats_from_encoded(encoded)
            prediction = model.cmif_residual_affinity_from_encoded(encoded).squeeze(1)
            arrays["rna_cross"].append(rna_cross.detach())
            arrays["molecule_cross"].append(molecule_cross.detach())
            arrays["views"].append(views.detach())
            arrays["contact_stats"].append(stats.detach())
            arrays["labels"].append(labels)
            direct_predictions.append(prediction.detach().cpu())
    rna_cross = torch.cat(arrays["rna_cross"], dim=0)
    molecule_cross = torch.cat(arrays["molecule_cross"], dim=0)
    views = torch.cat(arrays["views"], dim=0)
    contact_stats = torch.cat(arrays["contact_stats"], dim=0)
    labels = torch.cat(arrays["labels"], dim=0).numpy()
    direct_predictions = torch.cat(direct_predictions).numpy()

    with torch.no_grad():
        reconstructed = predict_from_representations(
            model, rna_cross, molecule_cross, views, contact_stats
        ).cpu().numpy()
    max_reconstruction_error = float(np.max(np.abs(direct_predictions - reconstructed)))
    if max_reconstruction_error > 1e-5:
        raise RuntimeError(f"Representation reconstruction mismatch: {max_reconstruction_error}")
    baseline = metrics(labels, reconstructed)

    perturbations = {
        "rna_representations": ("rna_joint", None),
        "ligand_representations": ("ligand_joint", None),
        "all_contact_statistics": ("stats_all", None),
        "density": ("stats_column", 0),
        "maxprob": ("stats_column", 1),
        "rnafocus": ("stats_column", 2),
        "atomfocus": ("stats_column", 3),
        "rna_sequence_view": ("view_column", 0),
        "rna_graph_view": ("view_column", 1),
        "molecule_sequence_view": ("view_column", 2),
        "molecule_graph_view": ("view_column", 3),
        "all_view_vectors": ("views_all", None),
    }
    rng = np.random.default_rng(args.seed)
    rows = []
    with torch.no_grad():
        for repetition in range(args.permutations):
            for name, (kind, index) in perturbations.items():
                permutation = torch.as_tensor(rng.permutation(len(labels)), device=device)
                rna_p = rna_cross
                molecule_p = molecule_cross
                views_p = views
                stats_p = contact_stats
                if kind == "rna_joint":
                    rna_p = rna_cross[permutation]
                    views_p = views.clone()
                    views_p[:, 0:2] = views[permutation, 0:2]
                elif kind == "ligand_joint":
                    molecule_p = molecule_cross[permutation]
                    views_p = views.clone()
                    views_p[:, 2:4] = views[permutation, 2:4]
                elif kind == "stats_all":
                    stats_p = contact_stats[permutation]
                elif kind == "stats_column":
                    stats_p = contact_stats.clone()
                    stats_p[:, index] = contact_stats[permutation, index]
                elif kind == "view_column":
                    views_p = views.clone()
                    views_p[:, index] = views[permutation, index]
                elif kind == "views_all":
                    views_p = views[permutation]
                prediction = predict_from_representations(
                    model, rna_p, molecule_p, views_p, stats_p
                ).cpu().numpy()
                value = metrics(labels, prediction)
                rows.append(
                    {
                        "perturbation": name,
                        "repetition": repetition,
                        **value,
                        "delta_pcc_drop": baseline["pcc"] - value["pcc"],
                        "delta_scc_drop": baseline["scc"] - value["scc"],
                        "delta_rmse_increase": value["rmse"] - baseline["rmse"],
                        "delta_mae_increase": value["mae"] - baseline["mae"],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "permutation_distributions.csv", index=False)
    summary_rows = []
    for perturbation, group in frame.groupby("perturbation"):
        row = {"perturbation": perturbation, "permutations": len(group)}
        for metric in (
            "delta_pcc_drop",
            "delta_scc_drop",
            "delta_rmse_increase",
            "delta_mae_increase",
        ):
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_ci95_low"] = float(group[metric].quantile(0.025))
            row[f"{metric}_ci95_high"] = float(group[metric].quantile(0.975))
            row[f"{metric}_empirical_p_le_zero"] = float(
                (1 + (group[metric] <= 0).sum()) / (len(group) + 1)
            )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "permutation_summary.csv", index=False)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get("validation_metrics"),
        "baseline_test_metrics": baseline,
        "permutations": args.permutations,
        "seed": args.seed,
        "analysis_level": "post-cross-fusion representation-level permutation",
        "reconstruction_max_abs_error": max_reconstruction_error,
        "contact_head_frozen_during_affinity_training": True,
    }
    (args.output_dir / "permutation_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "source_representations.npz",
        labels=labels,
        predictions=reconstructed,
        rna_cross=rna_cross.cpu().numpy(),
        molecule_cross=molecule_cross.cpu().numpy(),
        views=views.cpu().numpy(),
        contact_stats=contact_stats.cpu().numpy(),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
