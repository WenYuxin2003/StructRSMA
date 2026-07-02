import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import PDBContactPairDataset  # noqa: E402
from model import DeepRSMAContact  # noqa: E402
from scripts.build_pdb_contact_dataset import (  # noqa: E402
    group_rna_residues,
    read_structure_atoms,
    select_ligand_atoms,
)
from scripts.evaluate_contact_checkpoint import load_checkpoint, predict_single  # noqa: E402


ELEMENT_COLORS = {
    "C": "#2563EB",
    "N": "#1D4ED8",
    "O": "#DC2626",
    "S": "#CA8A04",
    "P": "#7C3AED",
}
COVALENT_RADII = {
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
}


def find_sample(dataset, pdb_id=None, ligand_resname=None, example_index=None):
    if example_index is not None:
        return example_index, dataset[example_index]
    pdb_id = pdb_id.lower() if pdb_id else None
    ligand_resname = ligand_resname.upper() if ligand_resname else None
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = sample[3]
        if pdb_id and str(meta.get("pdb_id", "")).lower() != pdb_id:
            continue
        if ligand_resname and str(meta.get("ligand_resname", "")).upper() != ligand_resname:
            continue
        return index, sample
    raise ValueError(f"Could not find sample pdb_id={pdb_id} ligand={ligand_resname}")


def atom_coords(atoms):
    return np.stack([atom["coord"] for atom in atoms]).astype(np.float32)


def residue_centers(residues):
    centers = []
    backbone = []
    labels = []
    for i, residue in enumerate(residues):
        coords = atom_coords(residue["atoms"])
        centers.append(coords.mean(axis=0))
        p_atoms = [atom for atom in residue["atoms"] if atom["atom_name"] == "P"]
        c4_atoms = [atom for atom in residue["atoms"] if atom["atom_name"] in {"C4'", "C4*"}]
        if p_atoms:
            backbone.append(p_atoms[0]["coord"])
        elif c4_atoms:
            backbone.append(c4_atoms[0]["coord"])
        else:
            backbone.append(coords.mean(axis=0))
        first_atom = residue["atoms"][0]
        labels.append(f"{residue['base']}{i + 1}")
    return np.stack(centers), np.stack(backbone), labels


def ligand_bonds(ligand_atoms):
    coords = atom_coords(ligand_atoms)
    bonds = []
    for i in range(len(ligand_atoms)):
        elem_i = ligand_atoms[i]["element"].strip().upper() or "C"
        for j in range(i + 1, len(ligand_atoms)):
            elem_j = ligand_atoms[j]["element"].strip().upper() or "C"
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            max_dist = 1.25 * (
                COVALENT_RADII.get(elem_i, 0.76) + COVALENT_RADII.get(elem_j, 0.76)
            )
            if 0.45 <= dist <= max_dist:
                bonds.append((i, j))
    return bonds


def pca_transform(reference_coords):
    center = reference_coords.mean(axis=0)
    shifted = reference_coords - center
    _, _, vt = np.linalg.svd(shifted, full_matrices=False)
    basis = vt.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1

    def transform(coords):
        return (coords - center) @ basis

    return transform


def set_equal_limits(ax, coords, margin=3.0):
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0 + margin, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.view_init(elev=18, azim=-62)


def topk_pairs(target, prob, k=None):
    positives = int(target.sum())
    k = positives if k is None else int(k)
    k = max(1, min(k, target.size))
    flat_prob = prob.reshape(-1)
    flat_target = target.reshape(-1)
    top_indices = np.argsort(flat_prob)[-k:][::-1]
    atom_count = target.shape[1]
    pairs = []
    for flat_index in top_indices:
        row = int(flat_index // atom_count)
        col = int(flat_index % atom_count)
        pairs.append(
            {
                "row": row,
                "col": col,
                "prob": float(flat_prob[flat_index]),
                "is_true": bool(flat_target[flat_index] > 0.5),
            }
        )
    return pairs


def draw_ligand(ax, ligand_coords, ligand_atoms, bonds, transform, atom_size=55, alpha=1.0):
    transformed = transform(ligand_coords)
    for i, j in bonds:
        seg = transformed[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#1E3A8A", linewidth=1.4, alpha=alpha)
    colors = [ELEMENT_COLORS.get(atom["element"].strip().upper(), "#2563EB") for atom in ligand_atoms]
    ax.scatter(
        transformed[:, 0],
        transformed[:, 1],
        transformed[:, 2],
        s=atom_size,
        c=colors,
        edgecolors="#111827",
        linewidths=0.35,
        depthshade=True,
        alpha=alpha,
    )
    return transformed


def draw_rna_backbone(ax, backbone, transform, color="#9CA3AF", linewidth=1.4, alpha=0.75):
    transformed = transform(backbone)
    ax.plot(
        transformed[:, 0],
        transformed[:, 1],
        transformed[:, 2],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.scatter(
        transformed[:, 0],
        transformed[:, 1],
        transformed[:, 2],
        s=13,
        c=color,
        edgecolors="none",
        alpha=min(alpha + 0.15, 1.0),
        depthshade=True,
    )
    return transformed


def draw_contact_residues(ax, centers, rows, transform, color="#EF4444", size=85, label_texts=None):
    if len(rows) == 0:
        return np.empty((0, 3))
    transformed = transform(centers[rows])
    ax.scatter(
        transformed[:, 0],
        transformed[:, 1],
        transformed[:, 2],
        s=size,
        c=color,
        edgecolors="#7F1D1D",
        linewidths=0.45,
        alpha=0.95,
        depthshade=True,
    )
    if label_texts is not None:
        for xyz, text in zip(transformed, label_texts):
            ax.text(xyz[0], xyz[1], xyz[2], text, fontsize=7, color="#7F1D1D")
    return transformed


def draw_prediction_lines(ax, centers, ligand_coords, pairs, transform, max_false_lines=12):
    false_count = 0
    hit_count = 0
    for pair in pairs:
        if pair["is_true"]:
            color = "#22C55E"
            linestyle = "-"
            linewidth = 1.35
            alpha = 0.8
            hit_count += 1
        else:
            if false_count >= max_false_lines:
                continue
            color = "#0EA5E9"
            linestyle = "--"
            linewidth = 1.0
            alpha = 0.55
            false_count += 1
        start = transform(centers[[pair["row"]]])[0]
        end = transform(ligand_coords[[pair["col"]]])[0]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
    return hit_count, false_count


def draw_backbone_2d(ax, backbone, transform, color="#9CA3AF", linewidth=1.7, alpha=0.75):
    coords = transform(backbone)
    ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=14, c=color, edgecolors="none", alpha=alpha, zorder=2)
    return coords


def draw_ligand_2d(ax, ligand_coords, ligand_atoms, bonds, transform, atom_size=72):
    coords = transform(ligand_coords)
    for i, j in bonds:
        seg = coords[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], color="#1E3A8A", linewidth=1.6, alpha=0.9, zorder=5)
    colors = [ELEMENT_COLORS.get(atom["element"].strip().upper(), "#2563EB") for atom in ligand_atoms]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=atom_size,
        c=colors,
        edgecolors="#111827",
        linewidths=0.45,
        alpha=0.98,
        zorder=7,
    )
    return coords


def draw_contact_residues_2d(ax, centers, rows, transform, size=125, labels=None):
    if len(rows) == 0:
        return np.empty((0, 3))
    coords = transform(centers[rows])
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=size,
        c="#EF4444",
        edgecolors="#7F1D1D",
        linewidths=0.7,
        alpha=0.95,
        zorder=6,
    )
    if labels is not None:
        label_offsets = {
            "G10": (10, 16),
            "G11": (26, -4),
            "G12": (24, -17),
            "A48": (-42, 8),
            "A49": (-36, -16),
        }
        default_offsets = [(8, 12), (10, -12), (-28, 8), (-34, -9)]
        for n, (xy, text) in enumerate(zip(coords, labels)):
            dx, dy = label_offsets.get(text, default_offsets[n % len(default_offsets)])
            ax.annotate(
                text,
                xy=(xy[0], xy[1]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                color="#7F1D1D",
                arrowprops={"arrowstyle": "-", "color": "#7F1D1D", "lw": 0.6, "alpha": 0.75},
                zorder=9,
            )
    return coords


def draw_prediction_lines_2d(ax, centers, ligand_coords, pairs, transform, max_false_lines=10):
    false_count = 0
    hit_count = 0
    transformed_centers = transform(centers)
    transformed_ligand = transform(ligand_coords)
    for pair in pairs:
        if pair["is_true"]:
            color = "#22C55E"
            linestyle = "-"
            linewidth = 1.85
            alpha = 0.86
            zorder = 9
            hit_count += 1
        else:
            if false_count >= max_false_lines:
                continue
            color = "#0EA5E9"
            linestyle = "--"
            linewidth = 1.35
            alpha = 0.82
            zorder = 8
            false_count += 1
        start = transformed_centers[pair["row"]]
        end = transformed_ligand[pair["col"]]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    return hit_count, false_count


def style_2d_axis(ax, coords, margin=3.0):
    mins = coords[:, :2].min(axis=0)
    maxs = coords[:, :2].max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max() + 2 * margin
    ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
    ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_structure_case(args):
    dataset = PDBContactPairDataset(args.data_dir)
    index, sample = find_sample(dataset, args.pdb_id, args.ligand_resname, args.example_index)
    rna, molecule, contact, meta = sample

    device = torch.device(args.device)
    model = DeepRSMAContact().to(device)
    checkpoint_info = load_checkpoint(model, args.checkpoint, device)
    target, prob, _ = predict_single(model, (rna, molecule, contact, meta), device)

    pdb_file = Path(meta.get("pdb_file", ""))
    if not pdb_file.is_absolute():
        pdb_file = ROOT / pdb_file
    atoms = read_structure_atoms(pdb_file)
    residues = group_rna_residues(atoms, chain=meta.get("rna_chain"))
    residues = residues[: target.shape[0]]
    ligand_atoms = select_ligand_atoms(atoms, meta)[: target.shape[1]]
    if len(residues) != target.shape[0] or len(ligand_atoms) != target.shape[1]:
        raise ValueError(
            f"Structure/map mismatch: residues={len(residues)} ligand_atoms={len(ligand_atoms)} "
            f"target={target.shape}"
        )

    centers, backbone, labels = residue_centers(residues)
    ligand_coords = atom_coords(ligand_atoms)
    bonds = ligand_bonds(ligand_atoms)
    pairs = topk_pairs(target, prob, k=args.top_k)
    true_rows = np.where(target.sum(axis=1) > 0.5)[0]
    true_cols = np.where(target.sum(axis=0) > 0.5)[0]
    hit_count = sum(1 for pair in pairs if pair["is_true"])
    positives = int(target.sum())
    topk_precision = hit_count / max(len(pairs), 1)

    reference_rows = sorted(set(true_rows.tolist() + [pair["row"] for pair in pairs]))
    reference_coords = np.vstack([centers[reference_rows], ligand_coords])
    transform = pca_transform(reference_coords)

    full_coords = transform(np.vstack([backbone, centers[true_rows], ligand_coords]))
    distances_to_ligand = np.linalg.norm(centers[:, None, :] - ligand_coords[None, :, :], axis=2).min(axis=1)
    zoom_rows = np.where(distances_to_ligand <= args.zoom_radius)[0]
    zoom_rows = np.array(sorted(set(zoom_rows.tolist() + reference_rows)), dtype=int)
    zoom_coords = transform(np.vstack([centers[zoom_rows], ligand_coords]))

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    draw_backbone_2d(ax_full, backbone, transform, linewidth=1.9, alpha=0.7)
    draw_contact_residues_2d(ax_full, centers, true_rows, transform, size=150)
    draw_ligand_2d(ax_full, ligand_coords, ligand_atoms, bonds, transform, atom_size=64)
    style_2d_axis(ax_full, full_coords, margin=4.5)
    ax_full.set_title("A. Full RNA-ligand complex", fontsize=11)

    draw_backbone_2d(ax_zoom, backbone, transform, linewidth=1.65, alpha=0.45)
    draw_prediction_lines_2d(
        ax_zoom,
        centers,
        ligand_coords,
        pairs,
        transform,
        max_false_lines=args.max_false_lines,
    )
    draw_contact_residues_2d(
        ax_zoom,
        centers,
        true_rows,
        transform,
        size=132,
        labels=[labels[i] for i in true_rows],
    )
    transformed_ligand = draw_ligand_2d(ax_zoom, ligand_coords, ligand_atoms, bonds, transform, atom_size=58)
    if len(true_cols):
        ax_zoom.scatter(
            transformed_ligand[true_cols, 0],
            transformed_ligand[true_cols, 1],
            s=96,
            facecolors="none",
            edgecolors="#F59E0B",
            linewidths=1.2,
            zorder=8,
        )
    style_2d_axis(ax_zoom, zoom_coords, margin=2.8)
    ax_zoom.set_title("B. Binding pocket with top-k predictions", fontsize=11)

    legend_handles = [
        Line2D([0], [0], color="#9CA3AF", lw=2, label="RNA backbone"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#EF4444", markeredgecolor="#7F1D1D", markersize=8, label="True contact nucleotide"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB", markeredgecolor="#111827", markersize=8, label="Ligand atom"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#F59E0B", markersize=8, label="True contact ligand atom"),
        Line2D([0], [0], color="#22C55E", lw=1.6, label="Top-k true pair"),
        Line2D([0], [0], color="#0EA5E9", lw=1.2, linestyle="--", label="Top-k false pair"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )
    title = (
        f"PDB {str(meta.get('pdb_id', '')).upper()} ligand {meta.get('ligand_resname', '')} "
        f"structure case: {positives} contacts, "
        f"top-k precision={topk_precision:.3f} ({hit_count}/{len(pairs)})"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.93))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset_index": int(index),
        "checkpoint_epoch": checkpoint_info.get("epoch"),
        "pdb_id": meta.get("pdb_id"),
        "ligand_resname": meta.get("ligand_resname"),
        "rna_chain": meta.get("rna_chain"),
        "rna_len": int(target.shape[0]),
        "atom_count": int(target.shape[1]),
        "positive_contacts": positives,
        "contact_rows": int(len(true_rows)),
        "contact_cols": int(len(true_cols)),
        "topk": int(len(pairs)),
        "topk_hits": int(hit_count),
        "topk_precision": float(topk_precision),
        "output_png": str(out_dir / f"{stem}.png"),
        "output_pdf": str(out_dir / f"{stem}.pdf"),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot a PDB 3D structure case for contact prediction.")
    parser.add_argument("--data-dir", default="dataset/pdb_contact_rna_only_500")
    parser.add_argument("--checkpoint", default="save/contact_pretrain_rna_only_500.pth")
    parser.add_argument("--pdb-id", default="3f4h")
    parser.add_argument("--ligand-resname", default="RS3")
    parser.add_argument("--example-index", type=int, default=None)
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--output-stem", default="fig_structure_case_3f4h")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-false-lines", type=int, default=6)
    parser.add_argument("--zoom-radius", type=float, default=18.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    summary = plot_structure_case(args)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
