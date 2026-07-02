import argparse
from pathlib import Path

import numpy as np
import torch


def load_torch(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def describe(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def format_desc(name, values, digits=2):
    desc = describe(values)
    return (
        f"{name}: min={desc['min']:.{digits}f} "
        f"mean={desc['mean']:.{digits}f} "
        f"median={desc['median']:.{digits}f} "
        f"max={desc['max']:.{digits}f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Summarize prebuilt PDB contact .pt samples.")
    parser.add_argument("--data-dir", default="dataset/pdb_contact")
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files found under {args.data_dir}")

    rna_lengths = []
    atom_counts = []
    positives = []
    totals = []
    ligands = []
    pdb_ids = []

    for path in files:
        item = load_torch(path)
        contact = item["contact_map"].float()
        meta = item.get("meta", {})
        rna_lengths.append(contact.size(0))
        atom_counts.append(contact.size(1))
        positives.append(float(contact.sum().item()))
        totals.append(float(contact.numel()))
        ligands.append(meta.get("ligand_resname", ""))
        pdb_ids.append(meta.get("pdb_id", path.stem))

    density = [pos / total if total else 0.0 for pos, total in zip(positives, totals)]
    print(f"samples: {len(files)}")
    print(f"unique_pdb: {len(set(pdb_ids))}")
    print(f"unique_ligand_resname: {len(set(ligands))}")
    print(format_desc("rna_len", rna_lengths, digits=1))
    print(format_desc("atom_count", atom_counts, digits=1))
    print(format_desc("positive_contacts", positives, digits=1))
    print(format_desc("contact_density", density, digits=4))
    print(f"total_positive_contacts: {int(sum(positives))}")
    print(f"total_pairs: {int(sum(totals))}")


if __name__ == "__main__":
    main()
