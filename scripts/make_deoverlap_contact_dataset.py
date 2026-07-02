import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from check_pdb_rsim_overlap import INDEPENDENT_RNA_SEQUENCE, best_window_identity, canonical_smiles


def load_torch(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def morgan_fp_from_canonical(canonical):
    mol = Chem.MolFromSmiles(canonical) if canonical else None
    if mol is None:
        return None
    try:
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol)
    except AttributeError:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def independent_ligands(independent_csv):
    df = pd.read_csv(independent_csv)
    records = []
    for index, row in df.iterrows():
        canonical = canonical_smiles(str(row["SMILES"]))
        records.append(
            {
                "index": int(index),
                "name": str(row.get("Name", "")),
                "smiles": str(row["SMILES"]),
                "canonical_smiles": canonical,
                "fingerprint": morgan_fp_from_canonical(canonical),
            }
        )
    return records


def best_ligand_similarity(smiles, independent_records):
    canonical = canonical_smiles(smiles)
    fp = morgan_fp_from_canonical(canonical)
    best = {
        "canonical_smiles": canonical,
        "best_independent_index": "",
        "best_independent_name": "",
        "best_ligand_tanimoto": 0.0,
        "ligand_exact_match": False,
    }
    if fp is None:
        return best

    for record in independent_records:
        if record["fingerprint"] is None:
            continue
        similarity = DataStructs.TanimotoSimilarity(fp, record["fingerprint"])
        if similarity > best["best_ligand_tanimoto"]:
            best.update(
                {
                    "best_independent_index": record["index"],
                    "best_independent_name": record["name"],
                    "best_ligand_tanimoto": float(similarity),
                    "ligand_exact_match": bool(canonical and canonical == record["canonical_smiles"]),
                }
            )
    return best


def classify_sample(path, independent_records, independent_rna_sequence, ligand_threshold, rna_threshold):
    item = load_torch(path)
    meta = item.get("meta", {})
    ligand = best_ligand_similarity(str(meta.get("smiles", "")), independent_records)
    rna_identity, window, window_start = best_window_identity(independent_rna_sequence, str(meta.get("sequence", "")))
    exclude_ligand = ligand["best_ligand_tanimoto"] >= ligand_threshold
    exclude_rna = rna_identity >= rna_threshold
    excluded = exclude_ligand or exclude_rna
    reasons = []
    if ligand["ligand_exact_match"]:
        reasons.append("ligand_exact")
    if exclude_ligand:
        reasons.append(f"ligand_tanimoto_ge_{ligand_threshold:.2f}")
    if exclude_rna:
        reasons.append(f"rna_identity_ge_{rna_threshold:.2f}")
    return {
        "sample_file": str(path),
        "file_name": path.name,
        "pdb_id": str(meta.get("pdb_id", "")),
        "ligand_resname": str(meta.get("ligand_resname", "")),
        "rna_chain": str(meta.get("rna_chain", "")),
        "rna_len": len(str(meta.get("sequence", ""))),
        "ligand_atom_count": int(item["contact_map"].shape[1]),
        "positive_contacts": int(item["contact_map"].sum().item()),
        **ligand,
        "best_rna_window_identity": float(rna_identity),
        "best_rna_window_start": int(window_start),
        "best_rna_window": window,
        "excluded": bool(excluded),
        "exclude_reason": ";".join(reasons),
    }


def main():
    parser = argparse.ArgumentParser(description="Create a PDB contact dataset with R-SIM independent-test overlaps removed.")
    parser.add_argument("--in-dir", default="dataset/pdb_contact_rna_only_500")
    parser.add_argument("--out-dir", default="dataset/pdb_contact_rna_only_500_deoverlap_l80_r80")
    parser.add_argument("--independent-csv", default="data/independent_data.csv")
    parser.add_argument("--independent-rna-sequence", default=INDEPENDENT_RNA_SEQUENCE)
    parser.add_argument("--ligand-threshold", type=float, default=0.80)
    parser.add_argument("--rna-threshold", type=float, default=0.80)
    parser.add_argument("--report-dir", default="docs")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    tables_dir = report_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    records = []
    independent_records = independent_ligands(args.independent_csv)
    for path in sorted(in_dir.glob("*.pt")):
        records.append(
            classify_sample(
                path,
                independent_records,
                args.independent_rna_sequence,
                args.ligand_threshold,
                args.rna_threshold,
            )
        )
    df = pd.DataFrame(records)

    for path in out_dir.glob("*.pt"):
        path.unlink()

    kept = df[df["excluded"] == False].copy()
    for _, row in kept.iterrows():
        shutil.copy2(row["sample_file"], out_dir / row["file_name"])

    summary = {
        "input_samples": int(len(df)),
        "kept_samples": int(len(kept)),
        "excluded_samples": int(df["excluded"].sum()),
        "ligand_threshold": float(args.ligand_threshold),
        "rna_threshold": float(args.rna_threshold),
        "excluded_ligand_tanimoto_ge_threshold": int((df["best_ligand_tanimoto"] >= args.ligand_threshold).sum()),
        "excluded_ligand_exact_match": int(df["ligand_exact_match"].sum()),
        "excluded_rna_identity_ge_threshold": int((df["best_rna_window_identity"] >= args.rna_threshold).sum()),
        "max_ligand_tanimoto_kept": float(kept["best_ligand_tanimoto"].max()) if len(kept) else 0.0,
        "max_rna_window_identity_kept": float(kept["best_rna_window_identity"].max()) if len(kept) else 0.0,
        "total_positive_contacts_kept": int(kept["positive_contacts"].sum()) if len(kept) else 0,
        "out_dir": str(out_dir),
    }

    stem = f"pdb_contact_deoverlap_l{int(args.ligand_threshold * 100)}_r{int(args.rna_threshold * 100)}"
    df.to_csv(tables_dir / f"{stem}_samples.csv", index=False)
    kept.to_csv(tables_dir / f"{stem}_kept.csv", index=False)
    df[df["excluded"] == True].to_csv(tables_dir / f"{stem}_excluded.csv", index=False)
    (report_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# De-overlapped PDB Contact Dataset",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Rule",
            "",
            f"A sample is removed if max ligand Morgan fingerprint Tanimoto to any independent-test ligand is >= {args.ligand_threshold:.2f}, or if the best RNA window identity to the independent HIV RNA sequence is >= {args.rna_threshold:.2f}.",
            "",
            "This de-overlapped subset is intended for leakage-robust contact pretraining; it does not overwrite the original Contact500 dataset or checkpoints.",
        ]
    )
    (report_dir / f"{stem}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
