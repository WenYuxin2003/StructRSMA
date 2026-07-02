import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


INDEPENDENT_RNA_SEQUENCE = "GGCAGAUCUGAGCCUGGGAGCUCUCUGCC"


def load_torch(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def canonical_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol)
    except AttributeError:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def best_window_identity(query, target):
    query = str(query).upper().replace("T", "U")
    target = str(target).upper().replace("T", "U")
    if not query or not target:
        return 0.0, "", 0
    q_len = len(query)
    if len(target) >= q_len:
        best_identity = -1.0
        best_window = ""
        best_start = 0
        for start in range(0, len(target) - q_len + 1):
            window = target[start : start + q_len]
            matches = sum(a == b for a, b in zip(query, window))
            identity = matches / q_len
            if identity > best_identity:
                best_identity = identity
                best_window = window
                best_start = start
        return best_identity, best_window, best_start

    best_identity = -1.0
    best_window = ""
    best_start = 0
    for start in range(0, q_len - len(target) + 1):
        window = query[start : start + len(target)]
        matches = sum(a == b for a, b in zip(window, target))
        identity = matches / len(target)
        if identity > best_identity:
            best_identity = identity
            best_window = target
            best_start = start
    return best_identity, best_window, best_start


def load_contact_samples(data_dir):
    rows = []
    for path in sorted(Path(data_dir).glob("*.pt")):
        item = load_torch(path)
        meta = item.get("meta", {})
        rows.append(
            {
                "sample_file": str(path),
                "pdb_id": str(meta.get("pdb_id", "")),
                "ligand_resname": str(meta.get("ligand_resname", "")),
                "smiles": str(meta.get("smiles", "")),
                "sequence": str(meta.get("sequence", "")),
                "rna_chain": str(meta.get("rna_chain", "")),
            }
        )
    return pd.DataFrame(rows)


def prepare_ligands(df, smiles_col="smiles", name_col=None, source=""):
    records = []
    for index, row in df.iterrows():
        smiles = str(row.get(smiles_col, ""))
        canonical = canonical_smiles(smiles)
        fp = morgan_fp(canonical) if canonical else None
        records.append(
            {
                "source": source,
                "index": int(index),
                "name": str(row.get(name_col, "")) if name_col else "",
                "smiles": smiles,
                "canonical_smiles": canonical,
                "has_fingerprint": fp is not None,
                "fingerprint": fp,
            }
        )
    return records


def ligand_overlap(independent_records, pdb_records):
    exact_matches = []
    top_similarity = []

    pdb_by_canonical = {}
    for record in pdb_records:
        if record["canonical_smiles"]:
            pdb_by_canonical.setdefault(record["canonical_smiles"], []).append(record)

    for ind in independent_records:
        canonical = ind["canonical_smiles"]
        if canonical and canonical in pdb_by_canonical:
            for pdb in pdb_by_canonical[canonical]:
                exact_matches.append(
                    {
                        "independent_index": ind["index"],
                        "independent_name": ind["name"],
                        "independent_smiles": ind["smiles"],
                        "pdb_index": pdb["index"],
                        "pdb_name": pdb["name"],
                        "canonical_smiles": canonical,
                    }
                )

        best = {
            "independent_index": ind["index"],
            "independent_name": ind["name"],
            "independent_smiles": ind["smiles"],
            "best_pdb_index": "",
            "best_pdb_name": "",
            "best_similarity": 0.0,
            "best_pdb_smiles": "",
        }
        if ind["fingerprint"] is not None:
            for pdb in pdb_records:
                if pdb["fingerprint"] is None:
                    continue
                similarity = DataStructs.TanimotoSimilarity(ind["fingerprint"], pdb["fingerprint"])
                if similarity > best["best_similarity"]:
                    best.update(
                        {
                            "best_pdb_index": pdb["index"],
                            "best_pdb_name": pdb["name"],
                            "best_similarity": float(similarity),
                            "best_pdb_smiles": pdb["smiles"],
                        }
                    )
        top_similarity.append(best)

    return pd.DataFrame(exact_matches), pd.DataFrame(top_similarity)


def rna_overlap(independent_sequence, pdb_df):
    rows = []
    for index, row in pdb_df.iterrows():
        identity, window, start = best_window_identity(independent_sequence, row["sequence"])
        rows.append(
            {
                "pdb_index": int(index),
                "pdb_id": row["pdb_id"],
                "ligand_resname": row["ligand_resname"],
                "rna_chain": row["rna_chain"],
                "pdb_sequence_length": len(str(row["sequence"])),
                "best_window_identity": float(identity),
                "best_window_start": int(start),
                "best_window": window,
            }
        )
    return pd.DataFrame(rows)


def fmt(value, digits=4):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isnan(value):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(summary, exact_df, ligand_sim_df, rna_df, out_path):
    top_ligands = ligand_sim_df.sort_values("best_similarity", ascending=False).head(10)
    top_rnas = rna_df.sort_values("best_window_identity", ascending=False).head(10)
    lines = [
        "# PDB Contact Pretraining vs R-SIM Independent Test Overlap Check",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {fmt(value)} |")

    lines.extend(
        [
            "",
            "## Ligand Exact Matches",
            "",
            f"Canonical SMILES exact matches: {len(exact_df)}",
            "",
        ]
    )
    if len(exact_df):
        lines.extend(["| independent_index | independent_name | pdb_name |", "|---:|---|---|"])
        for _, row in exact_df.head(20).iterrows():
            lines.append(f"| {row['independent_index']} | {row['independent_name']} | {row['pdb_name']} |")
    else:
        lines.append("No canonical-SMILES exact matches were found.")

    lines.extend(["", "## Top Ligand Fingerprint Similarities", "", "| independent_index | independent_name | best_pdb_name | Tanimoto |", "|---:|---|---|---:|"])
    for _, row in top_ligands.iterrows():
        lines.append(
            f"| {row['independent_index']} | {row['independent_name']} | "
            f"{row['best_pdb_name']} | {row['best_similarity']:.4f} |"
        )

    lines.extend(["", "## Top RNA Sequence Window Identities", "", "| pdb_id | ligand | length | best identity to HIV RNA | best window |", "|---|---|---:|---:|---|"])
    for _, row in top_rnas.iterrows():
        lines.append(
            f"| {row['pdb_id']} | {row['ligand_resname']} | {row['pdb_sequence_length']} | "
            f"{row['best_window_identity']:.4f} | {row['best_window']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This check compares the actual PDB-derived contact pretraining samples against the R-SIM independent-test ligands and the fixed HIV RNA sequence used by the independent test loader.",
            "Exact ligand matches, high ligand fingerprint similarity, and high RNA sequence identity should be considered potential leakage risks and can motivate a de-overlapped contact pretraining subset.",
        ]
    )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Check overlap between PDB contact pretraining data and R-SIM independent test.")
    parser.add_argument("--contact-data-dir", default="dataset/pdb_contact_rna_only_500")
    parser.add_argument("--independent-csv", default="data/independent_data.csv")
    parser.add_argument("--out-dir", default="docs")
    parser.add_argument("--independent-rna-sequence", default=INDEPENDENT_RNA_SEQUENCE)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pdb_df = load_contact_samples(args.contact_data_dir)
    independent_df = pd.read_csv(args.independent_csv)

    pdb_ligands = prepare_ligands(
        pdb_df.assign(name=pdb_df["pdb_id"] + ":" + pdb_df["ligand_resname"]),
        smiles_col="smiles",
        name_col="name",
        source="pdb_contact",
    )
    independent_ligands = prepare_ligands(
        independent_df,
        smiles_col="SMILES",
        name_col="Name",
        source="independent",
    )

    exact_df, ligand_sim_df = ligand_overlap(independent_ligands, pdb_ligands)
    rna_df = rna_overlap(args.independent_rna_sequence, pdb_df)

    ligand_sim_df = ligand_sim_df.sort_values("best_similarity", ascending=False)
    rna_df = rna_df.sort_values("best_window_identity", ascending=False)

    exact_df.to_csv(tables_dir / "overlap_ligand_exact_matches.csv", index=False)
    ligand_sim_df.to_csv(tables_dir / "overlap_ligand_fingerprint_similarity.csv", index=False)
    rna_df.to_csv(tables_dir / "overlap_rna_sequence_identity.csv", index=False)

    summary = {
        "contact_samples": int(len(pdb_df)),
        "independent_test_ligands": int(len(independent_df)),
        "pdb_ligands_with_valid_fingerprint": int(sum(record["has_fingerprint"] for record in pdb_ligands)),
        "independent_ligands_with_valid_fingerprint": int(sum(record["has_fingerprint"] for record in independent_ligands)),
        "canonical_smiles_exact_matches": int(len(exact_df)),
        "independent_ligands_with_tanimoto_ge_0.90": int((ligand_sim_df["best_similarity"] >= 0.90).sum()),
        "independent_ligands_with_tanimoto_ge_0.80": int((ligand_sim_df["best_similarity"] >= 0.80).sum()),
        "max_ligand_tanimoto": float(ligand_sim_df["best_similarity"].max()) if len(ligand_sim_df) else 0.0,
        "mean_ligand_tanimoto_best_match": float(ligand_sim_df["best_similarity"].mean()) if len(ligand_sim_df) else 0.0,
        "pdb_rnas_with_window_identity_ge_0.90": int((rna_df["best_window_identity"] >= 0.90).sum()),
        "pdb_rnas_with_window_identity_ge_0.80": int((rna_df["best_window_identity"] >= 0.80).sum()),
        "max_rna_window_identity_to_independent_hiv": float(rna_df["best_window_identity"].max()) if len(rna_df) else 0.0,
        "mean_rna_window_identity_to_independent_hiv": float(rna_df["best_window_identity"].mean()) if len(rna_df) else 0.0,
    }

    (out_dir / "pdb_rsim_overlap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, exact_df, ligand_sim_df, rna_df, out_dir / "pdb_rsim_overlap_check.md")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
