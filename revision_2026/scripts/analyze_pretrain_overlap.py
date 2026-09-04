import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[2]
INDEPENDENT_RNA = "GGCAGAUCUGAGCCUGGGAGCUCUCUGCC"
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def parse_args():
    parser = argparse.ArgumentParser(description="Systematic Contact500/R-SIM overlap audit.")
    parser.add_argument("--contact-data-dir", type=Path, default=ROOT / "dataset/pdb_contact_rna_only_500")
    parser.add_argument("--rsim-csv", type=Path, default=ROOT / "data/RSM_data/All_sf_dataset_v1.csv")
    parser.add_argument("--independent-csv", type=Path, default=ROOT / "data/independent_data.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rna-threshold", type=float, default=0.8)
    parser.add_argument("--ligand-threshold", type=float, default=0.8)
    return parser.parse_args()


def normalize_rna(sequence):
    return str(sequence).upper().replace("T", "U")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "", "", None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        scaffold = "ACYCLIC_EXACT::" + canonical
    try:
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprint = generator.GetFingerprint(mol)
    except AttributeError:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return canonical, scaffold, fingerprint


def make_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 0.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0
    return aligner


def global_identity(left, right, aligner):
    left = normalize_rna(left)
    right = normalize_rna(right)
    if not left or not right:
        return 0.0
    distance = -float(aligner.score(left, right))
    return max(0.0, 1.0 - distance / max(len(left), len(right)))


def load_contact_samples(path):
    records = []
    for file_path in sorted(path.glob("*.pt")):
        item = torch.load(file_path, map_location="cpu", weights_only=False)
        meta = item.get("meta", {})
        records.append(
            {
                "sample_file": file_path.name,
                "pdb_id": str(meta.get("pdb_id", "")),
                "rna_sequence": normalize_rna(meta.get("sequence", "")),
                "smiles": str(meta.get("smiles", "")),
                "ligand_resname": str(meta.get("ligand_resname", "")),
            }
        )
    frame = pd.DataFrame(records)
    ligand_data = frame["smiles"].map(canonicalize)
    frame["canonical_smiles"] = ligand_data.map(lambda value: value[0])
    frame["scaffold"] = ligand_data.map(lambda value: value[1])
    frame["fingerprint"] = ligand_data.map(lambda value: value[2])
    return frame


def similarity_bin(value):
    if value < 0.4:
        return "[0.0,0.4)"
    if value < 0.6:
        return "[0.4,0.6)"
    if value < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def build_downstream_frames(rsim_csv, independent_csv):
    rsim = pd.read_csv(rsim_csv, sep="\t")
    rsim = rsim.rename(columns={"Target_RNA_sequence": "rna_sequence", "SMILES": "smiles"})
    rsim["source"] = "R-SIM_all"
    rsim["downstream_id"] = rsim["Entry_ID"].map(lambda value: str(int(value)) if float(value).is_integer() else str(value))

    independent = pd.read_csv(independent_csv)
    independent = independent.rename(columns={"SMILES": "smiles"})
    independent["rna_sequence"] = INDEPENDENT_RNA
    independent["source"] = "independent_test"
    independent["downstream_id"] = [
        f"independent_{index:03d}_{str(name).replace(' ', '_')}"
        for index, name in enumerate(independent["Name"].tolist())
    ]
    independent["pKd"] = -np.log10(independent["KD"].astype(float))
    columns = ["source", "downstream_id", "rna_sequence", "smiles", "pKd"]
    return rsim[columns].copy(), independent[columns].copy()


def analyze_downstream(frame, contact, aligner, caches, contact_side):
    rows = []
    for _, row in frame.iterrows():
        sequence = normalize_rna(row["rna_sequence"])
        smiles = str(row["smiles"])
        if sequence not in caches["rna"]:
            caches["rna"][sequence] = np.asarray(
                [global_identity(sequence, other, aligner) for other in contact["rna_sequence"]],
                dtype=float,
            )
        if smiles not in caches["ligand"]:
            canonical, scaffold, fingerprint = canonicalize(smiles)
            if fingerprint is None:
                similarities = np.zeros(len(contact), dtype=float)
            else:
                valid_indices = [index for index, fp in enumerate(contact["fingerprint"]) if fp is not None]
                similarities = np.zeros(len(contact), dtype=float)
                valid_fps = [contact.iloc[index]["fingerprint"] for index in valid_indices]
                if valid_fps:
                    similarities[valid_indices] = DataStructs.BulkTanimotoSimilarity(fingerprint, valid_fps)
            caches["ligand"][smiles] = (canonical, scaffold, similarities)

        rna_similarities = caches["rna"][sequence]
        canonical, scaffold, ligand_similarities = caches["ligand"][smiles]
        pair_scores = np.minimum(rna_similarities, ligand_similarities)
        best_rna = int(np.argmax(rna_similarities))
        best_ligand = int(np.argmax(ligand_similarities))
        best_pair = int(np.argmax(pair_scores))
        contact_side["rna"] = np.maximum(contact_side["rna"], rna_similarities)
        contact_side["ligand"] = np.maximum(contact_side["ligand"], ligand_similarities)
        contact_side["pair"] = np.maximum(contact_side["pair"], pair_scores)
        result = {
            "source": row["source"],
            "downstream_id": row["downstream_id"],
            "pKd": float(row["pKd"]),
            "rna_length": len(sequence),
            "canonical_smiles": canonical,
            "scaffold": scaffold,
            "max_rna_global_identity": float(rna_similarities[best_rna]),
            "best_rna_pdb_id": contact.iloc[best_rna]["pdb_id"],
            "max_ligand_tanimoto": float(ligand_similarities[best_ligand]),
            "best_ligand_pdb_id": contact.iloc[best_ligand]["pdb_id"],
            "ligand_exact_match": bool(canonical and canonical in set(contact["canonical_smiles"])),
            "scaffold_exact_match": bool(scaffold and scaffold in set(contact["scaffold"])),
            "max_pair_min_similarity": float(pair_scores[best_pair]),
            "best_pair_pdb_id": contact.iloc[best_pair]["pdb_id"],
            "best_pair_rna_identity": float(rna_similarities[best_pair]),
            "best_pair_ligand_tanimoto": float(ligand_similarities[best_pair]),
        }
        result["rna_similarity_bin"] = similarity_bin(result["max_rna_global_identity"])
        result["ligand_similarity_bin"] = similarity_bin(result["max_ligand_tanimoto"])
        result["pair_similarity_bin"] = similarity_bin(result["max_pair_min_similarity"])
        rows.append(result)
    return pd.DataFrame(rows)


def summarize(frame, rna_threshold, ligand_threshold):
    return {
        "samples": len(frame),
        "rna_identity_ge_threshold": int((frame["max_rna_global_identity"] >= rna_threshold).sum()),
        "ligand_tanimoto_ge_threshold": int((frame["max_ligand_tanimoto"] >= ligand_threshold).sum()),
        "pair_min_similarity_ge_threshold": int(
            (frame["max_pair_min_similarity"] >= min(rna_threshold, ligand_threshold)).sum()
        ),
        "exact_ligand_matches": int(frame["ligand_exact_match"].sum()),
        "exact_scaffold_matches": int(frame["scaffold_exact_match"].sum()),
        "rna_bins": frame["rna_similarity_bin"].value_counts().to_dict(),
        "ligand_bins": frame["ligand_similarity_bin"].value_counts().to_dict(),
        "pair_bins": frame["pair_similarity_bin"].value_counts().to_dict(),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact = load_contact_samples(args.contact_data_dir)
    rsim, independent = build_downstream_frames(args.rsim_csv, args.independent_csv)
    aligner = make_aligner()
    caches = {"rna": {}, "ligand": {}}
    contact_side = {
        "rna": np.zeros(len(contact), dtype=float),
        "ligand": np.zeros(len(contact), dtype=float),
        "pair": np.zeros(len(contact), dtype=float),
    }

    rsim_overlap = analyze_downstream(rsim, contact, aligner, caches, contact_side)
    independent_overlap = analyze_downstream(independent, contact, aligner, caches, contact_side)
    overlap = pd.concat([rsim_overlap, independent_overlap], ignore_index=True)
    overlap.to_csv(args.output_dir / "downstream_to_contact500_overlap.csv", index=False)

    contact_report = contact.drop(columns=["fingerprint"]).copy()
    contact_report["max_rna_identity_to_any_downstream"] = contact_side["rna"]
    contact_report["max_ligand_tanimoto_to_any_downstream"] = contact_side["ligand"]
    contact_report["max_pair_min_similarity_to_any_downstream_pair"] = contact_side["pair"]
    contact_report["exclude_global_or_rule"] = (
        (contact_side["rna"] >= args.rna_threshold)
        | (contact_side["ligand"] >= args.ligand_threshold)
    )
    contact_report.to_csv(args.output_dir / "contact500_to_downstream_overlap.csv", index=False)

    summary = {
        "contact_samples": len(contact),
        "criteria": {
            "rna": f"global normalized edit identity >= {args.rna_threshold}",
            "ligand": f"Morgan radius-2 2048-bit Tanimoto >= {args.ligand_threshold}",
            "scaffold": "Bemis-Murcko exact match",
            "pair": "maximum over Contact500 complexes of min(RNA identity, ligand Tanimoto)",
        },
        "R-SIM_all": summarize(rsim_overlap, args.rna_threshold, args.ligand_threshold),
        "independent_test": summarize(
            independent_overlap, args.rna_threshold, args.ligand_threshold
        ),
        "global_or_rule_contact_samples_excluded": int(contact_report["exclude_global_or_rule"].sum()),
        "global_or_rule_contact_samples_retained": int((~contact_report["exclude_global_or_rule"]).sum()),
    }
    (args.output_dir / "overlap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
