import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build_pdb_contact_dataset import (  # noqa: E402
    RNA_BASES,
    WATER_AND_IONS,
    read_structure_atoms,
    residue_key,
    select_ligand_atoms,
)


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit the exact Contact500 artifacts used by StructRSMA.")
    parser.add_argument("--id-file", type=Path, default=ROOT / "data/pdb_contacts/pdb_ids_rna_only_500.txt")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/pdb_contacts/metadata_rna_only_500.csv")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset/pdb_contact_rna_only_500")
    parser.add_argument("--pdb-dir", type=Path, default=ROOT / "data/pdb_contacts/pdb")
    parser.add_argument("--prepare-log", type=Path, default=ROOT / "runs/prepare_pdb_contacts_rna_only_500.log")
    parser.add_argument("--build-log", type=Path, default=ROOT / "runs/build_pdb_contact_rna_only_500.log")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rna-identity-threshold", type=float, default=0.8)
    return parser.parse_args()


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_skip_log(path, stage):
    rows = []
    pattern = re.compile(r"\[skip\]\s+([A-Za-z0-9]+):\s+(.+)")
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            rows.append({"pdb_id": match.group(1).lower(), "stage": stage, "reason": match.group(2)})
    return rows


def pdb_header(path):
    method = "UNKNOWN"
    resolution = None
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "EXPDTA":
                method = line[10:].strip().rstrip(";") or "UNKNOWN"
            elif line.startswith("REMARK   2 RESOLUTION."):
                match = re.search(r"RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS", line)
                if match:
                    resolution = float(match.group(1))
            elif record in {"ATOM", "HETATM"}:
                break
    return method, resolution


def canonical_ligand(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return "", ""
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        scaffold = "ACYCLIC_EXACT::" + canonical
    return canonical, scaffold


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[right] = left


def rna_cluster_count(sequences, threshold):
    sequences = sorted(set(sequences), key=lambda value: (-len(value), value))
    union_find = UnionFind(len(sequences))
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 0.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0
    linked = 0
    for left in range(len(sequences)):
        for right in range(left + 1, len(sequences)):
            distance = -float(aligner.score(sequences[left], sequences[right]))
            identity = 1.0 - distance / max(len(sequences[left]), len(sequences[right]), 1)
            if identity >= threshold:
                union_find.union(left, right)
                linked += 1
    roots = {union_find.find(index) for index in range(len(sequences))}
    return {"clusters": len(roots), "linked_pairs": linked, "unique_sequences": len(sequences)}


def raw_altloc_count(path):
    count = 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line[:6].strip() in {"ATOM", "HETATM"} and len(line) > 16 and line[16].strip():
                count += 1
    return count


def ligand_link_flag(path, row):
    ligand = (
        str(row["ligand_resname"]).strip().upper(),
        str(row["ligand_chain"]).strip(),
        str(row["ligand_resseq"]).strip(),
    )
    links = 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("LINK"):
                continue
            first = (line[17:20].strip().upper(), line[21].strip(), line[22:26].strip())
            second = (line[47:50].strip().upper(), line[51].strip(), line[52:56].strip())
            if ligand in {first, second}:
                links += 1
    return links


def quantiles(values):
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if values.size == 0:
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_ids = [line.strip().lower() for line in args.id_file.read_text().splitlines() if line.strip()]
    metadata = pd.read_csv(args.metadata, dtype=str).fillna("")
    metadata_lookup = {row["pdb_id"].lower(): row for _, row in metadata.iterrows()}
    built_files = sorted(args.dataset_dir.glob("*.pt"))
    built_lookup = {path.stem.split("_", 1)[1].lower(): path for path in built_files}

    exclusions = parse_skip_log(args.prepare_log, "metadata_preparation")
    exclusions.extend(parse_skip_log(args.build_log, "contact_map_build"))
    pd.DataFrame(exclusions).to_csv(args.output_dir / "contact500_exclusions.csv", index=False)

    manifest = []
    method_counter = Counter()
    for pdb_id, built_path in built_lookup.items():
        row = metadata_lookup[pdb_id]
        item = load_torch(built_path)
        contact = item["contact_map"].float()
        meta = item.get("meta", {})
        sequence = str(meta.get("sequence", "")).upper().replace("T", "U")
        canonical_smiles, scaffold = canonical_ligand(row["smiles"])
        pdb_path = args.pdb_dir / row["pdb_file"]
        method, resolution = pdb_header(pdb_path)
        method_counter[method] += 1

        atoms = read_structure_atoms(pdb_path)
        ligand_atoms = select_ligand_atoms(atoms, row)
        ligand_residues = {
            residue_key(atom)
            for atom in atoms
            if atom["record"] == "HETATM"
            and atom["resname"].upper() == row["ligand_resname"].upper()
        }
        noncanonical_on_rna_chain = {
            atom["resname"].upper()
            for atom in atoms
            if atom["chain"] == row["rna_chain"]
            and atom["resname"].upper() not in RNA_BASES
            and atom["resname"].upper() not in WATER_AND_IONS
            and atom["resname"].upper() != row["ligand_resname"].upper()
        }
        positives = int(contact.sum().item())
        pairs = int(contact.numel())
        manifest.append(
            {
                "pdb_id": pdb_id,
                "pdb_file": row["pdb_file"],
                "experimental_method": method,
                "resolution_angstrom": resolution,
                "rna_chain": row["rna_chain"],
                "rna_length": int(contact.shape[0]),
                "rna_sequence": sequence,
                "ligand_resname": row["ligand_resname"],
                "ligand_chain": row["ligand_chain"],
                "ligand_resseq": row["ligand_resseq"],
                "canonical_smiles": canonical_smiles,
                "bemis_murcko_scaffold": scaffold,
                "ligand_heavy_atoms": int(contact.shape[1]),
                "positive_contacts": positives,
                "possible_pairs": pairs,
                "contact_density": positives / max(pairs, 1),
                "nearest_rna_distance_angstrom": float(row["nearest_rna_distance"]),
                "same_resname_copies_in_structure": len(ligand_residues),
                "selected_ligand_partial_occupancy_atoms": sum(
                    float(atom.get("occupancy", 1.0)) < 0.999 for atom in ligand_atoms
                ),
                "raw_altloc_atom_records": raw_altloc_count(pdb_path),
                "link_records_involving_selected_ligand": ligand_link_flag(pdb_path, row),
                "noncanonical_residue_names_on_rna_chain": ";".join(sorted(noncanonical_on_rna_chain)),
                "contact_cutoff_angstrom": float(meta.get("contact_cutoff", 4.0)),
                "built_file": built_path.name,
            }
        )

    manifest_frame = pd.DataFrame(manifest).sort_values("pdb_id")
    manifest_frame.to_csv(args.output_dir / "contact500_manifest.csv", index=False)
    rna_cluster_stats = rna_cluster_count(
        manifest_frame["rna_sequence"].tolist(), args.rna_identity_threshold
    )
    summary = {
        "initial_rcsb_ids": len(initial_ids),
        "metadata_rows": len(metadata),
        "built_complexes": len(manifest_frame),
        "excluded_during_metadata_preparation": sum(
            row["stage"] == "metadata_preparation" for row in exclusions
        ),
        "excluded_during_contact_build": sum(row["stage"] == "contact_map_build" for row in exclusions),
        "unique_exact_rna_sequences": int(manifest_frame["rna_sequence"].nunique()),
        "rna_sequence_clusters_at_threshold": {
            "threshold": args.rna_identity_threshold,
            "identity_definition": "1 - global_edit_distance / max(sequence lengths)",
            **rna_cluster_stats,
        },
        "unique_ligand_resnames": int(manifest_frame["ligand_resname"].nunique()),
        "unique_canonical_ligands": int(manifest_frame["canonical_smiles"].replace("", np.nan).nunique()),
        "unique_scaffolds": int(manifest_frame["bemis_murcko_scaffold"].replace("", np.nan).nunique()),
        "experimental_methods": dict(method_counter),
        "resolution_angstrom": quantiles(manifest_frame["resolution_angstrom"].tolist()),
        "rna_length": quantiles(manifest_frame["rna_length"].tolist()),
        "ligand_heavy_atoms": quantiles(manifest_frame["ligand_heavy_atoms"].tolist()),
        "positive_contacts": quantiles(manifest_frame["positive_contacts"].tolist()),
        "contact_density": quantiles(manifest_frame["contact_density"].tolist()),
        "complexes_with_multiple_same_resname_copies": int(
            (manifest_frame["same_resname_copies_in_structure"] > 1).sum()
        ),
        "complexes_with_partial_occupancy_selected_ligand": int(
            (manifest_frame["selected_ligand_partial_occupancy_atoms"] > 0).sum()
        ),
        "complexes_with_altloc_records": int((manifest_frame["raw_altloc_atom_records"] > 0).sum()),
        "complexes_with_link_records_involving_ligand": int(
            (manifest_frame["link_records_involving_selected_ligand"] > 0).sum()
        ),
        "complexes_with_noncanonical_residue_names_on_rna_chain": int(
            (manifest_frame["noncanonical_residue_names_on_rna_chain"] != "").sum()
        ),
    }
    (args.output_dir / "contact500_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    handling = f"""# Contact500 curation audit

- Initial RCSB identifiers: {summary['initial_rcsb_ids']}.
- Metadata rows after structure/ligand inference: {summary['metadata_rows']}.
- Final contact complexes: {summary['built_complexes']}.
- RNA definition: canonical A/U/G/C residue names and the RA/RU/RG/RC/ADE/URI/GUA/CYT aliases in `RNA_BASES`.
- Modified nucleotides outside this whitelist are not mapped to canonical RNA tokens; the manifest flags other residue names on the selected RNA chain.
- Water and the explicit ion list `HOH,WAT,H2O,NA,K,MG,MN,ZN,CA,CL,BR,IOD` are excluded from ligand candidates.
- Cofactors and crystallization additives are not removed by a semantic CCD category filter in the submitted Contact500 construction. A non-water/non-ion HETATM residue meeting the 4--128 heavy-atom and <=6 A RNA-distance rules can be selected. This boundary must be stated explicitly.
- At most one ligand residue is retained per PDB entry: candidates are sorted by heavy-atom count and the largest is selected. The manifest reports other copies with the same residue name.
- Alternate conformations: blank, A, or 1 are retained; other altloc identifiers are discarded. Only the first structural model is used.
- Hydrogens/deuteriums are discarded. Missing atoms are not imputed. Samples that cannot form an RDKit molecule with an atom count matching the selected PDB ligand are rejected during dataset construction.
- Covalent ligands were not explicitly excluded. `LINK` records involving the selected ligand are reported as a screening flag, not as a definitive covalent-bond annotation.
- Contact label: any heavy atom in nucleotide i within 4.0 A of ligand atom j gives C_ij=1.
"""
    (args.output_dir / "contact500_curation_audit.md").write_text(handling, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
