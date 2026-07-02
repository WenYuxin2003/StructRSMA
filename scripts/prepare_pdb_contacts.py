import argparse
import csv
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_pdb_contact_dataset import (  # noqa: E402
    RNA_BASES,
    WATER_AND_IONS,
    group_rna_residues,
    read_structure_atoms,
    residue_key,
)


PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
CCD_SDF_URL = "https://files.rcsb.org/ligands/download/{resname}_ideal.sdf"


def download(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with urllib.request.urlopen(url, timeout=60) as response:
        out_path.write_bytes(response.read())
    return out_path


def download_structure(pdb_id, pdb_dir):
    pdb_id = pdb_id.lower()
    pdb_path = Path(pdb_dir) / f"{pdb_id}.pdb"
    cif_path = Path(pdb_dir) / f"{pdb_id}.cif"
    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        return pdb_path
    if cif_path.exists() and cif_path.stat().st_size > 0:
        return cif_path
    try:
        return download(PDB_URL.format(pdb_id=pdb_id.upper()), pdb_path)
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 404}:
            raise

    return download(CIF_URL.format(pdb_id=pdb_id.upper()), cif_path)


def smiles_from_ccd(resname, cache_dir):
    resname = resname.upper()
    sdf_path = Path(cache_dir) / f"{resname}_ideal.sdf"
    try:
        download(CCD_SDF_URL.format(resname=resname), sdf_path)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return ""

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def ligand_groups(atoms):
    groups = defaultdict(list)
    for atom in atoms:
        resname = atom["resname"].upper()
        if atom["record"] != "HETATM":
            continue
        if resname in WATER_AND_IONS or resname in RNA_BASES:
            continue
        groups[residue_key(atom)].append(atom)
    return groups


def nearest_rna_chain_and_distance(ligand_atoms, rna_by_chain):
    if not rna_by_chain:
        return "", float("inf")
    ligand_coords = np.stack([atom["coord"] for atom in ligand_atoms])
    best_chain = ""
    best_distance = float("inf")
    for chain, residues in rna_by_chain.items():
        rna_coords = np.stack([atom["coord"] for residue in residues for atom in residue["atoms"]])
        for start in range(0, len(rna_coords), 10000):
            chunk = rna_coords[start : start + 10000]
            diff = chunk[:, None, :] - ligand_coords[None, :, :]
            distance = float(np.sqrt(np.sum(diff * diff, axis=2)).min())
            if distance < best_distance:
                best_distance = distance
                best_chain = chain
    return best_chain, best_distance


def infer_rows_for_pdb(
    pdb_path,
    ligand_smiles_dir,
    max_ligand_rna_distance=None,
    min_ligand_atoms=4,
    max_ligand_atoms=128,
):
    pdb_path = Path(pdb_path)
    pdb_id = pdb_path.stem.lower()
    atoms = read_structure_atoms(pdb_path)

    chain_counts = Counter(atom["chain"] for atom in atoms if atom["resname"].upper() in RNA_BASES)
    rna_by_chain = {}
    for chain in chain_counts:
        residues = group_rna_residues(atoms, chain=chain)
        if residues:
            rna_by_chain[chain] = residues

    rows = []
    for key, ligand_atoms in ligand_groups(atoms).items():
        chain, resseq, icode, resname = key
        if len(ligand_atoms) < min_ligand_atoms:
            continue
        if max_ligand_atoms > 0 and len(ligand_atoms) > max_ligand_atoms:
            continue
        rna_chain, nearest_distance = nearest_rna_chain_and_distance(ligand_atoms, rna_by_chain)
        if (
            max_ligand_rna_distance is not None
            and max_ligand_rna_distance > 0
            and nearest_distance > max_ligand_rna_distance
        ):
            continue
        smiles = smiles_from_ccd(resname, ligand_smiles_dir)
        rows.append(
            {
                "pdb_id": pdb_id,
                "pdb_file": pdb_path.name,
                "smiles": smiles,
                "rna_chain": rna_chain,
                "ligand_resname": resname,
                "ligand_chain": chain,
                "ligand_resseq": resseq,
                "embedding_file": "",
                "ligand_atom_count": str(len(ligand_atoms)),
                "nearest_rna_distance": f"{nearest_distance:.3f}",
            }
        )
    rows.sort(key=lambda row: int(row["ligand_atom_count"]), reverse=True)
    return rows


def read_pdb_ids(args):
    ids = []
    if args.pdb_ids:
        ids.extend(args.pdb_ids)
    if args.pdb_id_file:
        with open(args.pdb_id_file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
    return [pdb_id.strip().lower() for pdb_id in ids if pdb_id.strip()]


def main():
    parser = argparse.ArgumentParser(description="Download PDB files and prepare contact metadata CSV.")
    parser.add_argument("--pdb-ids", nargs="*", help="PDB IDs, e.g. 1fmn 1uud.")
    parser.add_argument("--pdb-id-file", help="Text file with one PDB ID per line.")
    parser.add_argument("--pdb-dir", default="data/pdb_contacts/pdb")
    parser.add_argument("--metadata", default="data/pdb_contacts/metadata.csv")
    parser.add_argument("--ccd-dir", default="data/pdb_contacts/ccd")
    parser.add_argument("--max-ligands-per-pdb", type=int, default=1)
    parser.add_argument("--min-ligand-atoms", type=int, default=4)
    parser.add_argument("--max-ligand-atoms", type=int, default=128)
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=20.0,
        help="Skip downloaded structure files larger than this. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-ligand-rna-distance",
        type=float,
        default=6.0,
        help="Skip ligand residues whose closest heavy atom is farther from RNA. Use 0 to disable.",
    )
    args = parser.parse_args()

    pdb_ids = read_pdb_ids(args)
    if not pdb_ids:
        raise SystemExit("No PDB IDs provided. Use --pdb-ids or --pdb-id-file.")

    out_path = Path(args.metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id",
        "pdb_file",
        "smiles",
        "rna_chain",
        "ligand_resname",
        "ligand_chain",
        "ligand_resseq",
        "embedding_file",
        "ligand_atom_count",
        "nearest_rna_distance",
    ]
    written = 0
    handle = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for pdb_id in pdb_ids:
        try:
            pdb_path = download_structure(pdb_id, args.pdb_dir)
            if args.max_file_mb > 0 and pdb_path.stat().st_size > args.max_file_mb * 1024 * 1024:
                raise ValueError(f"structure file too large: {pdb_path.stat().st_size / 1024 / 1024:.1f} MB")
            rows = infer_rows_for_pdb(
                pdb_path,
                args.ccd_dir,
                args.max_ligand_rna_distance,
                args.min_ligand_atoms,
                args.max_ligand_atoms,
            )
            if not rows:
                print(f"[skip] {pdb_id}: no RNA ligand rows inferred", flush=True)
                continue
            selected = rows[: args.max_ligands_per_pdb]
            writer.writerows(selected)
            handle.flush()
            written += len(selected)
            for row in selected:
                print(
                    f"[ok] {pdb_id} ligand={row['ligand_resname']} "
                    f"rna_chain={row['rna_chain']} atoms={row['ligand_atom_count']} "
                    f"nearest={row['nearest_rna_distance']}A smiles={'yes' if row['smiles'] else 'no'}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[skip] {pdb_id}: {exc}", flush=True)

    handle.close()
    print(f"metadata written: {out_path} rows={written}")


if __name__ == "__main__":
    main()
