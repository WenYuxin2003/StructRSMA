import argparse
import csv
import math
import os
import shlex
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.pdb_contact_dataset import molecule_to_data, rna_to_data  # noqa: E402


RNA_BASES = {
    "A": "A",
    "U": "U",
    "G": "G",
    "C": "C",
    "RA": "A",
    "RU": "U",
    "RG": "G",
    "RC": "C",
    "ADE": "A",
    "URI": "U",
    "GUA": "G",
    "CYT": "C",
}
WATER_AND_IONS = {
    "HOH",
    "WAT",
    "H2O",
    "NA",
    "K",
    "MG",
    "MN",
    "ZN",
    "CA",
    "CL",
    "BR",
    "IOD",
}


def normalize_cif_value(value):
    if value in {None, "", ".", "?"}:
        return ""
    return str(value)


def pdb_atom_line(atom, serial):
    atom_name = atom["atom_name"][:4]
    element = atom["element"][:2].strip().upper()
    if len(atom_name) < 4 and len(element) == 1:
        atom_field = f" {atom_name:<3}"
    else:
        atom_field = f"{atom_name:<4}"

    record = atom["record"][:6]
    altloc = (atom.get("altloc") or " ")[:1]
    resname = atom["resname"][:3]
    chain = (atom["chain"] or " ")[:1]
    resseq = str(atom["resseq"] or "1")[-4:]
    icode = (atom["icode"] or " ")[:1]
    x, y, z = atom["coord"].tolist()
    occupancy = float(atom.get("occupancy", 1.0))
    bfactor = float(atom.get("bfactor", 0.0))
    return (
        f"{record:<6}{serial:5d} {atom_field}{altloc}{resname:>3} {chain}"
        f"{resseq:>4}{icode:1}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element:>2}  "
    )


def parse_pdb_atom(line):
    record = line[0:6].strip()
    if record not in {"ATOM", "HETATM"}:
        return None
    altloc = line[16].strip()
    if altloc not in {"", "A", "1"}:
        return None
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    atom_name = line[12:16].strip()
    element = line[76:78].strip()
    if not element:
        element = "".join(ch for ch in atom_name if ch.isalpha())[:1]
    if element.upper() in {"H", "D"}:
        return None
    try:
        occupancy = float(line[54:60].strip() or 1.0)
        bfactor = float(line[60:66].strip() or 0.0)
    except ValueError:
        occupancy = 1.0
        bfactor = 0.0
    return {
        "record": record,
        "atom_name": atom_name,
        "altloc": altloc,
        "resname": line[17:20].strip(),
        "chain": line[21].strip(),
        "resseq": line[22:26].strip(),
        "icode": line[26].strip(),
        "coord": np.array([x, y, z], dtype=np.float32),
        "element": element,
        "occupancy": occupancy,
        "bfactor": bfactor,
        "line": line.rstrip("\n"),
    }


def read_pdb_atoms(path, first_model_only=True):
    atoms = []
    seen_model = False
    seen_atom_keys = set()
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[0:6].strip()
            if first_model_only and record == "MODEL":
                if seen_model:
                    break
                seen_model = True
                continue
            if first_model_only and seen_model and record == "ENDMDL":
                break

            atom = parse_pdb_atom(line)
            if atom is not None:
                atom_key = (
                    atom["record"],
                    atom["chain"],
                    atom["resseq"],
                    atom["icode"],
                    atom["resname"],
                    atom["atom_name"],
                )
                if atom_key in seen_atom_keys:
                    continue
                seen_atom_keys.add(atom_key)
                atoms.append(atom)
    return atoms


def split_cif_row(line):
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        return line.strip().split()


def iter_cif_atom_site_rows(path):
    tags = []
    in_loop = False
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "loop_":
                tags = []
                in_loop = True
                continue
            if not in_loop:
                continue
            if line.startswith("_"):
                tags.append(line.split()[0])
                continue
            if line == "#":
                tags = []
                in_loop = False
                continue
            if not tags or not tags[0].startswith("_atom_site."):
                continue

            values = split_cif_row(line)
            if len(values) < len(tags):
                continue
            row = {tag.replace("_atom_site.", ""): values[i] for i, tag in enumerate(tags)}
            yield row


def parse_cif_atom(row, serial):
    record = normalize_cif_value(row.get("group_PDB")).upper()
    if record not in {"ATOM", "HETATM"}:
        return None

    altloc = normalize_cif_value(row.get("label_alt_id"))
    if altloc not in {"", "A", "1"}:
        return None
    try:
        x = float(row["Cartn_x"])
        y = float(row["Cartn_y"])
        z = float(row["Cartn_z"])
    except (KeyError, ValueError):
        return None

    atom_name = normalize_cif_value(row.get("auth_atom_id")) or normalize_cif_value(row.get("label_atom_id"))
    element = normalize_cif_value(row.get("type_symbol")) or "".join(ch for ch in atom_name if ch.isalpha())[:1]
    if element.upper() in {"H", "D"}:
        return None

    resname = normalize_cif_value(row.get("auth_comp_id")) or normalize_cif_value(row.get("label_comp_id"))
    chain = normalize_cif_value(row.get("auth_asym_id")) or normalize_cif_value(row.get("label_asym_id"))
    resseq = normalize_cif_value(row.get("auth_seq_id")) or normalize_cif_value(row.get("label_seq_id"))
    icode = normalize_cif_value(row.get("pdbx_PDB_ins_code"))
    occupancy = float(normalize_cif_value(row.get("occupancy")) or 1.0)
    bfactor = float(normalize_cif_value(row.get("B_iso_or_equiv")) or 0.0)

    atom = {
        "record": record,
        "atom_name": atom_name,
        "altloc": altloc,
        "resname": resname,
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
        "coord": np.array([x, y, z], dtype=np.float32),
        "element": element,
        "occupancy": occupancy,
        "bfactor": bfactor,
    }
    atom["line"] = pdb_atom_line(atom, serial)
    return atom


def read_cif_atoms(path, first_model_only=True):
    atoms = []
    seen_atom_keys = set()
    first_model = None
    serial = 1
    for row in iter_cif_atom_site_rows(path):
        model = normalize_cif_value(row.get("pdbx_PDB_model_num"))
        if first_model_only and model:
            if first_model is None:
                first_model = model
            elif model != first_model:
                continue

        atom = parse_cif_atom(row, serial)
        if atom is None:
            continue
        atom_key = (
            atom["record"],
            atom["chain"],
            atom["resseq"],
            atom["icode"],
            atom["resname"],
            atom["atom_name"],
        )
        if atom_key in seen_atom_keys:
            continue
        seen_atom_keys.add(atom_key)
        atoms.append(atom)
        serial += 1
    return atoms


def read_structure_atoms(path, first_model_only=True):
    suffix = Path(path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return read_cif_atoms(path, first_model_only=first_model_only)
    return read_pdb_atoms(path, first_model_only=first_model_only)


def residue_key(atom):
    return atom["chain"], atom["resseq"], atom["icode"], atom["resname"]


def group_rna_residues(atoms, chain=None):
    residues = OrderedDict()
    for atom in atoms:
        base = RNA_BASES.get(atom["resname"].upper())
        if base is None:
            continue
        if chain and atom["chain"] != chain:
            continue
        key = residue_key(atom)
        residues.setdefault(key, {"base": base, "atoms": []})
        residues[key]["atoms"].append(atom)
    return list(residues.values())


def select_ligand_atoms(atoms, row):
    ligand_resname = row.get("ligand_resname", "").strip()
    ligand_chain = row.get("ligand_chain", "").strip()
    ligand_resseq = row.get("ligand_resseq", "").strip()

    candidates = []
    for atom in atoms:
        resname = atom["resname"].upper()
        if atom["record"] != "HETATM":
            continue
        if resname in WATER_AND_IONS or resname in RNA_BASES:
            continue
        if ligand_resname and resname != ligand_resname.upper():
            continue
        if ligand_chain and atom["chain"] != ligand_chain:
            continue
        if ligand_resseq and atom["resseq"] != ligand_resseq:
            continue
        candidates.append(atom)

    groups = defaultdict(list)
    for atom in candidates:
        groups[residue_key(atom)].append(atom)
    if not groups:
        return []

    # Use the largest ligand residue if the metadata did not fully specify one.
    return max(groups.values(), key=len)


def ligand_mol_from_pdb_atoms(ligand_atoms, smiles=None):
    pdb_block = "\n".join(atom["line"] for atom in ligand_atoms) + "\nEND\n"
    pdb_mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=True, proximityBonding=True)
    if pdb_mol is None:
        return None

    if not smiles:
        try:
            Chem.SanitizeMol(pdb_mol)
            return pdb_mol
        except Exception:
            return None

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        try:
            Chem.SanitizeMol(pdb_mol)
            return pdb_mol
        except Exception:
            return None

    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            Chem.SanitizeMol(pdb_mol)
            return pdb_mol
        except Exception:
            return None


def min_distance(coords_a, coords_b):
    best = float("inf")
    for a in coords_a:
        diff = coords_b - a
        dist = np.sqrt(np.sum(diff * diff, axis=1)).min()
        best = min(best, float(dist))
    return best


def build_rna_edges(residues, cutoff):
    edges = set()
    n = len(residues)
    for i in range(n - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))

    residue_coords = [np.stack([atom["coord"] for atom in residue["atoms"]]) for residue in residues]
    for i in range(n):
        for j in range(i + 1, n):
            if min_distance(residue_coords[i], residue_coords[j]) <= cutoff:
                edges.add((i, j))
                edges.add((j, i))
    return sorted(edges)


def build_contact_map(residues, ligand_atoms, cutoff):
    ligand_coords = np.stack([atom["coord"] for atom in ligand_atoms])
    contact = torch.zeros((len(residues), len(ligand_atoms)), dtype=torch.float32)
    for i, residue in enumerate(residues):
        rna_coords = [atom["coord"] for atom in residue["atoms"]]
        for j, ligand_atom in enumerate(ligand_atoms):
            if min_distance(rna_coords, ligand_atom["coord"][None, :]) <= cutoff:
                contact[i, j] = 1.0
    return contact


def load_embedding(row, embedding_dir, pdb_id, rna_chain, length):
    candidates = []
    if row.get("embedding_file"):
        candidates.append(Path(row["embedding_file"]))
    if embedding_dir:
        emb_dir = Path(embedding_dir)
        if pdb_id and rna_chain:
            candidates.append(emb_dir / f"{pdb_id}_{rna_chain}.npy")
        if pdb_id:
            candidates.append(emb_dir / f"{pdb_id}.npy")

    for path in candidates:
        if path.exists():
            emb = np.load(path)
            if emb.shape[0] >= length:
                return emb[:length]
            print(f"[warn] embedding too short for {pdb_id}: {path} shape={emb.shape}")
            break
    return np.zeros((length, 640), dtype=np.float32)


def iter_metadata(metadata_path, pdb_dir):
    if metadata_path:
        with open(metadata_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pdb_file = row.get("pdb_file") or f"{row.get('pdb_id', '')}.pdb"
                row["pdb_path"] = str(Path(pdb_dir) / pdb_file)
                yield row
    else:
        pdb_paths = list(Path(pdb_dir).glob("*.pdb")) + list(Path(pdb_dir).glob("*.cif"))
        for pdb_path in sorted(pdb_paths):
            yield {"pdb_id": pdb_path.stem, "pdb_file": pdb_path.name, "pdb_path": str(pdb_path)}


def build_one(row, args, sample_index):
    smiles = row.get("smiles", "").strip()

    pdb_path = Path(row["pdb_path"])
    atoms = read_structure_atoms(pdb_path)
    rna_chain = row.get("rna_chain", "").strip()
    residues = group_rna_residues(atoms, chain=rna_chain or None)
    if not residues:
        raise ValueError("no RNA residues found")
    if len(residues) > args.max_rna_len:
        residues = residues[: args.max_rna_len]

    ligand_atoms = select_ligand_atoms(atoms, row)
    if not ligand_atoms:
        raise ValueError("no ligand atoms found")

    mol = ligand_mol_from_pdb_atoms(ligand_atoms, smiles)
    if mol is None:
        raise ValueError("failed to build RDKit molecule from PDB ligand block")
    if not smiles:
        smiles = Chem.MolToSmiles(mol)
    if mol.GetNumAtoms() != len(ligand_atoms):
        raise ValueError(
            f"ligand atom count mismatch: RDKit={mol.GetNumAtoms()} PDB={len(ligand_atoms)}"
        )
    if len(ligand_atoms) > args.max_atoms:
        ligand_atoms = ligand_atoms[: args.max_atoms]

    sequence = "".join(residue["base"] for residue in residues)
    contact = build_contact_map(residues, ligand_atoms, args.contact_cutoff)
    positives = int(contact.sum().item())
    if positives < args.min_contacts:
        raise ValueError(f"too few contacts: {positives} < {args.min_contacts}")
    edges = build_rna_edges(residues, args.rna_contact_cutoff)

    pdb_id = row.get("pdb_id") or pdb_path.stem
    emb = load_embedding(row, args.embedding_dir, pdb_id, rna_chain, len(sequence))
    rna_data = rna_to_data(sequence, edges, emb=emb, entry_id=sample_index, target_id=pdb_id)
    molecule_data = molecule_to_data(mol, smiles, entry_id=sample_index, vocab_path=args.vocab_path)

    if molecule_data.x.size(0) != contact.size(1):
        contact = contact[:, : molecule_data.x.size(0)]

    meta = {
        "pdb_id": pdb_id,
        "pdb_file": str(pdb_path),
        "rna_chain": rna_chain,
        "ligand_resname": row.get("ligand_resname", ""),
        "smiles": smiles,
        "sequence": sequence,
        "contact_cutoff": args.contact_cutoff,
    }
    return {"rna": rna_data, "molecule": molecule_data, "contact_map": contact, "meta": meta}


def main():
    parser = argparse.ArgumentParser(description="Build nucleotide-atom contact labels from PDB RNA-ligand complexes.")
    parser.add_argument("--pdb-dir", required=True, help="Directory containing PDB files.")
    parser.add_argument("--metadata", help="CSV with pdb_id,pdb_file,smiles,rna_chain,ligand_resname,etc.")
    parser.add_argument("--out-dir", default="dataset/pdb_contact")
    parser.add_argument("--embedding-dir", help="Optional directory of RNA-FM .npy embeddings.")
    parser.add_argument("--vocab-path", default="data/smiles_vocab.pkl")
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--rna-contact-cutoff", type=float, default=8.0)
    parser.add_argument("--min-contacts", type=int, default=1)
    parser.add_argument("--max-rna-len", type=int, default=512)
    parser.add_argument("--max-atoms", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    for sample_index, row in enumerate(iter_metadata(args.metadata, args.pdb_dir)):
        try:
            sample = build_one(row, args, sample_index)
            pdb_id = sample["meta"]["pdb_id"]
            out_path = out_dir / f"{sample_index:05d}_{pdb_id}.pt"
            torch.save(sample, out_path)
            positives = int(sample["contact_map"].sum().item())
            total = sample["contact_map"].numel()
            print(f"[ok] {out_path.name} contacts={positives}/{total}")
            ok += 1
        except Exception as exc:
            skipped += 1
            label = row.get("pdb_id") or row.get("pdb_file") or row.get("pdb_path")
            print(f"[skip] {label}: {exc}")

    print(f"done: built={ok} skipped={skipped} out_dir={out_dir}")


if __name__ == "__main__":
    main()
