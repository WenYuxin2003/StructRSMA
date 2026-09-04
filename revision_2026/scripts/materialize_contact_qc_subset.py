import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "revision_2026/audit/contact500_v1/contact500_manifest.csv"
SOURCE = ROOT / "dataset/pdb_contact_rna_only_500"
OUT = ROOT / "revision_2026/data/contact246_conservative_qc"
AUDIT_OUT = ROOT / "revision_2026/audit/contact246_conservative_qc"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MANIFEST)
    frame["pass_resolution"] = (frame["experimental_method"] == "SOLUTION NMR") | (
        frame["resolution_angstrom"].fillna(float("inf")) <= 4.0
    )
    frame["pass_no_link_record"] = frame["link_records_involving_selected_ligand"] == 0
    frame["pass_full_ligand_occupancy"] = frame["selected_ligand_partial_occupancy_atoms"] == 0
    frame["pass_unambiguous_altloc"] = frame["raw_altloc_atom_records"] == 0
    rule_columns = [
        "pass_resolution",
        "pass_no_link_record",
        "pass_full_ligand_occupancy",
        "pass_unambiguous_altloc",
    ]
    frame["keep_conservative_qc"] = frame[rule_columns].all(axis=1)
    frame["exclusion_reasons"] = frame.apply(
        lambda row: ";".join(column.removeprefix("pass_") for column in rule_columns if not row[column]),
        axis=1,
    )
    selected = frame[frame["keep_conservative_qc"]].copy()
    for source_name in selected["built_file"]:
        shutil.copy2(SOURCE / source_name, OUT / source_name)
    frame.to_csv(AUDIT_OUT / "contact_qc_decisions.csv", index=False)
    selected.to_csv(AUDIT_OUT / "contact246_manifest.csv", index=False)
    summary = {
        "source_positive_contact_complexes": int(len(frame)),
        "retained_complexes": int(len(selected)),
        "excluded_complexes": int((~frame["keep_conservative_qc"]).sum()),
        "rules": {
            "resolution": "X-ray resolution <=4.0 A; solution NMR retained",
            "linked_ligands": "exclude any selected ligand with a PDB LINK record",
            "partial_occupancy": "exclude selected ligands containing atoms with occupancy <0.999",
            "alternate_locations": "exclude structures containing any alternate-location atom records",
        },
        "interpretation": (
            "Conservative sensitivity subset. LINK is treated as a broad linked/covalent-risk flag, "
            "so this filter may exclude noncovalent coordination records as well."
        ),
    }
    (AUDIT_OUT / "contact246_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUT / "subset_manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "files": selected["built_file"].tolist(),
                "pdb_ids": selected["pdb_id"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
