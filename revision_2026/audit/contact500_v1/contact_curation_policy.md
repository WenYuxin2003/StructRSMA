# Contact500 handling policy

This table records the behavior of the executable Contact500 construction code.
It intentionally distinguishes implemented filters from limitations that were
only audited or tested through a conservative sensitivity subset.

| Category | Contact500 behavior | Evidence status |
|---|---|---|
| Canonical and modified nucleotides | A/U/G/C and eight explicit aliases are canonicalized. Other modified residues are not silently converted and are flagged in the manifest. | Implemented and audited |
| Water and explicit ions | Water and the explicit ion list are excluded from ligand candidates. | Implemented |
| Cofactors and additives | No semantic CCD-category filter was used. Eligible hetero residues can therefore enter candidate selection. | Disclosed limitation |
| Covalent/linked ligands | Covalent ligands were not definitively excluded. Ligand-involving `LINK` records are reported as a broad risk flag. | Audited; conservative sensitivity filter available |
| Alternate locations | Blank/A/1 are retained and other altlocs discarded. | Implemented and audited |
| Missing atoms | Coordinates are not imputed; failed RDKit reconstruction or atom-count mismatch causes rejection. | Implemented |
| Multiple models/copies | First model only; largest valid ligand candidate only; repeated same-residue-name copies are counted in the audit. | Implemented and audited |

The full machine-readable version is `contact_curation_policy.csv`. The
conservative-QC subset excludes ligand `LINK` records, partial occupancy,
alternate-location records, and X-ray structures with resolution worse than
4.0 A. This reduces 484 positive-contact complexes to 246. Because PDB `LINK`
also describes some noncovalent coordination records, this is a deliberately
conservative sensitivity subset rather than a chemically definitive curation.
