# Contact500 curation audit

- Initial RCSB identifiers: 500.
- Metadata rows after structure/ligand inference: 486.
- Final contact complexes: 484.
- RNA definition: canonical A/U/G/C residue names and the RA/RU/RG/RC/ADE/URI/GUA/CYT aliases in `RNA_BASES`.
- Modified nucleotides outside this whitelist are not mapped to canonical RNA tokens; the manifest flags other residue names on the selected RNA chain.
- Water and the explicit ion list `HOH,WAT,H2O,NA,K,MG,MN,ZN,CA,CL,BR,IOD` are excluded from ligand candidates.
- Cofactors and crystallization additives are not removed by a semantic CCD category filter in the submitted Contact500 construction. A non-water/non-ion HETATM residue meeting the 4--128 heavy-atom and <=6 A RNA-distance rules can be selected. This boundary must be stated explicitly.
- At most one ligand residue is retained per PDB entry: candidates are sorted by heavy-atom count and the largest is selected. The manifest reports other copies with the same residue name.
- Alternate conformations: blank, A, or 1 are retained; other altloc identifiers are discarded. Only the first structural model is used.
- Hydrogens/deuteriums are discarded. Missing atoms are not imputed. Samples that cannot form an RDKit molecule with an atom count matching the selected PDB ligand are rejected during dataset construction.
- Covalent ligands were not explicitly excluded. `LINK` records involving the selected ligand are reported as a screening flag, not as a definitive covalent-bond annotation.
- Contact label: any heavy atom in nucleotide i within 4.0 A of ligand atom j gives C_ij=1.
