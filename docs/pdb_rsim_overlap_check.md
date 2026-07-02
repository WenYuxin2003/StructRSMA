# PDB Contact Pretraining vs R-SIM Independent Test Overlap Check

## Summary

| Item | Value |
|---|---:|
| contact_samples | 484 |
| independent_test_ligands | 48 |
| pdb_ligands_with_valid_fingerprint | 484 |
| independent_ligands_with_valid_fingerprint | 48 |
| canonical_smiles_exact_matches | 23 |
| independent_ligands_with_tanimoto_ge_0.90 | 8 |
| independent_ligands_with_tanimoto_ge_0.80 | 9 |
| max_ligand_tanimoto | 1.0000 |
| mean_ligand_tanimoto_best_match | 0.3969 |
| pdb_rnas_with_window_identity_ge_0.90 | 6 |
| pdb_rnas_with_window_identity_ge_0.80 | 8 |
| max_rna_window_identity_to_independent_hiv | 1.0000 |
| mean_rna_window_identity_to_independent_hiv | 0.4490 |

## Ligand Exact Matches

Canonical SMILES exact matches: 23

| independent_index | independent_name | pdb_name |
|---:|---|---|
| 0 | Neomycin B | 1ei2:NMY |
| 0 | Neomycin B | 1i9v:NMY |
| 0 | Neomycin B | 2a04:NMY |
| 0 | Neomycin B | 2et4:NMY |
| 0 | Neomycin B | 2fcy:NMY |
| 1 | Paromomycin | 1fyp:PAR |
| 1 | Paromomycin | 1j7t:PAR |
| 1 | Paromomycin | 2mxs:PAR |
| 1 | Paromomycin | 2o3w:PAR |
| 1 | Paromomycin | 3bnq:PAR |
| 1 | Paromomycin | 3bnr:PAR |
| 1 | Paromomycin | 3c44:PAR |
| 1 | Paromomycin | 4zc7:PAR |
| 1 | Paromomycin | 5zej:PAR |
| 2 | Sisomycin | 4f8u:SIS |
| 2 | Sisomycin | 4f8v:SIS |
| 4 | Tobramycin | 1lc4:TOY |
| 6 | Neamine | 2et8:XXX |
| 6 | Neamine | 2f4s:XXX |
| 6 | Neamine | 2fcx:XXX |

## Top Ligand Fingerprint Similarities

| independent_index | independent_name | best_pdb_name | Tanimoto |
|---:|---|---|---:|
| 0 | Neomycin B | 1ei2:NMY | 1.0000 |
| 1 | Paromomycin | 1fyp:PAR | 1.0000 |
| 2 | Sisomycin | 4f8u:SIS | 1.0000 |
| 4 | Tobramycin | 1lc4:TOY | 1.0000 |
| 7 | Kanamycin | 2esi:KAN | 1.0000 |
| 6 | Neamine | 2et8:XXX | 1.0000 |
| 8 | Amikacin | 4p20:AKN | 1.0000 |
| 21 | Mitoxantrone | 2kgp:MIX | 1.0000 |
| 3 | Streptomycin | 1nta:SRY | 0.8871 |
| 33 | Thiazole Orange (tosylate) | 6e84:J0D | 0.7500 |

## Top RNA Sequence Window Identities

| pdb_id | ligand | length | best identity to HIV RNA | best window |
|---|---|---:|---:|---|
| 2l8h | L8H | 29 | 1.0000 | GGCAGAUCUGAGCCUGGGAGCUCUCUGCC |
| 1uud | P14 | 29 | 1.0000 | GGCAGAUCUGAGCCUGGGAGCUCUCUGCC |
| 1uts | P13 | 29 | 1.0000 | GGCAGAUCUGAGCCUGGGAGCUCUCUGCC |
| 1uui | P12 | 29 | 1.0000 | GGCAGAUCUGAGCCUGGGAGCUCUCUGCC |
| 1arj | ARG | 29 | 1.0000 | GGCAGAUCUGAGCCUGGGAGCUCUCUGCC |
| 1lvj | PMZ | 31 | 0.9310 | GCCAGAUCUGAGCCUGGGAGCUCUCUGGC |
| 1qd3 | BDG | 29 | 0.8966 | GCCAGAUUUGAGCCUGGGAGCUCUCUGGC |
| 6qiq | J48 | 7 | 0.8571 | GCAGAGC |
| 1aju | ARG | 30 | 0.7931 | GGCCAGAUUGAGCCUGGGAGCUCUCUGGC |
| 1akx | ARG | 30 | 0.7931 | GGCCAGAUUGAGCCUGGGAGCUCUCUGGC |

## Interpretation

This check compares the actual PDB-derived contact pretraining samples against the R-SIM independent-test ligands and the fixed HIV RNA sequence used by the independent test loader.
Exact ligand matches, high ligand fingerprint similarity, and high RNA sequence identity should be considered potential leakage risks and can motivate a de-overlapped contact pretraining subset.
