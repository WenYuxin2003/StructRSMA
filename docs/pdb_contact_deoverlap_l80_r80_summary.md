# De-overlapped PDB Contact Dataset

| Item | Value |
|---|---:|
| input_samples | 484 |
| kept_samples | 440 |
| excluded_samples | 44 |
| ligand_threshold | 0.8 |
| rna_threshold | 0.8 |
| excluded_ligand_tanimoto_ge_threshold | 36 |
| excluded_ligand_exact_match | 23 |
| excluded_rna_identity_ge_threshold | 8 |
| max_ligand_tanimoto_kept | 0.775 |
| max_rna_window_identity_kept | 0.7931034482758621 |
| total_positive_contacts_kept | 12843 |
| out_dir | dataset\pdb_contact_rna_only_500_deoverlap_l80_r80 |

## Rule

A sample is removed if max ligand Morgan fingerprint Tanimoto to any independent-test ligand is >= 0.80, or if the best RNA window identity to the independent HIV RNA sequence is >= 0.80.

This de-overlapped subset is intended for leakage-robust contact pretraining; it does not overwrite the original Contact500 dataset or checkpoints.
