# Data placement

This repository releases identifiers, curation decisions, and fixed splits but does not redistribute all third-party or generated data.

Expected local layout:

```text
data/RSM_data/All_sf_dataset_v1.csv
data/RSM_data/Viral_RNA_independent_dataset_v1.csv
data/independent_data.csv
data/representations_cv/*.npy
data/representations_independent/*.npy
data/pdb_contacts/pdb/*.pdb
dataset/pdb_contact_rna_only_500/
```

The PDB identifiers are in `data/pdb_contacts/pdb_ids_rna_only_500.txt`. The accepted complex manifest, exclusions, and curation policy are in `revision_2026/audit/contact500_v1/`. Exact R-SIM split membership is in `revision_2026/splits_independent_scaffold_v1/` and `revision_2026/splits_strict_v1/`.

R-SIM labels are modeled as pKd. The local processed benchmark does not preserve complete assay-level provenance; see `revision_2026/audit/affinity_endpoints_v1/` for the available audit and its limitations.

