# StructRSMA

StructRSMA is a contact-supervised transfer framework for RNA-small-molecule binding-affinity prediction. It preserves the four-view DeepRSMA backbone (RNA sequence, RNA graph, molecule sequence, and molecule graph), adds nucleotide-atom contact pretraining from experimentally resolved PDB complexes, and uses a Structural Contact Adapter (SCA) for residual pKd calibration.

This release is organized for the ACS Omega revision. It contains executable preprocessing and de-overlap code, model and training code, fixed split definitions, experiment configurations, result tables, analysis scripts, compact reviewer figures, environment records, and PDB/R-SIM identifiers. Raw PDB structures, generated tensors, RNA-FM representations, and R-SIM source tables are not redistributed.

## Repository structure

```text
model/                         DeepRSMA backbone, contact head, and SCA
data/                          Dataset classes, feature code, and tokenizer
scripts/                       Contact-data construction utilities
revision_2026/scripts/         Strict training, audits, statistics, and figures
revision_2026/configs/         Executed experiment matrices
revision_2026/splits_*/        Fixed independent, cold, and contact splits
revision_2026/audit/           PDB curation, overlap, endpoint, and protocol audits
revision_2026/results/         Machine-readable final result tables
revision_2026/statistics/      Contact and similarity analyses
revision_2026/figures_r2/      Compact reviewer figures in PNG/SVG/PDF
checkpoints/                   Contact500 initializer and weight-release instructions
```

The manuscript source and private revision correspondence are intentionally not included.

## Environment

```bash
conda env create -f environment.yml
conda activate py37
```

The environment used for the revision is additionally recorded in `revision_2026/environment/`. GPU/CUDA details are descriptive; equivalent supported CUDA hardware may be used.

## Required external data

1. Obtain R-SIM from its original publication/DeepRSMA distribution and place the tables and precomputed RNA representations as described in `data/README.md`.
2. Download the PDB entries listed in `data/pdb_contacts/pdb_ids_rna_only_500.txt`.
3. The exact accepted Contact500 complexes and their metadata are listed in `revision_2026/audit/contact500_v1/contact500_manifest.csv`.

## Contact-map construction

The primary label is a binary nucleotide-atom matrix using a 4.0 A heavy-atom distance cutoff:

```bash
python scripts/build_pdb_contact_dataset.py \
  --pdb-dir data/pdb_contacts/pdb \
  --metadata data/pdb_contacts/metadata_rna_only_500.csv \
  --out-dir dataset/pdb_contact_rna_only_500 \
  --contact-cutoff 4.0

python revision_2026/scripts/audit_contact500.py \
  --output-dir revision_2026/audit/contact500_v1
```

Modified residues, ions, alternate conformations, missing atoms, linked ligands, and multiple copies are documented in `revision_2026/audit/contact500_v1/contact_curation_policy.md` and `.csv`.

## Leakage-aware overlap and de-overlap

The main similarity definitions are:

- RNA: normalized global edit identity, threshold 0.80.
- Ligand: Morgan fingerprint (radius 2, 2048 bits) Tanimoto, threshold 0.80.
- Scaffold: exact Bemis-Murcko scaffold.
- Pair-level similarity: minimum of RNA identity and ligand Tanimoto.
- Global de-overlap: exclude a contact complex when RNA identity >=0.80, ligand Tanimoto >=0.80, or exact scaffold overlap is present.

```bash
python revision_2026/scripts/analyze_pretrain_overlap.py \
  --output-dir revision_2026/audit/overlap_v1

python revision_2026/scripts/materialize_contact_subset.py
```

The resulting Contact484-to-Contact270 decisions are retained in the overlap audit and subset manifests.

## Fixed splits

- Independent scaffold-disjoint validation: `revision_2026/splits_independent_scaffold_v1/`.
- Cold-RNA, cold-scaffold, and double-cold: `revision_2026/splits_strict_v1/`.
- Grouped contact evaluation: `revision_2026/splits_contact_cutoff_common_v1/`.

Cold split seeds are 2026, 2027, and 2028. Exact R-SIM `Entry_ID` membership is stored in JSON/CSV files.

## Strict affinity protocol

All compared neural variants use the same split, optimization budget, early-stopping rule, and validation-only checkpoint selection. The test set is not evaluated during training and is evaluated once after restoring the validation-RMSE checkpoint. During affinity training, the pretrained contact head is frozen and the shared multiview encoders are fine-tuned.

```bash
python revision_2026/scripts/prepare_r2_matrices.py
python revision_2026/scripts/run_affinity_matrix.py \
  --matrix revision_2026/configs/r2_independent_ablation_matrix.json
python revision_2026/scripts/run_affinity_matrix.py \
  --matrix revision_2026/configs/r2_cold_structrsma_matrix.json
python revision_2026/scripts/run_classical_baseline_matrix.py
```

The consolidated fair-control, feature-ablation, view-randomization, and cold SCA results are in `revision_2026/results/unified_sca/`.

## Contact evaluation and statistics

```bash
python revision_2026/scripts/summarize_contact_results.py
python revision_2026/scripts/analyze_contact_strata.py
python revision_2026/scripts/summarize_r2_results.py \
  --bootstrap-iterations 2000 --bootstrap-seed 2026
python revision_2026/scripts/analyze_similarity_performance.py
```

Contact evaluation is performed per complex and macro-averaged. Reported metrics include AUPRC, AUROC, MCC, F1, precision, recall, P@k, R@k, confidence intervals, data strata, and 3.5/4.0/4.5/5.0 A cutoff sensitivity.

## Figures

Existing overlap, curation, and contact-evaluation figures:

```bash
python revision_2026/scripts/make_r2_figures.py \
  --figures similarity curation contact
```

Unified SCA mechanism figure from the released final summary tables:

```bash
python revision_2026/scripts/make_unified_sca_figure.py
```

## Checkpoints

`checkpoints/contact_pretrain_rna_only_500.pth` is the Contact500 pretrained initializer. Final affinity checkpoints should be attached to the GitHub release described in `checkpoints/README.md`; SHA-256 values must be updated after upload.

## Reproducibility index

See `REVIEWER_REPRODUCIBILITY.md` for the one-to-one mapping between Reviewer 1/2 requests and repository artifacts.

## Citation

Please cite the StructRSMA manuscript and the original DeepRSMA publication when using this code.

