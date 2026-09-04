# Reviewer reproducibility index

This index maps the reproducibility requests from Reviewer 1 and Reviewer 2 to public repository artifacts. It contains no manuscript correspondence.

## Reviewer 1

| Request | Repository evidence |
|---|---|
| Explain de-overlap performance and chemical-space coverage | `revision_2026/audit/overlap_v1/`, `revision_2026/statistics/similarity_performance/`, Figure R2-1 |
| Ablate the four SCA contact statistics | `revision_2026/results/unified_sca/feature_ablation.csv`, `view_randomization.csv`, `make_unified_sca_figure.py` |
| Distinguish retrospective and validation-selected checkpoints | `revision_2026/audit/r2_model_protocol_audit.csv`, `deeprsma_reproduction_note.md`, strict training scripts |
| Release Contact500 construction, thresholds, and weights | `scripts/build_pdb_contact_dataset.py`, `revision_2026/audit/contact500_v1/`, `checkpoints/` |
| ACS reference formatting | Manuscript-level request; no manuscript files are distributed in this code repository |

## Reviewer 2

| Request | Repository evidence |
|---|---|
| RNA/ligand/scaffold/pair overlap and similarity-performance | `revision_2026/scripts/analyze_pretrain_overlap.py`, `audit/overlap_v1/`, `statistics/similarity_performance/` |
| PDB contact-data curation | `data/pdb_contacts/`, `scripts/build_pdb_contact_dataset.py`, `audit/contact500_v1/`, `audit/contact246_conservative_qc/` |
| Cold-RNA, cold-scaffold, and double-cold | `revision_2026/splits_strict_v1/`, `results/non_sca/`, `results/unified_sca/cold_results.csv` |
| Fair controls and simple baselines | `revision_2026/configs/`, `results/unified_sca/fair_control_ablation.csv`, `results/non_sca/` |
| Endpoint and unit audit | `revision_2026/audit/affinity_endpoints_v1/` |
| Repeated runs, confidence intervals, paired tests, per-target and Fisher-z | `revision_2026/results/non_sca/`, `revision_2026/scripts/summarize_r2_results.py` |
| Complex-level contact metrics, strata, and cutoff sensitivity | `revision_2026/statistics/contact/`, `contact_stratified/`, Figure R2-5 |
| Complete executable repository | root `README.md`, `environment.yml`, fixed splits, configs, scripts, manifests, and checkpoint instructions |

## Known metadata items

Rfam family assignments, complete assay-level endpoint provenance, and CCD semantic ligand categories require external database annotation. The released identity clusters, endpoint audit, and structural-risk flags are reproducible proxies and are explicitly labeled as such.

## Checkpoint-selection policy

Strict runs select checkpoints using validation RMSE only. Test examples are not evaluated during optimization and are evaluated once after checkpoint restoration. The contact head is frozen during affinity fine-tuning; shared multiview encoders are fine-tuned. Retrospectively selected exploratory checkpoints must not be described as unbiased generalization estimates.

