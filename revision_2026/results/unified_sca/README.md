# Unified SCA mechanism and fair-control ablation

These tables provide the consolidated SCA fair-control, feature-ablation, view-randomization, and cold-generalization results:

- `fair_control_ablation.csv`: strict baseline, contact-only, no-pretraining, shuffled-contact, parameter-matched, and full SCA controls.
- `feature_ablation.csv`: removal, permutation, and zeroing of density, maxprob, rnafocus, and atomfocus.
- `view_randomization.csv`: RNA-side, ligand-side, all-view, contact-statistics, and combined randomization.
- `cold_results.csv`: Contact500-no-SCA and Full SCA under cold-RNA, cold-scaffold, and double-cold settings.

Blank uncertainty fields indicate unavailable values and must not be interpreted as zero. Claims of statistical significance require paired predictions or confidence intervals beyond these summary tables.
