# DeepRSMA reproduction status for Reviewer 2

## Code provenance

The baseline is a local reproduction built directly on the public DeepRSMA
implementation and data-processing classes available in this repository. The
four feature branches, cross-fusion transformer, and base affinity head are the
DeepRSMA implementation; the strict revision runner adds experiment control and
logging but does not redesign the baseline architecture.

## Published and local protocols

The DeepRSMA paper reports independent-test PCC 0.490, SCC 0.499, and RMSE
0.920, averaged over three seeds. Its independent setting uses 141 filtered
viral-RNA training pairs and 48 HIV-1 TAR test pairs.

The Reviewer-2 controlled reproduction uses the same 141/48 data construction,
but separates 28 of the 141 development pairs into a fixed scaffold-disjoint
validation set. Every neural variant therefore trains on 113 pairs, selects a
checkpoint using validation RMSE only, restores that checkpoint, and evaluates
the 48-pair independent test exactly once. The three-seed local DeepRSMA result
under this stricter rule is:

| Metric | Mean | SD | Published independent result |
|---|---:|---:|---:|
| PCC | 0.3875 | 0.0216 | 0.490 |
| SCC | 0.3448 | 0.0284 | 0.499 |
| RMSE | 0.9700 | 0.0120 | 0.920 |
| MAE | 0.7714 | 0.0131 | not reported |

## Interpretation of the difference

The local value is a reproduction under a deliberately stricter model-selection
protocol, not an exact rerun of the publication's checkpoint-selection path.
The smaller effective training set and validation-only checkpoint rule are the
main protocol differences. Consequently, the revision uses the locally
reproduced DeepRSMA values only for comparisons among variants that share the
same split, optimization budget, and selection rule; it does not present the
local result as a correction of the published benchmark.

## Machine-readable evidence

- Fixed development split: `revision_2026/splits_independent_scaffold_v1/split_seed2026.json`
- Baseline runs: `revision_2026/results/r2_independent_ablation/deeprsma/seed1` through `seed3`
- Fairness audit: `revision_2026/audit/r2_model_protocol_audit.csv`
