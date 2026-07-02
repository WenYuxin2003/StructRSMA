# Contact-Supervised DeepRSMA Paper Plan

## Working title

Contact-supervised pretraining improves RNA-small molecule affinity prediction.

## Core hypothesis

RNA-ligand binding affinity prediction benefits from structural supervision that
teaches the model where nucleotides and ligand atoms physically contact. Even
when the target affinity dataset has no binding-site annotation, contact labels
can be generated from independent PDB RNA-ligand complexes and used to pretrain
the DeepRSMA backbone.

## Current model story

Keep the original DeepRSMA architecture:

- RNA graph embedding
- RNA sequence embedding
- molecule graph embedding
- molecule sequence embedding
- cross-fusion module

Add an auxiliary nucleotide-atom contact prediction task after cross-fusion.
The contact head predicts a `RNA length x ligand atom count` map from pairwise
token features:

```text
score_ij = MLP([r_i, m_j, r_i * m_j])
```

Use PDB RNA-ligand complexes to pretrain the backbone and contact head, then
fine-tune on R-SIM affinity data. During affinity fine-tuning, load the
contact-pretrained backbone but skip the pKd head weights because they are not
supervised during contact pretraining.

## Current evidence

Independent-setting three-seed results:

| Method | Selection | Seeds | PCC | SCC | RMSE |
|---|---:|---:|---:|---:|---:|
| Original | best_pcc | 3 | 0.4866 +/- 0.0522 | 0.4912 +/- 0.0523 | 1.0584 +/- 0.1490 |
| Contact100SkipAff | best_pcc | 3 | 0.5195 +/- 0.0333 | 0.5257 +/- 0.0194 | 0.9754 +/- 0.0380 |
| Contact500SkipAff | best_pcc | 3 | 0.5816 +/- 0.0547 | 0.5749 +/- 0.0590 | 0.9152 +/- 0.0788 |
| Original | best_rmse | 3 | 0.4614 +/- 0.0613 | 0.4471 +/- 0.0855 | 0.9298 +/- 0.0318 |
| Contact100SkipAff | best_rmse | 3 | 0.4711 +/- 0.0250 | 0.4834 +/- 0.0306 | 0.9399 +/- 0.0226 |
| Contact500SkipAff | best_rmse | 3 | 0.5738 +/- 0.0495 | 0.5902 +/- 0.0204 | 0.8665 +/- 0.0400 |

Main positive result:

- Scaling RNA-ligand contact pretraining from 94 to 484 PDB complexes produces the current strongest result.
- Best-PCC protocol improves PCC by about 0.095 over the reproduced original.
- Best-PCC protocol improves SCC by about 0.084 over the reproduced original.
- Best-PCC protocol improves RMSE by about 0.143 over the reproduced original.
- Best-RMSE protocol improves PCC, SCC, and RMSE over the reproduced original.

## What is not yet the main claim

Contact-guided affinity heads were tested:

- Naive contact-guided head: weak.
- Residual contact calibration: strong in seed 1 but unstable across seed 2/3.

These variants are useful ablations, but the current robust claim is contact
pretraining plus clean transfer, not contact-guided affinity calibration.

Validation-selected refit was also tested for seed 1 before the 500-candidate
contact checkpoint was available. A 10% random validation
split was too small and selected checkpoints that did not reliably transfer to
the independent set. Under the full-train refit protocol, the original-style
model was stronger than the current ContactSkipAff model for this seed. This is
a useful negative result: the reproduced-protocol improvement is strong, but a
submission-quality claim still needs a more reliable validation protocol.

## Experiments needed before submission

1. Validation protocol

Use an internal validation split for epoch/model selection and evaluate the
independent test set only once. The current logs follow the original code style
of reporting test metrics every epoch, which is useful for reproduction but not
ideal for a paper claim.

2. Larger contact-pretraining corpus

Expand from 94 RNA-only PDB complexes to a larger filtered set. Track:

- number of complexes
- ligand diversity
- RNA length distribution
- contact density
- contact validation top-k precision / AUPRC

3. Contact-pretraining ablations

Run:

- original DeepRSMA
- contact-pretrained backbone with clean transfer
- full checkpoint load, as a negative/diagnostic control
- shuffled contact labels
- no graph view
- no sequence view
- no cross-fusion
- different contact cutoffs, for example 3.5, 4.0, 5.0 Angstrom

4. Contact task evaluation

Report the contact task separately:

- top-k precision
- AUPRC
- AUROC
- contact-map visualizations for representative PDB complexes

5. External comparison

Compare against available RNA-ligand affinity predictors or classical molecular
fingerprint baselines if they can be run on the same splits.

## Immediate coding roadmap

1. Run `main_independent_contact_val.py` for validation-selected independent results.
2. Add automatic result-table generation to every experiment run.
3. Scale PDB contact data beyond the first 100 RCSB hits.
4. Use a more robust validation strategy, for example larger validation splits,
   repeated validation splits, or validation-selected epoch ranges followed by
   full-train refit.
5. Add contact-task metrics and visualization scripts.
6. Re-run the final selected model with 3 to 5 seeds.

## Conservative paper claim

This work introduces structure-derived nucleotide-atom contact supervision as a
pretraining signal for RNA-small molecule affinity prediction. Without requiring
binding-site annotations in the affinity dataset, the method transfers physical
interaction knowledge from PDB RNA-ligand complexes into the DeepRSMA backbone
and improves independent-test affinity prediction under the reproduced
DeepRSMA protocol.
