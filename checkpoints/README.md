# Checkpoints

## Included

- `contact_pretrain_rna_only_500.pth`: Contact500 nucleotide-atom contact-pretraining initializer.
- `contact_pretrain_global_deoverlap_r80_l80.pth`: globally de-overlapped Contact270 initializer.
- `contact_pretrain_rna_only_500_shuffle.pth`: shuffled-contact negative-control initializer.

## Affinity checkpoints

Final validation-selected affinity checkpoints are distributed as GitHub Release assets:

https://github.com/WenYuxin2003/StructRSMA/releases

Recommended asset names:

```text
structrsma_full_sca_seed1.pt
structrsma_full_sca_seed2.pt
structrsma_full_sca_seed3.pt
deeprsma_strict_seed1.pt
deeprsma_strict_seed2.pt
deeprsma_strict_seed3.pt
```

Before replacing the public repository, upload the final files and add their exact asset URLs and SHA-256 values to `SHA256SUMS.txt`. Do not label a retrospective test-selected checkpoint as validation-selected.
