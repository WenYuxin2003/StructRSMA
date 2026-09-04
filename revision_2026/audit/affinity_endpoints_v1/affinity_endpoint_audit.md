# Affinity endpoint and unit audit

## Local benchmark

The distributed R-SIM files used by this code contain a single processed response column, `pKd`. They do not retain per-entry raw Kd values, units, assay methods, or literature identifiers. Consequently, the numerical range and downstream conversion can be checked, but the original assay provenance cannot be reconstructed entry by entry from this repository alone.

## Source-defined endpoint

The R-SIM database itself contains multiple affinity endpoint types (Ka, Kd, Ki, IC50, and EC50). However, the RSAPred model-development subset from which the benchmark was processed retained only dissociation-constant (Kd) measurements, converted each value to molar units, and used `pKd = -log10(Kd)`. DeepRSMA likewise describes the 1,439-pair response as negative-log dissociation constant.

## Independent set

The independent CSV contains a `KD` column. The executed preprocessing uses `pKd = -log10(KD)` directly. The magnitude of the stored KD values is consistent with molar units, although the local CSV has no explicit unit column.

## Modeling implication

The task should be described specifically as Kd-derived pKd prediction, not as a generic mixture of pKa, pKi, IC50, or EC50. A per-assay sensitivity analysis is not supported by the released local schema and this limitation must be disclosed rather than inferred away.

Sources: R-SIM, Journal of Molecular Biology, DOI 10.1016/j.jmb.2022.167914; RSAPred model-development methods, Briefings in Bioinformatics 25, bbae002; DeepRSMA, Bioinformatics 40, btae678.
