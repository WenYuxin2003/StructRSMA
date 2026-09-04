import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_2026/audit/affinity_endpoints_v1"


def describe_frame(name, frame):
    values = pd.to_numeric(frame["pKd"], errors="coerce")
    return {
        "dataset": name,
        "n_rows": int(len(frame)),
        "n_finite_pkd": int(np.isfinite(values).sum()),
        "pkd_min": float(values.min()),
        "pkd_median": float(values.median()),
        "pkd_max": float(values.max()),
        "pkd_mean": float(values.mean()),
        "pkd_sd": float(values.std(ddof=1)),
        "has_raw_endpoint_column": bool(
            any(column.lower() in {"kd", "ki", "ka", "ic50", "ec50"} for column in frame.columns)
        ),
        "has_raw_unit_column": bool(any("unit" in column.lower() for column in frame.columns)),
        "has_assay_method_column": bool(any("assay" in column.lower() for column in frame.columns)),
        "has_reference_column": bool(
            any(token in column.lower() for column in frame.columns for token in ("reference", "pubmed", "doi"))
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((ROOT / "data/RSM_data").glob("*_dataset_v1.csv")):
        frame = pd.read_csv(path, sep="\t")
        if "pKd" in frame:
            rows.append(describe_frame(path.stem, frame))

    independent = pd.read_csv(ROOT / "data/independent_data.csv")
    independent_pkd = -np.log10(pd.to_numeric(independent["KD"], errors="coerce"))
    rows.append(
        {
            "dataset": "independent_data",
            "n_rows": int(len(independent)),
            "n_finite_pkd": int(np.isfinite(independent_pkd).sum()),
            "pkd_min": float(independent_pkd.min()),
            "pkd_median": float(independent_pkd.median()),
            "pkd_max": float(independent_pkd.max()),
            "pkd_mean": float(independent_pkd.mean()),
            "pkd_sd": float(independent_pkd.std(ddof=1)),
            "has_raw_endpoint_column": True,
            "has_raw_unit_column": False,
            "has_assay_method_column": False,
            "has_reference_column": False,
        }
    )
    pd.DataFrame(rows).to_csv(OUT / "affinity_dataset_audit.csv", index=False)

    audit = {
        "local_rsim_schema": {
            "stored_response": "pKd only",
            "raw_Kd_present": False,
            "raw_units_present": False,
            "assay_method_present": False,
            "per_entry_reference_present": False,
        },
        "independent_schema": {
            "stored_response": "KD",
            "conversion_in_code": "pKd = -log10(KD)",
            "inferred_unit": "molar, consistent with the source values and DeepRSMA preprocessing",
            "unit_column_present": False,
        },
        "source_method": {
            "R_SIM_database_scope": "The database contains Ka, Kd, Ki, IC50, and EC50 records.",
            "RSAPred_model_subset": "Only Kd-quantified entries were retained; all Kd values were converted to molar before -log10 transformation.",
            "DeepRSMA_statement": "The processed 1,439-pair benchmark uses pKd as negative log10 dissociation constant.",
        },
        "auditable_conclusion": (
            "The local 1,439-pair benchmark is a Kd-derived pKd task according to the source methods, "
            "but its distributed CSV no longer contains per-entry raw Kd, unit, assay method, or citation fields. "
            "Therefore formula/range checks are possible, whereas per-entry assay and unit re-auditing is not."
        ),
    }
    (OUT / "affinity_endpoint_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    markdown = """# Affinity endpoint and unit audit

## Local benchmark

The distributed R-SIM files used by this code contain a single processed response column, `pKd`. They do not retain per-entry raw Kd values, units, assay methods, or literature identifiers. Consequently, the numerical range and downstream conversion can be checked, but the original assay provenance cannot be reconstructed entry by entry from this repository alone.

## Source-defined endpoint

The R-SIM database itself contains multiple affinity endpoint types (Ka, Kd, Ki, IC50, and EC50). However, the RSAPred model-development subset from which the benchmark was processed retained only dissociation-constant (Kd) measurements, converted each value to molar units, and used `pKd = -log10(Kd)`. DeepRSMA likewise describes the 1,439-pair response as negative-log dissociation constant.

## Independent set

The independent CSV contains a `KD` column. The executed preprocessing uses `pKd = -log10(KD)` directly. The magnitude of the stored KD values is consistent with molar units, although the local CSV has no explicit unit column.

## Modeling implication

The task should be described specifically as Kd-derived pKd prediction, not as a generic mixture of pKa, pKi, IC50, or EC50. A per-assay sensitivity analysis is not supported by the released local schema and this limitation must be disclosed rather than inferred away.

Sources: R-SIM, Journal of Molecular Biology, DOI 10.1016/j.jmb.2022.167914; RSAPred model-development methods, Briefings in Bioinformatics 25, bbae002; DeepRSMA, Bioinformatics 40, btae678.
"""
    (OUT / "affinity_endpoint_audit.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
