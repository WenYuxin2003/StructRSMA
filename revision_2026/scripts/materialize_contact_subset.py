import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Materialize an audited Contact500 subset without altering source files.")
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-column", default="exclude_global_or_rule")
    parser.add_argument("--keep-value", default="false")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.audit_csv)
    expected = str(args.keep_value).strip().lower()
    keep = frame[args.keep_column].map(lambda value: str(value).strip().lower() == expected)
    selected = frame[keep].copy()
    manifest = []
    for _, row in selected.iterrows():
        source = args.source_dir / row["sample_file"]
        destination = args.output_dir / source.name
        shutil.copy2(source, destination)
        manifest.append(
            {
                "file_name": source.name,
                "pdb_id": row["pdb_id"],
                "source_sha256": sha256(source),
                "copied_sha256": sha256(destination),
            }
        )
    payload = {
        "audit_csv": str(args.audit_csv.resolve()),
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "rule": f"{args.keep_column} == {args.keep_value}",
        "selected_samples": len(selected),
        "manifest": manifest,
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
