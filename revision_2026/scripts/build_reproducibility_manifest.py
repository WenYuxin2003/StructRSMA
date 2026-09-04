import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision_2026"
OUTPUT = REVISION / "reproducibility"


INCLUDE_ROOTS = [
    REVISION / "scripts",
    REVISION / "configs",
    REVISION / "splits_independent_scaffold_v1",
    REVISION / "splits_strict_v1",
    REVISION / "splits_contact_cutoff_common_v1",
    REVISION / "audit" / "contact500_v1",
    REVISION / "audit" / "overlap_v1",
    REVISION / "audit" / "affinity_endpoints_v1",
    REVISION / "environment",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for base in INCLUDE_ROOTS:
        if not base.exists():
            continue
        files = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in files:
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    csv_path = OUTPUT / "artifact_manifest_sha256.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "files": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "manifest": csv_path.relative_to(ROOT).as_posix(),
        "scope": [path.relative_to(ROOT).as_posix() for path in INCLUDE_ROOTS],
    }
    (OUTPUT / "artifact_manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
