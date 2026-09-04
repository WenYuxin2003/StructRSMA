import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "revision_2026/scripts/train_classical_baseline_strict.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run strict classical baseline matrices.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-cold", action="store_true")
    return parser.parse_args()


def build_runs(include_cold):
    runs = []
    models = ("sequence_smiles", "ligand_descriptor", "rna_ligand_descriptor")
    independent_split = ROOT / "revision_2026/splits_independent_scaffold_v1/split_seed2026.json"
    for model in models:
        for seed in (1, 2, 3):
            runs.append(
                {
                    "name": f"independent_{model}_seed{seed}",
                    "model": model,
                    "protocol": "independent",
                    "seed": seed,
                    "split_seed": 2026,
                    "independent_split_file": independent_split,
                }
            )
    if include_cold:
        split_types = {
            "cold_rna": "cold_rna",
            "cold_scaffold": "cold_scaffold",
            "double_cold": "double_cold",
        }
        for setting, stem in split_types.items():
            for split_seed in (2026, 2027, 2028):
                split_file = ROOT / f"revision_2026/splits_strict_v1/{stem}_seed{split_seed}.json"
                for model in models:
                    runs.append(
                        {
                            "name": f"{setting}_{model}_split{split_seed}",
                            "model": model,
                            "protocol": "clustered",
                            "seed": split_seed - 2025,
                            "split_seed": split_seed,
                            "split_file": split_file,
                        }
                    )
    return runs


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    statuses = []
    for run in build_runs(args.include_cold):
        output_dir = output_root / run["name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "summary.json").exists():
            statuses.append({"name": run["name"], "status": "SKIPPED_COMPLETED"})
            continue
        command = [
            sys.executable,
            str(TRAIN),
            "--model",
            run["model"],
            "--protocol",
            run["protocol"],
            "--output-dir",
            str(output_dir),
            "--seed",
            str(run["seed"]),
            "--split-seed",
            str(run["split_seed"]),
        ]
        if run.get("independent_split_file"):
            command.extend(["--independent-split-file", str(run["independent_split_file"])])
        if run.get("split_file"):
            command.extend(["--split-file", str(run["split_file"])])
        started = time.time()
        with (output_dir / "run.log").open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        status = {
            "name": run["name"],
            "status": "COMPLETED" if result.returncode == 0 else "FAILED",
            "return_code": result.returncode,
            "elapsed_seconds": time.time() - started,
        }
        statuses.append(status)
        (output_root / "matrix_status.json").write_text(
            json.dumps(statuses, indent=2), encoding="utf-8"
        )
        print(json.dumps(status), flush=True)
        if result.returncode != 0:
            raise SystemExit(f"Baseline failed: {run['name']}")


if __name__ == "__main__":
    main()
