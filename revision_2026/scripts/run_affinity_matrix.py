import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "revision_2026/scripts/train_affinity_strict.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a logged strict-affinity experiment matrix.")
    parser.add_argument("--matrix", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    matrix_path = args.matrix.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    status_path = matrix_path.with_name(matrix_path.stem + "_status.json")
    statuses = []

    for run in matrix["runs"]:
        output_dir = (ROOT / run["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            statuses.append(
                {
                    "run_name": run["run_name"],
                    "status": "SKIPPED_COMPLETED",
                    "output_dir": str(output_dir),
                }
            )
            continue

        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--protocol",
            run["protocol"],
            "--output-dir",
            str(output_dir),
            "--run-name",
            run["run_name"],
            "--model-label",
            run["model_label"],
            "--seed",
            str(run["seed"]),
            "--split-seed",
            str(run.get("split_seed", 2026)),
            "--epochs",
            str(run.get("epochs", matrix.get("epochs", 300))),
            "--patience",
            str(run.get("patience", matrix.get("patience", 40))),
            "--batch-size",
            str(run.get("batch_size", matrix.get("batch_size", 8))),
            "--lr",
            str(run.get("lr", matrix.get("lr", 6e-5))),
            "--weight-decay",
            str(run.get("weight_decay", matrix.get("weight_decay", 1e-5))),
            "--val-ratio",
            str(run.get("val_ratio", matrix.get("val_ratio", 0.2))),
        ]
        if run.get("split_file"):
            command.extend(["--split-file", str((ROOT / run["split_file"]).resolve())])
        if run.get("independent_split_file"):
            command.extend(
                [
                    "--independent-split-file",
                    str((ROOT / run["independent_split_file"]).resolve()),
                ]
            )
        if run.get("contact_checkpoint"):
            command.extend(
                ["--contact-checkpoint", str((ROOT / run["contact_checkpoint"]).resolve())]
            )
        if run.get("contact_mode"):
            command.extend(["--contact-mode", run["contact_mode"]])
        if run.get("contact_stat_mode"):
            command.extend(["--contact-stat-mode", run["contact_stat_mode"]])
        if run.get("freeze_contact_head", False):
            command.append("--freeze-contact-head")
        if run.get("shuffle_train", matrix.get("shuffle_train", False)):
            command.append("--shuffle-train")

        started = time.time()
        driver_log = output_dir / "driver.log"
        with driver_log.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        status = {
            "run_name": run["run_name"],
            "status": "COMPLETED" if completed.returncode == 0 else "FAILED",
            "return_code": completed.returncode,
            "elapsed_seconds": time.time() - started,
            "output_dir": str(output_dir),
            "command": command,
        }
        statuses.append(status)
        status_path.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
        print(json.dumps(status), flush=True)
        if completed.returncode != 0:
            raise SystemExit(
                f"Run {run['run_name']} failed. Inspect {driver_log}; no automatic retry was attempted."
            )

    status_path.write_text(json.dumps(statuses, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
