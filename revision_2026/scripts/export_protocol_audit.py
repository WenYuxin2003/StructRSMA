import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision_2026"
MATRIX = REVISION / "configs/r2_independent_ablation_matrix.json"
OUTPUT = REVISION / "audit/r2_model_protocol_audit.csv"


def main():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    runs = matrix["runs"]
    rows = []
    for run in runs:
        output_dir = ROOT / run["output_dir"]
        config_path = output_dir / "config.json"
        summary_path = output_dir / "summary.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        counts = config.get("parameter_counts", {})
        rows.append(
            {
                "run_name": config.get("run_name", output_dir.name),
                "model": config.get("model_label"),
                "seed": config.get("seed"),
                "status": summary.get("status", "PENDING"),
                "train_n": config.get("split_sizes", {}).get("train"),
                "validation_n": config.get("split_sizes", {}).get("validation"),
                "test_n": config.get("split_sizes", {}).get("test"),
                "epochs_max": config.get("epochs"),
                "early_stopping_patience": config.get("patience"),
                "selection_metric": config.get("selection_metric"),
                "test_evaluations_during_training": config.get("test_evaluations_during_training"),
                "test_evaluations_total": summary.get("test_evaluations_total"),
                "contact_mode": config.get("contact_mode"),
                "contact_stat_mode": config.get("contact_stat_mode"),
                "contact_checkpoint": config.get("contact_checkpoint"),
                "contact_head_frozen": config.get("contact_head_frozen"),
                "shared_encoders_finetuned": config.get("shared_encoders_finetuned"),
                "total_parameters": counts.get("total"),
                "trainable_parameters": counts.get("trainable"),
                "contact_head_parameters": counts.get("contact_head_total"),
                "selected_epoch": summary.get("selected_epoch"),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(row["status"] == "COMPLETED" for row in rows)
    print(f"wrote {OUTPUT}; completed={complete}/{len(rows)}")


if __name__ == "__main__":
    main()
