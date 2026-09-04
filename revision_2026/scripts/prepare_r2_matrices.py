import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "revision_2026/configs"


def independent_matrix():
    variants = [
        {
            "key": "deeprsma",
            "label": "DeepRSMA",
            "contact_mode": "none",
            "freeze_contact_head": True,
        },
        {
            "key": "contact500_no_sca",
            "label": "Contact500 pretraining without SCA",
            "checkpoint": "checkpoints/contact_pretrain_rna_only_500.pth",
            "contact_mode": "none",
            "freeze_contact_head": True,
        },
        {
            "key": "contact270_no_sca",
            "label": "Global-deoverlap Contact270 pretraining without SCA",
            "checkpoint": "checkpoints/contact_pretrain_global_deoverlap_r80_l80.pth",
            "contact_mode": "none",
            "freeze_contact_head": True,
        },
        {
            "key": "sca_random_head",
            "label": "SCA without contact pretraining",
            "contact_mode": "cmif_residual",
            "freeze_contact_head": True,
        },
        {
            "key": "structrsma_contact500",
            "label": "StructRSMA: Contact500 pretraining plus SCA",
            "checkpoint": "checkpoints/contact_pretrain_rna_only_500.pth",
            "contact_mode": "cmif_residual",
            "freeze_contact_head": True,
        },
        {
            "key": "structrsma_contact270",
            "label": "StructRSMA: Contact270 pretraining plus SCA",
            "checkpoint": "checkpoints/contact_pretrain_global_deoverlap_r80_l80.pth",
            "contact_mode": "cmif_residual",
            "freeze_contact_head": True,
        },
        {
            "key": "sca_shuffled_contact",
            "label": "Shuffled-contact pretraining plus SCA",
            "checkpoint": "checkpoints/contact_pretrain_rna_only_500_shuffle.pth",
            "contact_mode": "cmif_residual",
            "freeze_contact_head": True,
        },
        {
            "key": "matched_adapter_zero_contact",
            "label": "Parameter-matched residual adapter with zero contact statistics",
            "contact_mode": "cmif_residual",
            "contact_stat_mode": "zero",
            "freeze_contact_head": True,
        },
    ]
    runs = []
    split = "revision_2026/splits_independent_scaffold_v1/split_seed2026.json"
    for variant in variants:
        for seed in (1, 2, 3):
            run = {
                "protocol": "independent",
                "independent_split_file": split,
                "output_dir": f"revision_2026/results/r2_independent_ablation/{variant['key']}/seed{seed}",
                "run_name": f"r2_independent_{variant['key']}_seed{seed}",
                "model_label": variant["label"],
                "seed": seed,
                "split_seed": 2026,
                "epochs": 100,
                "patience": 15,
                "batch_size": 8,
                "lr": 6e-5,
                "weight_decay": 1e-5,
                "shuffle_train": True,
                "contact_mode": variant["contact_mode"],
                "contact_stat_mode": variant.get("contact_stat_mode", "predicted"),
                "freeze_contact_head": variant.get("freeze_contact_head", False),
            }
            if variant.get("checkpoint"):
                run["contact_checkpoint"] = variant["checkpoint"]
            runs.append(run)
    return {
        "name": "R2 strict independent ablation matrix",
        "selection_metric": "validation_rmse",
        "test_policy": "one evaluation after checkpoint selection",
        "epochs": 100,
        "patience": 15,
        "runs": runs,
    }


def cold_sca_matrix():
    runs = []
    for setting in ("cold_rna", "cold_scaffold", "double_cold"):
        for split_seed in (2026, 2027, 2028):
            training_seed = split_seed - 2025
            runs.append(
                {
                    "protocol": "clustered",
                    "split_file": f"revision_2026/splits_strict_v1/{setting}_seed{split_seed}.json",
                    "output_dir": (
                        f"revision_2026/results/r2_cold_structrsma/{setting}/split{split_seed}"
                    ),
                    "run_name": f"r2_{setting}_structrsma_split{split_seed}",
                    "model_label": "StructRSMA: Contact500 pretraining plus SCA",
                    "contact_checkpoint": "checkpoints/contact_pretrain_rna_only_500.pth",
                    "contact_mode": "cmif_residual",
                    "contact_stat_mode": "predicted",
                    "freeze_contact_head": True,
                    "seed": training_seed,
                    "split_seed": split_seed,
                    "epochs": 100,
                    "patience": 15,
                    "batch_size": 8,
                    "lr": 6e-5,
                    "weight_decay": 1e-5,
                    "shuffle_train": True,
                }
            )
    return {
        "name": "R2 cold generalization for full StructRSMA",
        "selection_metric": "validation_rmse",
        "test_policy": "one evaluation after checkpoint selection",
        "epochs": 100,
        "patience": 15,
        "runs": runs,
    }


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "r2_independent_ablation_matrix.json"
    path.write_text(json.dumps(independent_matrix(), indent=2), encoding="utf-8")
    print(path)
    cold_path = CONFIG_DIR / "r2_cold_structrsma_matrix.json"
    cold_path.write_text(json.dumps(cold_sca_matrix(), indent=2), encoding="utf-8")
    print(cold_path)


if __name__ == "__main__":
    main()
