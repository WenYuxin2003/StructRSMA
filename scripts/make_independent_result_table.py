import argparse
from statistics import mean, stdev

from summarize_independent_log import parse_log


SELECTIONS = {
    "best_pcc": lambda rows: max(rows, key=lambda row: row["pcc"]),
    "best_scc": lambda rows: max(rows, key=lambda row: row["scc"]),
    "best_rmse": lambda rows: min(rows, key=lambda row: row["rmse"]),
}


def fmt(values):
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +/- {stdev(values):.4f}"


def summarize_group(logs, selection):
    rows = []
    selector = SELECTIONS[selection]
    for log in logs:
        parsed = parse_log(log)
        if not parsed:
            raise ValueError(f"No epoch rows found in {log}")
        rows.append(selector(parsed))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Create a markdown mean/std table from DeepRSMA independent logs."
    )
    parser.add_argument(
        "--selection",
        choices=[*SELECTIONS.keys(), "all"],
        default="best_pcc",
        help="Which checkpoint-selection metric to summarize.",
    )
    parser.add_argument(
        "--group",
        action="append",
        nargs="+",
        metavar=("LABEL", "LOG"),
        required=True,
        help="Method label followed by one or more log paths.",
    )
    args = parser.parse_args()

    selections = list(SELECTIONS) if args.selection == "all" else [args.selection]
    print("| Method | Selection | Seeds | PCC | SCC | RMSE |")
    print("|---|---:|---:|---:|---:|---:|")
    for group in args.group:
        if len(group) < 2:
            raise ValueError("--group requires a label and at least one log")
        label, *logs = group
        for selection in selections:
            rows = summarize_group(logs, selection)
            print(
                f"| {label} | {selection} | {len(rows)} | "
                f"{fmt([row['pcc'] for row in rows])} | "
                f"{fmt([row['scc'] for row in rows])} | "
                f"{fmt([row['rmse'] for row in rows])} |"
            )


if __name__ == "__main__":
    main()
