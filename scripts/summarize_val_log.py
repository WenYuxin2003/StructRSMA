import argparse
import re
from pathlib import Path
from statistics import mean, stdev

from summarize_independent_log import read_text


BEST_VAL_PATTERN = re.compile(
    r"^BestVal:\s+epo:\s+(\d+)\s+pcc:\s+([-+0-9.eE]+)\s+scc:\s+([-+0-9.eE]+)\s+rmse:\s+([-+0-9.eE]+)"
)
FINAL_TEST_PATTERN = re.compile(
    r"^FinalTest:\s+selected_epoch:\s+(\d+)\s+pcc:\s+([-+0-9.eE]+)\s+scc:\s+([-+0-9.eE]+)\s+rmse:\s+([-+0-9.eE]+)"
)
REFIT_FINAL_TEST_PATTERN = re.compile(
    r"^RefitFinalTest:\s+selected_epoch:\s+(\d+)\s+pcc:\s+([-+0-9.eE]+)\s+scc:\s+([-+0-9.eE]+)\s+rmse:\s+([-+0-9.eE]+)"
)


def parse_val_log(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    best_val = None
    final_test = None
    refit_final_test = None
    for line in read_text(path).splitlines():
        best_match = BEST_VAL_PATTERN.search(line)
        if best_match:
            best_val = {
                "epoch": int(best_match.group(1)),
                "pcc": float(best_match.group(2)),
                "scc": float(best_match.group(3)),
                "rmse": float(best_match.group(4)),
            }
            continue
        final_match = FINAL_TEST_PATTERN.search(line)
        if final_match:
            final_test = {
                "selected_epoch": int(final_match.group(1)),
                "pcc": float(final_match.group(2)),
                "scc": float(final_match.group(3)),
                "rmse": float(final_match.group(4)),
            }
            continue
        refit_match = REFIT_FINAL_TEST_PATTERN.search(line)
        if refit_match:
            refit_final_test = {
                "selected_epoch": int(refit_match.group(1)),
                "pcc": float(refit_match.group(2)),
                "scc": float(refit_match.group(3)),
                "rmse": float(refit_match.group(4)),
            }
    return best_val, final_test, refit_final_test


def fmt(values):
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +/- {stdev(values):.4f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize validation-selected DeepRSMA logs.")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--prefer-refit", action="store_true")
    args = parser.parse_args()

    rows = []
    for log in args.logs:
        try:
            best_val, final_test, refit_final_test = parse_val_log(log)
        except FileNotFoundError:
            print(f"{log}: file not found")
            continue
        if args.prefer_refit and refit_final_test is not None:
            final_test = refit_final_test
        if final_test is None:
            print(f"{log}: FinalTest not found")
            continue
        rows.append((log, best_val, final_test, refit_final_test))

    if args.markdown:
        print("| Log | Selected epoch | Test PCC | Test SCC | Test RMSE |")
        print("|---|---:|---:|---:|---:|")
        for log, _, final_test, _ in rows:
            print(
                f"| {log} | {final_test['selected_epoch']} | "
                f"{final_test['pcc']:.4f} | {final_test['scc']:.4f} | {final_test['rmse']:.4f} |"
            )
        if rows:
            print(
                f"| mean +/- std | - | "
                f"{fmt([row[2]['pcc'] for row in rows])} | "
                f"{fmt([row[2]['scc'] for row in rows])} | "
                f"{fmt([row[2]['rmse'] for row in rows])} |"
            )
        return

    for log, best_val, final_test, refit_final_test in rows:
        print(f"\n{log}")
        if best_val is not None:
            print(
                f"best_val: epoch={best_val['epoch']} "
                f"pcc={best_val['pcc']:.10f} scc={best_val['scc']:.10f} rmse={best_val['rmse']:.10f}"
            )
        print(
            f"final_test: selected_epoch={final_test['selected_epoch']} "
            f"pcc={final_test['pcc']:.10f} scc={final_test['scc']:.10f} rmse={final_test['rmse']:.10f}"
        )
        if refit_final_test is not None:
            print(
                f"refit_final_test: selected_epoch={refit_final_test['selected_epoch']} "
                f"pcc={refit_final_test['pcc']:.10f} "
                f"scc={refit_final_test['scc']:.10f} rmse={refit_final_test['rmse']:.10f}"
            )
    if len(rows) > 1:
        print("\nFinalTest mean +/- std")
        print(f"PCC  {fmt([row[2]['pcc'] for row in rows])}")
        print(f"SCC  {fmt([row[2]['scc'] for row in rows])}")
        print(f"RMSE {fmt([row[2]['rmse'] for row in rows])}")


if __name__ == "__main__":
    main()
