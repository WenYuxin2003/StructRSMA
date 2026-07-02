import argparse
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"^epo:\s*(\d+)\s+pcc:\s*([-+0-9.eE]+)\s+scc:\s*([-+0-9.eE]+)\s+rmse:\s*([-+0-9.eE]+)"
)


def read_text(path):
    data = Path(path).read_bytes()
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def parse_log(path):
    rows = []
    for line in read_text(path).splitlines():
        match = EPOCH_PATTERN.search(line)
        if match:
            rows.append(
                {
                    "epoch": int(match.group(1)),
                    "pcc": float(match.group(2)),
                    "scc": float(match.group(3)),
                    "rmse": float(match.group(4)),
                }
            )
    return rows


def print_best(label, row):
    print(
        f"{label}: epoch={row['epoch']} "
        f"pcc={row['pcc']:.10f} scc={row['scc']:.10f} rmse={row['rmse']:.10f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Summarize DeepRSMA independent-setting logs.")
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    for log in args.logs:
        rows = parse_log(log)
        print(f"\n{log}")
        print(f"epochs: {len(rows)}")
        if not rows:
            continue
        print_best("best_pcc", max(rows, key=lambda row: row["pcc"]))
        print_best("best_scc", max(rows, key=lambda row: row["scc"]))
        print_best("best_rmse", min(rows, key=lambda row: row["rmse"]))


if __name__ == "__main__":
    main()
