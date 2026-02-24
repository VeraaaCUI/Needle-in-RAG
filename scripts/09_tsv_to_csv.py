from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a tab-delimited file to comma CSV for Excel.")
    ap.add_argument("--in", dest="inp", required=True, help="Input TSV-like file (tab-delimited).")
    ap.add_argument("--out", required=True, help="Output CSV file (comma-delimited).")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open("r", encoding="utf-8", newline="") as fi, out.open("w", encoding="utf-8", newline="") as fo:
        reader = csv.reader(fi, delimiter="\t")
        writer = csv.writer(fo, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            writer.writerow(row)

    print(f"Converted to: {out}")


if __name__ == "__main__":
    main()
