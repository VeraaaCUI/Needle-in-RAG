from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from rag_char_trace.eval.asr import load_run
from rag_char_trace.eval.grouped import group_asr_by_dataset_attack


def write_csv(path: str, datasets: List[str], attacks: List[str], cells: Dict[str, Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset"] + attacks)
        for d in datasets:
            row = [d]
            for a in attacks:
                row.append(f"{cells[d][a].asr:.3f}")
            w.writerow(row)


def write_latex(path: str, datasets: List[str], attacks: List[str], cells: Dict[str, Dict[str, Any]], caption: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    colspec = "l" + "".join(["c" for _ in attacks])
    lines: List[str] = []
    lines.append("\\begin{table}[t!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    header = "Dataset" + " & " + " & ".join(attacks) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for d in datasets:
        vals = [f"{cells[d][a].asr:.2f}" for a in attacks]
        lines.append(d + " & " + " & ".join(vals) + " \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run JSONL from scripts/02_run_rag.py")
    ap.add_argument("--out-json", default=None, help="Output grouped metrics JSON")
    ap.add_argument("--out-csv", default=None, help="Output CSV table")
    ap.add_argument("--out-tex", default=None, help="Output LaTeX table")
    ap.add_argument(
        "--caption",
        default="ASR of poisoning attacks without defenses.",
        help="LaTeX caption (used when --out-tex is set)",
    )
    args = ap.parse_args()

    rows = load_run(args.run)
    datasets, attacks, cells = group_asr_by_dataset_attack(rows)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "datasets": datasets,
            "attacks": attacks,
            "cells": {
                d: {a: {"n": cells[d][a].n, "asr": cells[d][a].asr} for a in attacks}
                for d in datasets
            },
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote grouped JSON to: {args.out_json}")

    if args.out_csv:
        write_csv(args.out_csv, datasets, attacks, cells)
        print(f"Wrote CSV to: {args.out_csv}")

    if args.out_tex:
        write_latex(args.out_tex, datasets, attacks, cells, caption=args.caption)
        print(f"Wrote LaTeX to: {args.out_tex}")

    # Always print a quick console view
    print("\nDatasets:", datasets)
    print("Attacks:", attacks)
    for d in datasets:
        row = [f"{cells[d][a].asr:.3f}" for a in attacks]
        print(d, " ".join(row))


if __name__ == "__main__":
    main()
