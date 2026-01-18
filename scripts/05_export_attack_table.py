from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag_char_trace.eval.asr import load_run
from rag_char_trace.eval.grouped import group_asr


def _sorted_keys(d: Dict[str, Any]) -> List[str]:
    return sorted([k for k in d.keys() if k is not None and k != ""])  # drop empty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run JSONL from scripts/02_run_rag.py")
    ap.add_argument("--out-json", required=True, help="Output grouped metrics JSON")
    ap.add_argument("--out-csv", required=False, help="Optional output CSV (wide)")
    ap.add_argument("--out-tex", required=False, help="Optional output LaTeX table")
    ap.add_argument("--dataset-key", default="dataset", help="meta key name for dataset")
    ap.add_argument("--attack-key", default="attack", help="meta key name for attack")
    ap.add_argument("--caption", default="ASR of poisoning attacks without defenses.")
    args = ap.parse_args()

    rows = load_run(args.run)
    grouped = group_asr(rows, dataset_key=args.dataset_key, attack_key=args.attack_key)

    datasets = _sorted_keys(grouped)
    attacks = _sorted_keys({a: 1 for ds in grouped.values() for a in ds.keys()})

    out_obj = {
        "datasets": datasets,
        "attacks": attacks,
        "cells": {
            ds: {atk: {"n": grouped[ds][atk].n, "asr": grouped[ds][atk].asr} for atk in grouped[ds]}
            for ds in grouped
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Dataset"] + attacks)
            for ds in datasets:
                row = [ds]
                for atk in attacks:
                    cell = grouped.get(ds, {}).get(atk)
                    row.append(f"{cell.asr:.2f}" if cell else "")
                w.writerow(row)

    if args.out_tex:
        Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("\\begin{table}[t!]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append(f"\\caption{{{args.caption}}}")
        colspec = "l" + "c" * len(attacks)
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append("\\toprule")
        lines.append("Dataset & " + " & ".join(attacks) + " \\")
        lines.append("\\midrule")
        for ds in datasets:
            vals = []
            for atk in attacks:
                cell = grouped.get(ds, {}).get(atk)
                vals.append(f"{cell.asr:.2f}" if cell else "-")
            lines.append(ds + " & " + " & ".join(vals) + " \\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        Path(args.out_tex).write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote grouped JSON to: {args.out_json}")
    if args.out_csv:
        print(f"Wrote CSV to: {args.out_csv}")
    if args.out_tex:
        print(f"Wrote LaTeX to: {args.out_tex}")


if __name__ == "__main__":
    main()
