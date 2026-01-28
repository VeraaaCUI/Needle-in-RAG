from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True, help="One or more metrics.json files")
    ap.add_argument("--out", required=True, help="Output .tex")
    args = ap.parse_args()

    rows = []
    for p in args.metrics:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        rows.append((Path(p).stem, obj))

    lines = []
    lines.append("\\begin{table}[t!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{ASR (attack success rate) of poisoning attacks without defenses (example exporter).}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Run & ASR \\uparrow & EM(correct) \\uparrow & RetrievedPoisonRate \\uparrow \\")
    lines.append("\\midrule")
    for name, obj in rows:
        lines.append(
            f"{name} & {obj['asr']:.3f} & {obj['em_correct']:.3f} & {obj['retrieved_poison_rate']:.3f} \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote LaTeX table to: {args.out}")


if __name__ == "__main__":
    main()
