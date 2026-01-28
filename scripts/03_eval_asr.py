from __future__ import annotations

import argparse
import json

from rag_char_trace.eval.asr import compute_metrics, load_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_run(args.run)
    m = compute_metrics(rows)

    # New naming scheme (recommended for papers / reports)
    out = {
        "n": m.n,

        # Attack success metrics
        "T_ASR": m.targeted_asr,        # Targeted Answer Attack Success Rate
        "PAR": m.any_poison_rate,       # Poison Adoption Rate (Non-targeted ASR; matches any incorrect in prompt-used hits)
        "P0_ASR": m.p0_poison_rate,     # Top-1 Poison Follow Rate (matches P0 incorrect)
        "UNK": m.unknown_rate,          # Abstention rate (UNKNOWN)
        "CONFUSION": m.confusion_rate,  # Neither gold nor any incorrect in prompt-used hits

        # Correctness (clean) metrics
        "EM": m.em_correct,
        "EM_contains": m.em_contains,

        # Diagnostics
        "target_in_prompt_rate": m.target_in_prompt_rate,
        "avg_target_rank_in_used": m.avg_target_rank_in_used,
        "avg_distinct_incorrects_in_used": m.avg_distinct_incorrects_in_used,
        "retrieved_poison_rate": m.retrieved_poison_rate,
        "prompt_poison_rate": m.prompt_poison_rate,

        # Backward-compatible keys (legacy)
        "asr": m.targeted_asr,
        "targeted_asr": m.targeted_asr,
        "any_poison_rate": m.any_poison_rate,
        "p0_poison_rate": m.p0_poison_rate,
        "unknown_rate": m.unknown_rate,
        "confusion_rate": m.confusion_rate,
        "em_correct": m.em_correct,
        "em_contains": m.em_contains,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to: {args.out}")


if __name__ == "__main__":
    main()
