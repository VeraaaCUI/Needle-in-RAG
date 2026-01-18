from __future__ import annotations

import argparse

from rag_char_trace.data.io import load_chunks
from rag_char_trace.index.tfidf import build_tfidf_index, save_tfidf_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--out", required=True, help="Output index directory")
    ap.add_argument("--max-features", type=int, default=200_000)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks for dataset={args.dataset}")

    index = build_tfidf_index(chunks, max_features=args.max_features)
    save_tfidf_index(index, args.out)
    print(f"Saved TF-IDF index to: {args.out}")


if __name__ == "__main__":
    main()
