## Notes

- This scaffold is intentionally minimal and focuses on reproducible trace.
- To reproduce paper-style ASR tables across multiple datasets/attacks, you can:
  1) iterate datasets under data/*
  2) run build_index once per dataset
  3) run rag for each attack/variant split (if desired)
  4) aggregate metrics and export tables

## Character-level (span-level) roadmap

- Add a module `src/rag_char_trace/spans/` that takes (question, retrieved chunks, completion) and outputs:
  - suspicious spans within retrieved chunks
  - attribution of generated answer spans to retrieved evidence spans

