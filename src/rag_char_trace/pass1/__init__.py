"""
Pass-1: Span-level poison traceback and mitigation (no LLM judge, no Shapley).

This package implements a practical, black-box influence-based pipeline:
  - Trigger on suspicious generation events (e.g., PAR/T_ASR).
  - Chunk-level occlusion to find the most influential passage.
  - Candidate span generation inside the passage (string matches + numeric/citation heuristics).
  - Targeted bisection refinement to localize a minimal char span.
  - Replay generation on sanitized prompt and log before/after signals.
"""
