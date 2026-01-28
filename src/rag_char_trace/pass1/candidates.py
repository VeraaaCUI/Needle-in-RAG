from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

from rag_char_trace.pass1.types import Span
from rag_char_trace.pass1.textutil import find_all, numeric_and_citation_spans, salient_terms


_CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")
_ALLCAP_RE = re.compile(r"\b[A-Z]{2,10}\b")


def _add_matches(
    spans: List[Span],
    *,
    chunk_id: str,
    rank: int,
    p_text_start: int,
    chunk_text: str,
    label: str,
    needle: str,
    base_score: float,
) -> None:
    if not needle:
        return
    needle = needle.strip()
    if not needle:
        return

    ms = find_all(chunk_text, needle, case_insensitive=False)
    if not ms and needle.lower() != needle:
        ms = find_all(chunk_text, needle, case_insensitive=True)

    for a, b in ms:
        spans.append(
            Span(
                chunk_id=chunk_id,
                rank=rank,
                chunk_start=a,
                chunk_end=b,
                prompt_start=p_text_start + a,
                prompt_end=p_text_start + b,
                label=label,
                score=base_score,
            )
        )


def candidate_spans_for_chunk(
    *,
    row: Dict[str, Any],
    block: Dict[str, Any],
    chunk_text: str,
    max_candidates: int = 50,
    mode: str = "event",
    target_text: Optional[str] = None,
) -> List[Span]:
    """Generate candidate spans inside a prompt-used chunk.

    Two modes:

    - mode="event" (default, recommended): no oracle labels required.
        Uses the *observed answer string* (target_text, typically the current model answer) as the
        "event target" to trace back, plus cheap lexical cues.
        This is practical when a user reports an incorrect output: we can chase the output itself.

    - mode="oracle" (upper bound / ablation): uses dataset-provided incorrect answers (target/pool)
        in addition to event cues.

    Always includes:
      - numeric/citation fingerprints
      - salient question terms
      - a small set of answer-like capitalized phrases (fallback)

    Returns spans with both chunk-local and prompt-global coordinates.
    """
    gen = (row.get("generation", {}) or {})
    sig = (gen.get("signals", {}) or {})
    pred = str(sig.get("pred", "") or "")
    oracle_target = str(sig.get("target", "") or "")
    oracle_pool = list(sig.get("pool_incorrect_answers", []) or [])
    question = str(row.get("question", "") or "")

    rank = int(block.get("rank", -1))
    chunk_id = str(block.get("chunk_id", "") or "")
    p_text_start = int(block.get("prompt_text_start", -1))

    spans: List[Span] = []

    # --- Event target: the observed answer string we want to "explain/remove" ---
    ev_target = (target_text or pred or "").strip()
    if 1 <= len(ev_target) <= 80:
        _add_matches(
            spans,
            chunk_id=chunk_id,
            rank=rank,
            p_text_start=p_text_start,
            chunk_text=chunk_text,
            label="event_target_answer",
            needle=ev_target,
            base_score=3.0,
        )

    # --- Oracle (optional): targeted + pool incorrect answers ---
    if (mode or "").lower() == "oracle":
        _add_matches(
            spans,
            chunk_id=chunk_id,
            rank=rank,
            p_text_start=p_text_start,
            chunk_text=chunk_text,
            label="target_incorrect_answer",
            needle=oracle_target,
            base_score=3.2,
        )
        for ia in oracle_pool:
            _add_matches(
                spans,
                chunk_id=chunk_id,
                rank=rank,
                p_text_start=p_text_start,
                chunk_text=chunk_text,
                label="pool_incorrect_answer",
                needle=str(ia or ""),
                base_score=2.3,
            )

    # --- Predicted answer (sometimes differs from target_text if caller passes a different target) ---
    if 1 <= len(pred) <= 80 and pred.strip() != ev_target:
        _add_matches(
            spans,
            chunk_id=chunk_id,
            rank=rank,
            p_text_start=p_text_start,
            chunk_text=chunk_text,
            label="predicted_answer",
            needle=pred,
            base_score=1.6,
        )

    # --- Numeric / citation fingerprints ---
    for a, b, lab in numeric_and_citation_spans(chunk_text):
        spans.append(
            Span(
                chunk_id=chunk_id,
                rank=rank,
                chunk_start=a,
                chunk_end=b,
                prompt_start=p_text_start + a,
                prompt_end=p_text_start + b,
                label=lab,
                score=1.0,
            )
        )

    # --- A few salient question terms (lexical heatmap seed) ---
    for t in salient_terms(question, max_terms=8):
        _add_matches(
            spans,
            chunk_id=chunk_id,
            rank=rank,
            p_text_start=p_text_start,
            chunk_text=chunk_text,
            label="question_term",
            needle=t,
            base_score=0.6,
        )

    # --- Answer-like phrases (fallback for cases where ev_target is not present as a substring) ---
    # We keep these low weight and cap count implicitly via max_candidates + dedupe.
    for m in _CAP_PHRASE_RE.finditer(chunk_text):
        s, e = m.start(), m.end()
        if e - s < 4 or e - s > 40:
            continue
        spans.append(
            Span(
                chunk_id=chunk_id,
                rank=rank,
                chunk_start=s,
                chunk_end=e,
                prompt_start=p_text_start + s,
                prompt_end=p_text_start + e,
                label="cap_phrase",
                score=0.4,
            )
        )

    for m in _ALLCAP_RE.finditer(chunk_text):
        s, e = m.start(), m.end()
        if e - s < 2 or e - s > 12:
            continue
        spans.append(
            Span(
                chunk_id=chunk_id,
                rank=rank,
                chunk_start=s,
                chunk_end=e,
                prompt_start=p_text_start + s,
                prompt_end=p_text_start + e,
                label="allcaps",
                score=0.35,
            )
        )

    # Deduplicate by (prompt_start,prompt_end,label) keeping max score
    best: Dict[Tuple[int, int, str], Span] = {}
    for sp in spans:
        k = (sp.prompt_start, sp.prompt_end, sp.label)
        if k not in best or sp.score > best[k].score:
            best[k] = sp
    out = list(best.values())
    out.sort(key=lambda s: (-s.score, s.rank, s.prompt_start))
    return out[: max(1, int(max_candidates))]
