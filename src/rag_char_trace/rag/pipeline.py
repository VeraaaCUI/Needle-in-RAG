from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rag_char_trace.data.io import QAItem
from rag_char_trace.index.tfidf import TfidfIndex
from rag_char_trace.llm.base import LLM, LLMConfig
from rag_char_trace.trace.logger import TraceRecord


@dataclass
class PromptSpec:
    system: str
    template: str


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def format_contexts(
    hits: List[Dict[str, Any]],
    *,
    prompt_chunk_chars: int,
    max_context_chars: int,
) -> Tuple[str, int]:
    """Format retrieved hits into a single context string.

    Returns (contexts_text, used_hits).

    Notes:
      - Each chunk text is truncated to prompt_chunk_chars for prompt-size control.
      - Total formatted context is capped by max_context_chars.
    """

    lines: List[str] = []
    used = 0
    cur_len = 0

    for h in hits:
        chunk_text = _truncate(h.get("text_full", ""), prompt_chunk_chars)

        # Stable passage id for citation in answers.
        # We use the retrieval rank as the passage id (P0, P1, ...).
        block = (
            f"[P{h['rank']}] score={h['score']:.4f} chunk_id={h['chunk_id']}\n"
            f"{chunk_text}"
        )

        # +2 for blank lines between blocks
        next_len = cur_len + len(block) + (2 if lines else 0)
        if max_context_chars > 0 and next_len > max_context_chars:
            break

        if lines:
            lines.append("")
        lines.append(block)
        cur_len = next_len
        used += 1

    return "\n".join(lines), used


def run_one(
    *,
    item: QAItem,
    index: TfidfIndex,
    llm: LLM,
    llm_cfg: LLMConfig,
    prompt: PromptSpec,
    top_k: int,
    prompt_max_hits: int | None = None,
    prompt_chunk_chars: int = 1000,
    max_context_chars: int = 12000,
    trace_chunk_chars: int = 512,
) -> TraceRecord:
    t0 = time.time()
    idx_scores = index.query(item.question, top_k=top_k)

    hits: List[Dict[str, Any]] = []
    for rank, (i, score) in enumerate(idx_scores):
        c = index.chunks[i]
        text_full = c.text or ""
        hits.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": c.chunk_id,
                "poisoned": bool((c.meta or {}).get("poisoned", False)),
                "meta": c.meta or {},
                # keep full text for prompt construction (not persisted to trace by default)
                "text_full": text_full,
                # persist only a preview to keep run logs compact
                "text_preview": _truncate(text_full, trace_chunk_chars),
                "text_len": len(text_full),
            }
        )

    hits_for_prompt = hits if prompt_max_hits is None else hits[: max(0, prompt_max_hits)]
    contexts, used_hits = format_contexts(
        hits_for_prompt,
        prompt_chunk_chars=prompt_chunk_chars,
        max_context_chars=max_context_chars,
    )
    user_prompt = prompt.template.format(question=item.question, contexts=contexts)

    t1 = time.time()
    completion = llm.generate(system=prompt.system, user=user_prompt, config=llm_cfg)
    t2 = time.time()

    record = TraceRecord(
        qid=item.qid,
        question=item.question,
        retrieval={
            "top_k": top_k,
            "prompt_max_hits": prompt_max_hits,
            "prompt_used_hits": used_hits,
            "prompt_chunk_chars": prompt_chunk_chars,
            "prompt_max_context_chars": max_context_chars,
            "trace_chunk_chars": trace_chunk_chars,
            "hits": [
                {
                    "rank": h["rank"],
                    "score": h["score"],
                    "chunk_id": h["chunk_id"],
                    "poisoned": h["poisoned"],
                    "meta": h["meta"],
                    "text": h["text_preview"],
                    "text_len": h["text_len"],
                }
                for h in hits
            ],
            "latency_s": t1 - t0,
        },
        generation={
            "system": prompt.system,
            "user": user_prompt,
            "completion": completion,
            "latency_s": t2 - t1,
        },
        meta={
            **item.meta,
            "answer": item.answer,
            "incorrect_answer": item.incorrect_answer,
        },
    )
    return record
