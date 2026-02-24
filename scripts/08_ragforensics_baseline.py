#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGForensics-style baseline (paper-faithful at chunk level) adapted for rag_char_trace.

What this script does:
- Reads an existing Pass0 run JSONL produced by scripts/02_run_rag.py.
- Defines an "event" when the model's extracted answer matches (leniently) at least one
  used hit's meta.incorrect_answer. Those matching chunks are the event-target set.
- Runs an LLM-judge (local model via Ollama) over the used hits, predicting YES/NO:
  does this context try to induce the model's answer?
- Produces a Pass1-like JSONL with:
    pass1.triggered
    pass1.chunk_tests (ranking of chunks by judge)
    pass1.selected_chunk (top responsible chunk: earliest retrieval-rank among predicted YES)

Char metrics:
- RAGForensics does NOT output character-level spans in the paper.
- To avoid N/A in your Char metrics, we optionally map the selected chunk to a conservative
  span baseline:
    --span-mode full_chunk  (default): attribution_span = entire chunk [0, len(text)]
    --span-mode none        : omit spans (Char metrics become N/A / n_char_evaluable=0)
    --span-mode answer      : (NOT conservative) answer-string match span (for debugging only)

This file does NOT modify/override any existing pass0/pass1 logic. It is a standalone baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import requests


_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^0-9a-zA-Z]+")


def _norm(s: str) -> str:
    s = (s or "").strip().casefold()
    if not s:
        return ""
    s = _NONWORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _match(pred: str, cand: str) -> bool:
    """Lenient match: cand in pred OR pred in cand after normalization."""
    p = _norm(pred)
    c = _norm(cand)
    if not p or not c:
        return False
    if c in p:
        return True
    if len(p) >= 3 and p in c:
        return True
    return False


def _extract_answer(completion: str) -> str:
    comp = completion or ""
    for ln in comp.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("answer:"):
            return s.split(":", 1)[1].strip()
    for ln in comp.splitlines():
        s = ln.strip()
        if s:
            return s
    return ""


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _used_hits(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    retr = rec.get("retrieval", {}) or {}
    hits = retr.get("hits", []) or []
    k_used = retr.get("prompt_used_hits", retr.get("prompt_max_hits", 10))
    k_used = _as_int(k_used, 10)
    return hits[: max(0, k_used)]


def _event_targets(event_answer: str, used_hits: List[Dict[str, Any]]) -> List[str]:
    pairs: List[Tuple[int, str]] = []
    for i, h in enumerate(used_hits):
        cid = (h.get("chunk_id", "") or "").strip()
        if not cid:
            continue
        m = h.get("meta", {}) or {}
        ia = (m.get("incorrect_answer", "") or "").strip()
        if ia and _match(event_answer, ia):
            r = _as_int(h.get("rank", i), i)
            pairs.append((r, cid))
    # fallback: match in hit.text
    if not pairs and event_answer:
        for i, h in enumerate(used_hits):
            cid = (h.get("chunk_id", "") or "").strip()
            txt = (h.get("text", "") or "")
            if cid and txt and _match(txt, event_answer):
                r = _as_int(h.get("rank", i), i)
                pairs.append((r, cid))
    return [cid for _, cid in sorted(pairs, key=lambda x: x[0])]


def _ollama_generate(
    *,
    base_url: str,
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response", "") or "")


def _judge_prompt(question: str, response: str, context: str) -> Tuple[str, str]:
    system = (
        "You are a strict binary classifier.\n"
        "You MUST output exactly one token: YES or NO.\n"
        "Do not add punctuation, explanations, or extra words.\n"
    )
    user = (
        "Decide whether the CONTEXT tries to induce an answer consistent with RESPONSE, regardless of correctness.\n"
        "Output YES if the context states/suggests/claims that the answer should be RESPONSE (or strongly supports it).\n"
        "Otherwise output NO.\n\n"
        f"QUESTION: {question}\n"
        f"RESPONSE: {response}\n"
        f"CONTEXT: {context}\n\n"
        "YES or NO:"
    )
    return system, user


def _parse_yes_no(text: str) -> str:
    t = (text or "").strip().upper()
    m = re.findall(r"\b(YES|NO)\b", t)
    if m:
        return m[-1]
    if "YES" in t and "NO" not in t:
        return "YES"
    if "NO" in t and "YES" not in t:
        return "NO"
    return "NO"


def _find_answer_span(text: str, answer: str) -> Optional[Tuple[int, int]]:
    """Find first occurrence of answer in text (case-insensitive), flexible whitespace."""
    if not text or not answer:
        return None
    ans = answer.strip()
    if not ans:
        return None
    toks = re.split(r"\s+", ans)
    toks = [t for t in toks if t]
    if not toks:
        return None
    pattern = r"\s+".join(re.escape(t) for t in toks)
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        idx = text.casefold().find(ans.casefold())
        if idx >= 0:
            return (idx, idx + len(ans))
        return None
    return (m.start(), m.end())


def _cache_key(question: str, response: str, context: str) -> str:
    h = hashlib.sha256()
    h.update((question or "").encode("utf-8"))
    h.update(b"\0")
    h.update((response or "").encode("utf-8"))
    h.update(b"\0")
    h.update((context or "").encode("utf-8"))
    return h.hexdigest()


def _load_cache(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with jsonlines.open(str(p), "r") as r:
        for obj in r:
            k = (obj.get("key", "") or "").strip()
            if k:
                cache[k] = obj
    return cache


def _append_cache(path: Optional[str], obj: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(p), "a") as w:
        w.write(obj)


def _load_chunk_texts(chunks_path: Optional[str]) -> Dict[str, str]:
    if not chunks_path:
        return {}
    p = Path(chunks_path)
    if not p.exists():
        return {}
    out: Dict[str, str] = {}
    with jsonlines.open(str(p), "r") as r:
        for j in r:
            cid = (j.get("chunk_id", "") or "").strip()
            if not cid:
                continue
            out[cid] = (j.get("text", "") or "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGForensics-style baseline over Pass0 run (local Ollama judge).")
    ap.add_argument("--run", required=True, help="Pass0 run JSONL (output of scripts/02_run_rag.py)")
    ap.add_argument("--out", required=True, help="Output JSONL with pass1 fields (RAGForensics baseline).")
    ap.add_argument("--max-hits", type=int, default=None, help="Max used hits to judge per record (default: prompt_used_hits).")
    ap.add_argument("--cache", default=None, help="Optional jsonl cache for judge calls (recommended).")

    # optional chunks file (for accurate full_chunk span length)
    ap.add_argument("--chunks", default=None, help="Optional chunks.jsonl to get full chunk text (recommended for span-mode full_chunk).")

    # span mode for conservative Char metrics (not in paper)
    ap.add_argument("--span-mode", choices=["full_chunk", "none", "answer"], default="full_chunk",
                    help="How to emit spans for Char metrics: full_chunk (conservative), none (omit spans), answer (debug/strong).")

    # local LLM (Ollama)
    ap.add_argument("--ollama-model", required=True, help="Ollama model name (e.g., gemma:7b, llama3:8b).")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Generation temperature for judge.")
    ap.add_argument("--max-tokens", type=int, default=4, help="Max tokens for judge output (YES/NO).")
    ap.add_argument("--timeout-s", type=float, default=120.0, help="HTTP timeout for Ollama calls.")
    args = ap.parse_args()

    cache = _load_cache(args.cache)
    chunk_texts = _load_chunk_texts(args.chunks)

    in_path = Path(args.run)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    total_llm_calls = 0

    with jsonlines.open(str(in_path), "r") as r, jsonlines.open(str(out_path), "w") as w:
        for rec in r:
            n += 1
            q = (rec.get("question", "") or "")
            comp = (rec.get("generation", {}) or {}).get("completion", "") or ""
            event_answer = _extract_answer(comp)

            used = _used_hits(rec)
            if args.max_hits is not None:
                used = used[: max(0, int(args.max_hits))]

            targets = _event_targets(event_answer, used)
            triggered = bool(event_answer) and (event_answer.strip().upper() != "UNKNOWN") and bool(targets)

            p1: Dict[str, Any] = {
                "triggered": triggered,
                "baseline": "ragforensics_paperfaithful_chunk",
                "event_answer": event_answer,
                "event_targets": targets,
                "span_mode": args.span_mode,
            }

            if not triggered:
                rec2 = dict(rec)
                rec2["pass1"] = p1
                w.write(rec2)
                continue

            before = total_llm_calls

            judged: List[Dict[str, Any]] = []
            for i, h in enumerate(used):
                cid = (h.get("chunk_id", "") or "").strip()
                ctx = (h.get("text", "") or "")
                rank = _as_int(h.get("rank", i), i)

                if not cid:
                    continue

                key = _cache_key(q, event_answer, ctx)
                if key in cache:
                    label = (cache[key].get("label", "NO") or "NO").upper()
                else:
                    system, prompt = _judge_prompt(q, event_answer, ctx)
                    raw = _ollama_generate(
                        base_url=args.ollama_url,
                        model=args.ollama_model,
                        system=system,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout_s,
                    )
                    total_llm_calls += 1
                    label = _parse_yes_no(raw)
                    obj = {"key": key, "label": label, "raw": raw}
                    cache[key] = obj
                    _append_cache(args.cache, obj)

                judged.append({"chunk_id": cid, "rank": rank, "judge_label": label, "judge_score": 1.0 if label == "YES" else 0.0})

            record_calls = total_llm_calls - before
            p1["llm_calls"] = record_calls

            # Rank chunk_tests: YES first, then NO; tie-break by retrieval rank
            chunk_tests = sorted(judged, key=lambda x: (-float(x.get("judge_score", 0.0)), int(x.get("rank", 10**9))))
            p1["chunk_tests"] = [{"chunk_id": x["chunk_id"]} for x in chunk_tests]

            # Responsible chunk: earliest retrieval rank among predicted YES
            yes_hits = [x for x in judged if x.get("judge_label") == "YES"]
            yes_hits = sorted(yes_hits, key=lambda x: int(x.get("rank", 10**9)))
            selected_chunk_id = yes_hits[0]["chunk_id"] if yes_hits else ""
            selected_rank = yes_hits[0]["rank"] if yes_hits else -1

            if selected_chunk_id:
                p1["selected_chunk"] = {"chunk_id": selected_chunk_id, "rank": int(selected_rank), "label": "judge_yes_top"}

                # Conservative span emission (NOT in paper), only for your Char metrics
                if args.span_mode != "none":
                    # prefer full text from chunks.jsonl if provided, otherwise fall back to run's hit.text
                    sel_text = chunk_texts.get(selected_chunk_id, "")
                    if not sel_text:
                        for h in used:
                            if (h.get("chunk_id", "") or "").strip() == selected_chunk_id:
                                sel_text = (h.get("text", "") or "")
                                break
                    if args.span_mode == "full_chunk":
                        span = (0, len(sel_text))
                        label = "full_chunk_baseline"
                    else:
                        # debug/strong baseline
                        found = _find_answer_span(sel_text, event_answer) if sel_text else None
                        span = found if found is not None else (0, 0)
                        label = "answer_string_match"

                    # Attach as attribution_span (and also as selected_span for compatibility)
                    p1["attribution_span"] = {"chunk_id": selected_chunk_id, "chunk_span": [int(span[0]), int(span[1])], "label": label}
                    p1["selected_span"] = {"chunk_id": selected_chunk_id, "chunk_span": [int(span[0]), int(span[1])], "label": label}

            rec2 = dict(rec)
            rec2["pass1"] = p1
            w.write(rec2)

    print(f"Wrote baseline pass1 to: {out_path}")
    print(f"Records: {n}")
    print(f"LLM judge calls (cache misses): {total_llm_calls}")
    if args.cache:
        print(f"Cache: {args.cache}")


if __name__ == "__main__":
    main()
