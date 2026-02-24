#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adapter: run CR-RAG defenses on our Pass0 run JSONL, then export:
  (1) defended run JSONL (completion replaced with defended response)
  (2) optional pass1-like JSONL for traceback-first evaluation (paper-faithful conservative)

Supports:
- Local HF models via vendored create_model(model_name=llama7b/mistral7b/...) (GPU or CPU)
- OpenAI via --llm-backend openai (uses vendored GPTModel which delegates to rag_char_trace OpenAILLM)

Defenses:
- none: vanilla single-shot query with retrieved contexts concatenated
- keyword: KeywordAgg
- decoding: DecodingAgg (HF only)

Important:
- Our EVT/traceback-first metrics expect "ANSWER: ..." in completion. We wrap defended response accordingly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines

from rag_char_trace.baselines.crrag_vendor.models import create_model
from rag_char_trace.baselines.crrag_vendor.defense import RRAG, KeywordAgg, DecodingAgg


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


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


def _used_hits(rec: Dict[str, Any], max_hits: Optional[int] = None) -> List[Dict[str, Any]]:
    retr = rec.get("retrieval", {}) or {}
    hits = retr.get("hits", []) or []
    k_used = retr.get("prompt_used_hits", retr.get("prompt_max_hits", 10))
    k_used = _as_int(k_used, 10)
    used = hits[: max(0, k_used)]
    if max_hits is not None:
        used = used[: max(0, int(max_hits))]
    return used


def _build_data_item(rec: Dict[str, Any], used_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = rec.get("meta", {}) or {}
    q = rec.get("question", "") or ""
    # CR-RAG expects answer as a list[str]
    ans = meta.get("answer", "") or ""
    # Some datasets store list; normalize
    if isinstance(ans, list):
        answers = [str(x) for x in ans if str(x).strip()]
    else:
        answers = [str(ans).strip()] if str(ans).strip() else [""]
    topk = [(h.get("text", "") or "") for h in used_hits]
    return {"question": q, "topk_content": topk, "answer": answers}


def _event_targets(event_answer: str, used_hits: List[Dict[str, Any]]) -> List[str]:
    # event target = hit whose meta.incorrect_answer equals event_answer (normalized exact match)
    import re
    ws = re.compile(r"\s+")
    nonword = re.compile(r"[^0-9a-zA-Z]+")
    def norm(s: str) -> str:
        s = (s or "").strip().casefold()
        s = nonword.sub(" ", s)
        s = ws.sub(" ", s).strip()
        return s
    ea = norm(event_answer)
    out: List[Tuple[int, str]] = []
    if not ea:
        return []
    for i, h in enumerate(used_hits):
        cid = (h.get("chunk_id", "") or "").strip()
        m = h.get("meta", {}) or {}
        ia = (m.get("incorrect_answer", "") or "")
        if cid and ia and norm(ia) == ea:
            out.append((_as_int(h.get("rank", i), i), cid))
    return [cid for _, cid in sorted(out, key=lambda x: x[0])]


def main() -> None:
    ap = argparse.ArgumentParser(description="CR-RAG adapter over rag_char_trace pass0 run")
    ap.add_argument("--run", required=True, help="Input pass0 run JSONL (from scripts/02_run_rag.py)")
    ap.add_argument("--out-run", required=True, help="Output defended run JSONL")
    ap.add_argument("--out-pass1", default=None, help="Optional output pass1-like JSONL for scripts/07_eval_pass1.py")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N records")

    # CR-RAG model backend
    ap.add_argument("--crrag-model", type=str, default="mistral7b",
                    help="CR-RAG underlying model: mistral7b/llama7b/gpt3.5/... (see vendored create_model)")
    ap.add_argument("--crrag-device", type=str, default=None, help="Set env CRRAG_DEVICE (e.g., cuda, cuda:0, cpu)")
    ap.add_argument("--use-cache", action="store_true", help="Enable CR-RAG model cache")
    ap.add_argument("--cache-dir", default="runs", help="Cache dir (when --use-cache)")

    # Defense
    ap.add_argument("--defense", choices=["none", "keyword", "decoding"], default="keyword")
    ap.add_argument("--alpha", type=float, default=0.3, help="KeywordAgg relative threshold alpha")
    ap.add_argument("--beta", type=float, default=3.0, help="KeywordAgg absolute threshold beta")
    ap.add_argument("--eta", type=float, default=0.0, help="DecodingAgg eta (k*eta in paper code)")
    ap.add_argument("--subsample-iter", type=int, default=1, help="DecodingAgg subsample_iter")
    ap.add_argument("--max-output-tokens", type=int, default=64, help="CR-RAG max output tokens")
    ap.add_argument("--max-hits", type=int, default=None, help="Max used hits to pass into CR-RAG (default prompt_used_hits)")

    # Pass1-like (paper-faithful conservative)
    ap.add_argument("--selected-chunk-mode", choices=["rank0", "none"], default="rank0")
    ap.add_argument("--span-mode", choices=["full_chunk", "none"], default="full_chunk")

    args = ap.parse_args()

    if args.crrag_device:
        import os
        os.environ["CRRAG_DEVICE"] = args.crrag_device

    cache_path = None
    if args.use_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = str(Path(args.cache_dir) / f"crrag_cache_{args.crrag_model}.z")

    llm = create_model(args.crrag_model, cache_path=cache_path, max_output_tokens=args.max_output_tokens)

    # Wrap defense
    if args.defense == "none":
        model = RRAG(llm)
    elif args.defense == "keyword":
        model = KeywordAgg(llm, relative_threshold=args.alpha, absolute_threshold=args.beta, longgen=False, certify_save_path="")
    else:
        # decoding: HF only (requires llm.model + tokenizer)
        # If user picked gpt3.5, this will fail; provide a clear message.
        if "gpt" in llm.model_name:
            raise RuntimeError("Decoding defense requires an HF model (llama/mistral), not GPT/OpenAI.")
        from types import SimpleNamespace
        cfg = SimpleNamespace(eta=args.eta, subsample_iter=args.subsample_iter, temperature=1.0)
        model = DecodingAgg(llm, cfg, eval_certify=True, certify_save_path="")

    in_path = Path(args.run)
    out_run = Path(args.out_run)
    out_run.parent.mkdir(parents=True, exist_ok=True)

    out_pass1 = Path(args.out_pass1) if args.out_pass1 else None
    if out_pass1:
        out_pass1.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with jsonlines.open(str(in_path), "r") as r, jsonlines.open(str(out_run), "w") as w_run:
        w_p1 = jsonlines.open(str(out_pass1), "w") if out_pass1 else None
        try:
            for rec in r:
                if args.limit is not None and n >= int(args.limit):
                    break
                n += 1

                used = _used_hits(rec, args.max_hits)
                data_item = _build_data_item(rec, used)

                # run defense -> defended response
                if args.defense == "none":
                    resp = model.query_undefended(data_item)
                else:
                    resp, _cert = model.query(data_item, corruption_size=0)

                resp = (resp or "").strip()
                defended_completion = f"ANSWER: {resp}"

                rec2 = dict(rec)
                gen = dict(rec2.get("generation", {}) or {})
                gen["completion"] = defended_completion
                rec2["generation"] = gen
                rec2["crrag"] = {
                    "defense": args.defense,
                    "crrag_model": args.crrag_model,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "eta": args.eta,
                    "subsample_iter": args.subsample_iter,
                    "max_output_tokens": args.max_output_tokens,
                }
                w_run.write(rec2)

                # Optional pass1-like output (conservative)
                if w_p1 is not None:
                    event_answer = _extract_answer(defended_completion)
                    targets = _event_targets(event_answer, used)
                    triggered = bool(event_answer) and (event_answer.strip().upper() != "UNKNOWN") and bool(targets)

                    p1 = {
                        "triggered": triggered,
                        "baseline": f"crrag_{args.defense}",
                        "event_answer": event_answer,
                        "event_targets": targets,
                        "chunk_tests": [{"chunk_id": (h.get("chunk_id","") or "").strip()} for h in used if (h.get("chunk_id","") or "").strip()],
                    }

                    if args.selected_chunk_mode == "rank0" and used:
                        sel = used[0]
                        sel_cid = (sel.get("chunk_id","") or "").strip()
                        p1["selected_chunk"] = {"chunk_id": sel_cid, "rank": _as_int(sel.get("rank", 0), 0), "label": "rank0"}
                        if args.span_mode == "full_chunk":
                            txt = (sel.get("text","") or "")
                            p1["attribution_span"] = {"chunk_id": sel_cid, "chunk_span": [0, len(txt)], "label": "full_chunk"}
                            p1["selected_span"] = {"chunk_id": sel_cid, "chunk_span": [0, len(txt)], "label": "full_chunk"}

                    rec3 = dict(rec2)
                    rec3["pass1"] = p1
                    w_p1.write(rec3)
        finally:
            if w_p1 is not None:
                w_p1.close()

    print(f"Wrote defended run to: {out_run}")
    if out_pass1:
        print(f"Wrote pass1-like baseline to: {out_pass1}")
    print(f"Records processed: {n}")


if __name__ == "__main__":
    main()
