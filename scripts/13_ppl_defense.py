#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/13_ppl_defense.py

Perplexity-based baseline for RAG poisoning defense / attribution.

This script is designed to *plug into the existing pipeline*:

  01_build_index -> 02_run_rag (pass0) -> 03_eval_asr -> 13_ppl_defense -> 07_eval_pass1

Key design goals:
  - Consume the JSONL produced by scripts/02_run_rag.py.
  - Produce a JSONL that is *schema-compatible* with scripts/07_eval_pass1.py
    (i.e., each record contains a `pass1` object). This is only for evaluation
    reuse; the method itself is NOT your "pass1" algorithm.
  - Chunk-wise scoring: compute a perplexity score for each retrieved chunk
    (typically only the prompt-used hits), rank them, and select the most
    suspicious chunk as the attribution target.
  - Span granularity: if we only identify a chunk, we output the whole chunk as
    a span (chunk_span = [0, very_large]) so the evaluator can clip to true
    chunk length.

Perplexity judge backends:
  - ollama: uses Ollama REST API (/api/generate) with `logprobs=true`.
            NOTE: Ollama returns logprobs for generated tokens; many versions
            do not expose prompt-token logprobs. We therefore try a cheap
            prompt-scoring call with num_predict=0; if no token logprobs are
            returned, we fall back to an "echo" scoring prompt.
  - hf: uses a local HuggingFace causal LM (e.g., gpt2) to compute true PPL.

The output record will include:
  - pass1.triggered: True only when a poisoning event is detected
    (i.e., the model answer matches any hit.meta.incorrect_answer among
    prompt-used hits). This matches your event-based evaluation setup.
  - pass1.chunk_tests: ranked list of chunks with their PPL scores.
  - pass1.attribution_span: selected chunk_id + full-chunk span.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import jsonlines
from tqdm import tqdm


# -----------------------------
# Text utils (keep consistent with eval scripts)

import re


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
    # prefer explicit ANSWER:
    for ln in comp.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("answer:"):
            return s.split(":", 1)[1].strip()
    # fallback: first non-empty line
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


def _event_found(pred_ans: str, used: List[Dict[str, Any]]) -> bool:
    """Return True if pred_ans matches any incorrect_answer among used hits."""
    if not pred_ans or pred_ans.strip().upper() == "UNKNOWN":
        return False
    for h in used:
        m = h.get("meta", {}) or {}
        ia = (m.get("incorrect_answer", "") or "").strip()
        if ia and _match(pred_ans, ia):
            return True
    return False


# -----------------------------
# PPL backends


def _extract_token_logprobs_from_ollama(obj: Any) -> List[float]:
    """Be robust to minor schema differences across Ollama versions."""
    if obj is None:
        return []
    lp = obj.get("logprobs") if isinstance(obj, dict) else None
    if lp is None:
        return []

    # Common shape: list[ {"token":..., "logprob": ... , ...}, ...]
    if isinstance(lp, list):
        out: List[float] = []
        for x in lp:
            if isinstance(x, dict) and "logprob" in x:
                try:
                    out.append(float(x["logprob"]))
                except Exception:
                    pass
        return out

    # Alternative: dict with "content": list[...]
    if isinstance(lp, dict):
        if isinstance(lp.get("content"), list):
            out2: List[float] = []
            for x in lp["content"]:
                if isinstance(x, dict) and "logprob" in x:
                    try:
                        out2.append(float(x["logprob"]))
                    except Exception:
                        pass
            return out2
        # Alternative: dict with token_logprobs list[float]
        if isinstance(lp.get("token_logprobs"), list):
            out3: List[float] = []
            for x in lp["token_logprobs"]:
                if x is None:
                    continue
                try:
                    out3.append(float(x))
                except Exception:
                    pass
            return out3
        # Alternative: dict with "logprobs": list[...]
        if isinstance(lp.get("logprobs"), list):
            out4: List[float] = []
            for x in lp["logprobs"]:
                if isinstance(x, dict) and "logprob" in x:
                    try:
                        out4.append(float(x["logprob"]))
                    except Exception:
                        pass
            return out4

    return []


def _ppl_from_logprobs(logprobs: List[float]) -> Optional[float]:
    if not logprobs:
        return None
    # average negative log-likelihood per token
    nll = -sum(logprobs) / float(len(logprobs))
    # guard for overflow
    try:
        return float(math.exp(nll))
    except OverflowError:
        return float("inf")


@dataclass
class OllamaPPLJudge:
    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 120.0
    # Cap generation used for pseudo-PPL. Higher is slower.
    # This is a hard cap; the script will still auto-scale below this.
    max_gen_tokens: int = 128
    # Warm up the model once to avoid a long first scored request.
    warmup: bool = True
    max_retries: int = 2
    retry_backoff_s: float = 0.5

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                # stream=false should yield a single JSON object, but be robust.
                lines = [ln for ln in body.splitlines() if ln.strip()]
                if not lines:
                    return {}
                if len(lines) == 1:
                    return json.loads(lines[0])
                # NDJSON: parse the last complete object
                return json.loads(lines[-1])
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * (2**attempt))
        if last_err is not None:
            raise last_err
        return {}

    def warmup_once(self) -> None:
        """Best-effort warmup.

        On the first request, Ollama may take a while to load the model into
        memory/VRAM. If you see the progress bar stuck at `0it` while the GPU is
        busy, it's often just this initial load.

        This function triggers a tiny bounded request so the first *scored* call
        is less likely to look like a hang.
        """
        if not self.warmup:
            return
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": "warmup",
                "stream": False,
                "raw": True,
                "options": {"temperature": 0.0, "num_predict": 8},
            }
            self._post_json("/api/generate", payload)
        except Exception:
            # Warmup is non-critical.
            return

    def perplexity(self, text: str, max_chars: Optional[int] = None) -> Optional[float]:
        """Approximate PPL by forcing the model to re-generate the same text and scoring output token logprobs.

        Notes:
        - Ollama's `logprobs` are for *generated* tokens, not prompt tokens.
        - We therefore use a copy/echo prompt and score the output tokens.
        - Generation is always capped to a positive value (never num_predict=0).
        """

        txt = (text or "")
        if max_chars is not None and max_chars > 0:
            txt = txt[:max_chars]
        txt = txt.strip("\n ")
        if not txt:
            return None

        # Echo scoring (single bounded request).
        approx_num_predict = int(len(txt) / 2) + 32
        max_cap = max(16, int(self.max_gen_tokens))
        approx_num_predict = max(16, min(max_cap, approx_num_predict))

        echo_prompt = (
            "Repeat the following text EXACTLY, verbatim. "
            "Output ONLY the text. Do not add quotes or any extra characters.\n"
            "---BEGIN---\n"
            + txt
            + "\n---END---\n"
        )
        payload2: Dict[str, Any] = {
            "model": self.model,
            "prompt": echo_prompt,
            "stream": False,
            "raw": True,
            "logprobs": True,
            "options": {
                "temperature": 0.0,
                "num_predict": approx_num_predict,
            },
        }
        try:
            obj2 = self._post_json("/api/generate", payload2)
            lps2 = _extract_token_logprobs_from_ollama(obj2)
            return _ppl_from_logprobs(lps2)
        except Exception:
            return None


@dataclass
class HFPerplexityJudge:
    model_name: str = "gpt2"
    device: str = "cpu"
    max_length: int = 512

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                "HF backend requires `transformers` and `torch`. "
                "Install them or use --judge-backend ollama."
            ) from e

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._mdl = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._mdl.eval()
        self._mdl.to(self.device)

    def perplexity(self, text: str, max_chars: Optional[int] = None) -> Optional[float]:
        text = (text or "").strip("\n ")
        if not text:
            return None
        if max_chars is not None and max_chars > 0:
            text = text[:max_chars]
        enc = self._tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        with self._torch.no_grad():
            out = self._mdl(input_ids=input_ids, labels=input_ids)
            loss = out.loss
            try:
                return float(self._torch.exp(loss).item())
            except Exception:
                return None


# -----------------------------
# Main pipeline logic


def _safe_chunk_id(hit: Dict[str, Any]) -> str:
    return (hit.get("chunk_id", "") or "").strip()


def _rank_of_hit(hit: Dict[str, Any], fallback: int) -> int:
    return _as_int(hit.get("rank", fallback), fallback)


def main() -> None:
    ap = argparse.ArgumentParser(description="Perplexity-based chunk filtering baseline (RAG security).")
    ap.add_argument("--run", required=True, help="Pass0 run JSONL (from scripts/02_run_rag.py)")
    ap.add_argument("--out-pass1", required=True, help="Output JSONL compatible with scripts/07_eval_pass1.py")

    ap.add_argument(
        "--chunks",
        default=None,
        help=(
            "Optional chunks.jsonl to load full chunk text by chunk_id. "
            "If omitted, uses hit.text from the run trace (often truncated)."
        ),
    )

    ap.add_argument("--max-hits", type=int, default=10, help="Only score the top-N prompt-used hits per example")
    ap.add_argument("--cand-k", type=int, default=5, help="How many top spans to emit into pass1.span_tests")
    ap.add_argument("--max-judge-chars", type=int, default=512, help="Truncate each chunk text before scoring")

    # Ollama-specific controls (for judge backend=ollama)
    ap.add_argument(
        "--judge-max-gen-tokens",
        type=int,
        default=128,
        help="Hard cap on generation tokens used for pseudo-PPL (smaller = faster).",
    )
    ap.add_argument(
        "--no-judge-warmup",
        action="store_true",
        help="Disable the one-time warmup request to Ollama judge model.",
    )

    ap.add_argument("--judge-backend", choices=["ollama", "hf"], default="ollama")
    # Default judge model requested for the PPL baseline (local Ollama).
    ap.add_argument("--judge-model", default="llama3.1:8b", help="Judge model name (Ollama or HF model id)")
    ap.add_argument("--judge-device", default=None, help="HF device, e.g. cpu/cuda. Default: auto")

    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=float, default=120.0)

    args = ap.parse_args()

    # overwrite output (keep deterministic)
    if os.path.exists(args.out_pass1):
        os.remove(args.out_pass1)

    if args.judge_backend == "hf":
        device = args.judge_device
        if not device:
            # auto
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        judge = HFPerplexityJudge(model_name=args.judge_model, device=device)
    else:
        judge = OllamaPPLJudge(
            model=args.judge_model,
            base_url=args.ollama_url,
            timeout_s=args.timeout,
            max_gen_tokens=args.judge_max_gen_tokens,
            warmup=(not args.no_judge_warmup),
        )
        if judge.warmup:
            print(f"Warming up Ollama judge model: {args.judge_model} ...")
        judge.warmup_once()
        if judge.warmup:
            print("Judge warmup done.")

    # Optional: full chunk texts
    full_text: Dict[str, str] = {}
    if args.chunks:
        try:
            with jsonlines.open(args.chunks, "r") as cr:
                for j in cr:
                    if not isinstance(j, dict):
                        continue
                    cid = (j.get("chunk_id", "") or "").strip()
                    if not cid:
                        continue
                    full_text[cid] = j.get("text", "") or ""
            print(f"Loaded {len(full_text)} chunks from: {args.chunks}")
        except FileNotFoundError:
            raise SystemExit(f"--chunks file not found: {args.chunks}")

    # cache: chunk_id -> ppl
    ppl_cache: Dict[str, float] = {}
    cache_hits = 0
    cache_miss = 0
    judge_calls = 0
    printed_first_scoring_hint = False

    def score_hit(hit: Dict[str, Any]) -> float:
        nonlocal cache_hits, cache_miss, judge_calls
        cid = _safe_chunk_id(hit)
        if cid and cid in ppl_cache:
            cache_hits += 1
            return ppl_cache[cid]
        cache_miss += 1
        txt = (full_text.get(cid) if cid else None) or (hit.get("text", "") or "")
        txt = txt.strip()
        ppl = judge.perplexity(txt, max_chars=args.max_judge_chars)
        judge_calls += 1
        # If scoring failed, treat as low suspiciousness.
        score = float(ppl) if ppl is not None else -1.0
        if cid:
            ppl_cache[cid] = score
        return score

    # process
    with jsonlines.open(args.run, "r") as r, jsonlines.open(args.out_pass1, "w") as w:
        for rec in tqdm(r, desc="PPL-Defense", ncols=100):
            # preserve original record and attach a pass1-like object
            if not isinstance(rec, dict):
                continue

            used = _used_hits(rec)
            if args.max_hits is not None:
                used = used[: max(0, args.max_hits)]

            comp = (rec.get("generation", {}) or {}).get("completion", "") or ""
            pred = _extract_answer(comp)
            triggered = _event_found(pred, used)

            p1: Dict[str, Any] = {
                "method": "ppl_defense",
                "triggered": bool(triggered),
                "n_rounds": 1 if triggered else 0,
                "llm_calls": 0,
                "mask_chars_total": 0,
                "chunk_tests": [],
                "span_tests": [],
            }

            if triggered and used:
                if not printed_first_scoring_hint:
                    printed_first_scoring_hint = True
                    print(
                        f"Scoring triggered records with judge model (per-chunk). "
                        f"First scored record may be slow; cache will help after that. "
                        f"hits_per_record={len(used)}, max_gen_tokens={getattr(judge, 'max_gen_tokens', 'NA')}, max_judge_chars={args.max_judge_chars}"
                    )
                scored: List[Tuple[float, int, Dict[str, Any]]] = []
                for i, h in enumerate(used):
                    s = score_hit(h)
                    scored.append((s, _rank_of_hit(h, i), h))

                # sort: highest perplexity = most suspicious
                scored.sort(key=lambda x: (-x[0], x[1]))

                chunk_tests: List[Dict[str, Any]] = []
                for j, (s, rnk, h) in enumerate(scored):
                    cid = _safe_chunk_id(h)
                    if not cid:
                        continue
                    chunk_tests.append({"chunk_id": cid, "score": s, "retrieval_rank": rnk, "rank": j})
                p1["chunk_tests"] = chunk_tests
                p1["llm_calls"] = len(scored)  # judge computations (before cache compression)

                if chunk_tests:
                    top = chunk_tests[0]
                    top_cid = top["chunk_id"]
                    # Whole-chunk span for compatibility; evaluator will clip to true text length.
                    full_span = [0, 10**9]
                    p1["selected_chunk"] = {"chunk_id": top_cid, "score": top.get("score")}
                    p1["attribution_span"] = {
                        "chunk_id": top_cid,
                        "chunk_span": full_span,
                        "score": top.get("score"),
                        "note": "chunk-level attribution (whole-chunk span)",
                    }

                    # Span candidates = top-k chunks as whole-chunk spans
                    span_tests: List[Dict[str, Any]] = []
                    for cand in chunk_tests[: max(0, args.cand_k)]:
                        span_tests.append(
                            {
                                "chunk_id": cand["chunk_id"],
                                "chunk_span": full_span,
                                "score": cand.get("score"),
                            }
                        )
                    p1["span_tests"] = span_tests

            rec["pass1"] = p1
            w.write(rec)

    # light summary
    print(f"Wrote PPL baseline pass1-compatible file to: {args.out_pass1}")
    print(
        json.dumps(
            {
                "judge_backend": args.judge_backend,
                "judge_model": args.judge_model,
                "judge_calls": judge_calls,
                "cache_entries": len(ppl_cache),
                "cache_hits": cache_hits,
                "cache_miss": cache_miss,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
