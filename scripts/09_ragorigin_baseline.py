#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGOrigin-style baseline adapted for rag_char_trace (local deployment).

Paper context:
- "Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation"
- Core idea (from the provided open-source main.py + helpers):
  * Use a proxy LM to compute per-context responsibility scores via losses:
      - loss(answer | context+question)
      - loss(question | context)
    combined with retrieval scores and normalized (variant).
  * Optionally narrow the scope via LLM probing (generator + judge).
  * Use dynamic thresholding (KMeans) or ranking to identify responsible contexts.

This script produces a Pass1-like JSONL compatible with scripts/07_eval_pass1.py (traceback-first EVT/Char metrics):
- selected_chunk: Top responsible chunk (by trace score)
- chunk_tests: ranking of candidate chunks by trace score
- attribution_span / selected_span: conservative span mode (default full_chunk) using the original chunks.jsonl text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import requests
from sklearn.cluster import KMeans



def _is_windows_drive_path(p: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", p or ""))


def _looks_like_ollama_tag(s: str) -> bool:
    # e.g. "gemma:7b", "llama3:8b". Avoid misclassifying Windows paths like "C:\...".
    s = (s or "").strip()
    if not s:
        return False
    if _is_windows_drive_path(s):
        return False
    if ":" in s and ("/" not in s and "\\" not in s):
        return True
    return False

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


def _candidate_hits(rec: Dict[str, Any], mode: str, max_hits: Optional[int]) -> List[Dict[str, Any]]:
    retr = rec.get("retrieval", {}) or {}
    hits = retr.get("hits", []) or []
    if mode == "used":
        hits = _used_hits(rec)
    elif mode == "topk":
        hits = hits[:]  # all retrieved
    else:
        raise ValueError(f"Unknown candidate source: {mode}")
    if max_hits is not None:
        hits = hits[: max(0, int(max_hits))]
    return hits


def _event_targets(event_answer: str, hits: List[Dict[str, Any]]) -> List[str]:
    pairs: List[Tuple[int, str]] = []
    for i, h in enumerate(hits):
        cid = (h.get("chunk_id", "") or "").strip()
        if not cid:
            continue
        m = h.get("meta", {}) or {}
        ia = (m.get("incorrect_answer", "") or "").strip()
        if ia and _match(event_answer, ia):
            r = _as_int(h.get("rank", i), i)
            pairs.append((r, cid))
    # fallback: match in hit.text
    if not pairs:
        for i, h in enumerate(hits):
            cid = (h.get("chunk_id", "") or "").strip()
            txt = (h.get("text", "") or "")
            if cid and txt and _match(txt, event_answer):
                r = _as_int(h.get("rank", i), i)
                pairs.append((r, cid))
    return [cid for _, cid in sorted(pairs, key=lambda x: x[0])]


def _cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


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


def _judge_true_false_prompt(reference_answer: str, other_answer: str) -> Tuple[str, str]:
    system = "You are a strict judge. Output exactly one token: TRUE or FALSE."
    user = (
        f'Given answer1 and answer2, answer1 is "{reference_answer}" and answer2 is "{other_answer}". '
        'If answer2 is exactly "I don\'t know", no matter what answer1 is, you must output FALSE. '
        'If answer1 is the same as or consistent with answer2, output TRUE. '
        'Otherwise output FALSE.\n'
        'TRUE or FALSE:'
    )
    return system, user


def _parse_true_false(text: str) -> bool:
    t = (text or "").strip().upper()
    m = re.findall(r"\b(TRUE|FALSE)\b", t)
    if m:
        return m[-1] == "TRUE"
    if "TRUE" in t and "FALSE" not in t:
        return True
    return False


def _gen_answer_from_contexts(question: str, contexts: List[str]) -> Tuple[str, str]:
    # Based on narrow_scope.py MULTIPLE_PROMPT
    context_str = "\n".join(contexts)
    prompt = (
        "Below is a query from a user and some relevant contexts. "
        "Answer the question given the information in those contexts. Your answer should be short and concise. "
        "If you cannot find the answer to the question, just say \"I don't know\".\n\n"
        f"Contexts: {context_str}\n\nQuery: {question}\n\nAnswer:"
    )
    system = "You are a helpful assistant."
    return system, prompt


def _z_score_normalize(arr: np.ndarray) -> np.ndarray:
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mean) / std


def _trace_scores(
    answer_losses: List[float],
    question_losses: List[float],
    retrieval_scores: List[float],
    variant: int,
) -> List[float]:
    # lower loss is better => negate before normalize
    a = _z_score_normalize(-np.array(answer_losses, dtype=np.float32))
    q = _z_score_normalize(-np.array(question_losses, dtype=np.float32))
    r = _z_score_normalize(np.array(retrieval_scores, dtype=np.float32))
    if variant == 0:
        return ((a + q + r) / 3.0).tolist()
    if variant == 1:
        return a.tolist()
    if variant == 2:
        return q.tolist()
    if variant == 3:
        return r.tolist()
    raise ValueError(f"Unsupported variant={variant} (supported: 0..3)")


def _kmeans_threshold(scores: List[float]) -> List[bool]:
    # True means "poison"/positive
    if len(scores) <= 1:
        return [True] * len(scores)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    x = np.array(scores, dtype=np.float32).reshape(-1, 1)
    kmeans.fit(x)
    centers = kmeans.cluster_centers_.flatten()
    pos_label = int(np.argmax(centers))
    y = (kmeans.labels_ == pos_label).astype(np.int32).tolist()
    return [bool(v) for v in y]


def _load_chunks_map(path: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            cid = (obj.get("chunk_id", "") or "").strip()
            txt = obj.get("text", "") or ""
            if cid:
                mp[cid] = txt
    return mp


@dataclass
class ProxyLM:
    model: Any
    tokenizer: Any
    device: str


def _load_proxy_model(proxy_model: str, device: str, dtype: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for proxy model. Install: pip install torch transformers"
        ) from e

    proxy_model = (proxy_model or "").strip()
    if not proxy_model:
        raise RuntimeError("Empty --proxy-model. Provide a HuggingFace model id (e.g. distilgpt2) or a local directory.")

    # Guardrail: users often pass Ollama tags (e.g. gemma:7b) by mistake.
    if _looks_like_ollama_tag(proxy_model) and not Path(proxy_model).exists():
        raise RuntimeError(
            f"Invalid --proxy-model='{proxy_model}'. This looks like an Ollama tag (e.g. gemma:7b), "
            "but RAGOrigin requires a HuggingFace/Transformers model id or a local directory path for the proxy LM. "
            r"Example: --proxy-model distilgpt2  (small)  or a local HF directory like C:\models\my_proxy_lm"
        )

    tok = AutoTokenizer.from_pretrained(proxy_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype = None
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = None  # auto

    mdl = AutoModelForCausalLM.from_pretrained(proxy_model, torch_dtype=torch_dtype)
    mdl.to(device)
    mdl.eval()
    return ProxyLM(model=mdl, tokenizer=tok, device=device)


def _calculate_loss(proxy: ProxyLM, context: str, response: str, max_input_tokens: int) -> float:
    import torch

    text = (context or "") + " " + (response or "")
    inputs = proxy.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = inputs["input_ids"].to(proxy.device)

    # Mask context tokens to compute loss only on response part
    context_ids = proxy.tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )["input_ids"]
    label_ids = input_ids.clone()
    ctx_len = min(label_ids.shape[1], context_ids.shape[1])
    label_ids[:, :ctx_len] = -100

    with torch.no_grad():
        out = proxy.model(input_ids, labels=label_ids)
    return float(out.loss.item())


def _calculate_scores(
    proxy: ProxyLM,
    contexts: List[str],
    question: str,
    rag_response: str,
    retrieval_scores: List[float],
    max_input_tokens: int,
) -> Tuple[List[float], List[float], List[float]]:
    answer_losses: List[float] = []
    question_losses: List[float] = []
    for ctx in contexts:
        prompt_1 = (
            "Below is a query from a user and a relevant context. "
            "Answer the question given the information in the context.\n\n"
            f"Context: {ctx}\n\nQuery: {question}\n\nAnswer:"
        )
        answer_losses.append(_calculate_loss(proxy, prompt_1, rag_response, max_input_tokens))

        prompt_2 = (
            "Below is a query from a user and a relevant context. "
            "Answer the question given the information in the context.\n\n"
            f"Context: {ctx}\n\nQuery:"
        )
        question_losses.append(_calculate_loss(proxy, prompt_2, question, max_input_tokens))

    return answer_losses, question_losses, retrieval_scores


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGOrigin baseline (local) -> pass1-like JSONL.")
    ap.add_argument("--run", required=True, help="Pass0 run JSONL (output of scripts/02_run_rag.py)")
    ap.add_argument("--out", required=True, help="Output JSONL with pass1 fields (RAGOrigin baseline).")
    ap.add_argument("--chunks", required=True, help="chunks.jsonl for this split (for full_chunk span length alignment).")

    ap.add_argument("--candidate-source", choices=["used", "topk"], default="used", help="Candidate contexts to score.")
    ap.add_argument("--max-hits", type=int, default=None, help="Max candidate hits to score (after candidate-source).")

    ap.add_argument("--top-K", type=int, default=5, help="Group size top_K (used by optional narrow-scope).")
    ap.add_argument("--enable-narrow-scope", action="store_true", help="Run narrow-scope probing (LLM generator+judge).")
    ap.add_argument("--max-scope-contexts", type=int, default=None, help="Cap contexts used for scope/scoring.")

    # local LLM for narrow-scope (defaults should match pass0 LLM)
    ap.add_argument("--ollama-model", required=False, default=None, help="Ollama model for generator/judge (e.g., gemma:7b).")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature for narrow-scope calls.")
    ap.add_argument("--max-tokens-gen", type=int, default=32, help="Max tokens for generator answer in narrow-scope.")
    ap.add_argument("--max-tokens-judge", type=int, default=8, help="Max tokens for judge output in narrow-scope.")
    ap.add_argument("--timeout-s", type=float, default=120.0, help="HTTP timeout for Ollama calls.")
    ap.add_argument("--cache", default=None, help="Optional jsonl cache for narrow-scope LLM calls.")

    # proxy model for responsibility measurement
    ap.add_argument("--proxy-model", dest="proxy_model", default="distilgpt2", help="Proxy LM (Transformers). HF model id or local path. Example: distilgpt2")
    ap.add_argument("--proxy-hf-model", dest="proxy_model", help="Alias of --proxy-model")
    ap.add_argument("--device", default=None, help="Proxy device, e.g. cuda:0 or cpu. Default: auto.")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto", help="Proxy dtype.")
    ap.add_argument("--max-input-tokens", type=int, default=1024, help="Max tokens for proxy scoring inputs.")
    ap.add_argument("--variant", type=int, default=0, help="Trace score variant (0..3).")
    ap.add_argument("--span-mode", choices=["full_chunk", "none"], default="full_chunk", help="Conservative span output.")
    args = ap.parse_args()

    # load chunks map for span alignment
    chunks_map = _load_chunks_map(args.chunks)

    # load proxy model
    if args.device is None:
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    proxy = _load_proxy_model(args.proxy_model, args.device, args.dtype)

    # LLM cache for narrow-scope
    cache: Dict[str, Dict[str, Any]] = {}
    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists():
        with jsonlines.open(str(cache_path), "r") as r:
            for obj in r:
                k = (obj.get("key", "") or "").strip()
                if k:
                    cache[k] = obj

    def cache_get(key: str) -> Optional[Dict[str, Any]]:
        return cache.get(key)

    def cache_put(obj: Dict[str, Any]) -> None:
        if not cache_path:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(str(cache_path), "a") as w:
            w.write(obj)
        cache[obj["key"]] = obj

    in_path = Path(args.run)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm_calls_total = 0
    n = 0

    with jsonlines.open(str(in_path), "r") as r, jsonlines.open(str(out_path), "w") as w:
        for rec in r:
            n += 1
            qid = (rec.get("qid", "") or "").strip()
            question = (rec.get("question", "") or "")
            comp = (rec.get("generation", {}) or {}).get("completion", "") or ""
            event_answer = _extract_answer(comp)
            if event_answer.strip().upper() == "UNKNOWN":
                event_answer = ""

            hits = _candidate_hits(rec, args.candidate_source, args.max_hits)
            targets = _event_targets(event_answer, hits)
            triggered = bool(event_answer) and bool(targets)

            p1: Dict[str, Any] = {
                "triggered": triggered,
                "baseline": "ragorigin_local",
                "event_answer": event_answer,
                "candidate_source": args.candidate_source,
                "variant": int(args.variant),
                "proxy_model": args.proxy_model,
                "proxy_device": args.device,
            }

            if not triggered or len(hits) == 0:
                rec2 = dict(rec)
                rec2["pass1"] = p1
                w.write(rec2)
                continue

            # Build contexts and retrieval scores
            context_texts = [(h.get("text", "") or "") for h in hits]
            retrieval_scores = [float(h.get("score", 0.0) or 0.0) for h in hits]
            chunk_ids = [(h.get("chunk_id", "") or "").strip() for h in hits]
            ranks = [int(h.get("rank", i) or i) for i, h in enumerate(hits)]

            # Optional narrow-scope probing to decide scope_size
            scope_size = len(context_texts)
            llm_calls = 0
            if args.max_scope_contexts is not None:
                scope_size = min(scope_size, int(args.max_scope_contexts))

            if args.enable_narrow_scope:
                if not args.ollama_model:
                    raise SystemExit("--enable-narrow-scope requires --ollama-model")
                topK = max(1, int(args.top_K))
                check_results: List[int] = [1]  # original response is consistent with itself

                # probe deeper groups until equal #consistent and #inconsistent
                # start from i=topK (skip first group)
                for i0 in range(topK, scope_size, topK):
                    if check_results.count(0) == check_results.count(1) and len(check_results) > 1:
                        break
                    group_contexts = context_texts[i0 : i0 + topK]
                    sys_gen, prompt_gen = _gen_answer_from_contexts(question, group_contexts)
                    key_gen = _cache_key("gen", question, event_answer, "\n".join(group_contexts))
                    cached = cache_get(key_gen)
                    if cached:
                        gen_ans = cached.get("text", "")
                    else:
                        gen_ans = _ollama_generate(
                            base_url=args.ollama_url,
                            model=args.ollama_model,
                            system=sys_gen,
                            prompt=prompt_gen,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens_gen,
                            timeout_s=args.timeout_s,
                        )
                        llm_calls += 1
                        cache_put({"key": key_gen, "text": gen_ans})
                    gen_ans = (gen_ans or "").strip()

                    sys_j, prompt_j = _judge_true_false_prompt(event_answer, gen_ans)
                    key_j = _cache_key("judge", event_answer, gen_ans)
                    cached2 = cache_get(key_j)
                    if cached2:
                        j_txt = cached2.get("text", "")
                    else:
                        j_txt = _ollama_generate(
                            base_url=args.ollama_url,
                            model=args.ollama_model,
                            system=sys_j,
                            prompt=prompt_j,
                            temperature=0.0,
                            max_tokens=args.max_tokens_judge,
                            timeout_s=args.timeout_s,
                        )
                        llm_calls += 1
                        cache_put({"key": key_j, "text": j_txt})
                    consistent = _parse_true_false(j_txt)
                    check_results.append(1 if consistent else 0)

                # compute group-based scope size (as in measure_responsibility.py)
                probe = check_results[:]
                g = 2
                while True:
                    # handle short probe list
                    s = sum(probe[:g])
                    if s == int(g / 2):
                        break
                    g += 2
                    if g > max(2, len(probe)):
                        break
                scope_size = min(scope_size, g * topK)
                scope_size = min(scope_size, len(context_texts))

                p1["check_results"] = probe
                p1["scope_size"] = int(scope_size)

            # Apply scope cap
            context_texts = context_texts[:scope_size]
            retrieval_scores = retrieval_scores[:scope_size]
            chunk_ids = chunk_ids[:scope_size]
            ranks = ranks[:scope_size]

            # Proxy scoring
            answer_losses, question_losses, retrieval_scores2 = _calculate_scores(
                proxy,
                context_texts,
                question,
                event_answer,
                retrieval_scores,
                args.max_input_tokens,
            )
            scores = _trace_scores(answer_losses, question_losses, retrieval_scores2, args.variant)

            # KMeans dynamic threshold (paper-style) for predicted poisoned contexts
            pred_poison = _kmeans_threshold(scores)
            p1["pred_poison_rate"] = float(sum(1 for v in pred_poison if v) / len(pred_poison)) if pred_poison else 0.0

            # Rank chunk tests by trace score desc (tie-break by retrieval rank)
            order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), int(ranks[i])))
            p1["chunk_tests"] = [{"chunk_id": chunk_ids[i]} for i in order if chunk_ids[i]]

            # Selected chunk: highest score (top-1)
            sel_i = order[0]
            selected_chunk_id = chunk_ids[sel_i]
            p1["selected_chunk"] = {"chunk_id": selected_chunk_id, "rank": int(ranks[sel_i]), "label": "trace_score_top1"}

            # Conservative span output
            if args.span_mode == "full_chunk":
                full_text = chunks_map.get(selected_chunk_id, "")
                if not full_text:
                    full_text = context_texts[sel_i]  # fallback (may be truncated)
                p1["attribution_span"] = {
                    "chunk_id": selected_chunk_id,
                    "chunk_span": [0, int(len(full_text))],
                    "label": "full_chunk",
                }
                p1["selected_span"] = {
                    "chunk_id": selected_chunk_id,
                    "chunk_span": [0, int(len(full_text))],
                    "label": "full_chunk",
                }

            p1["llm_calls"] = int(llm_calls)
            llm_calls_total += llm_calls

            rec2 = dict(rec)
            rec2["pass1"] = p1
            w.write(rec2)

    print(f"Wrote RAGOrigin baseline pass1 to: {out_path}")
    print(f"Records: {n}")
    print(f"Total narrow-scope LLM calls: {llm_calls_total}")
    if args.cache:
        print(f"Cache: {args.cache}")


if __name__ == "__main__":
    main()
