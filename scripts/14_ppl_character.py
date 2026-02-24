#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPL_Character baseline (character-level PPL traceback).

This is a *baseline* to compare against Pass-1 (our occlusion-based char traceback).

Pipeline integration:
  01_build_index -> 02_run_rag (pass0) -> 03_eval_asr -> 14_ppl_character -> 07_eval_pass1

Key idea:
- Use a local "judge" LLM (default: Ollama llama3.1:8b) to compute passage perplexity.
- Rank prompt-used chunks by perplexity (higher = more suspicious / less fluent).
- Pick the top chunk, then localize a character span by sliding-window perplexity search.

Outputs:
- Writes a JSONL file that keeps the original pass0 record and adds a `pass1` field
  compatible with scripts/07_eval_pass1.py.

Notes:
- This baseline does *not* run sanitize replay (no mitigation); it only attributes.
- If exact token-level offsets are unavailable, we localize via windowed substring PPL.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
from tqdm import tqdm


# -----------------------------
# Text utils (matching eval)
# -----------------------------

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
    """Lenient match (same as scripts/03_eval_asr.py)."""
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


def _used_hits(rec: Dict[str, Any], *, max_hits: int) -> List[Dict[str, Any]]:
    retr = rec.get("retrieval", {}) or {}
    hits = list(retr.get("hits", []) or [])

    k_used = retr.get("prompt_used_hits", None)
    if not isinstance(k_used, int) or k_used <= 0:
        k_used = retr.get("prompt_max_hits", None)
    if not isinstance(k_used, int) or k_used <= 0:
        k_used = max_hits

    k_used = min(int(k_used), int(max_hits))
    return hits[: max(0, k_used)]


def _event_found(pred_ans: str, used_hits: List[Dict[str, Any]]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Return (found, matched_incorrect, best_hit) if pred matches incorrect_answer in used hits."""
    matched: Optional[str] = None
    best_hit: Optional[Dict[str, Any]] = None

    # Prefer meta.incorrect_answer
    for h in used_hits:
        m = h.get("meta", {}) or {}
        ia = (m.get("incorrect_answer", "") or "").strip()
        if ia and _match(pred_ans, ia):
            matched = ia
            best_hit = h
            break

    # Fallback: match inside hit.text
    if matched is None:
        for h in used_hits:
            txt = (h.get("text", "") or "")
            if txt and _match(txt, pred_ans):
                matched = pred_ans
                best_hit = h
                break

    return (matched is not None), (matched or ""), best_hit


# -----------------------------
# Chunks loading / prompt block slicing
# -----------------------------


def load_chunks_text(chunks_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with jsonlines.open(chunks_path, "r") as r:
        for j in r:
            cid = (j.get("chunk_id", "") or "").strip()
            if not cid:
                continue
            out[cid] = j.get("text", "") or ""
    return out


def _block_text(user_prompt: str, block: Optional[Dict[str, Any]]) -> str:
    if not block:
        return ""
    a = _as_int(block.get("prompt_text_start", -1), -1)
    e = _as_int(block.get("prompt_text_end", -1), -1)
    if a < 0 or e < 0 or e <= a:
        return ""
    if not user_prompt:
        return ""
    a = max(0, min(a, len(user_prompt)))
    e = max(0, min(e, len(user_prompt)))
    if e <= a:
        return ""
    return user_prompt[a:e]


def _build_block_map(rec: Dict[str, Any]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    retr = rec.get("retrieval", {}) or {}
    blocks = list(retr.get("prompt_blocks", []) or [])
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for b in blocks:
        try:
            r = int(b.get("rank", -1))
        except Exception:
            r = -1
        cid = (b.get("chunk_id", "") or "").strip()
        if r >= 0 and cid:
            out[(r, cid)] = b
    return out


def _align_prompt_snippet(full_text: str, snippet: str) -> int:
    """Return offset of snippet inside full_text, or 0 if unknown."""
    if not full_text or not snippet:
        return 0
    idx = full_text.find(snippet)
    if idx != -1:
        return idx
    anchor = snippet[:200]
    if anchor:
        idx2 = full_text.find(anchor)
        if idx2 != -1:
            return idx2
    return 0


# -----------------------------
# Judge: perplexity
# -----------------------------


def _extract_token_logprobs_from_ollama(lp_obj: Any) -> List[float]:
    """Extract a flat list of token logprobs from Ollama's varying schemas."""
    if lp_obj is None:
        return []

    # Newer schema: {"content": [{"token": "...", "logprob": -0.1, ...}, ...]}
    if isinstance(lp_obj, dict) and "content" in lp_obj and isinstance(lp_obj["content"], list):
        out: List[float] = []
        for it in lp_obj["content"]:
            if isinstance(it, dict) and "logprob" in it:
                try:
                    out.append(float(it["logprob"]))
                except Exception:
                    pass
        return out

    # Sometimes list[dict]
    if isinstance(lp_obj, list):
        out2: List[float] = []
        for it in lp_obj:
            if isinstance(it, dict) and "logprob" in it:
                try:
                    out2.append(float(it["logprob"]))
                except Exception:
                    pass
            elif isinstance(it, (int, float)):
                out2.append(float(it))
        return out2

    # Fallback: dict with token_logprobs
    if isinstance(lp_obj, dict) and "token_logprobs" in lp_obj and isinstance(lp_obj["token_logprobs"], list):
        out3: List[float] = []
        for v in lp_obj["token_logprobs"]:
            try:
                out3.append(float(v))
            except Exception:
                pass
        return out3

    return []


class OllamaPPLJudge:
    def __init__(self, *, model: str, base_url: str, timeout_s: int = 600):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = int(timeout_s)
        self.calls = 0

    def warmup(self) -> None:
        _ = self._call_generate("warmup", num_predict=8)

    def _call_generate(self, prompt: str, *, num_predict: int) -> Dict[str, Any]:
        self.calls += 1
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.0,
                "num_predict": int(num_predict),
                "top_p": 1.0,
            },
            # raw=True helps avoid chat templating differences across models
            "raw": True,
            "logprobs": True,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        full = ""
        logprobs: List[float] = []
        last_obj: Dict[str, Any] = {}

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                last_obj = obj
                piece = obj.get("response", "")
                if isinstance(piece, str) and piece:
                    full += piece

                lp = obj.get("logprobs", None)
                if lp is not None:
                    logprobs.extend(_extract_token_logprobs_from_ollama(lp))

                if obj.get("done", False):
                    break

        return {"response": full, "logprobs": logprobs, "raw": last_obj}

    def _ppl_from_logprobs(self, logprobs: List[float]) -> Optional[float]:
        if not logprobs:
            return None
        mean_lp = sum(float(x) for x in logprobs) / float(len(logprobs))
        return float(math.exp(-mean_lp))

    def _score_with_num_predict0(self, text: str) -> Optional[float]:
        # Some Ollama builds return prompt-token logprobs with num_predict=0.
        try:
            out = self._call_generate(text, num_predict=0)
        except Exception:
            return None
        return self._ppl_from_logprobs(out.get("logprobs", []))

    def _score_with_echo(self, text: str) -> Optional[float]:
        prompt = (
            "Repeat the following text exactly. Output must match character-for-character.\n"
            "Do not add any prefix/suffix, do not explain.\n\n"
            + (text or "")
        )
        # Rough char->token estimate. Keep conservative to avoid truncation.
        char_est = max(16, (len(text) // 3) + 16)
        num_predict = min(4096, char_est)
        try:
            out = self._call_generate(prompt, num_predict=num_predict)
        except Exception:
            return None
        return self._ppl_from_logprobs(out.get("logprobs", []))

    def perplexity(self, text: str) -> Optional[float]:
        t = (text or "").strip("\n")
        if not t:
            return None
        p0 = self._score_with_num_predict0(t)
        if p0 is not None:
            return p0
        return self._score_with_echo(t)


class HFPPLJudge:
    def __init__(self, *, model: str, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.to(device)
        self.model.eval()
        self.calls = 0
        self._torch = torch

    def warmup(self) -> None:
        _ = self.perplexity("warmup")

    def perplexity(self, text: str) -> Optional[float]:
        self.calls += 1
        t = (text or "").strip("\n")
        if not t:
            return None
        enc = self.tokenizer(t, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._torch.no_grad():
            out = self.model(**enc, labels=enc["input_ids"])
            loss = out.loss
        return float(self._torch.exp(loss).item())


# -----------------------------
# Localization via sliding-window PPL
# -----------------------------


@dataclass(frozen=True)
class WindowScore:
    start: int
    end: int
    ppl: float


def _iter_window_starts(n: int, window: int, stride: int) -> List[int]:
    if n <= 0:
        return [0]
    window = max(1, min(window, n))
    stride = max(1, stride)
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, stride))
    last = n - window
    if starts and starts[-1] != last:
        starts.append(last)
    if not starts:
        starts = [0, last]
    out: List[int] = []
    seen = set()
    for s in starts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _ppl_cache_key(model_id: str, chunk_id: str, start: int, end: int) -> str:
    return f"{model_id}::{chunk_id}::{start}::{end}"


def _score_window(
    *,
    judge: Any,
    model_id: str,
    chunk_id: str,
    abs_start: int,
    abs_end: int,
    text: str,
    cache: Dict[str, float],
) -> Optional[float]:
    key = _ppl_cache_key(model_id, chunk_id, abs_start, abs_end)
    if key in cache:
        return cache[key]
    ppl = judge.perplexity(text)
    if ppl is None:
        return None
    cache[key] = float(ppl)
    return float(ppl)


def localize_span_by_ppl(
    *,
    judge: Any,
    model_id: str,
    chunk_id: str,
    full_text: str,
    snippet: str,
    snippet_offset: int,
    window_chars: int,
    min_window_chars: int,
    max_windows_per_level: int,
    topk_windows: int,
    cache: Dict[str, float],
) -> Tuple[Tuple[int, int], List[WindowScore]]:
    """Return ((chunk_abs_start, chunk_abs_end), top_windows_final_level)."""

    snip = snippet or ""
    n = len(snip)
    if n <= 0:
        return (0, min(len(full_text), max(1, window_chars))), []

    offset = max(0, min(int(snippet_offset), len(full_text)))

    region_start = 0
    region_end = n

    w = max(8, int(window_chars))
    w = min(w, n)
    w_min = max(8, int(min_window_chars))
    w_min = min(w_min, w)

    best_abs = (offset, offset + w)
    final_level_scores: List[WindowScore] = []

    while True:
        region_len = max(0, region_end - region_start)
        if region_len <= 0:
            break
        w_eff = min(w, region_len)
        stride = max(1, w_eff // 2)
        starts = _iter_window_starts(region_len, w_eff, stride)

        if max_windows_per_level > 0 and len(starts) > max_windows_per_level:
            keep = max(1, int(max_windows_per_level))
            step = max(1, len(starts) // keep)
            starts = starts[::step][:keep]
            last = region_len - w_eff
            if last not in starts:
                starts.append(last)

        scores: List[WindowScore] = []
        for s in starts:
            e = s + w_eff
            sub = snip[region_start + s : region_start + e]
            abs_s = offset + region_start + s
            abs_e = offset + region_start + e
            ppl = _score_window(
                judge=judge,
                model_id=model_id,
                chunk_id=chunk_id,
                abs_start=abs_s,
                abs_end=abs_e,
                text=sub,
                cache=cache,
            )
            if ppl is None:
                continue
            scores.append(WindowScore(start=abs_s, end=abs_e, ppl=float(ppl)))

        if not scores:
            break

        scores.sort(key=lambda x: (-x.ppl, x.start))
        best = scores[0]
        best_abs = (best.start, best.end)

        if w_eff <= w_min:
            final_level_scores = scores[: max(1, int(topk_windows))]
            break

        rel_best_start = best.start - offset
        rel_best_end = best.end - offset
        region_start = max(0, rel_best_start)
        region_end = min(n, rel_best_end)
        w = max(w_min, w_eff // 2)

    a, b = best_abs
    a = max(0, min(a, len(full_text)))
    b = max(0, min(b, len(full_text)))
    if b < a:
        a, b = b, a
    if b == a:
        b = min(len(full_text), a + 1)

    return (a, b), final_level_scores


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="PPL_Character baseline: chunk ranking + char localization via PPL.")
    ap.add_argument("--run", required=True, help="Input pass0 run JSONL from scripts/02_run_rag.py")
    ap.add_argument("--chunks", required=True, help="chunks.jsonl (for full chunk text + stable chunk_id)")
    ap.add_argument("--out-pass1", required=True, help="Output JSONL with pass1 field (for scripts/07_eval_pass1.py)")

    ap.add_argument("--max-hits", type=int, default=10, help="How many prompt-used hits to consider")
    ap.add_argument("--max-judge-chars", type=int, default=800, help="Max chars per chunk snippet for chunk-level scoring")
    ap.add_argument(
        "--max-loc-chars",
        type=int,
        default=1200,
        help="Max chars of snippet used for char-level localization (per selected chunk)",
    )

    ap.add_argument("--window-chars", type=int, default=256, help="Initial window size (chars) for localization")
    ap.add_argument("--min-window-chars", type=int, default=64, help="Minimum window size (chars) for localization")
    ap.add_argument("--max-windows-per-level", type=int, default=24, help="Cap windows per level to bound runtime")
    ap.add_argument("--topk-windows", type=int, default=10, help="How many best windows to export into pass1.span_tests")

    ap.add_argument(
        "--judge-backend",
        choices=["ollama", "hf"],
        default="ollama",
        help="Perplexity judge backend",
    )
    ap.add_argument("--judge-model", default="llama3.1:8b", help="Ollama judge model name")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--hf-model", default="gpt2")
    ap.add_argument("--hf-device", default="cpu")

    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable judge warmup (useful for debugging)",
    )

    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache dir for PPL scores (json). Greatly speeds up all-splits runs.",
    )

    args = ap.parse_args()

    out_path = Path(args.out_pass1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    chunks_text = load_chunks_text(args.chunks)

    if args.judge_backend == "ollama":
        judge: Any = OllamaPPLJudge(model=args.judge_model, base_url=args.ollama_url)
        model_id = f"ollama::{args.judge_model}@{args.ollama_url}".rstrip("/")
    else:
        judge = HFPPLJudge(model=args.hf_model, device=args.hf_device)
        model_id = f"hf::{args.hf_model}@{args.hf_device}"

    if not args.no_warmup:
        try:
            judge.warmup()
        except Exception:
            pass

    cache: Dict[str, float] = {}
    cache_path: Optional[Path] = None
    if args.cache_dir:
        cache_path = Path(args.cache_dir) / "ppl_character_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(cache, dict):
                    cache = {}
            except Exception:
                cache = {}

    n_total = 0
    n_triggered = 0

    start_t = time.time()

    with jsonlines.open(args.run, "r") as reader, jsonlines.open(out_path, "w") as writer:
        for rec in tqdm(reader, desc="PPL-Character", ncols=100):
            n_total += 1
            calls_before = int(getattr(judge, "calls", 0))

            gen = rec.get("generation", {}) or {}
            comp = (gen.get("completion", "") or "")
            pred_ans = _extract_answer(comp)

            used = _used_hits(rec, max_hits=int(args.max_hits))
            evt, matched_ia, _best_hit = _event_found(pred_ans, used)

            p1: Dict[str, Any] = {
                "method": "ppl_character",
                "judge_backend": args.judge_backend,
                "judge_model": (args.judge_model if args.judge_backend == "ollama" else args.hf_model),
                "max_hits": int(args.max_hits),
                "max_judge_chars": int(args.max_judge_chars),
                "max_loc_chars": int(args.max_loc_chars),
                "window_chars": int(args.window_chars),
                "min_window_chars": int(args.min_window_chars),
                "triggered": bool(evt),
                "baseline": {
                    "event_answer": pred_ans,
                    "matched_incorrect": matched_ia,
                },
                "chunk_tests": [],
                "selected_chunk": None,
                "attribution_span": None,
                "selected_span": None,
                "span_tests": [],
                "llm_calls": 0,
                "n_rounds": 1,
                "mask_chars_total": 0,
                "sanitized": None,
            }

            if not evt or not used:
                out_row = dict(rec)
                out_row["pass1"] = p1
                writer.write(out_row)
                continue

            n_triggered += 1

            user_prompt = str(gen.get("user", "") or "")
            block_map = _build_block_map(rec)

            # --------- Chunk ranking by PPL ---------
            chunk_scores: List[Dict[str, Any]] = []

            # Optional heuristic: prioritize chunks that contain the observed bad answer,
            # but still score all used hits to keep metrics fair.
            cand_used: List[Dict[str, Any]] = []
            other_used: List[Dict[str, Any]] = []
            for h in used:
                rank = _as_int(h.get("rank", -1), -1)
                cid = (h.get("chunk_id", "") or "").strip()
                block = block_map.get((rank, cid)) if (rank >= 0 and cid) else None
                snippet = _block_text(user_prompt, block) if block else ""
                if pred_ans and snippet and _match(snippet, pred_ans):
                    cand_used.append(h)
                else:
                    other_used.append(h)
            hits_to_score = cand_used + other_used
            if not hits_to_score:
                hits_to_score = used

            for h in tqdm(hits_to_score, desc="PPL-Chunks", ncols=100, leave=False):
                rank = _as_int(h.get("rank", -1), -1)
                cid = (h.get("chunk_id", "") or "").strip()
                if not cid:
                    continue

                block = block_map.get((rank, cid)) if (rank >= 0 and cid) else None
                snippet = _block_text(user_prompt, block) if block else ""

                full_text = chunks_text.get(cid, "") or (h.get("text", "") or "")
                text_for_score = snippet or full_text or ""
                if not text_for_score:
                    continue
                text_for_score = text_for_score[: int(args.max_judge_chars)]

                offset = _align_prompt_snippet(full_text, snippet) if snippet and full_text else 0
                abs_start = offset
                abs_end = offset + len(text_for_score)

                ppl = _score_window(
                    judge=judge,
                    model_id=model_id,
                    chunk_id=cid,
                    abs_start=abs_start,
                    abs_end=abs_end,
                    text=text_for_score,
                    cache=cache,
                )
                if ppl is None:
                    continue

                chunk_scores.append(
                    {
                        "chunk_id": cid,
                        "rank": int(rank),
                        "ppl": float(ppl),
                        "used_snippet": bool(bool(snippet)),
                        "scored_chars": int(len(text_for_score)),
                    }
                )

            if not chunk_scores:
                p1["llm_calls"] = int(getattr(judge, "calls", 0)) - calls_before
                out_row = dict(rec)
                out_row["pass1"] = p1
                writer.write(out_row)
                continue

            chunk_scores.sort(key=lambda x: (-float(x["ppl"]), int(x.get("rank", 10**9))))
            p1["chunk_tests"] = chunk_scores

            best_chunk = chunk_scores[0]
            sel_cid = str(best_chunk["chunk_id"])
            sel_rank = int(best_chunk.get("rank", -1))
            p1["selected_chunk"] = {
                "chunk_id": sel_cid,
                "rank": sel_rank,
                "score": float(best_chunk["ppl"]),
                "score_name": "ppl",
            }

            # --------- Char localization within selected chunk ---------
            sel_block = block_map.get((sel_rank, sel_cid)) if (sel_rank >= 0 and sel_cid) else None
            sel_snippet_prompt = _block_text(user_prompt, sel_block) if sel_block else ""

            sel_full_text = chunks_text.get(sel_cid, "") or ""
            if not sel_full_text:
                sel_full_text = sel_snippet_prompt or ""

            sel_snippet = (sel_snippet_prompt or sel_full_text)[: int(args.max_loc_chars)]
            sel_offset = _align_prompt_snippet(sel_full_text, sel_snippet_prompt) if sel_snippet_prompt else 0

            (abs_s, abs_e), top_windows = localize_span_by_ppl(
                judge=judge,
                model_id=model_id,
                chunk_id=sel_cid,
                full_text=sel_full_text,
                snippet=sel_snippet,
                snippet_offset=sel_offset,
                window_chars=int(args.window_chars),
                min_window_chars=int(args.min_window_chars),
                max_windows_per_level=int(args.max_windows_per_level),
                topk_windows=int(args.topk_windows),
                cache=cache,
            )

            snippet_text = sel_full_text[abs_s:abs_e] if sel_full_text else ""
            attr = {
                "chunk_id": sel_cid,
                "rank": sel_rank,
                "label": "ppl_window",
                "chunk_span": [int(abs_s), int(abs_e)],
                "snippet": snippet_text,
                "score": float(best_chunk["ppl"]),
                "score_name": "ppl_chunk",
            }
            p1["attribution_span"] = attr
            p1["selected_span"] = dict(attr)
            p1["mask_chars_total"] = int(max(0, abs_e - abs_s))

            span_tests: List[Dict[str, Any]] = []
            for ws in top_windows:
                snip2 = sel_full_text[ws.start:ws.end] if sel_full_text else ""
                span_tests.append(
                    {
                        "chunk_id": sel_cid,
                        "rank": sel_rank,
                        "label": "ppl_window",
                        "chunk_span": [int(ws.start), int(ws.end)],
                        "score": float(ws.ppl),
                        "score_name": "ppl",
                        "snippet": snip2,
                    }
                )
            span_tests.sort(key=lambda x: (-float(x["score"]), x["chunk_span"][0]))
            p1["span_tests"] = span_tests

            p1["llm_calls"] = int(getattr(judge, "calls", 0)) - calls_before

            out_row = dict(rec)
            out_row["pass1"] = p1
            writer.write(out_row)

    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    dt = time.time() - start_t
    print(
        json.dumps(
            {
                "n_total": n_total,
                "n_triggered": n_triggered,
                "judge_calls": int(getattr(judge, "calls", 0)),
                "seconds": dt,
                "out": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
