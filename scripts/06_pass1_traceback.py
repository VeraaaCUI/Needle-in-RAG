from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import jsonlines
from tqdm import tqdm

from rag_char_trace.llm.base import LLMConfig
from rag_char_trace.llm.ollama import OllamaLLM
from rag_char_trace.llm.openai_api import OpenAILLM

from rag_char_trace.pass1.trigger import should_trigger
from rag_char_trace.pass1.candidates import candidate_spans_for_chunk
from rag_char_trace.pass1.occlusion import (
    PromptEvaluator,
    influence_score,
    success_predicate,
    bisect_minimal_span,
)
from rag_char_trace.pass1.textutil import apply_masks, sentence_spans
from rag_char_trace.pass1.types import Signals
from rag_char_trace.utils.text import normalize_text, contains_normalized


def _used_k(row: Dict[str, Any], default_k: int = 10) -> int:
    retr = (row.get("retrieval", {}) or {})
    k = retr.get("prompt_used_hits", None)
    if isinstance(k, int) and k > 0:
        return k
    k = retr.get("prompt_max_hits", None)
    if isinstance(k, int) and k > 0:
        return k
    return default_k


def _signals_from_trace(
    row: Dict[str, Any], *, gold_fallback: str, target_fallback: str, pool_fallback: List[str]
) -> Optional[Signals]:
    """Build Signals from the stored trace without an extra LLM call."""
    gen = (row.get("generation", {}) or {})
    base_sig = None
    try:
        base_sig = gen.get("signals", None)
    except Exception:
        base_sig = None
    if not isinstance(base_sig, dict):
        return None
    pred = str(base_sig.get("pred", "") or "")
    return Signals(
        pred=pred,
        gold=str(base_sig.get("gold", "") or gold_fallback),
        target=str(base_sig.get("target", "") or target_fallback),
        T_ASR=bool(base_sig.get("T_ASR", False)),
        PAR=bool(base_sig.get("PAR", False)),
        P0_ASR=bool(base_sig.get("P0_ASR", False)),
        UNK=bool(base_sig.get("UNK", False)),
        CONFUSION=bool(base_sig.get("CONFUSION", False)),
        EM=bool(base_sig.get("EM", False)),
        EM_contains=bool(base_sig.get("EM_contains", False)),
        pool_incorrect_answers=list(base_sig.get("pool_incorrect_answers", []) or pool_fallback),
    )


def _trigger_on_signals(sig: Signals, trigger: str) -> bool:
    t = (trigger or "PAR").upper()
    if t == "ALWAYS":
        return True
    if t == "PAR":
        return bool(sig.PAR)
    if t == "T_ASR":
        return bool(sig.T_ASR)
    if t == "ANY":
        return bool(sig.PAR or sig.T_ASR or sig.CONFUSION)
    if t == "CONFUSION":
        return bool(sig.CONFUSION)
    return bool(sig.PAR)


def _overlaps_any(span: Tuple[int, int], regions: List[Tuple[int, int]]) -> bool:
    a, b = span
    for x, y in regions:
        if a < y and x < b:
            return True
    return False


def _block_text(prompt: str, block: Dict[str, Any]) -> str:
    a = int(block.get("prompt_text_start", -1))
    e = int(block.get("prompt_text_end", -1))
    if a < 0 or e < 0 or e <= a:
        return ""
    return prompt[a:e]



def _attribution_span_local(chunk_text: str, bad_answer: str) -> Tuple[int, int, str]:
    """Cheap heuristic attribution span in *chunk-local* character coords.

    This is intentionally label-free: it uses the observed bad_answer and simple text patterns.
    It is used for char-level localization reporting (guilty_spans overlap), not for
    mitigation masking decisions.
    """
    text = chunk_text or ""
    ans = (bad_answer or "").strip()

    # 1) Exact (case-insensitive) match of the observed bad answer in the chunk text.
    if ans:
        i = text.lower().find(ans.lower())
        if i != -1:
            return i, i + len(ans), "attr_event_answer"

    # 2) Prefix capitalized phrase (1-6 words): common for entity-style poisoned answers.
    m = re.match(r"^\s*([A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){0,5})", text)
    if m:
        s = text.find(m.group(1))
        return s, s + len(m.group(1)), "attr_prefix_cap"

    # 3) Fallback: first token.
    m2 = re.match(r"^\s*(\S+)", text)
    if m2:
        s = text.find(m2.group(1))
        return s, s + len(m2.group(1)), "attr_prefix_token"

    return 0, 0, ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pass-1: span-level poison traceback + sanitize replay (iterative, optimized)."
    )
    ap.add_argument("--run", required=True, help="Input Pass-0 run JSONL")
    ap.add_argument("--out", required=True, help="Output JSONL with pass1 field added")

    # Trigger controls which examples are processed (offline evaluation convenience).
    # For label-free usage, use --trigger ALWAYS and objective=event.
    ap.add_argument(
        "--trigger",
        default="PAR",
        choices=["PAR", "T_ASR", "ANY", "CONFUSION", "ALWAYS"],
        help="Which events to trace back",
    )

    # Optimization controls (defaults are cost-conscious; disable with --no-... flags)
    ap.add_argument(
        "--objective",
        default="event",
        choices=["event", "oracle"],
        help="Mask-scoring objective (event is label-free).",
    )
    ap.add_argument(
        "--candidate-mode",
        default="event",
        choices=["event", "oracle"],
        help="Candidate span generator mode.",
    )
    ap.add_argument("--max-rounds", type=int, default=2, help="Max iterative rounds.")
    ap.add_argument(
        "--stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Early stop chunk/sentence scans when a success is found.",
    )
    ap.add_argument(
        "--stop-when-trigger-resolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop iterative rounds once trigger condition becomes False.",
    )

    ap.add_argument(
        "--sentence-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Two-stage localization: sentence occlusion first, then (if needed) char refinement.",
    )
    ap.add_argument(
        "--max-sentences-to-test",
        type=int,
        default=5,
        help="Max sentence candidates to test (per round) inside selected chunk.",
    )
    ap.add_argument(
        "--top-sentences",
        type=int,
        default=2,
        help="If no sentence achieves success, restrict fine span search to top-N sentences by score.",
    )
    ap.add_argument("--sentence-min-len", type=int, default=8, help="Min sentence length to consider.")

    ap.add_argument("--max-examples", type=int, default=None, help="Process at most N triggered examples (None = all)")
    ap.add_argument("--k-use-default", type=int, default=10, help="Fallback used-hits if missing from trace")
    ap.add_argument("--max-chunks-to-test", type=int, default=5, help="How many prompt-used chunks to occlude-test per round")
    ap.add_argument("--max-span-candidates", type=int, default=12, help="How many fine-grained span candidates to score (only for hard cases)")
    ap.add_argument("--max-bisect-steps", type=int, default=6)
    ap.add_argument("--min-span-len", type=int, default=4)

    # LLM backend
    ap.add_argument("--llm-backend", choices=["ollama", "openai", "llama_cpp"], default="ollama")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=32)

    ap.add_argument("--ollama-model", default="llama3:8b")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--openai-model", default="gpt-4o-mini")

    args = ap.parse_args()

    llm_cfg = LLMConfig(temperature=args.temperature, max_tokens=args.max_tokens)

    if args.llm_backend == "ollama":
        llm = OllamaLLM(model=args.ollama_model, base_url=args.ollama_url)
    elif args.llm_backend == "openai":
        llm = OpenAILLM(model=args.openai_model)
    else:
        from rag_char_trace.llm.llama_cpp import LlamaCppLLM

        model_path = os.environ.get("LLAMA_CPP_MODEL_PATH")
        if not model_path:
            raise SystemExit("For llama_cpp backend, set env var LLAMA_CPP_MODEL_PATH to a local .gguf model path.")
        llm = LlamaCppLLM(model_path=model_path)

    evaluator = PromptEvaluator(llm=llm, llm_cfg=llm_cfg)

    triggered_seen = 0

    with jsonlines.open(args.run, mode="r") as reader, jsonlines.open(args.out, mode="w") as writer:
        for row in tqdm(reader, desc="Pass1", ncols=100):
            out_row = row

            if not should_trigger(row, args.trigger):
                writer.write(out_row)
                continue

            triggered_seen += 1
            if args.max_examples is not None and triggered_seen > args.max_examples:
                writer.write(out_row)
                continue

            gen = (row.get("generation", {}) or {})
            retr = (row.get("retrieval", {}) or {})
            meta = (row.get("meta", {}) or {})

            system = str(gen.get("system", "") or "")
            user_prompt = str(gen.get("user", "") or "")

            # oracle metadata (used only for analysis signals; the event objective does not depend on them)
            gold = str((meta.get("answer", "") or gen.get("signals", {}).get("gold", "") or "")).strip()
            target = str((meta.get("incorrect_answer", "") or gen.get("signals", {}).get("target", "") or "")).strip()
            pool_incorrect = list((gen.get("signals", {}) or {}).get("pool_incorrect_answers", []) or [])

            # used hits / prompt blocks
            used_k = _used_k(row, default_k=args.k_use_default)
            hits = list(retr.get("hits", []) or [])
            used_hits = hits[:used_k] if used_k > 0 else hits

            p0_incorrect = ""
            if used_hits:
                p0_incorrect = str(((used_hits[0].get("meta", {}) or {}).get("incorrect_answer", "") or "")).strip()

            prompt_blocks = list(retr.get("prompt_blocks", []) or [])
            used_ranks = set(int(h.get("rank", -1)) for h in used_hits)
            prompt_blocks_used = [b for b in prompt_blocks if int(b.get("rank", -1)) in used_ranks]
            prompt_blocks_used.sort(key=lambda b: int(b.get("rank", 10**9)))

            pass1: Dict[str, Any] = {
                "trigger": args.trigger,
                "objective": args.objective,
                "candidate_mode": args.candidate_mode,
                "sentence_first": bool(args.sentence_first),
                "triggered": True,
                "baseline": {},
                "chunk_tests": [],
                "sentence_tests": [],
                "span_tests": [],
                "selected_chunk": None,
                "attribution_span": None,  # cheap label-free attribution span (for char-level reporting)
                "selected_span": None,       # first-round mitigation span
                "final_selected_span": None, # last-round span (mitigation)
                "mask_prompt_regions": [],
                "rounds": [],
                "sanitized": None,
                "llm_calls": 0,
            }

            cache_before = len(evaluator.cache)

            # Baseline signals: reuse trace if possible (no extra call)
            base0 = _signals_from_trace(row, gold_fallback=gold, target_fallback=target, pool_fallback=pool_incorrect)
            if base0 is None:
                base0_eval = evaluator.eval(
                    system=system,
                    user=user_prompt,
                    gold=gold,
                    target=target,
                    pool_incorrect_answers=pool_incorrect,
                    p0_incorrect=p0_incorrect,
                )
                base0 = base0_eval.signals

            pass1["baseline"] = {
                "pred": base0.pred,
                "T_ASR": base0.T_ASR,
                "PAR": base0.PAR,
                "P0_ASR": base0.P0_ASR,
                "UNK": base0.UNK,
                "CONFUSION": base0.CONFUSION,
                "EM_contains": base0.EM_contains,
            }

            if not prompt_blocks_used:
                pass1["error"] = "No usable prompt_blocks in trace; cannot run Pass-1 occlusion."
                pass1["llm_calls"] = len(evaluator.cache) - cache_before
                out_row = dict(row)
                out_row["pass1"] = pass1
                writer.write(out_row)
                continue

            # Iterative masking loop (greedy)
            mask_regions: List[Tuple[int, int]] = []
            last_eval = None
            last_span = None

            for rnd in range(max(1, int(args.max_rounds))):
                cur_user = apply_masks(user_prompt, mask_regions) if mask_regions else user_prompt

                # For round 0, reuse baseline signals; for later rounds, evaluate current prompt.
                if rnd == 0:
                    base = base0
                else:
                    ev_base = evaluator.eval(
                        system=system,
                        user=cur_user,
                        gold=gold,
                        target=target,
                        pool_incorrect_answers=pool_incorrect,
                        p0_incorrect=p0_incorrect,
                    )
                    base = ev_base.signals

                # Stop if trigger resolved (optional)
                if rnd > 0 and args.stop_when_trigger_resolved and (not _trigger_on_signals(base, args.trigger)):
                    break

                bad_answer = base.pred  # observed answer at this round
                if not bad_answer.strip():
                    break

                round_info: Dict[str, Any] = {
                    "round": rnd,
                    "bad_answer": bad_answer,
                    "base": {
                        "pred": base.pred,
                        "T_ASR": base.T_ASR,
                        "PAR": base.PAR,
                        "EM_contains": base.EM_contains,
                        "CONFUSION": base.CONFUSION,
                    },
                    "chunk_tests": [],
                    "selected_chunk": None,
                    "sentence_tests": [],
                    "selected_sentence": None,
                    "span_tests": [],
                    "attribution_span": None,
                    "selected_span": None,
                    "mask_added": None,
                    "after": None,
                }

                # --- 1) Chunk-level occlusion tests ---
                best_chunk = None
                best_chunk_score = float("-inf")

                # Pre-filter chunks that contain the current bad answer (cost-saving + better attribution ordering).
                blocks_with_answer = []
                blocks_other = []
                for b in prompt_blocks_used:
                    txt = _block_text(cur_user, b)
                    if bad_answer and txt and contains_normalized(txt, bad_answer):
                        blocks_with_answer.append(b)
                    else:
                        blocks_other.append(b)
                blocks_to_test = blocks_with_answer if blocks_with_answer else prompt_blocks_used
                blocks_to_test = blocks_to_test[: max(0, int(args.max_chunks_to_test))]

                for b in blocks_to_test:
                    a = int(b.get("prompt_text_start", -1))
                    e = int(b.get("prompt_text_end", -1))
                    if a < 0 or e < 0 or e <= a:
                        continue

                    masked_user = apply_masks(cur_user, [(a, e)])
                    ev = evaluator.eval(
                        system=system,
                        user=masked_user,
                        gold=gold,
                        target=target,
                        pool_incorrect_answers=pool_incorrect,
                        p0_incorrect=p0_incorrect,
                    )
                    sc = influence_score(base, ev.signals, objective=args.objective, bad_answer=bad_answer, user_prompt=cur_user)
                    ok = success_predicate(base, ev.signals, args.trigger, objective=args.objective, bad_answer=bad_answer)

                    info = {
                        "round": rnd,
                        "rank": int(b.get("rank", -1)),
                        "chunk_id": str(b.get("chunk_id", "")),
                        "mask_prompt_region": [a, e],
                        "score": sc,
                        "success": bool(ok),
                        "after": {
                            "pred": ev.signals.pred,
                            "T_ASR": ev.signals.T_ASR,
                            "PAR": ev.signals.PAR,
                            "EM_contains": ev.signals.EM_contains,
                            "CONFUSION": ev.signals.CONFUSION,
                        },
                    }
                    round_info["chunk_tests"].append(info)

                    if (sc > best_chunk_score) or (
                        sc == best_chunk_score and int(b.get("rank", 10**9)) < int(best_chunk.get("rank", 10**9)) if best_chunk else False
                    ):
                        best_chunk_score = sc
                        best_chunk = b

                    if args.stop_on_success and ok:
                        break

                round_info["chunk_tests"].sort(key=lambda x: (-float(x["score"]), int(x["rank"])))

                if rnd == 0:
                    pass1["chunk_tests"] = list(round_info["chunk_tests"])

                if best_chunk is None:
                    round_info["error"] = "No usable chunk_text region in prompt_blocks."
                    pass1["rounds"].append(round_info)
                    break

                round_info["selected_chunk"] = {
                    "rank": int(best_chunk.get("rank", -1)),
                    "chunk_id": str(best_chunk.get("chunk_id", "")),
                    "best_score": best_chunk_score,
                }

                if rnd == 0:
                    pass1["selected_chunk"] = dict(round_info["selected_chunk"])

                # chunk text slice (raw / unmasked) for sentence splitting
                text_a = int(best_chunk.get("prompt_text_start", -1))
                text_e = int(best_chunk.get("prompt_text_end", -1))
                chunk_text_raw = user_prompt[text_a:text_e] if text_a >= 0 and text_e > text_a else ""

                # --- Attribution span (cheap, label-free) for char-level reporting ---
                attr_s, attr_e, attr_label = _attribution_span_local(chunk_text_raw, bad_answer)
                attr_span = None
                if attr_e > attr_s and text_a >= 0:
                    attr_span = {
                        "chunk_id": str(best_chunk.get("chunk_id", "")),
                        "rank": int(best_chunk.get("rank", -1)),
                        "label": str(attr_label),
                        "chunk_span": [int(attr_s), int(attr_e)],
                        "prompt_span": [int(text_a + attr_s), int(text_a + attr_e)],
                        "snippet": chunk_text_raw[attr_s:attr_e],
                        "round_bad_answer": bad_answer,
                    }
                round_info["attribution_span"] = attr_span
                if rnd == 0:
                    pass1["attribution_span"] = attr_span

                # --- 2) Sentence-level occlusion inside selected chunk (two-stage) ---
                focus_sentence_regions: List[Tuple[int, int]] = []
                best_success_sentence = None
                best_success_score = float("-inf")
                best_any_sentence = None
                best_any_score = float("-inf")

                if args.sentence_first and chunk_text_raw:
                    # generate sentence candidates in chunk-local coords then map to prompt coords
                    sent_local = sentence_spans(
                        chunk_text_raw,
                        max_sentences=50,
                        min_len=int(args.sentence_min_len),
                    )
                    sent_cands = []
                    for s0, e0 in sent_local:
                        ps = text_a + int(s0)
                        pe = text_a + int(e0)
                        if pe <= ps:
                            continue
                        if _overlaps_any((ps, pe), mask_regions):
                            continue
                        s_text = chunk_text_raw[s0:e0]
                        has_bad = bool(bad_answer and contains_normalized(s_text, bad_answer))
                        sent_cands.append((0 if has_bad else 1, -(e0 - s0), s0, e0, ps, pe, has_bad))

                    # Prefer sentences containing the bad answer; then longer; then earlier.
                    sent_cands.sort()
                    sent_cands = sent_cands[: max(0, int(args.max_sentences_to_test))]

                    for _, _, s0, e0, ps, pe, has_bad in sent_cands:
                        masked_user = apply_masks(cur_user, [(ps, pe)])
                        ev = evaluator.eval(
                            system=system,
                            user=masked_user,
                            gold=gold,
                            target=target,
                            pool_incorrect_answers=pool_incorrect,
                            p0_incorrect=p0_incorrect,
                        )
                        sc = influence_score(base, ev.signals, objective=args.objective, bad_answer=bad_answer, user_prompt=cur_user)
                        ok = success_predicate(base, ev.signals, args.trigger, objective=args.objective, bad_answer=bad_answer)

                        info = {
                            "round": rnd,
                            "chunk_id": str(best_chunk.get("chunk_id", "")),
                            "rank": int(best_chunk.get("rank", -1)),
                            "label": "sentence",
                            "contains_bad": bool(has_bad),
                            "score": sc,
                            "success": bool(ok),
                            "chunk_span": [int(s0), int(e0)],
                            "prompt_span": [int(ps), int(pe)],
                            "after": {
                                "pred": ev.signals.pred,
                                "T_ASR": ev.signals.T_ASR,
                                "PAR": ev.signals.PAR,
                                "EM_contains": ev.signals.EM_contains,
                                "CONFUSION": ev.signals.CONFUSION,
                            },
                        }
                        round_info["sentence_tests"].append(info)

                        if ok and sc > best_success_score:
                            best_success_score = sc
                            best_success_sentence = info
                        if sc > best_any_score:
                            best_any_score = sc
                            best_any_sentence = info

                        if args.stop_on_success and ok:
                            break

                    round_info["sentence_tests"].sort(key=lambda x: (-float(x["score"]), x["prompt_span"][0]))

                    if rnd == 0:
                        pass1["sentence_tests"] = list(round_info["sentence_tests"])

                    # Focus regions for fine-grained search (only used if we need it)
                    topn = max(0, int(args.top_sentences))
                    for info in round_info["sentence_tests"][:topn]:
                        ps = info.get("prompt_span", None)
                        if isinstance(ps, list) and len(ps) == 2:
                            focus_sentence_regions.append((int(ps[0]), int(ps[1])))

                # If we already have a successful sentence occlusion, jump straight to bisection refinement
                if args.sentence_first and best_success_sentence is not None:
                    ps = best_success_sentence["prompt_span"]
                    lo, hi, ev_ref = bisect_minimal_span(
                        evaluator=evaluator,
                        system=system,
                        user_prompt=cur_user,
                        base=base,
                        trigger=args.trigger,
                        gold=gold,
                        target=target,
                        pool_incorrect_answers=pool_incorrect,
                        p0_incorrect=p0_incorrect,
                        span_prompt_start=int(ps[0]),
                        span_prompt_end=int(ps[1]),
                        max_steps=int(args.max_bisect_steps),
                        min_len=int(args.min_span_len),
                        objective=args.objective,
                        bad_answer=bad_answer,
                    )

                    chunk_local_lo = max(0, int(lo) - int(text_a))
                    chunk_local_hi = max(0, int(hi) - int(text_a))

                    sel_span = {
                        "chunk_id": str(best_chunk.get("chunk_id", "")),
                        "rank": int(best_chunk.get("rank", -1)),
                        "label": "sentence_bisect",
                        "score": float(best_success_sentence.get("score", 0.0)),
                        "prompt_span": [int(lo), int(hi)],
                        "chunk_span": [int(chunk_local_lo), int(chunk_local_hi)],
                        "snippet": cur_user[lo:hi],
                        "round_bad_answer": bad_answer,
                    }

                    last_span = sel_span
                    last_eval = ev_ref

                    mask_regions.append((int(lo), int(hi)))

                    round_info["selected_sentence"] = {
                        "prompt_span": list(best_success_sentence.get("prompt_span", [])),
                        "chunk_span": list(best_success_sentence.get("chunk_span", [])),
                        "score": float(best_success_sentence.get("score", 0.0)),
                    }
                    round_info["selected_span"] = dict(sel_span)
                    round_info["mask_added"] = [int(lo), int(hi)]
                    round_info["after"] = {
                        "pred": ev_ref.signals.pred,
                        "T_ASR": ev_ref.signals.T_ASR,
                        "PAR": ev_ref.signals.PAR,
                        "EM_contains": ev_ref.signals.EM_contains,
                        "CONFUSION": ev_ref.signals.CONFUSION,
                    }

                    if rnd == 0:
                        pass1["selected_span"] = dict(sel_span)

                    pass1["rounds"].append(round_info)

                    # Stop if trigger resolved after this round
                    if args.stop_when_trigger_resolved and (not _trigger_on_signals(last_eval.signals, args.trigger)):
                        break

                    continue  # next round

                # --- 3) Fine-grained candidate spans inside selected chunk (hard cases) ---
                spans = candidate_spans_for_chunk(
                    row=row,
                    block=best_chunk,
                    chunk_text=chunk_text_raw,
                    max_candidates=int(args.max_span_candidates),
                    mode=args.candidate_mode,
                    target_text=bad_answer,
                )

                # Filter candidates that overlap already-masked regions
                spans = [sp for sp in spans if not _overlaps_any((sp.prompt_start, sp.prompt_end), mask_regions)]

                # If sentence_first ran but didn't find a success, restrict spans to top sentences (if available)
                if focus_sentence_regions:
                    filtered = []
                    for sp in spans:
                        for sa, sb in focus_sentence_regions:
                            if sp.prompt_start >= sa and sp.prompt_end <= sb:
                                filtered.append(sp)
                                break
                    if filtered:
                        spans = filtered

                # --- 4) Score spans by targeted masking ---
                best_span = None
                best_span_score = float("-inf")

                for sp in spans:
                    masked_user = apply_masks(cur_user, [(sp.prompt_start, sp.prompt_end)])
                    ev = evaluator.eval(
                        system=system,
                        user=masked_user,
                        gold=gold,
                        target=target,
                        pool_incorrect_answers=pool_incorrect,
                        p0_incorrect=p0_incorrect,
                    )
                    sc = influence_score(base, ev.signals, objective=args.objective, bad_answer=bad_answer, user_prompt=cur_user)
                    ok = success_predicate(base, ev.signals, args.trigger, objective=args.objective, bad_answer=bad_answer)

                    info = {
                        "round": rnd,
                        "chunk_id": sp.chunk_id,
                        "rank": sp.rank,
                        "label": sp.label,
                        "score": sc,
                        "success": bool(ok),
                        "chunk_span": [sp.chunk_start, sp.chunk_end],
                        "prompt_span": [sp.prompt_start, sp.prompt_end],
                        "after": {
                            "pred": ev.signals.pred,
                            "T_ASR": ev.signals.T_ASR,
                            "PAR": ev.signals.PAR,
                            "EM_contains": ev.signals.EM_contains,
                            "CONFUSION": ev.signals.CONFUSION,
                        },
                    }
                    round_info["span_tests"].append(info)

                    if sc > best_span_score:
                        best_span_score = sc
                        best_span = sp

                    if args.stop_on_success and ok:
                        break

                round_info["span_tests"].sort(key=lambda x: (-float(x["score"]), int(x["rank"]), x["prompt_span"][0]))

                if rnd == 0:
                    pass1["span_tests"] = list(round_info["span_tests"])

                # --- 5) If no span, fall back to masking the entire chunk text region ---
                if best_span is None:
                    a = int(best_chunk.get("prompt_text_start", -1))
                    e = int(best_chunk.get("prompt_text_end", -1))
                    if a >= 0 and e > a:
                        mask_regions.append((a, e))
                        cur_user2 = apply_masks(cur_user, [(a, e)])
                        ev2 = evaluator.eval(
                            system=system,
                            user=cur_user2,
                            gold=gold,
                            target=target,
                            pool_incorrect_answers=pool_incorrect,
                            p0_incorrect=p0_incorrect,
                        )
                        last_eval = ev2
                        round_info["mask_added"] = [a, e]
                        round_info["after"] = {
                            "pred": ev2.signals.pred,
                            "T_ASR": ev2.signals.T_ASR,
                            "PAR": ev2.signals.PAR,
                            "EM_contains": ev2.signals.EM_contains,
                            "CONFUSION": ev2.signals.CONFUSION,
                        }
                        pass1["rounds"].append(round_info)
                        continue
                    else:
                        round_info["error"] = "No span candidates and no valid chunk region to mask."
                        pass1["rounds"].append(round_info)
                        break

                # --- 6) Bisection refinement for a more minimal char span ---
                lo, hi, ev_ref = bisect_minimal_span(
                    evaluator=evaluator,
                    system=system,
                    user_prompt=cur_user,
                    base=base,
                    trigger=args.trigger,
                    gold=gold,
                    target=target,
                    pool_incorrect_answers=pool_incorrect,
                    p0_incorrect=p0_incorrect,
                    span_prompt_start=best_span.prompt_start,
                    span_prompt_end=best_span.prompt_end,
                    max_steps=int(args.max_bisect_steps),
                    min_len=int(args.min_span_len),
                    objective=args.objective,
                    bad_answer=bad_answer,
                )

                chunk_local_lo = max(0, int(lo) - int(best_chunk.get("prompt_text_start", 0)))
                chunk_local_hi = max(0, int(hi) - int(best_chunk.get("prompt_text_start", 0)))

                sel_span = {
                    "chunk_id": str(best_span.chunk_id),
                    "rank": int(best_span.rank),
                    "label": best_span.label,
                    "score": float(best_span_score),
                    "prompt_span": [int(lo), int(hi)],
                    "chunk_span": [int(chunk_local_lo), int(chunk_local_hi)],
                    "snippet": cur_user[lo:hi],
                    "round_bad_answer": bad_answer,
                }

                last_span = sel_span
                last_eval = ev_ref

                mask_regions.append((int(lo), int(hi)))

                round_info["selected_span"] = dict(sel_span)
                round_info["mask_added"] = [int(lo), int(hi)]
                round_info["after"] = {
                    "pred": ev_ref.signals.pred,
                    "T_ASR": ev_ref.signals.T_ASR,
                    "PAR": ev_ref.signals.PAR,
                    "EM_contains": ev_ref.signals.EM_contains,
                    "CONFUSION": ev_ref.signals.CONFUSION,
                }

                if rnd == 0:
                    pass1["selected_span"] = dict(sel_span)

                pass1["rounds"].append(round_info)

                if args.stop_when_trigger_resolved and (not _trigger_on_signals(last_eval.signals, args.trigger)):
                    break

            # Final sanitize evaluation
            final_user = apply_masks(user_prompt, mask_regions) if mask_regions else user_prompt
            if last_eval is None:
                last_eval = evaluator.eval(
                    system=system,
                    user=final_user,
                    gold=gold,
                    target=target,
                    pool_incorrect_answers=pool_incorrect,
                    p0_incorrect=p0_incorrect,
                )

            pass1["mask_prompt_regions"] = [[int(a), int(b)] for a, b in mask_regions]
            pass1["final_selected_span"] = last_span

            pass1["sanitized"] = {
                "mode": "mask_multi" if len(mask_regions) > 1 else ("mask_span" if len(mask_regions) == 1 else "none"),
                "mask_prompt_regions": [[int(a), int(b)] for a, b in mask_regions],
                "completion": last_eval.completion,
                "parsed": last_eval.parsed,
                "signals": {
                    "pred": last_eval.signals.pred,
                    "T_ASR": last_eval.signals.T_ASR,
                    "PAR": last_eval.signals.PAR,
                    "EM_contains": last_eval.signals.EM_contains,
                    "CONFUSION": last_eval.signals.CONFUSION,
                },
            }

            pass1["llm_calls"] = len(evaluator.cache) - cache_before

            out_row = dict(row)
            out_row["pass1"] = pass1
            writer.write(out_row)


if __name__ == "__main__":
    main()
