from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import jsonlines


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_answer(completion: str) -> str:
    """Best-effort answer extraction for readability."""
    if not completion:
        return ""
    # Prefer explicit "ANSWER:" line
    for ln in completion.splitlines():
        s = ln.strip()
        if s.lower().startswith("answer:"):
            return s.split(":", 1)[1].strip()
    # Try JSON
    try:
        j = json.loads(completion)
        if isinstance(j, dict) and "answer" in j:
            v = j.get("answer")
            return str(v).strip()
    except Exception:
        pass
    # Fallback: first non-empty line
    for ln in completion.splitlines():
        s = ln.strip()
        if s:
            return s
    return completion.strip()


def _merge_spans(spans: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s, e in spans:
        s = max(0, min(_as_int(s, 0), n))
        e = max(0, min(_as_int(e, 0), n))
        if e <= s:
            continue
        out.append((s, e))
    if not out:
        return []
    out.sort(key=lambda x: (x[0], x[1]))
    merged = [out[0]]
    for s, e in out[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _sanitize_text(text: str, spans: List[Tuple[int, int]], mask_token: str) -> str:
    n = len(text)
    spans = _merge_spans(spans, n)
    if not spans:
        return text
    parts: List[str] = []
    last = 0
    for s, e in spans:
        parts.append(text[last:s])
        parts.append(mask_token)
        last = e
    parts.append(text[last:])
    return "".join(parts)


def _collect_chunk_spans(obj: Any, chunk_id: str) -> List[Tuple[int, int]]:
    """Recursively collect all dicts containing (chunk_id, chunk_span) within pass1 payload."""
    spans: List[Tuple[int, int]] = []

    def rec(x: Any) -> None:
        if isinstance(x, dict):
            cid = (x.get("chunk_id", "") or "").strip()
            if cid and cid == chunk_id:
                cs = x.get("chunk_span", None)
                if isinstance(cs, list) and len(cs) >= 2:
                    spans.append((_as_int(cs[0], 0), _as_int(cs[1], 0)))
                # Alternative key shapes
                if "chunk_start" in x and "chunk_end" in x:
                    spans.append((_as_int(x.get("chunk_start"), 0), _as_int(x.get("chunk_end"), 0)))
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for it in x:
                rec(it)

    rec(obj)
    return spans


def _load_chunks(chunks_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with jsonlines.open(chunks_path, "r") as r:
        for j in r:
            cid = (j.get("chunk_id", "") or "").strip()
            if not cid:
                continue
            out[cid] = j
    return out


def _pick_cause_chunk_id(rec: Dict[str, Any]) -> str:
    p1 = rec.get("pass1", {}) or {}
    for path in [
        ("selected_chunk", "chunk_id"),
        ("selected_span", "chunk_id"),
        ("attribution_span", "chunk_id"),
    ]:
        d = p1.get(path[0], {}) or {}
        cid = (d.get(path[1], "") or "").strip()
        if cid:
            return cid

    # Fallback: top-1 used hit
    retr = rec.get("retrieval", {}) or {}
    hits = retr.get("hits", []) or []
    if hits:
        return (hits[0].get("chunk_id", "") or "").strip()
    return ""


def _get_hit_text(rec: Dict[str, Any], chunk_id: str) -> str:
    retr = rec.get("retrieval", {}) or {}
    hits = retr.get("hits", []) or []
    for h in hits:
        if (h.get("chunk_id", "") or "").strip() == chunk_id:
            return h.get("text", "") or ""
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a compact, human-readable JSONL from Pass1 traces.")
    ap.add_argument("--pass1", required=True, help="Pass1 JSONL (output of scripts/06_pass1_traceback.py).")
    ap.add_argument("--chunks", required=True, help="Chunks JSONL to recover full chunk text by chunk_id.")
    ap.add_argument("--out", required=True, help="Output readable JSONL path.")
    ap.add_argument("--limit", type=int, default=None, help="Optional max number of records to export.")
    ap.add_argument("--only-triggered", action="store_true", help="Only export records where pass1.triggered==True.")
    ap.add_argument("--mask-token", default="[[MASK]]", help="Mask token inserted into sanitized chunk text.")
    ap.add_argument("--chunk-max-chars", type=int, default=600, help="Truncate chunk texts to this many chars for readability.")
    args = ap.parse_args()

    chunks = _load_chunks(args.chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with jsonlines.open(args.pass1, "r") as r, jsonlines.open(out_path, "w") as w:
        for rec in r:
            if args.limit is not None and n >= args.limit:
                break

            p1 = rec.get("pass1", {}) or {}
            if args.only_triggered and not bool(p1.get("triggered", False)):
                continue

            qid = (rec.get("qid", "") or "").strip()
            q = rec.get("question", "") or ""

            # Baseline (Pass0)
            base_comp = ((rec.get("generation", {}) or {}).get("completion", "") or "").strip()
            base_ans = _extract_answer(base_comp)

            # Sanitized (Pass1)
            san_comp = (((p1.get("sanitized", {}) or {}).get("completion", "") or "").strip())
            san_ans = _extract_answer(san_comp)

            # Cause chunk + sanitized chunk view
            cid = _pick_cause_chunk_id(rec)
            chunk_text = ""
            if cid and cid in chunks:
                chunk_text = chunks[cid].get("text", "") or ""
            if not chunk_text:
                chunk_text = _get_hit_text(rec, cid)

            spans = _collect_chunk_spans(p1, cid) if cid else []
            san_chunk = _sanitize_text(chunk_text, spans, args.mask_token) if chunk_text else ""

            # Truncate for readability (optionally)
            mx = max(0, int(args.chunk_max_chars))
            if mx and len(chunk_text) > mx:
                chunk_text = chunk_text[:mx] + "…"
            if mx and len(san_chunk) > mx:
                san_chunk = san_chunk[:mx] + "…"

            out_rec = {
                "qid": qid,
                "question": q,
                "llm_wrong_answer": base_ans,
                "cause_chunk_id": cid,
                "cause_chunk": chunk_text,
                "sanitized_chunk": san_chunk,
                "llm_answer_after_sanitize": san_ans,
            }
            w.write(out_rec)
            n += 1

    print(f"Wrote {n} readable records to: {out_path}")


if __name__ == "__main__":
    main()
