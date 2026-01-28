from __future__ import annotations

import re
from typing import Iterable, List, Tuple


def mask_keep_newlines(s: str) -> str:
    """Mask a string with spaces but keep newline characters."""
    if not s:
        return s
    return "".join("\n" if c == "\n" else " " for c in s)


def apply_masks(prompt: str, regions: List[Tuple[int, int]]) -> str:
    """Apply multiple [start,end) masks to the prompt. Masking preserves length."""
    if not regions:
        return prompt
    n = len(prompt)
    # normalize and merge overlaps
    rs = []
    for a, b in regions:
        a = max(0, min(int(a), n))
        b = max(0, min(int(b), n))
        if b <= a:
            continue
        rs.append((a, b))
    if not rs:
        return prompt
    rs.sort()
    merged = []
    cur_a, cur_b = rs[0]
    for a, b in rs[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))

    out = prompt
    # apply from left to right is safe because we preserve length; still, apply from right for clarity
    for a, b in reversed(merged):
        out = out[:a] + mask_keep_newlines(out[a:b]) + out[b:]
    return out


def find_all(haystack: str, needle: str, *, case_insensitive: bool = False) -> List[Tuple[int, int]]:
    if not needle:
        return []
    if case_insensitive:
        h = haystack.lower()
        n = needle.lower()
    else:
        h = haystack
        n = needle

    res = []
    start = 0
    while True:
        i = h.find(n, start)
        if i < 0:
            break
        res.append((i, i + len(needle)))
        start = i + max(1, len(needle))
    return res


_NUM_RE = re.compile(r"(?<!\w)(?:\d{4}|\d+(?:\.\d+)?)(?!\w)")
_CITE_RE = re.compile(r"\[[0-9]{1,3}\]|\([0-9]{4}\)")


def numeric_and_citation_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return (start,end,label) spans for simple numeric/citation fingerprints."""
    spans = []
    for m in _NUM_RE.finditer(text):
        spans.append((m.start(), m.end(), "numeric"))
    for m in _CITE_RE.finditer(text):
        spans.append((m.start(), m.end(), "citation"))
    return spans


def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


_STOP = {
    "the","a","an","of","to","in","on","for","and","or","is","are","was","were","be",
    "what","who","when","where","which","why","how","does","do","did","at","by","from",
    "with","as","it","this","that","these","those"
}


def salient_terms(question: str, max_terms: int = 8) -> List[str]:
    toks = [t for t in tokenize_words(question) if len(t) >= 4 and t not in _STOP]
    # preserve order but dedupe
    out = []
    seen = set()
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= max_terms:
            break
    return out


_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_spans(text: str, *, max_sentences: int = 20, min_len: int = 8) -> List[Tuple[int, int]]:
    """Return rough sentence spans (start,end) within `text`.

    Heuristic splitter:
      - splits on whitespace following [.?!]
      - splits on one-or-more newlines

    The returned spans are trimmed for leading/trailing whitespace, but indices
    refer to the original `text`.

    If no boundaries are found, returns a single span covering the whole text.
    """
    if not text:
        return []

    spans: List[Tuple[int, int]] = []
    start = 0

    def _trim(a: int, b: int) -> Tuple[int, int]:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        return a, b

    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.start()
        a, b = _trim(start, end)
        if b - a >= int(min_len):
            spans.append((a, b))
            if len(spans) >= int(max_sentences):
                return spans
        start = m.end()

    a, b = _trim(start, len(text))
    if b - a >= int(min_len):
        spans.append((a, b))

    if not spans:
        spans = [(0, len(text))]
    return spans
