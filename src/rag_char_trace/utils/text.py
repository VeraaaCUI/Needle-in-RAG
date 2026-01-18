from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Normalize text for matching (ASR/EM).

    - lower
    - collapse whitespace
    - strip
    """
    if s is None:
        return ""
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def contains_normalized(haystack: str, needle: str) -> bool:
    hs = normalize_text(haystack)
    nd = normalize_text(needle)
    if not nd:
        return False
    return nd in hs
